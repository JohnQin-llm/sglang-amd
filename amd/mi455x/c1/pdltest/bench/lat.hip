// PDL-style release on gfx1250: release latency, stream-wait cost, graph-setting cost (one binary, env picks setting).
//   lat calib        : are wall clocks comparable across blocks/XCDs? (all blocks read the clock right after a barrier)
//   lat release K    : flag write -> consumer first instruction; K = 0 sys-release store, 1 agent-release, 2 relaxed
//                      store to uncached memory. eager and graph.
//   lat waitcost     : per-kernel time of a chain of tiny kernels: plain / + satisfied hipStreamWaitValue32 /
//                      + cross-stream event ping-pong, eager and graph.
//   lat graph TAIL   : decode-like graph, 62 layers x (10 small kernels + K2-shaped persistent kernel + QKV-shaped
//                      kernel), linear vs PDL topology (QKV on a 2nd stream behind the K2 trigger flag). TAIL = number
//                      of K2 CTAs still running during the last hand-off (the others exit at the trigger).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <unistd.h>

#define CHECK(x) do { hipError_t err_ = (x); if (err_ != hipSuccess) { printf("ERR %s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(err_)); exit(1); } } while (0)
typedef unsigned long long u64;
__device__ inline u64 now() { return wall_clock64(); }
__device__ inline unsigned ldr(unsigned* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ inline unsigned lda(unsigned* p) { return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT); }
static double TICK_US;
static u64 us2t(double us) { return (u64)(us / TICK_US); }

// ---------------- calib ----------------
__global__ void calib(unsigned* arrive, u64* t, int nb) {
  if (threadIdx.x == 0) {
    atomicAdd(arrive, 1u);
    while (ldr(arrive) < (unsigned)nb) {}
    t[blockIdx.x] = now();
  }
}

// ---------------- release ----------------
__global__ void rel_producer(unsigned* flag, unsigned val, int kind, u64 pre, u64 post, u64* tw_out, u64* prog) {
  if (threadIdx.x) return;
  u64 t0 = now();
  if (blockIdx.x == 0) {
    while (now() - t0 < pre) {}
    u64 tw = now();
    if (kind == 0) __hip_atomic_store(flag, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    else if (kind == 1) __hip_atomic_store(flag, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    else __hip_atomic_store(flag, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    *tw_out = tw;
    u64 t;
    while ((t = now()) - tw < post) __hip_atomic_store(prog, t - tw, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } else {
    while (now() - t0 < pre + post) {}
  }
}
__global__ void reset_flag(unsigned* f) { if (threadIdx.x == 0) __hip_atomic_store(f, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
__global__ void rel_consumer(u64* tc, u64* pc, u64* prog, unsigned* flag, int reset) {
  if (threadIdx.x) return;
  u64 t = now();
  pc[blockIdx.x] = __hip_atomic_load(prog, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  tc[blockIdx.x] = t;
  if (reset && blockIdx.x == 0) __hip_atomic_store(flag, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// ---------------- waitcost ----------------
__global__ void tiny(unsigned* c) { if (threadIdx.x == 0 && blockIdx.x == 0) c[0]++; }

// ---------------- graph ----------------
#define VG(n) asm volatile("v_mov_b32 v" #n ", 0" ::: "v" #n);
__global__ __launch_bounds__(256) void small_k(u64 iters, float* scratch) {
  float a = threadIdx.x, b = 0.999f;
  for (u64 i = 0; i < iters; ++i) {
#pragma unroll 16
    for (int j = 0; j < 16; ++j) a = a * b + 0.5f;
  }
  if (a == 12345.f) scratch[blockIdx.x] = a;
}
// K2-shaped: 256 x 512, 145 KB LDS, 251 VGPRs. Waits for all CTAs to be resident, runs `body` ticks, then the
// trigger (flag=1, after all CTAs started); CTAs >= tail exit, CTAs < tail run the last hand-off (`hand` ticks).
// Last CTA out bumps done[0].
__device__ unsigned g_k2guard;
template <int V>
__global__ __launch_bounds__(512) void k2_k(unsigned* started, unsigned* flag, unsigned* done, u64 body, u64 hand, int tail, int np) {
  extern __shared__ float lds[];
  if constexpr (V == 251) { VG(250) } else { VG(223) }
  if (threadIdx.x == 0) {
    atomicAdd(started, 1u);
    u64 t = now();
    while (ldr(started) < (unsigned)np) { if (now() - t > 100000000ull) { atomicAdd(&g_k2guard, 1u); break; } }
    while (now() - t < body) __builtin_amdgcn_s_sleep(1);
    if (blockIdx.x == 0) __hip_atomic_store(flag, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    if ((int)blockIdx.x < tail) { u64 t1 = now(); while (now() - t1 < hand) __builtin_amdgcn_s_sleep(1); }
    lds[0] = 1;
    if (__hip_atomic_fetch_add(done, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT) == (unsigned)np - 1) {
      *started = 0;
    }
  }
}
// QKV-shaped: 576 x 128, 43.5 KB LDS, 104 VGPRs. mode 0: read own weight slice once (the projection).
// mode 1 (PDL): read slice (prefetch) -> wait until K2 done == np -> read slice again (warm) -> reset flag/done.
__device__ unsigned g_guard;
__device__ unsigned g_ovl[64];
__device__ inline void wait_done(unsigned* done, int np) {
  if (threadIdx.x == 0) {
    u64 t = now(); bool to = false;
    while (ldr(done) < (unsigned)np) { __builtin_amdgcn_s_sleep(1); if (now() - t > 2000000ull) { to = true; break; } }
    if (to) atomicAdd(&g_guard, 1u);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
  }
  __syncthreads();
}
// QKV-shaped: 576 x 128, 43.5 KB LDS, 104 VGPRs. mode 0: read own weight slice once (the projection).
// 1: prefetch-read the first pf/8 of the slice -> wait until K2 done == np -> read the whole slice.
// 2: early launch, wait K2 done, then the single read.
__global__ __launch_bounds__(128) void qkv_k(const uint4* w, size_t per_cta, unsigned* flag, unsigned* done, int np, int mode, unsigned* sink, int pf) {
  extern __shared__ float lds[];
  VG(103)
  const uint4* p = w + blockIdx.x * per_cta;
  unsigned acc = 0;
  if (threadIdx.x == 0) atomicMin(&g_ovl[pf >> 8], ldr(done));
  pf &= 255;
  if (mode == 1) {
    size_t n = per_cta * pf / 8;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) { uint4 v = p[i]; acc ^= v.z; }
  }
  if (mode >= 1) wait_done(done, np);
  for (size_t i = threadIdx.x; i < per_cta; i += blockDim.x) { uint4 v = p[i]; acc ^= v.x ^ v.w; }
  lds[threadIdx.x] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    if (lds[5] == 1234.5f) sink[blockIdx.x] = acc;
    if (atomicAdd(sink + 4096, 1u) == gridDim.x - 1) { sink[4096] = 0; if (flag) { *flag = 0; *done = 0; } }
  }
}
__global__ void reset_layer(unsigned* flag, unsigned* done) {
  if (threadIdx.x == 0) { __hip_atomic_store(flag, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); __hip_atomic_store(done, 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); }
}
static int g_proto = 0;  // 0: QKV resets flag/done at its end (plain stores); 1: atomic reset kernel before the fork
__global__ void rescue(unsigned* f, int n) { if (threadIdx.x < n) __hip_atomic_store(f + threadIdx.x, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
static unsigned* g_flags = nullptr; static int g_nflags = 0; static hipStream_t g_s3; static int g_deadlocks = 0;
static void host_wait(hipEvent_t b) {  // poll; if stuck 3 s, a wait-value node is ordered ahead of its producer: unblock it
  for (int i = 0; ; ++i) {
    hipError_t q = hipEventQuery(b);
    if (q == hipSuccess) return;
    if (q != hipErrorNotReady) { printf("ERR event query %s\n", hipGetErrorString(q)); exit(1); }
    if (i == 3000 && g_flags) { g_deadlocks++; printf("DEADLOCK: replay stuck 3 s, releasing flags from a third stream\n"); fflush(stdout); rescue<<<1, 64, 0, g_s3>>>(g_flags, g_nflags); }
    if (i > 10000) { printf("ERR stuck even after rescue\n"); exit(2); }
    usleep(1000);
  }
}
static double replay_us(hipGraphExec_t e, hipStream_t s, int reps = 15) {
  hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
  CHECK(hipGraphLaunch(e, s)); CHECK(hipEventRecord(b, s)); host_wait(b);
  std::vector<double> v;
  for (int i = 0; i < reps; ++i) {
    CHECK(hipEventRecord(a, s)); CHECK(hipGraphLaunch(e, s)); CHECK(hipEventRecord(b, s)); host_wait(b);
    float ms; CHECK(hipEventElapsedTime(&ms, a, b)); v.push_back(ms * 1000);
  }
  std::sort(v.begin(), v.end()); return v[v.size() / 2];
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "calib";
  int rate; CHECK(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0)); TICK_US = 1000.0 / rate;
  hipStream_t s1, s2; CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking)); CHECK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
  hipEvent_t ev[4]; for (auto& e : ev) CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
  const char* env[] = {"DEBUG_HIP_GRAPH_CLASSIC_PATH", "DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING", "DEBUG_HIP_FORCE_GRAPH_QUEUES"};
  char setting[128] = ""; for (auto n : env) if (getenv(n)) { strcat(setting, n + 16); strcat(setting, "="); strcat(setting, getenv(n)); strcat(setting, " "); }
  if (!setting[0]) strcpy(setting, "default");

  if (!strcmp(mode, "calib")) {
    int nb = 64; unsigned* arr; u64* t; CHECK(hipMalloc(&arr, 4)); CHECK(hipMalloc(&t, nb * 8));
    std::vector<double> off(8, 0);
    for (int it = 0; it < 5; ++it) {
      CHECK(hipMemset(arr, 0, 4)); calib<<<nb, 64>>>(arr, t, nb); CHECK(hipDeviceSynchronize());
      std::vector<u64> h(nb); CHECK(hipMemcpy(h.data(), t, nb * 8, hipMemcpyDeviceToHost));
      printf("RESULT calib it%d: clock(block b) - clock(block 0) us by b%%8 (median over b):", it);
      for (int c = 0; c < 8; ++c) { std::vector<double> d; for (int b = c; b < nb; b += 8) d.push_back(((double)(long long)(h[b] - h[0])) * TICK_US); std::sort(d.begin(), d.end()); printf(" %+.2f", d[d.size() / 2]); }
      double mn = 1e9, mx = -1e9; for (int b = 0; b < nb; ++b) { double d = (double)(long long)(h[b] - h[0]) * TICK_US; mn = std::min(mn, d); mx = std::max(mx, d); }
      printf(" | all-block spread %.2f..%.2f\n", mn, mx);
    }
    return 0;
  }

  if (!strcmp(mode, "release")) {
    int kind = argc > 2 ? atoi(argv[2]) : 0;
    int nbp = 64, nbc = 64;
    unsigned* flag; u64 *tw, *tc, *pc, *prog;
    if (kind == 2) CHECK(hipExtMallocWithFlags((void**)&flag, 4, hipDeviceMallocUncached)); else CHECK(hipMalloc(&flag, 4));
    CHECK(hipMalloc(&tw, 8)); CHECK(hipMalloc(&tc, nbc * 8)); CHECK(hipMalloc(&pc, nbc * 8)); CHECK(hipMalloc(&prog, 8));
    CHECK(hipMemset(flag, 0, 4));
    u64 pre = us2t(60), post = us2t(60);
    for (int g = 0; g < 2; ++g) {
      hipGraphExec_t ex = nullptr;
      if (g) {
        hipGraph_t gr;
        CHECK(hipMemset(flag, 0, 4)); CHECK(hipDeviceSynchronize());
        CHECK(hipStreamBeginCapture(s1, hipStreamCaptureModeGlobal));
        reset_flag<<<1, 64, 0, s1>>>(flag);
        CHECK(hipEventRecord(ev[0], s1)); CHECK(hipStreamWaitEvent(s2, ev[0], 0));
        rel_producer<<<nbp, 64, 0, s1>>>(flag, 1, kind, pre, post, tw, prog);
        CHECK(hipStreamWaitValue32(s2, flag, 1, hipStreamWaitValueGte, 0xffffffff));
        rel_consumer<<<nbc, 64, 0, s2>>>(tc, pc, prog, flag, 0);
        CHECK(hipEventRecord(ev[1], s2)); CHECK(hipStreamWaitEvent(s1, ev[1], 0));
        CHECK(hipStreamEndCapture(s1, &gr)); CHECK(hipGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
      }
      std::vector<double> lc, lp, lall;
      for (int it = 0; it < 12; ++it) {
        unsigned val = g ? 1 : 1000 * (kind + 1) + it + 1;
        CHECK(hipMemset(prog, 0, 8)); CHECK(hipMemset(tc, 0, nbc * 8)); CHECK(hipDeviceSynchronize());
        if (g) CHECK(hipGraphLaunch(ex, s1));
        else {
          rel_producer<<<nbp, 64, 0, s1>>>(flag, val, kind, pre, post, tw, prog);
          CHECK(hipStreamWaitValue32(s2, flag, val, hipStreamWaitValueGte, 0xffffffff));
          rel_consumer<<<nbc, 64, 0, s2>>>(tc, pc, prog, flag, 0);
        }
        CHECK(hipDeviceSynchronize());
        u64 htw; std::vector<u64> htc(nbc), hpc(nbc);
        CHECK(hipMemcpy(&htw, tw, 8, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(htc.data(), tc, nbc * 8, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(hpc.data(), pc, nbc * 8, hipMemcpyDeviceToHost));
        if (it < 2) continue;
        double mc = 1e9, ma = 1e9, mp = 1e9;
        for (int b = 0; b < nbc; ++b) {
          double d = (double)(long long)(htc[b] - htw) * TICK_US;
          ma = std::min(ma, d); if (b % 8 == 0) mc = std::min(mc, d);
          mp = std::min(mp, hpc[b] * TICK_US);
        }
        lc.push_back(mc); lall.push_back(ma); lp.push_back(mp);
      }
      auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
      auto mn = [](std::vector<double> v) { return *std::min_element(v.begin(), v.end()); };
      auto mx = [](std::vector<double> v) { return *std::max_element(v.begin(), v.end()); };
      int before = 0; for (double v : lp) before += v == 0;
      printf("RESULT release kind=%d (%s) %s [%s]: flag write -> first consumer instr, same-XCD-class clock: median %.2f us (min %.2f max %.2f) | "
             "all-block clock min %.2f | progress counter (lower bound): median %.2f us (min %.2f) | consumer started BEFORE the flag write in %d/%zu\n",
             kind, kind == 0 ? "sys-release" : kind == 1 ? "agent-release" : "relaxed/uncached", g ? "graph" : "eager", setting,
             med(lc), mn(lc), mx(lc), med(lall), med(lp), mn(lp), before, lp.size());
    }
    return 0;
  }

  if (!strcmp(mode, "waitcost")) {
    int N = 400; unsigned *c, *flag; CHECK(hipMalloc(&c, 4)); CHECK(hipMalloc(&flag, 4)); CHECK(hipMemset(flag, 0, 4));
    for (int variant = 0; variant < 3; ++variant) {
      auto body = [&](hipStream_t a, hipStream_t b) {
        for (int i = 0; i < N; ++i) {
          if (variant == 0) tiny<<<1, 64, 0, a>>>(c);
          else if (variant == 1) { CHECK(hipStreamWaitValue32(a, flag, 0, hipStreamWaitValueGte, 0xffffffff)); tiny<<<1, 64, 0, a>>>(c); }
          else {  // ping-pong: alternate streams with an event edge each step
            hipStream_t x = (i & 1) ? b : a, y = (i & 1) ? a : b;
            CHECK(hipEventRecord(ev[i & 1], y)); CHECK(hipStreamWaitEvent(x, ev[i & 1], 0)); tiny<<<1, 64, 0, x>>>(c);
          }
        }
      };
      // eager
      hipEvent_t ta, tb; CHECK(hipEventCreate(&ta)); CHECK(hipEventCreate(&tb));
      std::vector<double> ve;
      for (int r = 0; r < 6; ++r) {
        CHECK(hipDeviceSynchronize()); CHECK(hipEventRecord(ta, s1)); body(s1, s2);
        CHECK(hipEventRecord(ev[2], s2)); CHECK(hipStreamWaitEvent(s1, ev[2], 0)); CHECK(hipEventRecord(tb, s1)); CHECK(hipDeviceSynchronize());
        float ms; CHECK(hipEventElapsedTime(&ms, ta, tb)); if (r) ve.push_back(ms * 1000 / N);
      }
      std::sort(ve.begin(), ve.end());
      hipGraph_t gr; hipGraphExec_t ex;
      CHECK(hipStreamBeginCapture(s1, hipStreamCaptureModeGlobal));
      CHECK(hipEventRecord(ev[3], s1)); CHECK(hipStreamWaitEvent(s2, ev[3], 0));
      body(s1, s2);
      CHECK(hipEventRecord(ev[2], s2)); CHECK(hipStreamWaitEvent(s1, ev[2], 0));
      CHECK(hipStreamEndCapture(s1, &gr)); CHECK(hipGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
      double gt = replay_us(ex, s1) / N;
      printf("RESULT waitcost [%s] %s: eager %.2f us/kernel, graph %.2f us/kernel\n", setting,
             variant == 0 ? "plain chain" : variant == 1 ? "chain + satisfied hipStreamWaitValue32 per kernel" : "cross-stream event ping-pong", ve[ve.size() / 2], gt);
    }
    return 0;
  }

  if (!strcmp(mode, "chain")) {  // per-kernel cost of a dependent chain: blocks x spin
    float* scratch; CHECK(hipMalloc(&scratch, 4096 * 4));
    int N = 600;
    int bl[] = {1, 64, 256, 1024}; int sp[] = {0, 60};
    for (int b : bl) for (int it : sp) {
      u64 t = it; double us = 0;
      hipGraph_t gr; hipGraphExec_t ex;
      CHECK(hipStreamBeginCapture(s1, hipStreamCaptureModeGlobal));
      for (int i = 0; i < N; ++i) small_k<<<b, 256, 0, s1>>>(t, scratch);
      CHECK(hipStreamEndCapture(s1, &gr)); CHECK(hipGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
      double g = replay_us(ex, s1) / N;
      hipEvent_t a, c; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&c));
      std::vector<double> v;
      for (int r = 0; r < 5; ++r) { CHECK(hipEventRecord(a, s1)); for (int i = 0; i < N; ++i) small_k<<<b, 256, 0, s1>>>(t, scratch);
        CHECK(hipEventRecord(c, s1)); CHECK(hipStreamSynchronize(s1)); float ms; CHECK(hipEventElapsedTime(&ms, a, c)); if (r) v.push_back(ms * 1000 / N); }
      std::sort(v.begin(), v.end());
      printf("RESULT chain [%s] %4d blocks x256, alu iters %d: graph %.2f us/kernel, eager %.2f us/kernel\n", setting, b, it, g, v[v.size() / 2]);
      CHECK(hipGraphExecDestroy(ex)); CHECK(hipGraphDestroy(gr));
    }
    return 0;
  }

  if (!strcmp(mode, "graph")) {
    int tail = argc > 2 ? atoi(argv[2]) : 256;
    double body_us = argc > 3 ? atof(argv[3]) : 40, hand_us = argc > 4 ? atof(argv[4]) : 4;
    int k2v = argc > 5 ? atoi(argv[5]) : 251;
    g_proto = argc > 6 ? atoi(argv[6]) : 0;
    void (*k2f)(unsigned*, unsigned*, unsigned*, u64, u64, int, int) = k2v == 251 ? k2_k<251> : k2_k<224>;
    int L = 62, S = 10, NP = 256, NQ = 576;
    size_t wbytes = (size_t)4608 * 8192 * 2, per_cta = wbytes / 16 / NQ;
    std::vector<uint4*> W(L);
    for (auto& w : W) { CHECK(hipMalloc(&w, wbytes)); CHECK(hipMemset(w, 1, wbytes)); }
    unsigned *started, *flags, *dones, *sink; float* scratch;
    CHECK(hipMalloc(&started, 4 * L)); CHECK(hipMalloc(&flags, 4 * L)); CHECK(hipMalloc(&dones, 4 * L)); CHECK(hipMalloc(&sink, 4 * 8192)); CHECK(hipMalloc(&scratch, 4096));
    g_flags = flags; g_nflags = L; CHECK(hipStreamCreateWithPriority(&g_s3, hipStreamNonBlocking, -1));
    CHECK(hipMemset(started, 0, 4 * L)); CHECK(hipMemset(flags, 0, 4 * L)); CHECK(hipMemset(dones, 0, 4 * L)); CHECK(hipMemset(sink, 0, 4 * 8192));
    CHECK(hipFuncSetAttribute((void*)k2_k<251>, hipFuncAttributeMaxDynamicSharedMemorySize, 145 * 1024));
    CHECK(hipFuncSetAttribute((void*)k2_k<224>, hipFuncAttributeMaxDynamicSharedMemorySize, 145 * 1024));
    CHECK(hipFuncSetAttribute((void*)qkv_k, hipFuncAttributeMaxDynamicSharedMemorySize, 43 * 1024));
    u64 small_t = 60; double small_us = 2.0;  // ~2 us of ALU per small kernel (64 x 256)
    u64 body = us2t(body_us), hand = us2t(hand_us);
    std::vector<hipEvent_t> evs(4 * L + 4); for (auto& e : evs) CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
    // 0 small chain only; 1 linear; 2 two-stream fork/join, QKV on s2 after K2 (event, no wait value);
    // 3 PDL: QKV on s2 behind the trigger flag, prefetch its weights, wait K2 done, read again (warm);
    // 4 PDL without prefetch: early launch, wait K2 done, then the single read
    const char* vname[] = {"small-chain-only", "linear", "2stream-event", "PDL+prefetch-all", "PDL-noprefetch", "PDL-nowaitdone(unsafe)",
                           "linear+waitdone", "PDL+prefetch-1/4"};
    for (int variant = 0; variant < 8; ++variant) {
      hipGraph_t gr; hipGraphExec_t ex;
      CHECK(hipStreamBeginCapture(s1, hipStreamCaptureModeGlobal));
      int nk = 0;
      for (int l = 0; l < L; ++l) {
        for (int i = 0; i < S; ++i) { small_k<<<64, 256, 0, s1>>>(small_t, scratch); nk++; }
        if (variant == 0) continue;
        if (g_proto) reset_layer<<<1, 64, 0, s1>>>(flags + l, dones + l);
        if (variant >= 3 && variant != 6) { CHECK(hipEventRecord(evs[2 * l], s1)); CHECK(hipStreamWaitEvent(s2, evs[2 * l], 0)); }
        hipLaunchKernelGGL(k2f, dim3(NP), dim3(512), 145 * 1024, s1, started + l, flags + l, dones + l, body, hand, tail, NP); nk++;
        if (variant == 1 || variant == 6) { qkv_k<<<NQ, 128, 43 * 1024, s1>>>(W[l], per_cta, flags + l, dones + l, NP, variant == 6 ? 2 : 0, sink, l << 8); nk++; }
        else if (variant == 2) {
          CHECK(hipEventRecord(evs[2 * l], s1)); CHECK(hipStreamWaitEvent(s2, evs[2 * l], 0));
          qkv_k<<<NQ, 128, 43 * 1024, s2>>>(W[l], per_cta, flags + l, dones + l, NP, 0, sink, l << 8); nk++;
          CHECK(hipEventRecord(evs[2 * l + 1], s2)); CHECK(hipStreamWaitEvent(s1, evs[2 * l + 1], 0));
        } else {
          CHECK(hipStreamWaitValue32(s2, flags + l, 1, hipStreamWaitValueGte, 0xffffffff));
          qkv_k<<<NQ, 128, 43 * 1024, s2>>>(W[l], per_cta, g_proto ? nullptr : flags + l, dones + l, NP, variant == 3 || variant == 7 ? 1 : variant == 4 ? 2 : 0, sink, (l << 8) | (variant == 7 ? 2 : 8)); nk++;
          CHECK(hipEventRecord(evs[2 * l + 1], s2)); CHECK(hipStreamWaitEvent(s1, evs[2 * l + 1], 0));
        }
      }
      CHECK(hipStreamEndCapture(s1, &gr)); CHECK(hipGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
      unsigned zero[64] = {0}, ff[64]; memset(ff, 0xff, sizeof(ff));
      CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_guard), zero, 4)); CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_k2guard), zero, 4));
      double t = replay_us(ex, s1, 10);
      unsigned guard, k2guard; CHECK(hipMemcpyFromSymbol(&guard, HIP_SYMBOL(g_guard), 4)); CHECK(hipMemcpyFromSymbol(&k2guard, HIP_SYMBOL(g_k2guard), 4));
      CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_ovl), ff, sizeof(ff)));
      CHECK(hipGraphLaunch(ex, s1)); CHECK(hipEventRecord(ev[3], s1)); host_wait(ev[3]);
      unsigned ovl[64]; CHECK(hipMemcpyFromSymbol(ovl, HIP_SYMBOL(g_ovl), sizeof(ovl)));
      int early = 0; double dsum = 0; for (int l = 0; l < L; ++l) if (ovl[l] < (unsigned)NP) { early++; dsum += ovl[l]; }
      double work = L * S * small_us + (variant ? L * (body_us + hand_us) : 0);
      printf("RESULT graph [%s] k2vgpr=%d tail=%d body=%.0f hand=%.0f %s: %d kernels, replay %.1f us, per layer %.2f us, "
             "(replay - spin work)/kernel %.2f us | deadlock rescues %d | QKV started before K2 finished in %d/%d layers (mean K2 CTAs done then %.0f) | done-wait guard timeouts %u | K2 residency-barrier timeouts (CTAs) %u | reset protocol %s\n", setting, k2v, tail, body_us, hand_us,
             vname[variant], nk, t, t / L, (t - work) / nk, g_deadlocks, variant ? early : 0, L, early ? dsum / early : 0.0, guard, k2guard, g_proto ? "atomic-before-fork" : "plain-at-QKV-end");
      CHECK(hipGraphExecDestroy(ex)); CHECK(hipGraphDestroy(gr));
    }
    // eager reference of the linear layer chain
    return 0;
  }
  printf("unknown mode\n");
  return 1;
}
