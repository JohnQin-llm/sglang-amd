// S0 XGMI sync benchmarks for the TP4/EP1 K2 on 4x MI455X (gfx1250).
//
//  1. pair ping-pong variants: push (store to peer, poll own memory) vs pull (store own, poll peer);
//     memory kinds uncached / fine-grained / coarse; s_sleep(0) vs busy poll.
//  2. 4-rank barrier: each rank writes epoch into its slot of every rank's flag line, polls its own 4 slots.
//     Rounds back to back; us per barrier = slowest rank time / rounds.
//  3. tagged-mailbox exchange (the K2 hand-off): every rank sends P dwords to each of the 3 peers as 64-bit
//     {value, tag} pairs (tag = epoch+1), double-buffered by epoch parity, then waits for all tags of its own
//     inbox. G CTAs x 256 threads per rank. Reports us per exchange for several (P, G).
//  4. device-scope floor: same-GPU two-CTA ping-pong.
//
// Build: hipcc -O3 --offload-arch=gfx1250 xgmi_bench.hip -o xgmi_bench
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

#define TMO 200000000LL
#ifndef RSLEEP
#define RSLEEP 0
#endif
__device__ int g_timeout = 0;
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); exit(1);} } while (0)

static int g_rate_khz = 100000;
static FILE* g_json = nullptr;
static bool g_first = true;
static void jrec(const std::string& name, double us) {
  if (!g_json) return;
  fprintf(g_json, "%s\n  \"%s\": %.4f", g_first ? "" : ",", name.c_str(), us);
  g_first = false;
  fflush(g_json);
}

// ---------------------------------------------------------------- 1. pairs
template <bool SLEEP, int ORD = __ATOMIC_ACQUIRE, int ORDS = __ATOMIC_RELEASE>
__global__ void player(int* poll_flag, int* store_flag, int pos, int n, int rounds, long long* t) {
  long long t0 = wall_clock64();
  for (int r = 0; r < rounds; ++r) {
    int want = r * n + pos;
    if (!(r == 0 && pos == 0)) {
      long long ts = wall_clock64();
      while (__hip_atomic_load(poll_flag, ORD, __HIP_MEMORY_SCOPE_SYSTEM) < want) {
        if (SLEEP) __builtin_amdgcn_s_sleep(0);
        if (wall_clock64() - ts > TMO) { atomicAdd(&g_timeout, 1); t[0] = -1; return; }
      }
    }
    __hip_atomic_store(store_flag, want + 1, ORDS, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if (pos == 0) {
    long long ts = wall_clock64();
    while (__hip_atomic_load(poll_flag, ORD, __HIP_MEMORY_SCOPE_SYSTEM) < rounds * n) {
      if (SLEEP) __builtin_amdgcn_s_sleep(0);
      if (wall_clock64() - ts > TMO) { t[0] = -1; return; }
    }
  }
  t[0] = wall_clock64() - t0;
}

struct Bufs { std::vector<int*> f; };

static Bufs alloc_flags(unsigned flags_kind, size_t bytes) {
  Bufs b; b.f.resize(4);
  for (int g = 0; g < 4; ++g) {
    CK(hipSetDevice(g));
    if (flags_kind == 0xFFFFFFFFu) CK(hipMalloc((void**)&b.f[g], bytes));
    else CK(hipExtMallocWithFlags((void**)&b.f[g], bytes, flags_kind));
    CK(hipMemset(b.f[g], 0, bytes));
  }
  return b;
}
static void free_flags(Bufs& b) { for (int g = 0; g < 4; ++g) { CK(hipSetDevice(g)); CK(hipFree(b.f[g])); } }

// push: pos i polls its own flag, stores into next's flag.  pull: pos i stores into its own "out" word,
// polls prev's out word (remote read).
static double run_pair(int a, int b, Bufs& fl, int rounds, bool pull, bool sleep, bool relaxed = false) {
  int gpus[2] = {a, b};
  long long* t[2];
  for (int i = 0; i < 2; ++i) { CK(hipSetDevice(gpus[i])); CK(hipMemset(fl.f[gpus[i]], 0, 256)); CK(hipMalloc(&t[i], 8)); CK(hipDeviceSynchronize()); }
  for (int i = 1; i >= 0; --i) {
    CK(hipSetDevice(gpus[i]));
    int* poll = pull ? fl.f[gpus[(i + 1) % 2]] : fl.f[gpus[i]];
    int* store = pull ? fl.f[gpus[i]] : fl.f[gpus[(i + 1) % 2]];
    if (relaxed) hipLaunchKernelGGL((player<true, __ATOMIC_RELAXED, __ATOMIC_RELAXED>), dim3(1), dim3(1), 0, 0, poll, store, i, 2, rounds, t[i]);
    else if (sleep) hipLaunchKernelGGL(player<true>, dim3(1), dim3(1), 0, 0, poll, store, i, 2, rounds, t[i]);
    else hipLaunchKernelGGL(player<false>, dim3(1), dim3(1), 0, 0, poll, store, i, 2, rounds, t[i]);
  }
  for (int i = 0; i < 2; ++i) { CK(hipSetDevice(gpus[i])); CK(hipDeviceSynchronize()); }
  long long ticks; CK(hipSetDevice(a)); CK(hipMemcpy(&ticks, t[0], 8, hipMemcpyDeviceToHost));
  for (int i = 0; i < 2; ++i) { CK(hipSetDevice(gpus[i])); CK(hipFree(t[i])); }
  return ticks * 1e3 / g_rate_khz / (2.0 * rounds);
}

// same GPU, 2 CTAs
static double run_self(int g, Bufs& fl, int rounds) {
  CK(hipSetDevice(g));
  long long* t; CK(hipMalloc(&t, 16));
  CK(hipMemset(fl.f[g], 0, 256));
  int* f0 = fl.f[g]; int* f1 = fl.f[g] + 32;
  hipStream_t s1, s2; CK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking)); CK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
  hipLaunchKernelGGL(player<true>, dim3(1), dim3(1), 0, s1, f1, f0, 1, 2, rounds, t + 1);
  hipLaunchKernelGGL(player<true>, dim3(1), dim3(1), 0, s2, f0, f1, 0, 2, rounds, t);
  CK(hipDeviceSynchronize());
  CK(hipStreamDestroy(s1)); CK(hipStreamDestroy(s2));
  long long ticks; CK(hipMemcpy(&ticks, t, 8, hipMemcpyDeviceToHost));
  CK(hipFree(t));
  return ticks * 1e3 / g_rate_khz / (2.0 * rounds);
}

// ---------------------------------------------------------------- 2. 4-rank barrier
// line layout per rank: 4 slots, 64 B apart (slot s written by rank s).
__global__ void barrier_kernel(int* const* lines, int me, int rounds, long long* t) {
  int lane = threadIdx.x;  // 4 threads: thread p stores to rank p's line, polls slot p of own line
  int* mine = lines[me];
  long long t0 = wall_clock64();
  for (int r = 1; r <= rounds; ++r) {
    if (lane < 4) {
      __hip_atomic_store(lines[lane] + me * 16, r, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      long long ts = wall_clock64();
      while (__hip_atomic_load(mine + lane * 16, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < r) {
        __builtin_amdgcn_s_sleep(0);
        if (wall_clock64() - ts > TMO) { t[0] = -1; return; }
      }
    }
    __syncthreads();
  }
  if (lane == 0) t[0] = wall_clock64() - t0;
}

static double run_barrier(Bufs& fl, int rounds) {
  int** dlines[4]; long long* t[4];
  for (int g = 0; g < 4; ++g) {
    CK(hipSetDevice(g)); CK(hipMemset(fl.f[g], 0, 256));
    CK(hipMalloc(&dlines[g], 4 * sizeof(int*)));
    CK(hipMemcpy(dlines[g], fl.f.data(), 4 * sizeof(int*), hipMemcpyHostToDevice));
    CK(hipMalloc(&t[g], 8)); CK(hipDeviceSynchronize());
  }
  for (int g = 0; g < 4; ++g) { CK(hipSetDevice(g)); hipLaunchKernelGGL(barrier_kernel, dim3(1), dim3(32), 0, 0, dlines[g], g, rounds, t[g]); }
  double worst = 0;
  for (int g = 0; g < 4; ++g) {
    CK(hipSetDevice(g)); CK(hipDeviceSynchronize());
    long long ticks; CK(hipMemcpy(&ticks, t[g], 8, hipMemcpyDeviceToHost));
    worst = std::max(worst, ticks * 1e3 / g_rate_khz / rounds);
    CK(hipFree(t[g])); CK(hipFree(dlines[g]));
  }
  return worst;
}

// ---------------------------------------------------------------- 3. tagged mailbox exchange
// inbox per rank: [parity 2][src 4][P] uint64 {lo=value, hi=tag}.  CTA c of G handles dwords c*chunk..
__global__ void exchange_kernel(uint64_t* const* inbox, int me, int P, int rounds, int intra_barrier,
                                unsigned* gbar, long long* t, unsigned* bad) {
  const int G = gridDim.x, c = blockIdx.x, tid = threadIdx.x, T = blockDim.x;
  const int chunk = (P + G - 1) / G;
  const int lo = c * chunk, hi = min(P, lo + chunk);
  uint64_t* mine = inbox[me];
  long long t0 = wall_clock64();
  unsigned nbad = 0;
  for (int r = 0; r < rounds; ++r) {
    const unsigned tag = (unsigned)r + 1;
    const int par = r & 1;
    // send: to every peer (and self) write my P dwords into its inbox slot [par][me]
    for (int dst = 0; dst < 4; ++dst) {
      uint64_t* box = inbox[dst] + ((size_t)par * 4 + me) * P;
      for (int i = lo + tid; i < hi; i += T) {
        unsigned v = (unsigned)(me * 1000003 + i * 7 + r);
        uint64_t w = ((uint64_t)tag << 32) | v;
        __hip_atomic_store(box + i, w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
    // receive: poll own inbox [par][src] for my chunk until all tags match
    for (int src = 0; src < 4; ++src) {
      uint64_t* box = mine + ((size_t)par * 4 + src) * P;
      for (int i = lo + tid; i < hi; i += T) {
        uint64_t w;
        long long ts = wall_clock64();
        do { w = __hip_atomic_load(box + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
             if (RSLEEP && (unsigned)(w >> 32) != tag) __builtin_amdgcn_s_sleep(RSLEEP);
             if (wall_clock64() - ts > TMO) { t[0] = -1; atomicAdd(bad + 1, 1u); return; } } while ((unsigned)(w >> 32) != tag);
        unsigned v = (unsigned)(src * 1000003 + i * 7 + r);
        if ((unsigned)w != v) { ++nbad; if (atomicAdd(bad + 3, 1u) < 8) { unsigned k = atomicAdd(bad + 4, 1u); if (k < 8) { bad[8 + k * 4] = src; bad[9 + k * 4] = i; bad[10 + k * 4] = r; bad[11 + k * 4] = (unsigned)w - v; } } }
      }
    }
    if (intra_barrier) {  // grid-wide fan-in (device scope), as K2 does between stages
      __syncthreads();
      if (tid == 0) {
        __hip_atomic_fetch_add(gbar, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        unsigned want = (unsigned)(r + 1) * G;
        long long ts = wall_clock64();
        while (__hip_atomic_load(gbar, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < want) {
          __builtin_amdgcn_s_sleep(0);
          if (wall_clock64() - ts > TMO) { t[0] = -1; atomicAdd(bad + 2, 1u); break; }
        }
      }
      __syncthreads();
    }
  }
  __syncthreads();
  if (tid == 0 && c == 0) t[0] = wall_clock64() - t0;
  if (nbad) atomicAdd(bad, nbad);
}

static double run_exchange(Bufs& fl, int P, int G, int rounds, int intra, unsigned* nbad_out) {
  uint64_t** dbox[4]; long long* t[4]; unsigned* bad[4]; unsigned* gbar[4];
  size_t bytes = (size_t)2 * 4 * P * 8;
  for (int g = 0; g < 4; ++g) {
    CK(hipSetDevice(g)); CK(hipMemset(fl.f[g], 0, bytes));
    CK(hipMalloc(&dbox[g], 4 * sizeof(void*)));
    CK(hipMemcpy(dbox[g], fl.f.data(), 4 * sizeof(void*), hipMemcpyHostToDevice));
    CK(hipMalloc(&t[g], 8)); CK(hipMalloc(&bad[g], 256)); CK(hipMemset(bad[g], 0, 256));
    CK(hipMalloc(&gbar[g], 4)); CK(hipMemset(gbar[g], 0, 4));
    CK(hipDeviceSynchronize());
  }
  for (int g = 0; g < 4; ++g) {
    CK(hipSetDevice(g));
    hipLaunchKernelGGL(exchange_kernel, dim3(G), dim3(256), 0, 0, dbox[g], g, P, rounds, intra, gbar[g], t[g], bad[g]);
  }
  double worst = 0; unsigned nb = 0;
  for (int g = 0; g < 4; ++g) {
    CK(hipSetDevice(g)); CK(hipDeviceSynchronize());
    long long ticks; unsigned b[64];
    CK(hipMemcpy(&ticks, t[g], 8, hipMemcpyDeviceToHost)); CK(hipMemcpy(b, bad[g], 256, hipMemcpyDeviceToHost));
    nb += b[0] + b[1] + b[2];
    if (b[0] + b[1] + b[2]) {
      printf("    rank %d: value-mismatch %u, recv-timeouts %u, barrier-timeouts %u\n", g, b[0], b[1], b[2]);
      for (int k = 0; k < 8 && k < (int)b[4]; ++k) printf("      bad src=%u i=%u r=%u got-want=%d\n", b[8+k*4], b[9+k*4], b[10+k*4], (int)b[11+k*4]);
    }
    worst = std::max(worst, ticks * 1e3 / g_rate_khz / rounds);
    CK(hipFree(t[g])); CK(hipFree(dbox[g])); CK(hipFree(bad[g])); CK(hipFree(gbar[g]));
  }
  *nbad_out = nb;
  return worst;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  int rounds = argc > 1 ? atoi(argv[1]) : 20000;
  int only = argc > 3 ? atoi(argv[3]) : -1;  // -1 all, 0..2 one memory kind, 3 exchange only
  if (argc > 2) g_json = fopen(argv[2], "w");
  if (g_json) fprintf(g_json, "{");
  CK(hipDeviceGetAttribute(&g_rate_khz, hipDeviceAttributeWallClockRate, 0));
  for (int g = 0; g < 4; ++g) {
    CK(hipSetDevice(g));
    for (int p = 0; p < 4; ++p) if (p != g) {
      hipError_t e = hipDeviceEnablePeerAccess(p, 0);
      if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CK(e);
    }
  }
  struct Kind { const char* name; unsigned flags; } kinds[] = {
    {"uncached", hipDeviceMallocUncached}, {"finegrained", hipDeviceMallocFinegrained}, {"coarse", 0xFFFFFFFFu}};
  printf("wall clock %d kHz, rounds %d\n", g_rate_khz, rounds);
  printf("== 1. pair ping-pong, us per one-way hop (pairs 0-1, 0-2, 0-3, 1-2, 1-3, 2-3)\n");
  for (int ki = 0; ki < 3; ++ki) {
    if (only >= 0 && only != ki) continue;
    auto& k = kinds[ki];
    Bufs fl = alloc_flags(k.flags, 1 << 16);
    for (int pull = 0; pull < 2; ++pull) for (int sl = 1; sl >= 0; --sl) {
      printf("  %-12s %-4s %-6s:", k.name, pull ? "pull" : "push", sl ? "sleep" : "busy");
      double s = 0; int n = 0;
      for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b) { double us = run_pair(a, b, fl, rounds, pull, sl); printf(" %.3f", us); s += us; ++n; }
      printf("  mean %.3f\n", s / n);
      jrec(std::string("pair_") + k.name + (pull ? "_pull" : "_push") + (sl ? "_sleep" : "_busy"), s / n);
    }
    {
      printf("  %-12s push relaxed:", k.name);
      double s = 0; int n = 0;
      for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b) { double us = run_pair(a, b, fl, rounds, false, true, true); printf(" %.3f", us); s += us; ++n; }
      printf("  mean %.3f\n", s / n);
      jrec(std::string("pair_") + k.name + "_push_relaxed", s / n);
    }
    double self = run_self(0, fl, rounds);
    printf("  %-12s same-GPU two-CTA hop: %.3f us\n", k.name, self);
    jrec(std::string("self_") + k.name, self);
    double bar = run_barrier(fl, rounds);
    printf("  %-12s 4-rank barrier (push, 1 CTA/rank): %.3f us per barrier\n", k.name, bar);
    jrec(std::string("barrier4_") + k.name, bar);
    free_flags(fl);
  }
  if (only == 5) {  // soak in the K2 regime: grid fan-in between epochs
    Bufs fl = alloc_flags(hipDeviceMallocUncached, (size_t)2 * 4 * 131072 * 8);
    int cfg[][4] = {{32768,256,1,1000000},{8192,256,1,1000000},{1,256,1,1000000},{32768,256,0,200000}};
    for (auto& c : cfg) {
      unsigned nb = 0; double us = run_exchange(fl, c[0], c[1], c[3], c[2], &nb);
      printf("  SOAK P=%d G=%d bar=%d rounds=%d: %.3f us/exchange bad=%u\n", c[0], c[1], c[2], c[3], us, nb);
    }
    free_flags(fl);
  }
  if (only == 4) {  // repeat the big exchange
    Bufs fl = alloc_flags(hipDeviceMallocUncached, (size_t)2 * 4 * 131072 * 8);
    int cfg[][3] = {{131072,256,0},{131072,256,1},{65536,256,0},{131072,128,0},{32768,256,0}};
    for (auto& c : cfg) for (int rep = 0; rep < 3; ++rep) {
      unsigned nb = 0; double us = run_exchange(fl, c[0], c[1], 200, c[2], &nb);
      printf("  P=%d G=%d bar=%d rep %d: %.3f us bad=%u\n", c[0], c[1], c[2], rep, us, nb);
    }
    free_flags(fl);
  }
  if (only == -1 || only == 3) {
  printf("== 3. tagged mailbox exchange (all-to-all incl. self), uncached, us per exchange (slowest rank)\n");
  {
    Bufs fl = alloc_flags(hipDeviceMallocUncached, (size_t)2 * 4 * 131072 * 8);
    int Ps[] = {1, 64, 1024, 8192, 32768, 131072};
    int Gs[] = {1, 16, 64, 256};
    for (int intra = 0; intra < 2; ++intra)
      for (int P : Ps) for (int G : Gs) {
        if (G > P && P > 1) continue;
        if (P == 1 && G > 1 && !intra) continue;
        int r = std::max(200, std::min(rounds, (int)(2e8 / ((double)P * 16 + 1e5))));
        unsigned nb = 0;
        double us = run_exchange(fl, P, G, r, intra, &nb);
        printf("  P=%6d dwords/peer (%7.1f KB payload/peer) G=%3d intra_barrier=%d: %8.3f us  (%d rounds, bad=%u)\n",
               P, P * 4 / 1024.0, G, intra, us, r, nb);
        char nm[128]; snprintf(nm, sizeof nm, "xchg_P%d_G%d_bar%d", P, G, intra); jrec(nm, us);
        if (nb) printf("  !! STALE/BAD READS\n");
      }
    free_flags(fl);
  }
  }
  if (g_json) { fprintf(g_json, "\n}\n"); fclose(g_json); }
  return 0;
}
