// Co-residency of an early-launched consumer beside a K2-shaped persistent producer (gfx1250).
// Producer: NP CTAs x 512 threads, PLDS bytes dynamic LDS, VGPRs forced (template). Each CTA announces itself;
// once all NP have started (all resident), CTA 0 writes the trigger flag. Then every CTA does a fixed ALU+LDS
// work loop and records its own duration (same-CTA clock) and its HW_ID1 (SE/SA/WGP).
// Consumer: real QKV-partials launch shape (NC CTAs x 256 threads, CLDS LDS, VGPRs forced), on a second stream
// behind hipStreamWaitValue32(trigger >= epoch). Each CTA records how many producer CTAs had finished at its start
// and at its end (clock-free), so "ran while the producer was fully alive" = end count 0.
// Usage: cores <prod_vgpr: 251|128|64> <cons_vgpr: 0(natural)|64|128|256> <cons_lds_kb> <cons_ctas> [prod_lds_kb=145] [nprod=256]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("ERR %s -> %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)
__device__ inline unsigned long long now() { return wall_clock64(); }
__device__ inline unsigned hwid() { unsigned h; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(h)); return h; }
__device__ inline unsigned ld(unsigned* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }

#define VG(n) if constexpr (V == n + 1) asm volatile("v_mov_b32 v" #n ", 0" ::: "v" #n);
template <int V> __device__ inline void use_vgprs() {
  VG(250) VG(239) VG(231) VG(223) VG(215) VG(199) VG(191) VG(127) VG(103) VG(63)
}

struct PRec { unsigned long long t0, t1; unsigned hw, pad; };
struct CRec { unsigned done_at_start, done_at_end, hw, alive_at_start; };

template <int V>
__global__ __launch_bounds__(512) void producer(unsigned* trig, unsigned epoch, unsigned* started, unsigned* pdone,
                                                PRec* rec, int iters, int np, unsigned* timeout) {
  extern __shared__ float lds[];
  use_vgprs<V>();
  __shared__ int ok;
  if (threadIdx.x == 0) {
    atomicAdd(started, 1u);
    unsigned long long t = now(); ok = 1;
    while (ld(started) < (unsigned)np) { if (now() - t > 200000000ull) { ok = 0; atomicAdd(timeout, 1u); break; } }  // 2 s guard
    if (blockIdx.x == 0 && ok) __hip_atomic_store(trig, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  __syncthreads();
  unsigned long long t0 = now();
  float a = threadIdx.x * 1e-3f, b = 1.0001f;
  int n = blockDim.x;
  for (int i = 0; i < iters; ++i) {
    lds[(threadIdx.x + i * 32) % (n * 16)] = a;
#pragma unroll 8
    for (int j = 0; j < 32; ++j) a = a * b + 1e-7f;
    __syncthreads();
    a += lds[(threadIdx.x * 7 + i) % (n * 16)] * 1e-9f;
  }
  if (a == 12345.f) lds[0] = a;
  __syncthreads();
  if (threadIdx.x == 0) {
    rec[blockIdx.x].t0 = t0; rec[blockIdx.x].t1 = now(); rec[blockIdx.x].hw = hwid();
    __hip_atomic_fetch_add(pdone, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
}

__device__ unsigned g_live, g_maxlive;
template <int V>
__global__ __launch_bounds__(256) void consumer(unsigned* started, unsigned* pdone, CRec* rec, int spin_ticks, const uint4* wbuf, size_t per_cta) {
  extern __shared__ float lds[];
  use_vgprs<V>();
  unsigned s = 0, a = 0;
  if (threadIdx.x == 0) { s = ld(pdone); a = ld(started); unsigned l = atomicAdd(&g_live, 1u) + 1; atomicMax(&g_maxlive, l); }
  lds[threadIdx.x] = threadIdx.x;
  if (wbuf) {  // prefetch: stream this CTA's weight slice (the QKV-weight warm-up)
    const uint4* p = wbuf + blockIdx.x * per_cta;
    unsigned acc = 0;
    for (size_t i = threadIdx.x; i < per_cta; i += blockDim.x) { uint4 v = p[i]; acc ^= v.x ^ v.w; }
    if (acc == 0x12345678u) lds[2] = acc;
  } else {
    unsigned long long t = now();
    while (now() - t < (unsigned long long)spin_ticks) __builtin_amdgcn_s_sleep(1);
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicSub(&g_live, 1u);
  if (threadIdx.x == 0) { CRec r; r.done_at_start = s; r.done_at_end = ld(pdone); r.hw = hwid(); r.alive_at_start = a + (unsigned)lds[1]; rec[blockIdx.x] = r; }
}

template <int V> void* prod_ptr() { return (void*)producer<V>; }
template <int V> void* cons_ptr() { return (void*)consumer<V>; }

int main(int argc, char** argv) {
  int pv = argc > 1 ? atoi(argv[1]) : 251, cv = argc > 2 ? atoi(argv[2]) : 0;
  int clds = (argc > 3 ? atoi(argv[3]) : 0) * 1024, nc = argc > 4 ? atoi(argv[4]) : 576;
  int plds = (argc > 5 ? atoi(argv[5]) : 145) * 1024, np = argc > 6 ? atoi(argv[6]) : 256;
  int iters = argc > 7 ? atoi(argv[7]) : 20000;
  int nthr_c = argc > 8 ? atoi(argv[8]) : 256;
  int stream_mb = argc > 9 ? atoi(argv[9]) : 0;
#define PICK(T, v) (v == 251 ? T<251>() : v == 240 ? T<240>() : v == 232 ? T<232>() : v == 224 ? T<224>() : v == 216 ? T<216>() : \
  v == 200 ? T<200>() : v == 192 ? T<192>() : v == 128 ? T<128>() : v == 104 ? T<104>() : v == 64 ? T<64>() : T<0>())
  void* pk = PICK(prod_ptr, pv);
  void* ck = PICK(cons_ptr, cv);
  uint4* wbuf = nullptr; size_t per_cta = 0;
  if (stream_mb) { size_t bytes = (size_t)stream_mb << 20; per_cta = bytes / 16 / nc; CHECK(hipMalloc(&wbuf, bytes)); CHECK(hipMemset(wbuf, 1, bytes)); }
  hipFuncAttributes fa;
  CHECK(hipFuncGetAttributes(&fa, pk));
  int pvg = fa.numRegs;
  CHECK(hipFuncSetAttribute(pk, hipFuncAttributeMaxDynamicSharedMemorySize, plds));
  CHECK(hipFuncGetAttributes(&fa, ck));
  int cvg = fa.numRegs;
  if (clds > 0) CHECK(hipFuncSetAttribute(ck, hipFuncAttributeMaxDynamicSharedMemorySize, clds));
  int clds_l = std::max(clds, 1024);
  unsigned *trig, *started, *pdone, *tmo; PRec* pr; CRec* cr;
  CHECK(hipMalloc(&trig, 4)); CHECK(hipMalloc(&started, 4)); CHECK(hipMalloc(&pdone, 4)); CHECK(hipMalloc(&tmo, 4));
  CHECK(hipMalloc(&pr, np * sizeof(PRec))); CHECK(hipMalloc(&cr, nc * sizeof(CRec)));
  CHECK(hipMemset(trig, 0, 4)); CHECK(hipMemset(tmo, 0, 4));
  hipStream_t s1, s2;
  CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking)); CHECK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
  int rate; CHECK(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0));
  double tick_us = 1000.0 / rate;
  int spin = (int)(2.0 / tick_us);  // consumer CTA lives ~2 us
  printf("CONFIG prod %d CTAs x512 vgpr=%d lds=%dKB | cons %d CTAs x%d vgpr=%d lds=%dKB stream %d MB\n", np, pvg, plds / 1024, nc, nthr_c, cvg, clds_l / 1024, stream_mb);
  for (int mode = 0; mode < 2; ++mode) {  // 0: producer alone, 1: producer + early consumer
    std::vector<double> pdur; int full = 0, any = 0, tot = 0, tmo_h = 0; std::vector<unsigned> cus; unsigned maxlive = 0;
    for (int it = 0; it < 6; ++it) {
      unsigned epoch = 1 + it + 100 * mode;
      CHECK(hipMemset(started, 0, 4)); CHECK(hipMemset(pdone, 0, 4)); CHECK(hipMemset(cr, 0xff, nc * sizeof(CRec)));
      unsigned z = 0; CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_maxlive), &z, 4)); CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_live), &z, 4));
      CHECK(hipDeviceSynchronize());
      hipLaunchKernelGGL((void (*)(unsigned*, unsigned, unsigned*, unsigned*, PRec*, int, int, unsigned*))pk, dim3(np), dim3(512), plds, s1,
                         trig, epoch, started, pdone, pr, iters, np, tmo);
      if (mode == 1) {
        CHECK(hipStreamWaitValue32(s2, trig, epoch, hipStreamWaitValueGte, 0xffffffff));
        hipLaunchKernelGGL((void (*)(unsigned*, unsigned*, CRec*, int, const uint4*, size_t))ck, dim3(nc), dim3(nthr_c), clds_l, s2, started, pdone, cr, spin, (const uint4*)wbuf, per_cta);
      }
      CHECK(hipDeviceSynchronize());
      std::vector<PRec> p(np); CHECK(hipMemcpy(p.data(), pr, np * sizeof(PRec), hipMemcpyDeviceToHost));
      double mx = 0; for (auto& r : p) mx = std::max(mx, (r.t1 - r.t0) * tick_us);
      if (it > 0) pdur.push_back(mx);
      if (it == 1) { for (auto& r : p) cus.push_back(r.hw & 0x000fff00); }
      if (mode == 1) {
        std::vector<CRec> c(nc); CHECK(hipMemcpy(c.data(), cr, nc * sizeof(CRec), hipMemcpyDeviceToHost));
        unsigned ml; CHECK(hipMemcpyFromSymbol(&ml, HIP_SYMBOL(g_maxlive), 4)); if (it > 0) maxlive = std::max(maxlive, ml);
        if (it > 0) for (auto& r : c) { tot++; if (r.done_at_end == 0) full++; if (r.done_at_start < (unsigned)np) any++; }
      }
    }
    CHECK(hipMemcpy(&tmo_h, tmo, 4, hipMemcpyDeviceToHost));
    std::sort(pdur.begin(), pdur.end()); std::sort(cus.begin(), cus.end());
    int distinct = std::unique(cus.begin(), cus.end()) - cus.begin();
    printf("RESULT pv=%d cv=%d clds=%dKB mode=%s: producer CTA work time median %.1f us (max over CTAs)", pvg, cvg, clds_l / 1024,
           mode ? "with-consumer" : "alone", pdur[pdur.size() / 2], distinct);
    if (mode) printf(" | consumer CTAs started+finished while ALL producer CTAs alive: %.1f%% (%d/%d), started before producer end: %.1f%%",
                     100.0 * full / tot, full, tot, 100.0 * any / tot);
    if (mode) printf(" | max concurrent consumer CTAs %u", maxlive);
    printf(" | guard timeouts %d\n", tmo_h);
  }
  return 0;
}
