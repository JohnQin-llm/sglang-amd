// Does a dependent kernel start before its producer ends on gfx1250?
// A: 8 blocks, each spins ~SPIN_US, writes *flag = 1 at its trigger point (start), records start/end times.
// B: 8 blocks, records its start time. Overlap = B.start < A.end.
// Modes: plain (same stream), attr6 (hipLaunchKernelExC with launch attribute id 6 = CUDA's
// ProgrammaticStreamSerialization), waitvalue (B on a second stream behind hipStreamWaitValue32 on the flag).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("  %s -> %s\n", #x, hipGetErrorString(e)); } } while (0)

__device__ inline unsigned long long now() { return wall_clock64(); }

__global__ void producer(volatile unsigned int* flag, unsigned long long* t, unsigned long long spin_ticks, unsigned int* done) {
  unsigned long long t0 = now();
  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) { __threadfence_system(); *flag = 1; __threadfence_system(); }
    t[2 * blockIdx.x] = t0;
  }
  while (now() - t0 < spin_ticks) { __builtin_amdgcn_s_sleep(2); }
  if (threadIdx.x == 0) { t[2 * blockIdx.x + 1] = now(); __threadfence(); atomicAdd(done, 1u); }
}

// clock-free: each B block records how many A blocks had finished when it started
__global__ void consumer(unsigned long long* t, unsigned int* done) {
  if (threadIdx.x == 0) t[blockIdx.x] = __hip_atomic_load(done, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "plain";
  int nb = 8, spin_us = 200;
  int rate_khz = 0;
  CHECK(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, 0));
  unsigned long long spin_ticks = (unsigned long long)spin_us * rate_khz / 1000;
  unsigned int* flag; unsigned int* done; unsigned long long *ta, *tb;
  CHECK(hipMalloc(&flag, 4)); CHECK(hipMalloc(&done, 4)); CHECK(hipMalloc(&ta, 2 * nb * 8)); CHECK(hipMalloc(&tb, nb * 8));
  hipStream_t s1, s2; CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking)); CHECK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
  double worst = 1e9, best = -1e9;
  for (int it = 0; it < 6; ++it) {
    CHECK(hipMemset(flag, 0, 4)); CHECK(hipMemset(done, 0, 4)); CHECK(hipMemset(ta, 0, 2 * nb * 8)); CHECK(hipMemset(tb, 0, nb * 8)); CHECK(hipDeviceSynchronize());
    if (strcmp(mode, "graph")) hipLaunchKernelGGL(producer, dim3(nb), dim3(64), 0, s1, (volatile unsigned int*)flag, ta, spin_ticks, done);
    if (!strcmp(mode, "plain")) {
      hipLaunchKernelGGL(consumer, dim3(nb), dim3(64), 0, s1, tb, done);
    } else if (!strcmp(mode, "attr6")) {
      hipLaunchAttribute attr; memset(&attr, 0, sizeof(attr));
      attr.id = (hipLaunchAttributeID)6; attr.val.priority = 1;  // val is a union; 1 = enable
      hipLaunchConfig_t cfg; memset(&cfg, 0, sizeof(cfg));
      cfg.gridDim = dim3(nb); cfg.blockDim = dim3(64); cfg.stream = s1; cfg.attrs = &attr; cfg.numAttrs = 1;
      void* args[] = {&tb, &done};
      hipError_t e = hipLaunchKernelExC(&cfg, (const void*)consumer, args);
      if (it == 0) printf("  hipLaunchKernelExC(attr id 6) -> %s\n", hipGetErrorString(e));
      if (e != hipSuccess) hipLaunchKernelGGL(consumer, dim3(nb), dim3(64), 0, s1, tb, done);
    } else if (!strcmp(mode, "graph")) {  // the waitvalue pattern captured into one graph and replayed
      static hipGraphExec_t exec = nullptr;
      if (!exec) {
        CHECK(hipDeviceSynchronize());
        hipStream_t c1 = s1, c2 = s2; hipEvent_t fork, join;
        CHECK(hipEventCreateWithFlags(&fork, hipEventDisableTiming)); CHECK(hipEventCreateWithFlags(&join, hipEventDisableTiming));
        hipGraph_t g;
        CHECK(hipStreamBeginCapture(c1, hipStreamCaptureModeGlobal));
        CHECK(hipEventRecord(fork, c1)); CHECK(hipStreamWaitEvent(c2, fork, 0));
        hipLaunchKernelGGL(producer, dim3(nb), dim3(64), 0, c1, (volatile unsigned int*)flag, ta, spin_ticks, done);
        hipError_t e = hipStreamWaitValue32(c2, flag, 1, hipStreamWaitValueGte, 0xffffffff);
        printf("  hipStreamWaitValue32 inside capture -> %s\n", hipGetErrorString(e));
        hipLaunchKernelGGL(consumer, dim3(nb), dim3(64), 0, c2, tb, done);
        CHECK(hipEventRecord(join, c2)); CHECK(hipStreamWaitEvent(c1, join, 0));
        hipError_t ec = hipStreamEndCapture(c1, &g);
        printf("  hipStreamEndCapture -> %s\n", hipGetErrorString(ec));
        if (ec != hipSuccess) { printf("RESULT graph: capture unsupported\n"); return 0; }
        CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
        CHECK(hipMemset(flag, 0, 4)); CHECK(hipMemset(done, 0, 4)); CHECK(hipMemset(tb, 0, nb * 8)); CHECK(hipDeviceSynchronize());
      }
      CHECK(hipGraphLaunch(exec, s1));
    } else {  // waitvalue
      CHECK(hipStreamWaitValue32(s2, flag, 1, hipStreamWaitValueGte, 0xffffffff));
      hipLaunchKernelGGL(consumer, dim3(nb), dim3(64), 0, s2, tb, done);
    }
    CHECK(hipDeviceSynchronize());
    unsigned long long b[8];
    CHECK(hipMemcpy(b, tb, nb * 8, hipMemcpyDeviceToHost));
    unsigned long long mn = ~0ull, mx = 0;
    for (int i = 0; i < nb; ++i) { if (b[i] < mn) mn = b[i]; if (b[i] > mx) mx = b[i]; }
    printf("  iter %d: A blocks finished when B blocks started: %llu..%llu of %d (%s)\n", it, mn, mx, nb, mn < (unsigned long long)nb ? "B started BEFORE A ended" : "B started after A ended");
    if (it > 0) { if ((double)mn < worst) worst = (double)mn; if ((double)mn > best) best = (double)mn; }
  }
  printf("RESULT %s: overlap %s (A blocks done at B start, min over iters %.0f, max %.0f, of %d)\n", mode, best < nb ? "YES" : (worst < nb ? "SOMETIMES" : "NO"), worst, best, nb);
  return 0;
}
