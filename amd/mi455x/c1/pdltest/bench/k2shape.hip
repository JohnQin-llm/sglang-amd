// K2-shaped persistent kernel + flag helpers as a shared library for torch (ctypes), so the real Triton QKV kernel
// can be measured behind it. Epoch-safe: every counter is reset by an atomic store in pdl_reset (ordered before the
// fork); the K2 kernel never resets its own counters.
#include <hip/hip_runtime.h>
typedef unsigned long long u64;
__device__ inline u64 now() { return wall_clock64(); }
__device__ inline unsigned ldr(unsigned* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ unsigned g_k2guard;

// ctr[0] started, ctr[1] trigger flag, ctr[2] done
template <int V>
__global__ __launch_bounds__(512) void k2_shape(unsigned* ctr, u64 body, u64 hand, int np) {
  extern __shared__ float lds[];
  if constexpr (V == 251) asm volatile("v_mov_b32 v250, 0" ::: "v250");
  else asm volatile("v_mov_b32 v223, 0" ::: "v223");
  if (threadIdx.x == 0) {
    atomicAdd(ctr, 1u);
    u64 t = now();
    while (ldr(ctr) < (unsigned)np) { if (now() - t > 100000000ull) { atomicAdd(&g_k2guard, 1u); break; } }
    while (now() - t < body) __builtin_amdgcn_s_sleep(1);
    if (blockIdx.x == 0) __hip_atomic_store(ctr + 1, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    u64 t1 = now();
    while (now() - t1 < hand) __builtin_amdgcn_s_sleep(1);
    lds[0] = 1;
  }
  __syncthreads();  // all 16 waves (and their VGPRs) stay resident through the last hand-off, as in K2
  if (threadIdx.x == 0) __hip_atomic_fetch_add(ctr + 2, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
__global__ void reset3(unsigned* ctr) {
  if (threadIdx.x < 3) __hip_atomic_store(ctr + threadIdx.x, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

extern "C" {
int k2_launch(void* stream, unsigned* ctr, double body_us, double hand_us, int vgpr) {
  static int init = 0;
  if (!init) {
    hipFuncSetAttribute((const void*)k2_shape<251>, hipFuncAttributeMaxDynamicSharedMemorySize, 145 * 1024);
    hipFuncSetAttribute((const void*)k2_shape<224>, hipFuncAttributeMaxDynamicSharedMemorySize, 145 * 1024);
    init = 1;
  }
  int rate; hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0);
  u64 b = (u64)(body_us * rate / 1000), h = (u64)(hand_us * rate / 1000);
  if (vgpr == 251) hipLaunchKernelGGL(k2_shape<251>, dim3(256), dim3(512), 145 * 1024, (hipStream_t)stream, ctr, b, h, 256);
  else hipLaunchKernelGGL(k2_shape<224>, dim3(256), dim3(512), 145 * 1024, (hipStream_t)stream, ctr, b, h, 256);
  return (int)hipGetLastError();
}
int k2_reset(void* stream, unsigned* ctr) { hipLaunchKernelGGL(reset3, dim3(1), dim3(64), 0, (hipStream_t)stream, ctr); return (int)hipGetLastError(); }
int k2_wait_trigger(void* stream, unsigned* ctr) { return (int)hipStreamWaitValue32((hipStream_t)stream, ctr + 1, 1, hipStreamWaitValueGte, 0xffffffff); }
unsigned k2_guard_read_and_clear() { unsigned v = 0, z = 0; hipMemcpyFromSymbol(&v, HIP_SYMBOL(g_k2guard), 4); hipMemcpyToSymbol(HIP_SYMBOL(g_k2guard), &z, 4); return v; }
}
