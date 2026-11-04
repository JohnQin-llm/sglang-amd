// What the gfx1250 memory model emits for each hand-off primitive K2 uses.
// hipcc -O3 --offload-arch=gfx1250 -S --cuda-device-only memmodel_probe.hip -o memmodel_probe.s
#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void relaxed_sys_pair(uint64_t* box, uint64_t w, uint64_t* out) {
  __hip_atomic_store(box, w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  out[0] = __hip_atomic_load(box + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
extern "C" __global__ void relaxed_dev_pair(uint64_t* box, uint64_t w, uint64_t* out) {
  __hip_atomic_store(box, w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  out[0] = __hip_atomic_load(box + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
extern "C" __global__ void relacq_sys(int* f, int* out) {
  __hip_atomic_store(f, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  out[0] = __hip_atomic_load(f + 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
extern "C" __global__ void fence_rel_agent(int* d, int* f) {
  d[threadIdx.x] = 7;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  f[0] = 1;
}
extern "C" __global__ void fence_rel_sys(int* d, int* f) {
  d[threadIdx.x] = 7;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
  f[0] = 1;
}
extern "C" __global__ void fence_acq_agent(int* d, int* out) {
  int v = d[0];
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  out[0] = v + d[1];
}
extern "C" __global__ void clocks(long long* out) {
  out[0] = wall_clock64();
  __builtin_amdgcn_s_sleep(1);
  out[1] = clock64();
  out[2] = __builtin_readsteadycounter();
}
