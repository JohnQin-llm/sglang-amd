#include <hip/hip_runtime.h>
typedef int v16i __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
__global__ void k(const v16i* a, const v16i* b, const int* sa, const int* sb, v8f* d) {
  int l = threadIdx.x;
  v8f acc = {};
  acc = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(4, a[l], 0, b[l], (short)0, acc, 0, 0, sa[l], 0, 0, sb[l], false, false);
  d[l] = acc;
}
