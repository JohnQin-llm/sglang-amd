// Diagnostic (not a serving candidate): one block a _dense_partial program p = pn + NT pk (grid (NT, 1, SK)),
// touching the weight a program reads (rows bn pn .. + bn, K slice pk):
//   mode 0  global_prefetch_b8 scope:SCOPE_DEV, every 128 B line
//   mode 1  a dword load, every line
//   mode 2  a dword load a row (row stride 16 KB: one a 4 KB page)       mode 3  every 4th row (one a 64 KB)
//   mode 4  one dword a program (row 0: one a 1 MB tile)                  mode 5  scalar loads (s_load_b32), a row
#include <hip/hip_runtime.h>
extern "C" __global__ void pf_qkv(const char* w, int K, int bn, int sk, int mode, unsigned* sink) {
  int nt = gridDim.x / sk;
  int pn = blockIdx.x % nt, pk = blockIdx.x / nt;
  size_t rowb = (size_t)K * 2, kb = (size_t)(K / sk) * 2;
  const char* base = w + (size_t)(pn * bn) * rowb + pk * kb;
  int lpr = (int)(kb / 128), total = bn * lpr;
  unsigned a = 0;
  if (mode <= 1) {
    for (int l = threadIdx.x; l < total; l += blockDim.x) {
      const char* p = base + (size_t)(l / lpr) * rowb + (l % lpr) * 128;
      if (mode == 0)
        __builtin_amdgcn_global_prefetch(p, 16);
      else
        a += *(const volatile unsigned*)p;
    }
  } else if (mode <= 4) {
    int step = mode == 2 ? 1 : mode == 3 ? 4 : bn;
    if (threadIdx.x * step < bn) a += *(const volatile unsigned*)(base + (size_t)(threadIdx.x * step) * rowb);
  } else if (threadIdx.x < 32) {
    for (int r = 0; r < bn; r++) {
      unsigned v;
      const char* p = base + (size_t)r * rowb;
      asm volatile("s_load_b32 %0, %1, 0x0\n s_wait_kmcnt 0x0" : "=s"(v) : "s"(p));
      a += v;
    }
  }
  if (mode && a == 0x9e3779b9u) sink[0] = a;
}
