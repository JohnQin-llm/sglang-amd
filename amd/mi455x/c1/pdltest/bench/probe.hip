#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k(unsigned* o) {
  unsigned x, h;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID2)" : "=s"(x));
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(h));
  if (threadIdx.x == 0) { o[2*blockIdx.x] = x; o[2*blockIdx.x+1] = h; }
}
int main() {
  hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
  int r; hipDeviceGetAttribute(&r, hipDeviceAttributeWallClockRate, 0);
  printf("%s %s CUs %d smem/blk %zu smem/CU %zu regs/blk %d regs/CU %d maxthr/CU %d warp %d l2 %d wallclk_khz %d clk_khz %d\n", p.name, p.gcnArchName, p.multiProcessorCount, p.sharedMemPerBlock, p.maxSharedMemoryPerMultiProcessor, p.regsPerBlock, p.regsPerMultiprocessor, p.maxThreadsPerMultiProcessor, p.warpSize, p.l2CacheSize, r, p.clockRate);
  unsigned* o; hipMalloc(&o, 4096*8); k<<<64,64>>>(o); hipDeviceSynchronize();
  unsigned h[128]; hipMemcpy(h, o, 128*4, hipMemcpyDeviceToHost);
  for (int i=0;i<64;i++) printf("%d:x%u/h%08x ", i, h[2*i]&0xf, h[2*i+1]); printf("\n");
}
