// Row hand-off between the four ranks of a Muse Spark TP4 group, for kernels that exchange rows through uncached
// peer workspaces registered with AITER's custom all-reduce (tp4_peer_workspace.py).
//
// A writer stores rows into uncached memory, waits for the stores to complete, then stores the call's generation
// into the reader's flag; the reader polls its local flag. Every kernel keeps a call counter (epoch) in its
// rank-local flag words: generation = epoch + 1, the same on every rank because every rank launches the same
// sequence of calls. Generations compare modulo 2^32, so the counters may wrap.
#pragma once

#ifndef USE_ROCM
#error "tp4_row_exchange_hip.cuh is written for ROCm"
#endif

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sglang {
namespace muse_tp4_exchange {

constexpr int kRanks = 4;

struct alignas(16) Bf16x8 {
  uint16_t v[8];
};

__device__ __forceinline__ float bf16_to_float(uint16_t u) {
  return __uint_as_float(uint32_t(u) << 16);
}

// Round to nearest even, gfx950's conversion instruction.
__device__ __forceinline__ uint16_t float_to_bf16(float f) {
  return __builtin_bit_cast(uint16_t, static_cast<__bf16>(f));
}

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int offset = 32; offset >= 1; offset >>= 1)
    v = v + __shfl_xor(v, offset, 64);
  return v;
}

// Sum over the block's waves in order; `cells` holds kThreads / 64 floats used by this reduction only.
template <int kThreads>
__device__ __forceinline__ float block_sum(float v, float* cells) {
  v = wave_sum(v);
  if (threadIdx.x % 64 == 0) cells[threadIdx.x / 64] = v;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < kThreads / 64; ++w)
    total = total + cells[w];
  return total;
}

// Every rank's flag words of one workspace (peer-mapped), passed by value.
struct Flags {
  uint32_t* rank[kRanks];
};

// The flag words at byte `offset` of every rank's workspace, from the workspaces' peer addresses.
inline Flags flags_at(const int64_t* workspaces, size_t offset) {
  Flags flags;
  for (int r = 0; r < kRanks; ++r)
    flags.rank[r] = reinterpret_cast<uint32_t*>(workspaces[r] + offset);
  return flags;
}

__device__ __forceinline__ void publish(uint32_t* flag, uint32_t generation) {
  __scoped_atomic_store_n(flag, generation, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// Polls with a short back-off between reads (s_sleep 2, about 128 cycles): 0.4-0.7 us per call faster than a
// tight spin at 128 rows.
__device__ __forceinline__ void wait_for(const uint32_t* flag, uint32_t generation) {
  while (static_cast<int32_t>(__scoped_atomic_load_n(flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) - generation) < 0)
    __builtin_amdgcn_s_sleep(2);
}

// The stores this block issued are complete (uncached memory: visible to the peers) before any thread goes on.
__device__ __forceinline__ void wait_for_stores() {
  __builtin_amdgcn_s_waitcnt(0);
  __syncthreads();
}

// This call's generation from the kernel's epoch word. An atomic load, so it cannot come from the scalar cache: a
// stale epoch on one rank would put it out of step with the others.
__device__ __forceinline__ uint32_t begin_call(const uint32_t* local, int epoch) {
  return __scoped_atomic_load_n(&local[epoch], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1;
}

// The last block of the call to finish advances the epoch.
__device__ __forceinline__ void end_call(uint32_t* local, int epoch, int done, uint32_t generation) {
  __syncthreads();
  if (threadIdx.x == 0 &&
      __scoped_atomic_fetch_add(&local[done], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == gridDim.x - 1) {
    __scoped_atomic_store_n(&local[done], 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    __scoped_atomic_store_n(&local[epoch], generation, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
}

}  // namespace muse_tp4_exchange
}  // namespace sglang
