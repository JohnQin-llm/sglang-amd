// Muse Spark TP4 target verify: the two all-reduces of a decoder layer, each fused with its neighbouring kernel.
//
// * finalize_all_reduce: the MoE finalize (per-slot RMSNorm with input scale, route-weighted sum of this rank's
//   slots) and the all-reduce of the [M, 4096] partial sums.
// * all_reduce_boundary: the all-reduce of the attention o_proj partials [M, 8192] and the attention boundary
//   (post-norm, gated residual, pre-FFN norm).
//
// Both run on AITER's custom all-reduce: peer addresses of IPC-registered buffers (including CUDA-graph inputs)
// and the per-rank tmp buffer. Rank r reduces rows [r M/4, (r+1) M/4), in FP32 rounded to BF16, in AITER's order for
// the size: ranks 0..3 below 160 KB (its one-stage kernel: 8 and 16 MoE rows, 8 attention rows), from r upwards
// above (its two-stage kernel). The finalize reproduces FlyDSL's muse_moe_finalize_vector arithmetic (thread t holds
// columns 4 t + 1024 c + j, wave butterfly, waves summed in order), so the fused MoE output is bitwise equal to the
// finalize followed by AITER's all-reduce. The boundary keeps the Triton kernel's formula, but its outputs are not
// bitwise equal to the Triton kernel's.
//
// The three phases (local work, the owner's reduction, the copy of the reduced rows) hand rows over with per-row
// flags (tp4_row_exchange_hip.cuh), so every block works in every phase.
//
// No buffer is rewritten while a peer still reads it: a rank writes the next call's partials only after it has
// copied every reduced row of this call (so every owner has read its partials), and an owner overwrites its tmp
// rows only after every rank has entered the next call (so every rank has copied them).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include "tp4_row_exchange_hip.cuh"
#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER csrc/include: CustomAllreduce, RankData, get_tmp_buf

namespace sglang {
namespace muse_tp4_fused_all_reduce {

using namespace muse_tp4_exchange;
using aiter::RankData;
using aiter::RankSignals;

constexpr int kThreads = 512;
constexpr int kMoeHidden = 4096;
constexpr int kHidden = 8192;
constexpr int kTopK = 8;
constexpr int kMaxRows = 128;

// A rank's workspace: the finalized MoE partial rows, then the flag words. Peers write the ready, partial and
// arrive flags; the counters are rank-local.
constexpr size_t kPartialBytes = size_t(kMaxRows) * kMoeHidden * sizeof(uint16_t);
namespace flag {
constexpr int kReadyMoe = 0;                           // [row]: row reduced on its owner (MoE)
constexpr int kReadyAttention = kMaxRows;              // [row]: row reduced on its owner (attention)
constexpr int kPartial = 2 * kMaxRows;                 // [row][rank]: the rank's finalized row is written
constexpr int kArrive = kPartial + kMaxRows * kRanks;  // [rank]: the rank entered the attention call
constexpr int kCountMoe = kArrive + kRanks;            // [row]: slices of the row reduced here
constexpr int kCountAttention = kCountMoe + kMaxRows;
constexpr int kEpochMoe = kCountAttention + kMaxRows, kDoneMoe = kEpochMoe + 1;
constexpr int kEpochAttention = kEpochMoe + 2, kDoneAttention = kEpochMoe + 3;
constexpr int kWords = kEpochMoe + 4;
}  // namespace flag
constexpr size_t kWorkspaceBytes = kPartialBytes + flag::kWords * sizeof(uint32_t);

__device__ __forceinline__ Bf16x8 sum_packs(const Bf16x8* in) {
  Bf16x8 out;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    float acc = bf16_to_float(in[0].v[k]);
#pragma unroll
    for (int i = 1; i < kRanks; ++i)
      acc += bf16_to_float(in[i].v[k]);
    out.v[k] = float_to_bf16(acc);
  }
  return out;
}

// AITER's all-reduce at 4 ranks runs its one-stage kernel below this size (ranks 0..3 in order on every rank) and its
// two-stage kernel from it on (each row from its owner upwards).
constexpr int kOneStageBytes = 160 * 1024;

// One pack summed over the ranks in order first, first + 1, ...
__device__ __forceinline__ Bf16x8 reduce_pack(const Bf16x8* const* src, int first, int idx) {
  Bf16x8 in[kRanks];
#pragma unroll
  for (int i = 0; i < kRanks; ++i)
    in[i] = src[(first + i) % kRanks][idx];
  return sum_packs(in);
}

// The owner's reduction of one slice of one of its rows into its tmp buffer, in AITER's order for the size of the
// call; the last slice of a row to complete flags the row to every rank. kGroups thread groups split the ranks: a
// slice is kThreads / kGroups packs, group g loads ranks first + g * (4 / kGroups) + i into LDS and group 0 sums all
// four in that order. One group (each thread
// loads all four ranks) is 0.4-0.5 us faster for the MoE rows of 4096 columns at 16..80 rows; four groups are
// 1.3-4.6 us faster for the attention rows of 8192 columns at 16..128 rows, where they give four times as many
// blocks a slice.
template <int kRowPacks, int kGroups>
struct SliceReduce {
  static constexpr int kSlicePacks = kThreads / kGroups, kSlices = kRowPacks / kSlicePacks;
  static constexpr int kRanksPerGroup = kRanks / kGroups;

  __device__ static void
  run(const Bf16x8* const* src,
      Bf16x8* tmp,
      Bf16x8* stage,
      const Flags& flags,
      int ready,
      int count,
      int rank,
      int owned,
      int task,
      uint32_t generation) {
    const int j = task / kSlices, g = threadIdx.x / kSlicePacks, lane = threadIdx.x % kSlicePacks;
    const int idx = (task % kSlices) * kSlicePacks + lane, row = rank * owned + j;
    const int first = owned * kRanks * kRowPacks * (int)sizeof(Bf16x8) < kOneStageBytes ? 0 : rank;
    if constexpr (kGroups == 1) {
      tmp[j * kRowPacks + idx] = reduce_pack(src, first, row * kRowPacks + idx);
    } else {
#pragma unroll
      for (int i = 0; i < kRanksPerGroup; ++i) {
        const int k = g * kRanksPerGroup + i;
        stage[k * kSlicePacks + lane] = src[(first + k) % kRanks][row * kRowPacks + idx];
      }
      __syncthreads();
      if (g == 0) {
        Bf16x8 in[kRanks];
#pragma unroll
        for (int k = 0; k < kRanks; ++k)
          in[k] = stage[k * kSlicePacks + lane];
        tmp[j * kRowPacks + idx] = sum_packs(in);
      }
    }
    wait_for_stores();  // the slice is in (uncached) tmp before it is counted
    if (threadIdx.x == 0) {
      uint32_t* local = flags.rank[rank];
      const bool last =
          kSlices == 1 ||
          __scoped_atomic_fetch_add(&local[count + row], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) % kSlices ==
              kSlices - 1;
      if (last) {
#pragma unroll
        for (int r = 0; r < kRanks; ++r)
          publish(&flags.rank[r][ready + row], generation);
      }
    }
  }
};

// ---------------------------------------------------------------------------------------------------------------
// MoE finalize + all-reduce
// ---------------------------------------------------------------------------------------------------------------

struct FinalizeArgs {
  const uint16_t* y;  // [M * 8, 4096] per-slot expert rows
  const float* scores;
  const int* ids;
  int local_start, num_local;
  float eps;
};

// One row's finalize on the 256 threads of a half block, in FlyDSL's layout: thread t holds columns
// 4 t + 1024 c + j. All of the row's local slot data is loaded before any of it is used.
struct RowFinalizer {
  float scale[16];  // bf16(gamma + gain)
  uint2 raw[kTopK][4];
  float score[kTopK];
  uint32_t local = 0;  // bit s: slot s is routed to one of this rank's experts

  __device__ void load_scale(const uint16_t* gamma, float gain, int t) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint2 g = *reinterpret_cast<const uint2*>(gamma + t * 4 + c * 1024);
      const uint16_t* gv = reinterpret_cast<const uint16_t*>(&g);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        scale[c * 4 + j] = bf16_to_float(float_to_bf16(bf16_to_float(gv[j]) + gain));
    }
  }

  __device__ void load_row(const FinalizeArgs& a, int row, int t) {
    const int4 i0 = reinterpret_cast<const int4*>(a.ids + row * kTopK)[0];
    const int4 i1 = reinterpret_cast<const int4*>(a.ids + row * kTopK)[1];
    const float4 s0 = reinterpret_cast<const float4*>(a.scores + row * kTopK)[0];
    const float4 s1 = reinterpret_cast<const float4*>(a.scores + row * kTopK)[1];
    const int ids[kTopK] = {i0.x, i0.y, i0.z, i0.w, i1.x, i1.y, i1.z, i1.w};
    const float sc[kTopK] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
    local = 0;
#pragma unroll
    for (int s = 0; s < kTopK; ++s) {
      score[s] = sc[s];
      // Padded rows carry expert id -1, which no rank holds.
      if (ids[s] >= a.local_start && ids[s] < a.local_start + a.num_local) local |= 1u << s;
    }
#pragma unroll
    for (int s = 0; s < kTopK; ++s)
      if (local >> s & 1) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          raw[s][c] = *reinterpret_cast<const uint2*>(a.y + (size_t)(row * kTopK + s) * kMoeHidden + t * 4 + c * 1024);
      }
  }

  // bf16(y * scale)
  __device__ float value(int s, int c, int j) const {
    const uint16_t* yv = reinterpret_cast<const uint16_t*>(&raw[s][c]);
    return bf16_to_float(float_to_bf16(bf16_to_float(yv[j]) * scale[c * 4 + j]));
  }

  // Per-slot sum of squares in FlyDSL's order: this thread's 16 values, the wave butterfly, then the four waves
  // (summed in weighted_sum). Written to sums[slot][wave].
  __device__ void square_sums(int t, float (*sums)[4]) const {
#pragma unroll
    for (int s = 0; s < kTopK; ++s) {
      if (!(local >> s & 1)) continue;
      float ss = 0.0f;
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          ss = ss + value(s, c, j) * value(s, c, j);
      ss = wave_sum(ss);
      if (t % 64 == 0) sums[s][t / 64] = ss;
    }
  }

  // The route-weighted sum of the normed local slots, stored to `row_out`.
  __device__ void weighted_sum(int t, const float (*sums)[4], float eps, uint16_t* row_out) const {
    float state[16];
#pragma unroll
    for (int k = 0; k < 16; ++k)
      state[k] = 0.0f;
#pragma unroll
    for (int s = 0; s < kTopK; ++s) {
      if (!(local >> s & 1)) continue;
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < 4; ++w)
        total = total + sums[s][w];
      const float mean = total * (1.0f / kMoeHidden) + eps;
      const float sc = score[s] * (1.0f / __builtin_amdgcn_sqrtf(mean));
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          state[c * 4 + j] = state[c * 4 + j] + sc * value(s, c, j);
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      uint16_t o[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        o[j] = float_to_bf16(state[c * 4 + j]);
      *reinterpret_cast<uint2*>(row_out + t * 4 + c * 1024) = *reinterpret_cast<const uint2*>(o);
    }
  }
};

// (1) Every half block finalizes whole rows into this rank's partial buffer and flags each to its owner; (2) the
// owner reduces its rows in slices on every block, and the last slice of a row flags the row to all ranks; (3)
// every rank copies each row from its owner's tmp buffer.
__global__ void __launch_bounds__(kThreads, 1) finalize_all_reduce_kernel(
    FinalizeArgs a,
    const uint16_t* gamma,
    float gain,
    const RankData* partials,
    RankSignals sg,
    Flags flags,
    uint16_t* __restrict__ out,
    int rank,
    int m) {
  __shared__ float sums[2][kTopK][4];
  constexpr int kPacks = kMoeHidden / 8;  // 512
  using Reduce = SliceReduce<kPacks, 1>;
  const int owned = m / kRanks;
  uint32_t* local = flags.rank[rank];
  const uint32_t generation = begin_call(local, flag::kEpochMoe);

  uint16_t* partial = static_cast<uint16_t*>(const_cast<void*>(partials->ptrs[rank]));
  const int half = threadIdx.x / 256, t = threadIdx.x % 256;
  RowFinalizer fin;
  fin.load_scale(gamma, gain, t);
  for (int base = 2 * blockIdx.x; base < m; base += 2 * gridDim.x) {
    const int row = base + half;
    const bool valid = row < m;
    if (valid) {
      fin.load_row(a, row, t);
      fin.square_sums(t, sums[half]);
    }
    __syncthreads();
    if (valid) fin.weighted_sum(t, sums[half], a.eps, partial + (size_t)row * kMoeHidden);
    wait_for_stores();
    if (valid && t == 0) publish(&flags.rank[row / owned][flag::kPartial + row * kRanks + rank], generation);
  }

  const Bf16x8* src[kRanks];
#pragma unroll
  for (int r = 0; r < kRanks; ++r)
    src[r] = static_cast<const Bf16x8*>(partials->ptrs[r]);
  Bf16x8* tmp = aiter::get_tmp_buf<Bf16x8>(sg.signals[rank]);
  for (int task = blockIdx.x; task < owned * Reduce::kSlices; task += gridDim.x) {
    const int row = rank * owned + task / Reduce::kSlices;
    if (threadIdx.x < kRanks) wait_for(&local[flag::kPartial + row * kRanks + threadIdx.x], generation);
    __syncthreads();
    Reduce::run(src, tmp, nullptr, flags, flag::kReadyMoe, flag::kCountMoe, rank, owned, task, generation);
  }

  for (int row = blockIdx.x; row < m; row += gridDim.x) {
    if (threadIdx.x == 0) wait_for(&local[flag::kReadyMoe + row], generation);
    __syncthreads();
    const int owner = row / owned;
    const Bf16x8* owner_tmp = aiter::get_tmp_buf<Bf16x8>(sg.signals[owner]);
    reinterpret_cast<Bf16x8*>(out)[row * kPacks + threadIdx.x] =
        owner_tmp[(row - owner * owned) * kPacks + threadIdx.x];
  }
  end_call(local, flag::kEpochMoe, flag::kDoneMoe, generation);
}

// ---------------------------------------------------------------------------------------------------------------
// Attention all-reduce + boundary
// ---------------------------------------------------------------------------------------------------------------

// Thread t's local operands of one boundary row: 16 columns (t + 512 p) * 8 + k, p < 2, k < 8.
struct BoundaryOperands {
  float stream[16], alpha[16], beta[16], weight[16];

  __device__ void load(
      int row,
      const float* __restrict__ stream_in,
      const float* __restrict__ alpha_in,
      const float* __restrict__ beta_in,
      const uint16_t* __restrict__ pre_weight) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int col = (threadIdx.x + p * kThreads) * 8;
      const float4* st = reinterpret_cast<const float4*>(stream_in + (size_t)row * kHidden + col);
      const float4* al = reinterpret_cast<const float4*>(alpha_in + col);
      const float4* be = reinterpret_cast<const float4*>(beta_in + col);
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const float4 sv = st[q], av = al[q], bv = be[q];
        const int i = p * 8 + q * 4;
        stream[i] = sv.x, stream[i + 1] = sv.y, stream[i + 2] = sv.z, stream[i + 3] = sv.w;
        alpha[i] = av.x, alpha[i + 1] = av.y, alpha[i + 2] = av.z, alpha[i + 3] = av.w;
        beta[i] = bv.x, beta[i + 1] = bv.y, beta[i + 2] = bv.z, beta[i + 3] = bv.w;
      }
      const Bf16x8 w = *reinterpret_cast<const Bf16x8*>(pre_weight + col);
#pragma unroll
      for (int k = 0; k < 8; ++k)
        weight[p * 8 + k] = bf16_to_float(w.v[k]);
    }
  }
};

struct BoundaryArgs {
  const float* stream;
  const float* alpha;
  const float* beta;
  const uint16_t* pre_weight;
  uint16_t* ffn_input;
  float* new_stream;
  float post_eps, pre_eps;
};

// post_norm (no weight) -> bf16, gated residual alpha * stream + beta * branch, pre_norm * weight -> bf16: the
// formula of the Triton _fused_norm_residual_triton_kernel for the attention boundary, with its fused multiply-add
// in the residual.
__device__ __forceinline__ void attention_boundary(
    const Bf16x8 (&reduced)[2], int row, const BoundaryOperands& op, const BoundaryArgs& a, float* cells) {
  float b[16];
  float ss = 0.0f;
#pragma unroll
  for (int p = 0; p < 2; ++p)
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      b[p * 8 + k] = bf16_to_float(reduced[p].v[k]);
      ss = ss + b[p * 8 + k] * b[p * 8 + k];
    }
  const float post_rstd = __builtin_amdgcn_rsqf(block_sum<kThreads>(ss, cells) / kHidden + a.post_eps);
  float s[16];
  ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float normed = bf16_to_float(float_to_bf16(b[i] * post_rstd));
    s[i] = __builtin_fmaf(op.beta[i], normed, op.alpha[i] * op.stream[i]);
    ss = ss + s[i] * s[i];
  }
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const int col = (threadIdx.x + p * kThreads) * 8;
    float4* ns = reinterpret_cast<float4*>(a.new_stream + (size_t)row * kHidden + col);
    ns[0] = make_float4(s[p * 8 + 0], s[p * 8 + 1], s[p * 8 + 2], s[p * 8 + 3]);
    ns[1] = make_float4(s[p * 8 + 4], s[p * 8 + 5], s[p * 8 + 6], s[p * 8 + 7]);
  }
  const float pre_rstd = __builtin_amdgcn_rsqf(block_sum<kThreads>(ss, cells + 8) / kHidden + a.pre_eps);
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const int col = (threadIdx.x + p * kThreads) * 8;
    Bf16x8 o;
#pragma unroll
    for (int k = 0; k < 8; ++k)
      o.v[k] = float_to_bf16(s[p * 8 + k] * pre_rstd * op.weight[p * 8 + k]);
    *reinterpret_cast<Bf16x8*>(a.ffn_input + (size_t)row * kHidden + col) = o;
  }
  __syncthreads();  // the next row reuses `cells`
}

// Every rank flags its arrival (its o_proj partials are complete and it is done with the previous call's rows), the
// owner reduces its rows in slices on every block, the last slice of a row flags the row to all ranks, and each
// row's boundary runs on one block once the row is flagged.
__global__ void __launch_bounds__(kThreads, 1)
    all_reduce_boundary_kernel(const RankData* inputs, RankSignals sg, Flags flags, BoundaryArgs a, int rank, int m) {
  constexpr int kGroups = 4;
  __shared__ float cells[16];
  __shared__ Bf16x8 stage[kRanks * (kThreads / kGroups)];
  constexpr int kPacks = kHidden / 8;  // 1024
  using Reduce = SliceReduce<kPacks, kGroups>;
  constexpr int kSlices = Reduce::kSlices;
  const int owned = m / kRanks;
  uint32_t* local = flags.rank[rank];
  const uint32_t generation = begin_call(local, flag::kEpochAttention);
  if (blockIdx.x == 0 && threadIdx.x < kRanks) publish(&flags.rank[threadIdx.x][flag::kArrive + rank], generation);
  const Bf16x8* src[kRanks];
#pragma unroll
  for (int r = 0; r < kRanks; ++r)
    src[r] = static_cast<const Bf16x8*>(inputs->ptrs[r]);
  Bf16x8* tmp = aiter::get_tmp_buf<Bf16x8>(sg.signals[rank]);
  if ((int)blockIdx.x < owned * kSlices) {
    if (threadIdx.x < kRanks) wait_for(&local[flag::kArrive + threadIdx.x], generation);
    __syncthreads();
  }
  for (int task = blockIdx.x; task < owned * kSlices; task += gridDim.x)
    Reduce::run(src, tmp, stage, flags, flag::kReadyAttention, flag::kCountAttention, rank, owned, task, generation);
  BoundaryOperands op;
  for (int row = blockIdx.x; row < m; row += gridDim.x) {
    op.load(row, a.stream, a.alpha, a.beta, a.pre_weight);
    if (threadIdx.x == 0) wait_for(&local[flag::kReadyAttention + row], generation);
    __syncthreads();
    const int owner = row / owned;
    const Bf16x8* from = aiter::get_tmp_buf<Bf16x8>(sg.signals[owner]) + (row - owner * owned) * kPacks;
    Bf16x8 reduced[2];
#pragma unroll
    for (int p = 0; p < 2; ++p)
      reduced[p] = from[threadIdx.x + p * kThreads];
    attention_boundary(reduced, row, op, a, cells);
  }
  end_call(local, flag::kEpochAttention, flag::kDoneAttention, generation);
}

// ---------------------------------------------------------------------------------------------------------------
// Host entry points. `comm` is AITER's CustomAllreduce*; `workspace` is this rank's workspace, registered with it
// (tp4_peer_workspace.py), and `peers` (CPU int64 [4]) every rank's workspace address. Rows: a multiple of 8 in
// 8..128.
// ---------------------------------------------------------------------------------------------------------------

inline aiter::CustomAllreduce* tp4_communicator(int64_t comm, int64_t workspace_bytes) {
  using namespace host;
  auto* state = reinterpret_cast<aiter::CustomAllreduce*>(comm);
  RuntimeCheck(state->world_size_ == kRanks, "Muse TP4 fused all-reduce requires TP", kRanks);
  RuntimeCheck(
      workspace_bytes >= static_cast<int64_t>(kWorkspaceBytes), "Muse TP4 fused all-reduce workspace is too small");
  return state;
}

inline Flags workspace_flags(const tvm::ffi::TensorView peers) {
  using namespace host;
  TensorMatcher({kRanks}).with_dtype<int64_t>().with_device<kDLCPU>().verify(peers);
  return flags_at(static_cast<const int64_t*>(peers.data_ptr()), kPartialBytes);
}

inline int64_t check_rows(int64_t m) {
  using namespace host;
  RuntimeCheck(
      m % (2 * kRanks) == 0 && 0 < m && m <= kMaxRows, "rows must be a multiple of 8 in 8..", kMaxRows, ", got ", m);
  return m;
}

struct FusedAllReduce {
  static void finalize_all_reduce(
      int64_t comm,
      int64_t workspace,
      int64_t workspace_bytes,
      const tvm::ffi::TensorView peers,
      const tvm::ffi::TensorView y,
      const tvm::ffi::TensorView gamma,
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView out,
      int64_t local_start,
      int64_t num_local,
      double eps,
      double gain) {
    using namespace host;
    auto M = SymbolicSize{"num_rows"};
    auto MK = SymbolicSize{"num_slots"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({MK, kMoeHidden}).with_dtype<bf16_t>().with_device(device).verify(y);
    TensorMatcher({kMoeHidden}).with_dtype<bf16_t>().with_device(device).verify(gamma);
    TensorMatcher({M, kTopK}).with_dtype<float>().with_device(device).verify(scores);
    TensorMatcher({M, kTopK}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({M, kMoeHidden}).with_dtype<bf16_t>().with_device(device).verify(out);
    const int64_t m = check_rows(M.unwrap());
    RuntimeCheck(MK.unwrap() == m * kTopK, "y must hold 8 slot rows per row");

    auto* state = tp4_communicator(comm, workspace_bytes);
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    const FinalizeArgs args{
        static_cast<const uint16_t*>(y.data_ptr()),
        static_cast<const float*>(scores.data_ptr()),
        static_cast<const int*>(ids.data_ptr()),
        static_cast<int>(local_start),
        static_cast<int>(num_local),
        static_cast<float>(eps)};
    const RankData* partials = state->get_buffer_RD(stream, reinterpret_cast<void*>(workspace));
    // One block a row (measured fastest in 4-rank graphs).
    LaunchKernel(static_cast<uint32_t>(m), kThreads, stream)(
        finalize_all_reduce_kernel,
        args,
        static_cast<const uint16_t*>(gamma.data_ptr()),
        static_cast<float>(gain),
        partials,
        state->sg_,
        workspace_flags(peers),
        static_cast<uint16_t*>(out.data_ptr()),
        state->rank_,
        static_cast<int>(m));
  }

  static void all_reduce_boundary(
      int64_t comm,
      int64_t workspace_bytes,
      const tvm::ffi::TensorView peers,
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView stream_in,
      const tvm::ffi::TensorView alpha,
      const tvm::ffi::TensorView beta,
      const tvm::ffi::TensorView pre_weight,
      const tvm::ffi::TensorView ffn_input,
      const tvm::ffi::TensorView new_stream,
      double post_eps,
      double pre_eps) {
    using namespace host;
    auto M = SymbolicSize{"num_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device).verify(input).verify(ffn_input);
    TensorMatcher({M, kHidden}).with_dtype<float>().with_device(device).verify(stream_in).verify(new_stream);
    TensorMatcher({kHidden}).with_dtype<float>().with_device(device).verify(alpha).verify(beta);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(pre_weight);
    const int64_t m = check_rows(M.unwrap());

    auto* state = tp4_communicator(comm, workspace_bytes);
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    const BoundaryArgs args{
        static_cast<const float*>(stream_in.data_ptr()),
        static_cast<const float*>(alpha.data_ptr()),
        static_cast<const float*>(beta.data_ptr()),
        static_cast<const uint16_t*>(pre_weight.data_ptr()),
        static_cast<uint16_t*>(ffn_input.data_ptr()),
        static_cast<float*>(new_stream.data_ptr()),
        static_cast<float>(post_eps),
        static_cast<float>(pre_eps)};
    const RankData* inputs = state->get_buffer_RD(stream, input.data_ptr());
    // Two blocks a row up to 48 rows, one above (measured fastest in 4-rank graphs).
    LaunchKernel(static_cast<uint32_t>(m <= 48 ? 2 * m : m), kThreads, stream)(
        all_reduce_boundary_kernel,
        inputs,
        state->sg_,
        workspace_flags(peers),
        args,
        state->rank_,
        static_cast<int>(m));
  }
};

}  // namespace muse_tp4_fused_all_reduce
}  // namespace sglang
