// Muse Spark TP4 target verify at 48, 56, 64, 80 and 96 rows: the MoE pooler and unpool split four ways across the
// TP ranks.
//
// Each rank projects a quarter of the columns (pooler 1024 of 4096, unpool 2048 of 8192) with a split-K GEMM. The
// two kernels here sum that GEMM's FP32 partials, exchange the BF16 slices between the four ranks and run the
// consumer that needs whole rows:
// * front_epilogue: per row, this rank's pooled slice; the router top-8 (the router is replicated: its high and low
//   BF16 parts come out of the same GEMM as the pooler slice); then, once all four slices of the row are exchanged,
//   the pre-expert RMSNorm of the 4096 pooled columns -> the expert input.
// * tail_boundary: per row, this rank's branch slice; once all four are exchanged, the FFN boundary (input-scale
//   post-norm, gated residual, the next layer's pre-attention norm) on the 8192 branch columns.
//
// The router and the pre-expert norm reproduce _joint_epilogue (router_pooler_topk.py) bitwise for the same pooled
// values and router logits. The split-K partials are summed in split order 0, 1, ...; the FFN boundary keeps the
// Triton formula (fused_norm_residual.py) with its two row sums in this kernel's order.
//
// Exchange (tp4_row_exchange_hip.cuh): a rank stores its slice of a row into every rank's workspace, waits for
// the stores, then raises the row's flag in every rank's flag words; a reader polls its local flags and reads the
// whole row from its own buffer. Pushing the slices this way made the tail kernel 13.5 -> 12.6 us at 48 rows,
// 17.0 -> 14.8 at 80 and 21.7 -> 18.5 at 128 against pulling them from the owners' buffers, and the front within
// 0.4 us. The buffers alternate by generation parity: a rank is at most one call ahead of a peer (it cannot finish a
// call before every peer has published that call's rows), so it never overwrites data the peer still reads.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "tp4_row_exchange_hip.cuh"
#include <cstdint>
#include <type_traits>

namespace sglang {
namespace muse_tp4_split_pool_unpool {

using namespace muse_tp4_exchange;

constexpr int kMaxRows = 128;
constexpr int kExperts = 256;
constexpr int kTopK = 8;
constexpr int kPooled = 4096;                 // pooler output columns (the experts' hidden size)
constexpr int kPoolSlice = kPooled / kRanks;  // 1024 a rank
constexpr int kHidden = 8192;
constexpr int kBranchSlice = kHidden / kRanks;  // 2048 a rank
constexpr int kFrontThreads = 256;
constexpr int kTailThreads = 512;
constexpr float kLog2e = 1.4426950408889634f;

// A rank's workspace: two generations of whole pooled rows, then two of whole branch rows (every rank writes its
// slice of a row into every rank's buffer), then the flag words. The ready flags are written by the peers.
constexpr size_t kPoolElems = size_t(2) * kMaxRows * kPooled;
constexpr size_t kBranchElems = size_t(2) * kMaxRows * kHidden;
constexpr size_t kFlagOffset = (kPoolElems + kBranchElems) * sizeof(uint16_t);
namespace flag {
constexpr int kPoolReady = 0;                    // [row][source rank]
constexpr int kBranchReady = kMaxRows * kRanks;  // [row][source rank]
constexpr int kEpochFront = 2 * kMaxRows * kRanks, kDoneFront = kEpochFront + 1;
constexpr int kEpochTail = kEpochFront + 2, kDoneTail = kEpochFront + 3;
constexpr int kWords = kEpochFront + 4;
}  // namespace flag
constexpr size_t kWorkspaceBytes = kFlagOffset + flag::kWords * sizeof(uint32_t);

// Every rank's workspace, passed by value: the exchange rows and the flag words.
struct Peers {
  uint16_t* data[kRanks];
  Flags flags;

  static Peers of(const int64_t* workspaces) {
    Peers peers;
    for (int r = 0; r < kRanks; ++r)
      peers.data[r] = reinterpret_cast<uint16_t*>(workspaces[r]);
    peers.flags = flags_at(workspaces, kFlagOffset);
    return peers;
  }
  __device__ uint16_t* pooled(int rank, uint32_t generation, int row) const {
    return data[rank] + ((generation & 1) * kMaxRows + row) * kPooled;
  }
  __device__ uint16_t* branch(int rank, uint32_t generation, int row) const {
    return data[rank] + kPoolElems + ((generation & 1) * kMaxRows + row) * kHidden;
  }
};

// This block's slice of a row is in every rank's buffer before the row's flag goes to every rank.
__device__ __forceinline__ void announce(const Peers& peers, int ready, int row, int rank, uint32_t generation) {
  wait_for_stores();
  if (threadIdx.x < kRanks) publish(&peers.flags.rank[threadIdx.x][ready + row * kRanks + rank], generation);
}

__device__ __forceinline__ void await_row(const Peers& peers, int ready, int row, int rank, uint32_t generation) {
  if (threadIdx.x < kRanks) wait_for(&peers.flags.rank[rank][ready + row * kRanks + threadIdx.x], generation);
  __syncthreads();
}

// FP32 split-K partials [kSplits][rows][cols], summed in split order. The split count is a template parameter so
// every load is issued before the first add: with a runtime loop each load waited for the previous add, and the
// front kernel took 30.7 instead of 14.0 us at 48 rows and 16 splits.
template <int kSplits>
__device__ __forceinline__ float4 sum_splits4(const float* partial, size_t split_stride, size_t idx) {
  float4 v[kSplits];
#pragma unroll
  for (int s = 0; s < kSplits; ++s)
    v[s] = *reinterpret_cast<const float4*>(partial + s * split_stride + idx);
  float4 acc = v[0];
#pragma unroll
  for (int s = 1; s < kSplits; ++s) {
    acc.x = acc.x + v[s].x;
    acc.y = acc.y + v[s].y;
    acc.z = acc.z + v[s].z;
    acc.w = acc.w + v[s].w;
  }
  return acc;
}

template <int kSplits>
__device__ __forceinline__ float sum_splits(const float* partial, size_t split_stride, size_t idx) {
  float v[kSplits];
#pragma unroll
  for (int s = 0; s < kSplits; ++s)
    v[s] = partial[s * split_stride + idx];
  float acc = v[0];
#pragma unroll
  for (int s = 1; s < kSplits; ++s)
    acc = acc + v[s];
  return acc;
}

__device__ __forceinline__ uint2 pack4(float4 v) {
  const uint16_t o[4] = {float_to_bf16(v.x), float_to_bf16(v.y), float_to_bf16(v.z), float_to_bf16(v.w)};
  return *reinterpret_cast<const uint2*>(o);
}

// ---------------------------------------------------------------------------------------------------------------
// Front: pooled-slice exchange, router top-8, pre-expert norm. One block of 256 threads (4 waves) per row.
// ---------------------------------------------------------------------------------------------------------------

struct FrontParams {
  const float* pool_partial;    // [splits][m][1024] FP32: this rank's pooler columns
  const float* router_partial;  // [splits][m][512] FP32: router high, then router low
  const float* bias;            // [256] FP32 expert bias
  const uint16_t* norm;         // [4096] BF16 effective pre-expert norm weight
  const int* num_real;          // device count of real rows (later rows are graph padding), or null
  float* scores;                // [m][8]
  int* ids;                     // [m][8]
  uint16_t* expert_input;       // [m][4096] BF16
  float multiplier, scale, eps;
  int rank, m;
};

// One expert's router score for one row, as in _joint_epilogue: logit = high + low,
// score = 1 / (1 + 2^(logit * multiplier * -log2 e)). The selection key is score + bias. A NaN score ranks last with
// weight 0.
template <int kSplits>
__device__ __forceinline__ void expert_score(const FrontParams& p, int row, int e, float* key, float* score) {
  const size_t stride = size_t(p.m) * 2 * kExperts, base = size_t(row) * 2 * kExperts;
  const float logit = sum_splits<kSplits>(p.router_partial, stride, base + e) +
                      sum_splits<kSplits>(p.router_partial, stride, base + kExperts + e);
  const float x = logit * p.multiplier;
  const float s = 1.0f / (1.0f + __builtin_amdgcn_exp2f(x * -kLog2e));
  const float biased = s + p.bias[e];
  key[e] = biased == biased ? biased : -3.0e38f;
  score[e] = s == s ? s : 0.0f;
}

__device__ __forceinline__ float pick4(const float (&v)[4], int i) {
  return i == 0 ? v[0] : i == 1 ? v[1] : i == 2 ? v[2] : v[3];
}

// The top-8 of one row on one wave from the 256 keys and scores in LDS (lane l: experts l + 64 i), as in
// _joint_epilogue: selection by key (ties: the lower expert id), weight = score * (scale / (sum of the selected scores
// in selection order + 1e-15)).
__device__ void route_row(const FrontParams& p, int row, int lane, const float* lds_key, const float* lds_score) {
  float key[4], score[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    key[i] = lds_key[lane + 64 * i];
    score[i] = lds_score[lane + 64 * i];
  }
  float denominator = 0.0f, mine_score = 0.0f;
  int mine_id = 0;
#pragma unroll
  for (int j = 0; j < kTopK; ++j) {
    float best = key[0];
    int best_id = lane;
#pragma unroll
    for (int i = 1; i < 4; ++i)
      if (key[i] > best) {
        best = key[i];
        best_id = lane + 64 * i;
      }
#pragma unroll
    for (int offset = 32; offset >= 1; offset >>= 1) {
      const float other = __shfl_xor(best, offset, 64);
      const int other_id = __shfl_xor(best_id, offset, 64);
      if (other > best || (other == best && other_id < best_id)) {
        best = other;
        best_id = other_id;
      }
    }
    const int owner = best_id % 64, slot = best_id / 64;
    const float selected = __shfl(pick4(score, slot), owner, 64);
    if (lane == owner) key[slot] = -INFINITY;
    denominator = denominator + selected;
    if (lane == j) {
      mine_id = best_id;
      mine_score = selected;
    }
  }
  if (lane < kTopK) {
    const bool real = p.num_real == nullptr || row < *p.num_real;
    p.ids[row * kTopK + lane] = real ? mine_id : -1;  // -1: no rank's expert (graph padding)
    p.scores[row * kTopK + lane] = mine_score * (p.scale / (denominator + 1e-15f));
  }
}

// The pre-expert RMSNorm of one exchanged row: thread t holds columns 1024 r + 4 t + e (r: the source rank), in
// _joint_epilogue's order: per-thread pairs fma(x0, x0, x2 x2) and fma(x1, x1, x3 x3), lane butterfly 8, 4, 2, 1,
// 16, 32, waves (0 + 2) + (1 + 3), then bf16((x * rstd) * weight).
__device__ void norm_row(const FrontParams& p, const Peers& peers, uint32_t generation, int row, float* cells) {
  const int t = threadIdx.x, lane = t % 64, wave = t / 64;
  float x[kRanks][4];
#pragma unroll
  for (int r = 0; r < kRanks; ++r) {
    const uint2 raw = *reinterpret_cast<const uint2*>(peers.pooled(p.rank, generation, row) + r * kPoolSlice + 4 * t);
    const uint16_t* v = reinterpret_cast<const uint16_t*>(&raw);
#pragma unroll
    for (int e = 0; e < 4; ++e)
      x[r][e] = bf16_to_float(v[e]);
  }
  float lo[kRanks], hi[kRanks];
#pragma unroll
  for (int r = 0; r < kRanks; ++r) {
    lo[r] = __builtin_fmaf(x[r][0], x[r][0], x[r][2] * x[r][2]);
    hi[r] = __builtin_fmaf(x[r][1], x[r][1], x[r][3] * x[r][3]);
  }
  float ss = ((lo[0] + lo[1]) + (lo[2] + lo[3])) + ((hi[0] + hi[1]) + (hi[2] + hi[3]));
  constexpr int kButterfly[6] = {8, 4, 2, 1, 16, 32};
#pragma unroll
  for (int i = 0; i < 6; ++i)
    ss = ss + __shfl_xor(ss, kButterfly[i], 64);
  if (lane == 0) cells[wave] = ss;
  __syncthreads();
  const float total = (cells[0] + cells[2]) + (cells[1] + cells[3]);
  const float rstd = __builtin_amdgcn_rsqf(total * (1.0f / kPooled) + p.eps);
#pragma unroll
  for (int r = 0; r < kRanks; ++r) {
    const int col = r * kPoolSlice + 4 * t;
    const uint2 w = *reinterpret_cast<const uint2*>(p.norm + col);
    const uint16_t* wv = reinterpret_cast<const uint16_t*>(&w);
    const float4 y = make_float4(
        (x[r][0] * rstd) * bf16_to_float(wv[0]),
        (x[r][1] * rstd) * bf16_to_float(wv[1]),
        (x[r][2] * rstd) * bf16_to_float(wv[2]),
        (x[r][3] * rstd) * bf16_to_float(wv[3]));
    *reinterpret_cast<uint2*>(p.expert_input + size_t(row) * kPooled + col) = pack4(y);
  }
}

template <int kSplits>
__global__ void __launch_bounds__(kFrontThreads, 1) front_epilogue_kernel(FrontParams p, Peers peers) {
  __shared__ float cells[4], key[kExperts], score[kExperts];
  const int row = blockIdx.x, t = threadIdx.x;
  uint32_t* local = peers.flags.rank[p.rank];
  const uint32_t generation = begin_call(local, flag::kEpochFront);
  // 1. This rank's pooled slice of the row (thread t: columns 4 t .. 4 t + 3) to every rank, and expert t's score.
  const uint2 packed =
      pack4(sum_splits4<kSplits>(p.pool_partial, size_t(p.m) * kPoolSlice, size_t(row) * kPoolSlice + 4 * t));
#pragma unroll
  for (int r = 0; r < kRanks; ++r)
    *reinterpret_cast<uint2*>(peers.pooled(r, generation, row) + p.rank * kPoolSlice + 4 * t) = packed;
  expert_score<kSplits>(p, row, t, key, score);
  announce(peers, flag::kPoolReady, row, p.rank, generation);
  // 2. The top-8 needs no peer data: it runs while the other ranks' slices arrive.
  if (t < 64) route_row(p, row, t, key, score);
  // 3. The pre-expert norm over the four slices.
  await_row(peers, flag::kPoolReady, row, p.rank, generation);
  norm_row(p, peers, generation, row, cells);
  end_call(local, flag::kEpochFront, flag::kDoneFront, generation);
}

// ---------------------------------------------------------------------------------------------------------------
// Tail: branch-slice exchange and the FFN boundary. One block of 512 threads (8 waves) per row.
// ---------------------------------------------------------------------------------------------------------------

struct TailParams {
  const float* partial;         // [splits][m][2048] FP32: this rank's unpool columns
  const float* stream;          // [m][8192] FP32 residual stream
  const float* alpha;           // [8192] residual gate coefficients
  const float* beta;            // [8192]
  const uint16_t* post_weight;  // [8192] BF16 effective post-norm (input scale) weight
  const uint16_t* pre_weight;   // [8192] BF16 effective next pre-attention norm weight
  uint16_t* next_input;         // [m][8192] BF16
  float* new_stream;            // [m][8192] FP32
  float post_eps, pre_eps;
  bool pre_bf16;  // the last layer rounds the stream to BF16 before the final norm
  int rank, m;
};

// Thread t's 16 columns of a row: (t + 512 q) * 8 + k, q < 2, k < 8 (each pack lies in one rank's slice).
struct TailOperands {
  float stream[16], alpha[16], beta[16], post_w[16], pre_w[16];

  __device__ void load(const TailParams& p, int row) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int col = (threadIdx.x + q * kTailThreads) * 8;
      const float4* st = reinterpret_cast<const float4*>(p.stream + size_t(row) * kHidden + col);
      const float4* al = reinterpret_cast<const float4*>(p.alpha + col);
      const float4* be = reinterpret_cast<const float4*>(p.beta + col);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float4 sv = st[h], av = al[h], bv = be[h];
        const int i = q * 8 + h * 4;
        stream[i] = sv.x, stream[i + 1] = sv.y, stream[i + 2] = sv.z, stream[i + 3] = sv.w;
        alpha[i] = av.x, alpha[i + 1] = av.y, alpha[i + 2] = av.z, alpha[i + 3] = av.w;
        beta[i] = bv.x, beta[i + 1] = bv.y, beta[i + 2] = bv.z, beta[i + 3] = bv.w;
      }
      const Bf16x8 pw = *reinterpret_cast<const Bf16x8*>(p.post_weight + col);
      const Bf16x8 qw = *reinterpret_cast<const Bf16x8*>(p.pre_weight + col);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        post_w[q * 8 + k] = bf16_to_float(pw.v[k]);
        pre_w[q * 8 + k] = bf16_to_float(qw.v[k]);
      }
    }
  }
};

// The formula of the Triton _fused_norm_residual_triton_kernel for the FFN boundary: post = bf16(branch * post_w),
// normed = bf16(post * rsqrt(mean(post^2) + post_eps)), stream' = alpha * stream + beta * normed,
// next input = bf16(pre * rsqrt(mean(pre^2) + pre_eps) * pre_w) with pre = stream' (or bf16(stream')).
__device__ void
ffn_boundary(const Bf16x8 (&branch)[2], int row, const TailOperands& op, const TailParams& p, float* cells) {
  float post[16];
  float ss = 0.0f;
#pragma unroll
  for (int q = 0; q < 2; ++q)
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = q * 8 + k;
      post[i] = bf16_to_float(float_to_bf16(bf16_to_float(branch[q].v[k]) * op.post_w[i]));
      ss = ss + post[i] * post[i];
    }
  const float post_rstd = __builtin_amdgcn_rsqf(block_sum<kTailThreads>(ss, cells) / kHidden + p.post_eps);
  float stream[16], pre[16];
  ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float normed = bf16_to_float(float_to_bf16(post[i] * post_rstd));
    stream[i] = op.alpha[i] * op.stream[i] + op.beta[i] * normed;
    pre[i] = p.pre_bf16 ? bf16_to_float(float_to_bf16(stream[i])) : stream[i];
    ss = ss + pre[i] * pre[i];
  }
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int col = (threadIdx.x + q * kTailThreads) * 8;
    float4* ns = reinterpret_cast<float4*>(p.new_stream + size_t(row) * kHidden + col);
    ns[0] = make_float4(stream[q * 8 + 0], stream[q * 8 + 1], stream[q * 8 + 2], stream[q * 8 + 3]);
    ns[1] = make_float4(stream[q * 8 + 4], stream[q * 8 + 5], stream[q * 8 + 6], stream[q * 8 + 7]);
  }
  const float pre_rstd = __builtin_amdgcn_rsqf(block_sum<kTailThreads>(ss, cells + 8) / kHidden + p.pre_eps);
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int col = (threadIdx.x + q * kTailThreads) * 8;
    Bf16x8 o;
#pragma unroll
    for (int k = 0; k < 8; ++k)
      o.v[k] = float_to_bf16(pre[q * 8 + k] * pre_rstd * op.pre_w[q * 8 + k]);
    *reinterpret_cast<Bf16x8*>(p.next_input + size_t(row) * kHidden + col) = o;
  }
}

template <int kSplits>
__global__ void __launch_bounds__(kTailThreads, 1) tail_boundary_kernel(TailParams p, Peers peers) {
  __shared__ float cells[16];
  const int row = blockIdx.x, t = threadIdx.x;
  uint32_t* local = peers.flags.rank[p.rank];
  const uint32_t generation = begin_call(local, flag::kEpochTail);
  // 1. This rank's branch slice of the row (thread t: columns 4 t .. 4 t + 3) to every rank.
  const uint2 packed =
      pack4(sum_splits4<kSplits>(p.partial, size_t(p.m) * kBranchSlice, size_t(row) * kBranchSlice + 4 * t));
#pragma unroll
  for (int r = 0; r < kRanks; ++r)
    *reinterpret_cast<uint2*>(peers.branch(r, generation, row) + p.rank * kBranchSlice + 4 * t) = packed;
  // 2. The row's own operands load while the stores complete and the other ranks' slices arrive.
  TailOperands op;
  op.load(p, row);
  announce(peers, flag::kBranchReady, row, p.rank, generation);
  await_row(peers, flag::kBranchReady, row, p.rank, generation);
  Bf16x8 branch[2];
#pragma unroll
  for (int q = 0; q < 2; ++q)
    branch[q] = *reinterpret_cast<const Bf16x8*>(peers.branch(p.rank, generation, row) + (t + q * kTailThreads) * 8);
  ffn_boundary(branch, row, op, p, cells);
  end_call(local, flag::kEpochTail, flag::kDoneTail, generation);
}

// ---------------------------------------------------------------------------------------------------------------
// Host entry points. `peers` (CPU int64 [4]) is every rank's workspace address (tp4_peer_workspace.py) and `rank`
// this rank. Rows: 1..128, one block each; split-K counts 4, 8 or 16.
// ---------------------------------------------------------------------------------------------------------------

inline Peers workspace_peers(const tvm::ffi::TensorView peers, int64_t workspace_bytes) {
  using namespace host;
  TensorMatcher({kRanks}).with_dtype<int64_t>().with_device<kDLCPU>().verify(peers);
  RuntimeCheck(
      workspace_bytes >= static_cast<int64_t>(kWorkspaceBytes), "Muse TP4 split pooler/unpool workspace is too small");
  return Peers::of(static_cast<const int64_t*>(peers.data_ptr()));
}

// Calls `launch` with the split-K count as a compile-time constant.
template <typename Launch>
inline void dispatch_splits(int64_t splits, Launch&& launch) {
  using namespace host;
  switch (splits) {
    case 4:
      return launch(std::integral_constant<int, 4>{});
    case 8:
      return launch(std::integral_constant<int, 8>{});
    case 16:
      return launch(std::integral_constant<int, 16>{});
    default:
      RuntimeCheck(false, "split-K count must be 4, 8 or 16, got ", splits);
  }
}

struct SplitPoolUnpool {
  static void front_epilogue(
      const tvm::ffi::TensorView peers,
      int64_t workspace_bytes,
      int64_t rank,
      const tvm::ffi::TensorView pool_partial,
      const tvm::ffi::TensorView router_partial,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView norm,
      const tvm::ffi::Optional<tvm::ffi::TensorView> num_real,
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView expert_input,
      double multiplier,
      double scale,
      double eps) {
    using namespace host;
    auto S = SymbolicSize{"splits"};
    auto M = SymbolicSize{"num_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({S, M, kPoolSlice}).with_dtype<float>().with_device(device).verify(pool_partial);
    TensorMatcher({S, M, 2 * kExperts}).with_dtype<float>().with_device(device).verify(router_partial);
    TensorMatcher({kExperts}).with_dtype<float>().with_device(device).verify(bias);
    TensorMatcher({kPooled}).with_dtype<bf16_t>().with_device(device).verify(norm);
    TensorMatcher({M, kTopK}).with_dtype<float>().with_device(device).verify(scores);
    TensorMatcher({M, kTopK}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({M, kPooled}).with_dtype<bf16_t>().with_device(device).verify(expert_input);
    const int* real = nullptr;
    if (num_real.has_value()) {
      TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_real.value());
      real = static_cast<const int*>(num_real.value().data_ptr());
    }
    const int64_t m = M.unwrap();
    RuntimeCheck(0 < m && m <= kMaxRows, "rows must be in 1..", kMaxRows, ", got ", m);

    const Peers exchange = workspace_peers(peers, workspace_bytes);
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    const FrontParams params{
        static_cast<const float*>(pool_partial.data_ptr()),
        static_cast<const float*>(router_partial.data_ptr()),
        static_cast<const float*>(bias.data_ptr()),
        static_cast<const uint16_t*>(norm.data_ptr()),
        real,
        static_cast<float*>(scores.data_ptr()),
        static_cast<int*>(ids.data_ptr()),
        static_cast<uint16_t*>(expert_input.data_ptr()),
        static_cast<float>(multiplier),
        static_cast<float>(scale),
        static_cast<float>(eps),
        static_cast<int>(rank),
        static_cast<int>(m)};
    dispatch_splits(S.unwrap(), [&](auto splits) {
      LaunchKernel(static_cast<uint32_t>(m), kFrontThreads, stream)(
          front_epilogue_kernel<decltype(splits)::value>, params, exchange);
    });
  }

  static void tail_boundary(
      const tvm::ffi::TensorView peers,
      int64_t workspace_bytes,
      int64_t rank,
      const tvm::ffi::TensorView partial,
      const tvm::ffi::TensorView stream_in,
      const tvm::ffi::TensorView alpha,
      const tvm::ffi::TensorView beta,
      const tvm::ffi::TensorView post_weight,
      const tvm::ffi::TensorView pre_weight,
      const tvm::ffi::TensorView next_input,
      const tvm::ffi::TensorView new_stream,
      double post_eps,
      double pre_eps,
      bool pre_bf16) {
    using namespace host;
    auto S = SymbolicSize{"splits"};
    auto M = SymbolicSize{"num_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({S, M, kBranchSlice}).with_dtype<float>().with_device(device).verify(partial);
    TensorMatcher({M, kHidden}).with_dtype<float>().with_device(device).verify(stream_in).verify(new_stream);
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device).verify(next_input);
    TensorMatcher({kHidden}).with_dtype<float>().with_device(device).verify(alpha).verify(beta);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).verify(post_weight).verify(pre_weight);
    const int64_t m = M.unwrap();
    RuntimeCheck(0 < m && m <= kMaxRows, "rows must be in 1..", kMaxRows, ", got ", m);

    const Peers exchange = workspace_peers(peers, workspace_bytes);
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    const TailParams params{
        static_cast<const float*>(partial.data_ptr()),
        static_cast<const float*>(stream_in.data_ptr()),
        static_cast<const float*>(alpha.data_ptr()),
        static_cast<const float*>(beta.data_ptr()),
        static_cast<const uint16_t*>(post_weight.data_ptr()),
        static_cast<const uint16_t*>(pre_weight.data_ptr()),
        static_cast<uint16_t*>(next_input.data_ptr()),
        static_cast<float*>(new_stream.data_ptr()),
        static_cast<float>(post_eps),
        static_cast<float>(pre_eps),
        pre_bf16,
        static_cast<int>(rank),
        static_cast<int>(m)};
    dispatch_splits(S.unwrap(), [&](auto splits) {
      LaunchKernel(static_cast<uint32_t>(m), kTailThreads, stream)(
          tail_boundary_kernel<decltype(splits)::value>, params, exchange);
    });
  }
};

}  // namespace muse_tp4_split_pool_unpool
}  // namespace sglang
