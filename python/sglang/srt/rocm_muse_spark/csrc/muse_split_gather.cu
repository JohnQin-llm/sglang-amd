// Muse Spark TP4 target verify at 48..128 rows: the MoE pooler and unpool split four ways across the TP ranks.
//
// Each rank projects a quarter of the columns (pooler 1024 of 4096, unpool 2048 of 8192) with a split-K GEMM.
// The two kernels here sum that GEMM's FP32 partials, exchange the BF16 slices between the four ranks and run
// the consumer that needs whole rows:
// * front_epilogue: per row, this rank's pooled slice; the router top-8 (the router is replicated: its high and
//   low BF16 parts come out of the same GEMM as the pooler slice); then, once all four slices of the row are
//   exchanged, the pre-expert RMSNorm of the 4096 pooled columns -> the expert input.
// * tail_boundary: per row, this rank's branch slice; once all four are exchanged, the FFN boundary (input-scale
//   post-norm, gated residual, the next layer's pre-attention norm) on the 8192 branch columns.
//
// Arithmetic: the router and the pre-expert norm follow _joint_epilogue (router_pooler_topk.py) with the
// instruction choices and summation order the TP4 K2 front uses to reproduce it bitwise (tp4mono tpm_front.py
// _route_token / stage_xq). The split-K partials are summed in split order 0, 1, ...; the FFN boundary keeps the
// Triton formula (fused_norm_residual.py) with its two row sums in this kernel's order.
//
// Exchange (push): a rank stores its slice of a row into every rank's uncached exchange buffer, waits for the
// stores, then raises the row's flag in every rank's flag buffer; a reader polls its local flags and reads the whole
// row from its own buffer. Measured against pulling the slices from the owners' buffers (results/bench-layer-pull.log
// vs bench-layer-push.log): the tail kernel 13.5 -> 12.6 us at 48 rows, 17.0 -> 14.8 at 80, 21.7 -> 18.5 at 128; the
// front kernel within 0.4 us. Each kernel keeps a call counter (epoch) in the flag buffer; generation = epoch + 1,
// the same on every rank because all ranks launch the same sequence. The buffers alternate by generation parity: a
// rank is at most one call ahead of a peer (it cannot finish a call before every peer has published that call's
// rows), so it never overwrites data the peer still reads.
#include <ATen/hip/HIPContext.h>
#include <torch/extension.h>

#include <optional>

namespace {

constexpr int kRanks = 4;
constexpr int kMaxRows = 128;
constexpr int kExperts = 256;
constexpr int kTopK = 8;
constexpr int kPooled = 4096;                           // pooler output columns (expert hidden size)
constexpr int kPoolSlice = kPooled / kRanks;            // 1024 a rank
constexpr int kFrontCols = kPoolSlice + 2 * kExperts;   // front GEMM output: pool slice, router high, router low
constexpr int kHidden = 8192;
constexpr int kBranchSlice = kHidden / kRanks;          // 2048 a rank
constexpr float kLog2e = 1.4426950408889634f;

// uint32 offsets in a rank's flag buffer. The ready flags are written by the peers.
namespace flag {
constexpr int kPoolReady = 0;                           // [row][source rank]
constexpr int kBranchReady = kMaxRows * kRanks;         // [row][source rank]
constexpr int kEpochFront = 2 * kMaxRows * kRanks, kDoneFront = kEpochFront + 1;
constexpr int kEpochTail = kEpochFront + 2, kDoneTail = kEpochFront + 3;
constexpr int kWords = kEpochFront + 4;
}  // namespace flag

// A rank's exchange buffer (BF16 elements): two generations of whole pooled rows, then two of whole branch rows;
// every rank writes its slice of a row into every rank's buffer.
constexpr size_t kPoolElems = size_t(2) * kMaxRows * kPooled;
constexpr size_t kBranchElems = size_t(2) * kMaxRows * kHidden;

struct Peers {
    uint16_t* data[kRanks];   // every rank's exchange buffer (IPC-mapped)
    uint32_t* flags[kRanks];  // every rank's flag buffer (IPC-mapped)

    __device__ uint16_t* pooled(int rank, uint32_t generation, int row) const {
        return data[rank] + ((generation & 1) * kMaxRows + row) * kPooled;
    }
    __device__ uint16_t* branch(int rank, uint32_t generation, int row) const {
        return data[rank] + kPoolElems + ((generation & 1) * kMaxRows + row) * kHidden;
    }
};

struct alignas(16) Pack8 {
    uint16_t v[8];
};

__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float(uint32_t(u) << 16); }

__device__ __forceinline__ uint16_t f2bf(float f) {
    return __builtin_bit_cast(uint16_t, static_cast<__bf16>(f));  // RNE, the gfx950 conversion instruction
}

__device__ __forceinline__ void publish(uint32_t* flag, uint32_t generation) {
    __scoped_atomic_store_n(flag, generation, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// Polls with a short back-off (s_sleep 2), as boundary-fusion's kernels do.
__device__ __forceinline__ void wait_for(const uint32_t* flag, uint32_t generation) {
    while (__scoped_atomic_load_n(flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < generation)
        __builtin_amdgcn_s_sleep(2);
}

// This call's generation. Read with an atomic so the load cannot come from the scalar cache.
__device__ __forceinline__ uint32_t begin_call(const uint32_t* local, int epoch) {
    return __scoped_atomic_load_n(&local[epoch], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1;
}

// The last block of a call to finish advances the kernel's epoch.
__device__ __forceinline__ void end_call(uint32_t* local, int epoch, int done, uint32_t generation) {
    __syncthreads();
    if (threadIdx.x == 0 && __scoped_atomic_fetch_add(&local[done], 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) ==
                                gridDim.x - 1) {
        __scoped_atomic_store_n(&local[done], 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
        __scoped_atomic_store_n(&local[epoch], generation, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
}

// Stores the row's slice (written by this block before the call) is complete, then flags it to every rank.
__device__ __forceinline__ void announce(const Peers& peers, int ready, int row, int rank, uint32_t generation) {
    __builtin_amdgcn_s_waitcnt(0);  // the slice is in (uncached) memory before any flag goes out
    __syncthreads();
    if (threadIdx.x < kRanks) publish(&peers.flags[threadIdx.x][ready + row * kRanks + rank], generation);
}

__device__ __forceinline__ void await_row(const Peers& peers, int ready, int row, int rank, uint32_t generation) {
    if (threadIdx.x < kRanks) wait_for(&peers.flags[rank][ready + row * kRanks + threadIdx.x], generation);
    __syncthreads();
}

// Four consecutive columns of one row of FP32 split-K partials [kSplits][m][cols], summed in split order. The split
// count is a template parameter so every load is issued before the first add. With a runtime-length loop each load
// waited for the previous add: the front kernel took 30.7 us at 48 rows / 16 splits, 14.0 us templated
// (results/bench-layer-v1.log vs bench-layer-pull.log; that change also moved the router scores onto all 256 threads).
template <int kSplits>
__device__ __forceinline__ float4 sum_splits4(const float* partial, size_t split_stride, size_t idx) {
    float4 v[kSplits];
#pragma unroll
    for (int s = 0; s < kSplits; ++s) v[s] = *reinterpret_cast<const float4*>(partial + s * split_stride + idx);
    float4 acc = v[0];
#pragma unroll
    for (int s = 1; s < kSplits; ++s) acc.x = acc.x + v[s].x, acc.y = acc.y + v[s].y, acc.z = acc.z + v[s].z,
                                      acc.w = acc.w + v[s].w;
    return acc;
}

template <int kSplits>
__device__ __forceinline__ float sum_splits(const float* partial, size_t split_stride, size_t idx) {
    float v[kSplits];
#pragma unroll
    for (int s = 0; s < kSplits; ++s) v[s] = partial[s * split_stride + idx];
    float acc = v[0];
#pragma unroll
    for (int s = 1; s < kSplits; ++s) acc = acc + v[s];
    return acc;
}

__device__ __forceinline__ uint2 pack4(float4 v) {
    const uint16_t o[4] = {f2bf(v.x), f2bf(v.y), f2bf(v.z), f2bf(v.w)};
    return *reinterpret_cast<const uint2*>(o);
}

// ------------------------------------------------------------------------------------------------------------
// Front: pooled-slice exchange, router top-8, pre-expert norm. One block of 256 threads (4 waves) per row.
// ------------------------------------------------------------------------------------------------------------

struct FrontArgs {
    const float* partial;  // [splits][m][kFrontCols] FP32: pool slice, router high, router low
    const float* bias;     // [256] FP32 expert bias
    const uint16_t* norm;  // [4096] BF16 effective pre-expert norm weight
    float multiplier, scale, eps;
    const int* num_real;   // device count of real rows (rows after it are graph padding), or null
    float* scores;         // [m][8]
    int* ids;              // [m][8]
    uint16_t* xe;          // [m][4096] BF16
};

// One expert's router score for one row (thread t: expert t), as in _joint_epilogue: logit = high + low,
// score = 1 / (1 + 2^(logit * multiplier * -log2 e)). The selection key is score + bias; a NaN score ranks last with
// weight 0, as the K2 route does.
template <int kSplits>
__device__ __forceinline__ void expert_score(const FrontArgs& a, int row, int m, int e, float* key, float* score) {
    const size_t stride = size_t(m) * kFrontCols, base = size_t(row) * kFrontCols + kPoolSlice;
    const float logit = sum_splits<kSplits>(a.partial, stride, base + e) +
                        sum_splits<kSplits>(a.partial, stride, base + kExperts + e);
    const float x = logit * a.multiplier;
    const float s = 1.0f / (1.0f + __builtin_amdgcn_exp2f(x * -kLog2e));
    const float biased = s + a.bias[e];
    key[e] = biased == biased ? biased : -3.0e38f;
    score[e] = s == s ? s : 0.0f;
}

__device__ __forceinline__ float pick4(const float (&v)[4], int i) {
    return i == 0 ? v[0] : i == 1 ? v[1] : i == 2 ? v[2] : v[3];
}

// The top-8 of one row on one wave from the 256 keys / scores in LDS (lane l: experts l + 64 i), as in
// _joint_epilogue: selection by key (ties: the lower expert id), weight = score * (scale / (sum of the selected
// scores in selection order + 1e-15)).
__device__ void route_row(const FrontArgs& a, int row, int lane, const float* lds_key, const float* lds_score) {
    float key[4], score[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) key[i] = lds_key[lane + 64 * i], score[i] = lds_score[lane + 64 * i];
    float denominator = 0.0f, mine_score = 0.0f;
    int mine_id = 0;
#pragma unroll
    for (int j = 0; j < kTopK; ++j) {
        float best = key[0];
        int best_id = lane;
#pragma unroll
        for (int i = 1; i < 4; ++i)
            if (key[i] > best) best = key[i], best_id = lane + 64 * i;
#pragma unroll
        for (int offset = 32; offset >= 1; offset >>= 1) {
            const float other = __shfl_xor(best, offset, 64);
            const int other_id = __shfl_xor(best_id, offset, 64);
            if (other > best || (other == best && other_id < best_id)) best = other, best_id = other_id;
        }
        const int owner = best_id % 64, slot = best_id / 64;
        const float selected = __shfl(pick4(score, slot), owner, 64);
        if (lane == owner) key[slot] = -INFINITY;
        denominator = denominator + selected;
        if (lane == j) mine_id = best_id, mine_score = selected;
    }
    if (lane < kTopK) {
        const bool real = a.num_real == nullptr || row < *a.num_real;
        a.ids[row * kTopK + lane] = real ? mine_id : -1;  // -1: no rank's expert (graph padding)
        a.scores[row * kTopK + lane] = mine_score * (a.scale / (denominator + 1e-15f));
    }
}

// The pre-expert RMSNorm of one exchanged row: thread t holds columns r * 1024 + 4 t + e (r: the source rank),
// _joint_epilogue's order -- per-thread pairs fma(x0, x0, x2 x2) and fma(x1, x1, x3 x3), lane butterfly
// 8, 4, 2, 1, 16, 32, waves (0 + 2) + (1 + 3) -- then bf16((x * rstd) * weight).
__device__ void norm_row(const FrontArgs& a, const Peers& peers, int rank, uint32_t generation, int row, float* cells) {
    const int t = threadIdx.x, lane = t % 64, wave = t / 64;
    float x[kRanks][4];
#pragma unroll
    for (int r = 0; r < kRanks; ++r) {
        const uint2 raw = *reinterpret_cast<const uint2*>(peers.pooled(rank, generation, row) + r * kPoolSlice + 4 * t);
        const uint16_t* v = reinterpret_cast<const uint16_t*>(&raw);
#pragma unroll
        for (int e = 0; e < 4; ++e) x[r][e] = bf2f(v[e]);
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
    for (int i = 0; i < 6; ++i) ss = ss + __shfl_xor(ss, kButterfly[i], 64);
    if (lane == 0) cells[wave] = ss;
    __syncthreads();
    const float total = (cells[0] + cells[2]) + (cells[1] + cells[3]);
    const float rstd = __builtin_amdgcn_rsqf(total * (1.0f / kPooled) + a.eps);
#pragma unroll
    for (int r = 0; r < kRanks; ++r) {
        const int col = r * kPoolSlice + 4 * t;
        const uint2 w = *reinterpret_cast<const uint2*>(a.norm + col);
        const uint16_t* wv = reinterpret_cast<const uint16_t*>(&w);
        const float4 y = make_float4((x[r][0] * rstd) * bf2f(wv[0]), (x[r][1] * rstd) * bf2f(wv[1]),
                                     (x[r][2] * rstd) * bf2f(wv[2]), (x[r][3] * rstd) * bf2f(wv[3]));
        *reinterpret_cast<uint2*>(a.xe + size_t(row) * kPooled + col) = pack4(y);
    }
}

template <int kSplits>
__global__ void __launch_bounds__(256, 1) front_epilogue_kernel(FrontArgs a, Peers peers, int rank, int m) {
    __shared__ float cells[4], key[kExperts], score[kExperts];
    const int row = blockIdx.x, t = threadIdx.x;
    uint32_t* local = peers.flags[rank];
    const uint32_t generation = begin_call(local, flag::kEpochFront);
    // 1. This rank's pooled slice of the row: thread t sums columns 4t..4t+3 over the splits, rounds to BF16; and
    //    the router score of expert t.
    const float4 v =
        sum_splits4<kSplits>(a.partial, size_t(m) * kFrontCols, size_t(row) * kFrontCols + 4 * t);
    const uint2 packed = pack4(v);
#pragma unroll
    for (int r = 0; r < kRanks; ++r)
        *reinterpret_cast<uint2*>(peers.pooled(r, generation, row) + rank * kPoolSlice + 4 * t) = packed;
    expert_score<kSplits>(a, row, m, t, key, score);
    announce(peers, flag::kPoolReady, row, rank, generation);
    // 2. The top-8 needs no peer data: it runs while the other ranks' slices arrive.
    if (t < 64) route_row(a, row, t, key, score);
    // 3. The pre-expert norm over the four slices.
    await_row(peers, flag::kPoolReady, row, rank, generation);
    norm_row(a, peers, rank, generation, row, cells);
    end_call(local, flag::kEpochFront, flag::kDoneFront, generation);
}

// ------------------------------------------------------------------------------------------------------------
// Tail: branch-slice exchange and the FFN boundary. One block of 512 threads (8 waves) per row.
// ------------------------------------------------------------------------------------------------------------

constexpr int kTailThreads = 512;

struct TailArgs {
    const float* partial;  // [splits][m][kBranchSlice] FP32
    const float* stream;   // [m][8192] FP32 residual stream
    const float* alpha;    // [8192] residual gate coefficients
    const float* beta;
    const uint16_t* post_weight;  // [8192] BF16 effective post-norm (input scale) weight
    const uint16_t* pre_weight;   // [8192] BF16 effective next pre-attention norm weight
    float post_eps, pre_eps;
    bool pre_bf16;          // the last layer rounds the stream to BF16 before the final norm
    uint16_t* next_input;  // [m][8192] BF16
    float* new_stream;     // [m][8192] FP32
};

// Thread t's 16 columns of a row: (t + p * 512) * 8 + k, p < 2, k < 8 (each pack lies in one rank's slice).
struct TailOperands {
    float stream[16], alpha[16], beta[16], post_w[16], pre_w[16];

    __device__ void load(const TailArgs& a, int row) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int col = (threadIdx.x + p * kTailThreads) * 8;
            const float4* st = reinterpret_cast<const float4*>(a.stream + size_t(row) * kHidden + col);
            const float4* al = reinterpret_cast<const float4*>(a.alpha + col);
            const float4* be = reinterpret_cast<const float4*>(a.beta + col);
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const float4 sv = st[q], av = al[q], bv = be[q];
                const int i = p * 8 + q * 4;
                stream[i] = sv.x, stream[i + 1] = sv.y, stream[i + 2] = sv.z, stream[i + 3] = sv.w;
                alpha[i] = av.x, alpha[i + 1] = av.y, alpha[i + 2] = av.z, alpha[i + 3] = av.w;
                beta[i] = bv.x, beta[i + 1] = bv.y, beta[i + 2] = bv.z, beta[i + 3] = bv.w;
            }
            const Pack8 pw = *reinterpret_cast<const Pack8*>(a.post_weight + col);
            const Pack8 qw = *reinterpret_cast<const Pack8*>(a.pre_weight + col);
#pragma unroll
            for (int k = 0; k < 8; ++k) post_w[p * 8 + k] = bf2f(pw.v[k]), pre_w[p * 8 + k] = bf2f(qw.v[k]);
        }
    }
};

// Sum over the block's 8 waves; `cells` holds 8 floats used by this reduction only.
__device__ __forceinline__ float block_sum(float v, float* cells) {
#pragma unroll
    for (int offset = 32; offset >= 1; offset >>= 1) v = v + __shfl_xor(v, offset, 64);
    if (threadIdx.x % 64 == 0) cells[threadIdx.x / 64] = v;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < kTailThreads / 64; ++w) total = total + cells[w];
    return total;
}

// The Triton _fused_norm_residual_triton_kernel formula for the FFN boundary: post_input = bf16(branch * post_w),
// normed = bf16(post_input * rsqrt(mean(post_input^2) + post_eps)), stream' = alpha * stream + beta * normed,
// next input = bf16(pre * rsqrt(mean(pre^2) + pre_eps) * pre_w) with pre = stream' (or bf16(stream')).
__device__ void ffn_boundary(const Pack8 (&branch)[2], int row, const TailOperands& op, const TailArgs& a,
                             float* cells) {
    float post[16];
    float ss = 0.0f;
#pragma unroll
    for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int i = p * 8 + k;
            post[i] = bf2f(f2bf(bf2f(branch[p].v[k]) * op.post_w[i]));
            ss = ss + post[i] * post[i];
        }
    const float post_rstd = __builtin_amdgcn_rsqf(block_sum(ss, cells) / kHidden + a.post_eps);
    float stream[16], pre[16];
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float normed = bf2f(f2bf(post[i] * post_rstd));
        stream[i] = op.alpha[i] * op.stream[i] + op.beta[i] * normed;
        pre[i] = a.pre_bf16 ? bf2f(f2bf(stream[i])) : stream[i];
        ss = ss + pre[i] * pre[i];
    }
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int col = (threadIdx.x + p * kTailThreads) * 8;
        float4* ns = reinterpret_cast<float4*>(a.new_stream + size_t(row) * kHidden + col);
        ns[0] = make_float4(stream[p * 8 + 0], stream[p * 8 + 1], stream[p * 8 + 2], stream[p * 8 + 3]);
        ns[1] = make_float4(stream[p * 8 + 4], stream[p * 8 + 5], stream[p * 8 + 6], stream[p * 8 + 7]);
    }
    const float pre_rstd = __builtin_amdgcn_rsqf(block_sum(ss, cells + 8) / kHidden + a.pre_eps);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int col = (threadIdx.x + p * kTailThreads) * 8;
        Pack8 o;
#pragma unroll
        for (int k = 0; k < 8; ++k) o.v[k] = f2bf(pre[p * 8 + k] * pre_rstd * op.pre_w[p * 8 + k]);
        *reinterpret_cast<Pack8*>(a.next_input + size_t(row) * kHidden + col) = o;
    }
}

template <int kSplits>
__global__ void __launch_bounds__(kTailThreads, 1) tail_boundary_kernel(TailArgs a, Peers peers, int rank, int m) {
    __shared__ float cells[16];
    const int row = blockIdx.x, t = threadIdx.x;
    uint32_t* local = peers.flags[rank];
    const uint32_t generation = begin_call(local, flag::kEpochTail);
    // 1. This rank's branch slice of the row: thread t sums columns 4t..4t+3 over the splits, rounds to BF16.
    const float4 v = sum_splits4<kSplits>(a.partial, size_t(m) * kBranchSlice, size_t(row) * kBranchSlice + 4 * t);
    const uint2 packed = pack4(v);
#pragma unroll
    for (int r = 0; r < kRanks; ++r)
        *reinterpret_cast<uint2*>(peers.branch(r, generation, row) + rank * kBranchSlice + 4 * t) = packed;
    // 2. The row's local operands load while the stores complete and the other ranks' slices arrive.
    TailOperands op;
    op.load(a, row);
    announce(peers, flag::kBranchReady, row, rank, generation);
    await_row(peers, flag::kBranchReady, row, rank, generation);
    Pack8 branch[2];
#pragma unroll
    for (int p = 0; p < 2; ++p)
        branch[p] = *reinterpret_cast<const Pack8*>(peers.branch(rank, generation, row) + (t + p * kTailThreads) * 8);
    ffn_boundary(branch, row, op, a, cells);
    end_call(local, flag::kEpochTail, flag::kDoneTail, generation);
}

Peers make_peers(const std::vector<int64_t>& data, const std::vector<int64_t>& flags) {
    TORCH_CHECK(data.size() == kRanks && flags.size() == kRanks);
    Peers peers;
    for (int r = 0; r < kRanks; ++r) {
        peers.data[r] = reinterpret_cast<uint16_t*>(data[r]);
        peers.flags[r] = reinterpret_cast<uint32_t*>(flags[r]);
    }
    return peers;
}

void check_partial(const torch::Tensor& partial, int64_t cols) {
    TORCH_CHECK(partial.dtype() == torch::kFloat32 && partial.is_contiguous() && partial.dim() == 3);
    TORCH_CHECK(partial.size(2) == cols && partial.size(0) >= 1);
    const int64_t m = partial.size(1);
    TORCH_CHECK(0 < m && m <= kMaxRows, "rows must be in 1..", kMaxRows);
}

}  // namespace

// Bytes of one rank's exchange and flag buffers.
int64_t data_bytes() { return int64_t(kPoolElems + kBranchElems) * 2; }
int64_t flag_bytes() { return int64_t(flag::kWords) * 4; }

std::vector<int64_t> open_peer_buffers(int64_t self_ptr, const std::vector<torch::Tensor>& handles,
                                       const std::vector<int64_t>& offsets, int64_t rank) {
    std::vector<int64_t> ptrs(handles.size());
    for (size_t r = 0; r < handles.size(); ++r) {
        if ((int64_t)r == rank) {
            ptrs[r] = self_ptr;
            continue;
        }
        hipIpcMemHandle_t handle;
        std::memcpy(&handle, handles[r].data_ptr(), sizeof(handle));
        void* ptr = nullptr;
        TORCH_CHECK(hipIpcOpenMemHandle(&ptr, handle, hipIpcMemLazyEnablePeerAccess) == hipSuccess,
                    "hipIpcOpenMemHandle failed");
        ptrs[r] = reinterpret_cast<int64_t>(ptr) + offsets[r];
    }
    return ptrs;
}

void zero_flags(int64_t ptr) {
    TORCH_CHECK(hipMemset(reinterpret_cast<void*>(ptr), 0, flag_bytes()) == hipSuccess);
    TORCH_CHECK(hipDeviceSynchronize() == hipSuccess);
}

void front_epilogue(const torch::Tensor& partial, const torch::Tensor& bias, const torch::Tensor& norm,
                    torch::Tensor& scores, torch::Tensor& ids, torch::Tensor& xe, double multiplier, double scale,
                    double eps, const std::optional<torch::Tensor>& num_real, const std::vector<int64_t>& data,
                    const std::vector<int64_t>& flags, int64_t rank) {
    check_partial(partial, kFrontCols);
    const int m = partial.size(1);
    TORCH_CHECK(bias.dtype() == torch::kFloat32 && bias.numel() == kExperts && bias.is_contiguous());
    TORCH_CHECK(norm.dtype() == torch::kBFloat16 && norm.numel() == kPooled && norm.is_contiguous());
    TORCH_CHECK(scores.dtype() == torch::kFloat32 && scores.is_contiguous() && scores.numel() == m * kTopK);
    TORCH_CHECK(ids.dtype() == torch::kInt32 && ids.is_contiguous() && ids.numel() == m * kTopK);
    TORCH_CHECK(xe.dtype() == torch::kBFloat16 && xe.is_contiguous() && xe.numel() == int64_t(m) * kPooled);
    const int* nr = nullptr;
    if (num_real.has_value()) {
        TORCH_CHECK(num_real->dtype() == torch::kInt32 && num_real->numel() >= 1);
        nr = num_real->data_ptr<int>();
    }
    const FrontArgs args{partial.data_ptr<float>(), bias.data_ptr<float>(),
                         reinterpret_cast<const uint16_t*>(norm.data_ptr()), (float)multiplier, (float)scale,
                         (float)eps, nr, scores.data_ptr<float>(), ids.data_ptr<int>(),
                         reinterpret_cast<uint16_t*>(xe.data_ptr())};
    const Peers peers = make_peers(data, flags);
    hipStream_t stream = at::hip::getCurrentHIPStream();
    switch (partial.size(0)) {
        case 4: front_epilogue_kernel<4><<<m, 256, 0, stream>>>(args, peers, (int)rank, m); break;
        case 8: front_epilogue_kernel<8><<<m, 256, 0, stream>>>(args, peers, (int)rank, m); break;
        case 16: front_epilogue_kernel<16><<<m, 256, 0, stream>>>(args, peers, (int)rank, m); break;
        default: TORCH_CHECK(false, "front epilogue: split-K count must be 4, 8 or 16");
    }
}

void tail_boundary(const torch::Tensor& partial, const torch::Tensor& stream, const torch::Tensor& alpha,
                   const torch::Tensor& beta, const torch::Tensor& post_weight, const torch::Tensor& pre_weight,
                   double post_eps, double pre_eps, bool pre_bf16, torch::Tensor& next_input,
                   torch::Tensor& new_stream, const std::vector<int64_t>& data, const std::vector<int64_t>& flags,
                   int64_t rank) {
    check_partial(partial, kBranchSlice);
    const int m = partial.size(1);
    TORCH_CHECK(stream.dtype() == torch::kFloat32 && stream.is_contiguous() && stream.numel() == int64_t(m) * kHidden);
    TORCH_CHECK(alpha.dtype() == torch::kFloat32 && beta.dtype() == torch::kFloat32);
    TORCH_CHECK(alpha.numel() == kHidden && beta.numel() == kHidden && alpha.is_contiguous() && beta.is_contiguous());
    TORCH_CHECK(post_weight.dtype() == torch::kBFloat16 && post_weight.numel() == kHidden);
    TORCH_CHECK(pre_weight.dtype() == torch::kBFloat16 && pre_weight.numel() == kHidden);
    TORCH_CHECK(next_input.dtype() == torch::kBFloat16 && next_input.numel() == int64_t(m) * kHidden);
    TORCH_CHECK(new_stream.dtype() == torch::kFloat32 && new_stream.numel() == int64_t(m) * kHidden);
    const TailArgs args{partial.data_ptr<float>(), stream.data_ptr<float>(),
                        alpha.data_ptr<float>(), beta.data_ptr<float>(),
                        reinterpret_cast<const uint16_t*>(post_weight.data_ptr()),
                        reinterpret_cast<const uint16_t*>(pre_weight.data_ptr()), (float)post_eps, (float)pre_eps,
                        pre_bf16, reinterpret_cast<uint16_t*>(next_input.data_ptr()), new_stream.data_ptr<float>()};
    const Peers peers = make_peers(data, flags);
    hipStream_t hip_stream = at::hip::getCurrentHIPStream();
    switch (partial.size(0)) {
        case 4: tail_boundary_kernel<4><<<m, kTailThreads, 0, hip_stream>>>(args, peers, (int)rank, m); break;
        case 8: tail_boundary_kernel<8><<<m, kTailThreads, 0, hip_stream>>>(args, peers, (int)rank, m); break;
        case 16: tail_boundary_kernel<16><<<m, kTailThreads, 0, hip_stream>>>(args, peers, (int)rank, m); break;
        default: TORCH_CHECK(false, "tail boundary: split-K count must be 4, 8 or 16");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("data_bytes", &data_bytes);
    mod.def("flag_bytes", &flag_bytes);
    mod.def("open_peer_buffers", &open_peer_buffers);
    mod.def("zero_flags", &zero_flags);
    mod.def("front_epilogue", &front_epilogue);
    mod.def("tail_boundary", &tail_boundary);
}
