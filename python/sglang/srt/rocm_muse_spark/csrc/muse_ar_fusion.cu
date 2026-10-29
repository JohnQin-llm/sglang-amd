// Muse Spark TP4 decode: the two per-layer collectives with their neighbouring kernel folded in.
//
// * finalize_all_reduce: the MoE finalize (per-slot RMSNorm with input scale, route-weighted sum of the local
//   slots) followed by the all-reduce of the [M, 4096] partial sums.
// * all_reduce_boundary: the all-reduce of the attention o_proj partials [M, 8192] followed by the attention
//   boundary (post-norm, gated residual, pre-FFN norm).
//
// Both run on AITER's custom all-reduce state (its rank signals, tmp buffers and IPC-registered inputs) and sum
// the ranks like AITER's kernels: in FP32, rounded to BF16, in rank order 0..3 for the one-stage form and
// rank-rotated from the owner for the two-stage form (rank r reduces rows [r M/4, (r+1) M/4), AITER's partition
// for M a multiple of 4). The finalize reproduces AITER's FlyDSL muse_moe_finalize_vector arithmetic (thread t
// holds columns t*4 + c*1024 + j, wave butterfly, waves summed in order), so the fused MoE output is bitwise equal
// to finalize + all-reduce. The attention boundary keeps the Triton kernel's formula; only its two row sums run
// in a different order.
//
// Synchronisation. The one-stage forms (small M) use AITER's per-block barriers. The two-stage forms hand rows
// over with flags instead, so every block works in every phase: a writer stores a row into uncached memory, waits
// for its stores, then stores the call's generation into the reader's flag; the reader polls its local flag.
// Each kernel keeps a launch counter (epoch) in the flag buffer; generation = epoch + 1, the same on every rank
// because all ranks launch the same sequence. A rank enters its next call only after it saw every row of this call
// reduced, so no buffer is rewritten while a peer still reads it.
#include <ATen/hip/HIPContext.h>
#include <torch/extension.h>

#include "custom_all_reduce.cuh"

using aiter::RankData;
using aiter::RankSignals;
using aiter::Signal;

namespace {

constexpr int kRanks = 4;
constexpr int kThreads = 512;
constexpr int kMoeHidden = 4096;
constexpr int kHidden = 8192;
constexpr int kTopK = 8;
constexpr int kMaxRows = 256;

// Flag buffer of one rank (uint32 offsets). ready / partial / arrive are written by peers.
namespace flag {
constexpr int kReadyMoe = 0;                          // [row]: row reduced on its owner (MoE)
constexpr int kReadyAttn = kMaxRows;                  // [row]: row reduced on its owner (attention)
constexpr int kPartial = 2 * kMaxRows;                // [row][rank]: rank's finalized partial row written
constexpr int kArrive = kPartial + kMaxRows * kRanks;  // [rank]: rank entered the attention call
constexpr int kCountMoe = kArrive + 64;               // [row]: slices of the row reduced here (local)
constexpr int kCountAttn = kCountMoe + kMaxRows;
constexpr int kEpochMoe = kCountAttn + kMaxRows, kDoneMoe = kEpochMoe + 1;
constexpr int kEpochAttn = kEpochMoe + 2, kDoneAttn = kEpochMoe + 3;
}  // namespace flag

struct Flags {
    uint32_t* rank[kRanks];  // every rank's flag buffer (IPC-mapped)
};

struct alignas(16) Pack8 {
    uint16_t v[8];
};

__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float(uint32_t(u) << 16); }

__device__ __forceinline__ uint16_t f2bf(float f) {
    return __builtin_bit_cast(uint16_t, static_cast<__bf16>(f));  // RNE, the gfx950 conversion instruction
}

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int shift = 32; shift >= 1; shift >>= 1) v = v + __shfl_xor(v, shift, 64);
    return v;
}

__device__ __forceinline__ void publish(uint32_t* flag, uint32_t generation) {
    __scoped_atomic_store_n(flag, generation, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// Polls back off between reads (s_sleep 2, ~128 cycles): 0-0.7 us faster than a tight spin at 80..128 rows,
// the same below.
__device__ __forceinline__ void wait_for(const uint32_t* flag, uint32_t generation) {
    while (__scoped_atomic_load_n(flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < generation)
        __builtin_amdgcn_s_sleep(2);
}

// This call's generation: the kernel's epoch + 1. Read with an atomic so the load cannot come from the scalar
// cache (a stale epoch on one rank would desynchronise the generations of all ranks).
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

__device__ __forceinline__ Pack8 sum_packs(const Pack8* in) {
    Pack8 out;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float acc = bf2f(in[0].v[k]);
#pragma unroll
        for (int i = 1; i < kRanks; ++i) acc += bf2f(in[i].v[k]);
        out.v[k] = f2bf(acc);
    }
    return out;
}

// All ranks' pack idx summed in rank order 0..3 (AITER's one-stage order).
__device__ __forceinline__ Pack8 reduce_pack(const Pack8* const* src, int idx) {
    Pack8 in[kRanks];
#pragma unroll
    for (int i = 0; i < kRanks; ++i) in[i] = src[i][idx];
    return sum_packs(in);
}

// One pack summed over the ranks in AITER's two-stage order, rotated from the owner: rank, rank + 1, ...
__device__ __forceinline__ Pack8 reduce_pack_rotated(const Pack8* const* src, int rank, int idx) {
    Pack8 in[kRanks];
#pragma unroll
    for (int i = 0; i < kRanks; ++i) in[i] = src[(rank + i) % kRanks][idx];
    return sum_packs(in);
}

// The owner's reduction of one slice of its row block into its tmp buffer; the last slice of a row to complete
// flags the row to every rank. kGroups thread groups split the ranks: a slice is kThreads / kGroups packs, group g
// loads ranks rank + g * (4 / kGroups) + i (rotated order) and group 0 sums all four in that order.
template <int kRowPacks, int kGroups>
struct SliceReduce {
    static constexpr int kSlicePacks = kThreads / kGroups, kSlices = kRowPacks / kSlicePacks;
    static constexpr int kRanksPerGroup = kRanks / kGroups;

    __device__ static void run(const Pack8* const* src, Pack8* tmp, Pack8* stage, const Flags& flags, int ready,
                               int count, int rank, int owned, int task, uint32_t generation) {
        const int j = task / kSlices, g = threadIdx.x / kSlicePacks, lane = threadIdx.x % kSlicePacks;
        const int idx = (task % kSlices) * kSlicePacks + lane, row = rank * owned + j;
        if constexpr (kGroups == 1) {
            tmp[j * kRowPacks + idx] = reduce_pack_rotated(src, rank, row * kRowPacks + idx);
        } else {
#pragma unroll
            for (int i = 0; i < kRanksPerGroup; ++i) {
                const int k = g * kRanksPerGroup + i;
                stage[k * kSlicePacks + lane] = src[(rank + k) % kRanks][row * kRowPacks + idx];
            }
            __syncthreads();
            if (g == 0) {
                Pack8 in[kRanks];
#pragma unroll
                for (int k = 0; k < kRanks; ++k) in[k] = stage[k * kSlicePacks + lane];
                tmp[j * kRowPacks + idx] = sum_packs(in);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);  // the slice is in (uncached) tmp before it is counted
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t* local = flags.rank[rank];
            const bool last = kSlices == 1 || __scoped_atomic_fetch_add(&local[count + row], 1u, __ATOMIC_RELAXED,
                                                                        __MEMORY_SCOPE_DEVICE) % kSlices == kSlices - 1;
            if (last)
#pragma unroll
                for (int r = 0; r < kRanks; ++r) publish(&flags.rank[r][ready + row], generation);
        }
    }
};

// AITER's relaxed per-block barrier after this block stored data peers will read (uncached buffers: a completed
// store is visible).
__device__ __forceinline__ void barrier_after_stores(const RankSignals& sg, Signal* self_sg, int rank) {
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    aiter::start_sync<kRanks>(sg, self_sg, rank);
}

// ------------------------------------------------------------------------------------------------------------
// MoE finalize + all-reduce
// ------------------------------------------------------------------------------------------------------------

struct FinalizeArgs {
    const uint16_t* y;  // [M * 8, 4096] per-slot expert rows
    const float* scores;
    const int* ids;
    int local_start, num_local;
    float eps;
};

// One row's finalize on the 256 threads of a half block, FlyDSL's layout: thread t holds columns
// t*4 + c*1024 + j. All of the row's local slot data is loaded before any of it is used.
struct RowFinalizer {
    float scale[16];  // bf16(gamma + gain)
    uint2 raw[kTopK][4];
    float score[kTopK];
    uint32_t local = 0;  // bit s: slot s is routed to this rank

    __device__ void load_scale(const uint16_t* gamma, float gain, int t) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint2 g = *reinterpret_cast<const uint2*>(gamma + t * 4 + c * 1024);
            const uint16_t* gv = reinterpret_cast<const uint16_t*>(&g);
#pragma unroll
            for (int j = 0; j < 4; ++j) scale[c * 4 + j] = bf2f(f2bf(bf2f(gv[j]) + gain));
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
            if (ids[s] >= a.local_start && ids[s] < a.local_start + a.num_local) local |= 1u << s;
        }
#pragma unroll
        for (int s = 0; s < kTopK; ++s)
            if (local >> s & 1)
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    raw[s][c] = *reinterpret_cast<const uint2*>(a.y + (size_t)(row * kTopK + s) * kMoeHidden + t * 4 +
                                                                c * 1024);
    }

    __device__ float value(int s, int c, int j) const {  // bf16(y * scale)
        const uint16_t* yv = reinterpret_cast<const uint16_t*>(&raw[s][c]);
        return bf2f(f2bf(bf2f(yv[j]) * scale[c * 4 + j]));
    }

    // Per-slot sum of squares over the row, in FlyDSL's order: this thread's 16 values, the wave butterfly, then
    // the four waves (summed in weighted_sum). Written to sums[slot][wave].
    __device__ void square_sums(int t, float (*sums)[4]) const {
#pragma unroll
        for (int s = 0; s < kTopK; ++s) {
            if (!(local >> s & 1)) continue;
            float ss = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; ++c)
#pragma unroll
                for (int j = 0; j < 4; ++j) ss = ss + value(s, c, j) * value(s, c, j);
            ss = wave_sum(ss);
            if (t % 64 == 0) sums[s][t / 64] = ss;
        }
    }

    // Route-weighted sum of the normed local slots into the row of `partial`.
    __device__ void weighted_sum(int t, const float (*sums)[4], float eps, uint16_t* row_out) const {
        float state[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) state[k] = 0.0f;
#pragma unroll
        for (int s = 0; s < kTopK; ++s) {
            if (!(local >> s & 1)) continue;
            float total = 0.0f;
#pragma unroll
            for (int w = 0; w < 4; ++w) total = total + sums[s][w];
            const float mean = total * (1.0f / kMoeHidden) + eps;
            const float sc = score[s] * (1.0f / __builtin_amdgcn_sqrtf(mean));
#pragma unroll
            for (int c = 0; c < 4; ++c)
#pragma unroll
                for (int j = 0; j < 4; ++j) state[c * 4 + j] = state[c * 4 + j] + sc * value(s, c, j);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            uint16_t o[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = f2bf(state[c * 4 + j]);
            *reinterpret_cast<uint2*>(row_out + t * 4 + c * 1024) = *reinterpret_cast<const uint2*>(o);
        }
    }
};

// Rows row0 + 2 * grid * i + (2 * blockIdx.x + half): every half block finalizes whole rows. `after_row` runs on
// every thread once the pair of rows of an iteration is stored.
template <typename AfterRow>
__device__ __forceinline__ void finalize_rows(const FinalizeArgs& a, const uint16_t* gamma, float gain, int m,
                                              uint16_t* partial, AfterRow after_row) {
    __shared__ float sums[2][kTopK][4];
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
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        after_row(row, valid, t);
    }
}

// One-stage form (M <= 16): finalize, AITER barrier, every rank sums the rows the same block finalized everywhere.
__global__ void __launch_bounds__(kThreads, 1) finalize_all_reduce_1stage(FinalizeArgs a, const uint16_t* gamma,
                                                                          float gain, RankData* partial_dp,
                                                                          RankSignals sg, Signal* self_sg,
                                                                          uint16_t* __restrict__ out, int rank, int m) {
    uint16_t* partial = reinterpret_cast<uint16_t*>(const_cast<void*>(partial_dp->ptrs[rank]));
    finalize_rows(a, gamma, gain, m, partial, [](int, bool, int) {});
    barrier_after_stores(sg, self_sg, rank);
    const Pack8* src[kRanks];
#pragma unroll
    for (int r = 0; r < kRanks; ++r) src[r] = reinterpret_cast<const Pack8*>(partial_dp->ptrs[r]);
    constexpr int kPacks = kMoeHidden / 8;  // 512: one pack per thread
    const int half = threadIdx.x / 256, t = threadIdx.x % 256;
    for (int base = 2 * blockIdx.x; base < m; base += 2 * gridDim.x) {
        const int row = base + half;
        if (row < m)
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int idx = row * kPacks + t + p * 256;
                reinterpret_cast<Pack8*>(out)[idx] = reduce_pack(src, idx);
            }
    }
    aiter::end_sync<kRanks, true>(sg, self_sg, rank);  // peers are done reading this rank's partials
}

// Two-stage form: (1) finalize whole rows, flag each to its owner; (2) the owner reduces its rows in slices on
// every block, the last slice of a row flags the row to all ranks; (3) every rank copies each row from
// its owner.
template <int kGroups>
__global__ void __launch_bounds__(kThreads, 1) finalize_all_reduce_2stage(FinalizeArgs a, const uint16_t* gamma,
                                                                          float gain, RankData* partial_dp,
                                                                          RankSignals sg, Flags flags,
                                                                          uint16_t* __restrict__ out, int rank, int m) {
    __shared__ Pack8 stage[kGroups > 1 ? kRanks * (kThreads / kGroups) : 1];
    constexpr int kPacks = kMoeHidden / 8;  // 512
    using Reduce = SliceReduce<kPacks, kGroups>;
    const int owned = m / kRanks;
    uint32_t* local = flags.rank[rank];
    const uint32_t generation = begin_call(local, flag::kEpochMoe);
    uint16_t* partial = reinterpret_cast<uint16_t*>(const_cast<void*>(partial_dp->ptrs[rank]));
    finalize_rows(a, gamma, gain, m, partial, [&](int row, bool valid, int t) {
        if (valid && t == 0) publish(&flags.rank[row / owned][flag::kPartial + row * kRanks + rank], generation);
    });
    const Pack8* src[kRanks];
#pragma unroll
    for (int r = 0; r < kRanks; ++r) src[r] = reinterpret_cast<const Pack8*>(partial_dp->ptrs[r]);
    Pack8* tmp = aiter::get_tmp_buf<Pack8>(sg.signals[rank]);
    for (int task = blockIdx.x; task < owned * Reduce::kSlices; task += gridDim.x) {
        const int row = rank * owned + task / Reduce::kSlices;
        if (threadIdx.x < kRanks) wait_for(&local[flag::kPartial + row * kRanks + threadIdx.x], generation);
        __syncthreads();
        Reduce::run(src, tmp, stage, flags, flag::kReadyMoe, flag::kCountMoe, rank, owned, task, generation);
    }
    for (int row = blockIdx.x; row < m; row += gridDim.x) {
        if (threadIdx.x == 0) wait_for(&local[flag::kReadyMoe + row], generation);
        __syncthreads();
        const int owner = row / owned;
        const Pack8* owner_tmp = aiter::get_tmp_buf<Pack8>(sg.signals[owner]);
        reinterpret_cast<Pack8*>(out)[row * kPacks + threadIdx.x] = owner_tmp[(row - owner * owned) * kPacks + threadIdx.x];
    }
    end_call(local, flag::kEpochMoe, flag::kDoneMoe, generation);
}

// ------------------------------------------------------------------------------------------------------------
// Attention all-reduce + boundary
// ------------------------------------------------------------------------------------------------------------

// Sum over the block's 8 waves; `cells` holds 8 floats used by this reduction only.
__device__ __forceinline__ float block_sum(float v, float* cells) {
    v = wave_sum(v);
    if (threadIdx.x % 64 == 0) cells[threadIdx.x / 64] = v;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < kThreads / 64; ++w) total = total + cells[w];
    return total;
}

// This thread's local operands of one boundary row: 16 columns (threadIdx.x + p * 512) * 8 + k, p < 2, k < 8.
struct BoundaryOperands {
    float stream[16], alpha[16], beta[16], weight[16];

    __device__ void load(int row, const float* __restrict__ stream_in, const float* __restrict__ alpha_in,
                         const float* __restrict__ beta_in, const uint16_t* __restrict__ pre_weight) {
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
            const Pack8 w = *reinterpret_cast<const Pack8*>(pre_weight + col);
#pragma unroll
            for (int k = 0; k < 8; ++k) weight[p * 8 + k] = bf2f(w.v[k]);
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

// post_norm (no weight) -> bf16, gated residual alpha * stream + beta * branch, pre_norm * weight -> bf16:
// the Triton _fused_norm_residual_triton_kernel formula for the attention boundary.
__device__ __forceinline__ void attention_boundary(const Pack8 (&reduced)[2], int row, const BoundaryOperands& op,
                                                   const BoundaryArgs& a, float* cells) {
    float b[16];
    float ss = 0.0f;
#pragma unroll
    for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            b[p * 8 + k] = bf2f(reduced[p].v[k]);
            ss = ss + b[p * 8 + k] * b[p * 8 + k];
        }
    const float post_rstd = __builtin_amdgcn_rsqf(block_sum(ss, cells) / kHidden + a.post_eps);
    float s[16];
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float normed = bf2f(f2bf(b[i] * post_rstd));
        s[i] = op.alpha[i] * op.stream[i] + op.beta[i] * normed;
        ss = ss + s[i] * s[i];
    }
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int col = (threadIdx.x + p * kThreads) * 8;
        float4* ns = reinterpret_cast<float4*>(a.new_stream + (size_t)row * kHidden + col);
        ns[0] = make_float4(s[p * 8 + 0], s[p * 8 + 1], s[p * 8 + 2], s[p * 8 + 3]);
        ns[1] = make_float4(s[p * 8 + 4], s[p * 8 + 5], s[p * 8 + 6], s[p * 8 + 7]);
    }
    const float pre_rstd = __builtin_amdgcn_rsqf(block_sum(ss, cells + 8) / kHidden + a.pre_eps);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int col = (threadIdx.x + p * kThreads) * 8;
        Pack8 o;
#pragma unroll
        for (int k = 0; k < 8; ++k) o.v[k] = f2bf(s[p * 8 + k] * pre_rstd * op.weight[p * 8 + k]);
        *reinterpret_cast<Pack8*>(a.ffn_input + (size_t)row * kHidden + col) = o;
    }
    __syncthreads();  // cells are reused by the next row
}

// One-stage form: block b reads rows b, b + grid, ... from all ranks and applies the boundary.
__global__ void __launch_bounds__(kThreads, 1) all_reduce_boundary_1stage(RankData* input_dp, RankSignals sg,
                                                                          Signal* self_sg, BoundaryArgs a, int rank,
                                                                          int m) {
    __shared__ float cells[16];
    constexpr int kPacks = kHidden / 8;  // 1024: two packs per thread
    const Pack8* src[kRanks];
#pragma unroll
    for (int r = 0; r < kRanks; ++r) src[r] = reinterpret_cast<const Pack8*>(input_dp->ptrs[r]);
    BoundaryOperands op;
    op.load(blockIdx.x, a.stream, a.alpha, a.beta, a.pre_weight);
    aiter::start_sync<kRanks>(sg, self_sg, rank);  // the o_proj partials come from an earlier kernel
    for (int row = blockIdx.x; row < m; row += gridDim.x) {
        if (row != (int)blockIdx.x) op.load(row, a.stream, a.alpha, a.beta, a.pre_weight);
        Pack8 reduced[2];
#pragma unroll
        for (int p = 0; p < 2; ++p) reduced[p] = reduce_pack(src, row * kPacks + threadIdx.x + p * kThreads);
        attention_boundary(reduced, row, op, a, cells);
    }
    aiter::end_sync<kRanks, true>(sg, self_sg, rank);
}

// Two-stage form: every rank flags its arrival (its o_proj partials are complete), the owner reduces its rows in
// slices on every block, the last slice of a row flags the row to all ranks, and each row's boundary runs
// on one block once the row is flagged.
template <int kGroups>
__global__ void __launch_bounds__(kThreads, 1) all_reduce_boundary_2stage(RankData* input_dp, RankSignals sg,
                                                                          Flags flags, BoundaryArgs a, int rank,
                                                                          int m) {
    __shared__ float cells[16];
    __shared__ Pack8 stage[kGroups > 1 ? kRanks * (kThreads / kGroups) : 1];
    constexpr int kPacks = kHidden / 8;  // 1024
    using Reduce = SliceReduce<kPacks, kGroups>;
    constexpr int kSlices = Reduce::kSlices;
    const int owned = m / kRanks;
    uint32_t* local = flags.rank[rank];
    const uint32_t generation = begin_call(local, flag::kEpochAttn);
    if (blockIdx.x == 0 && threadIdx.x < kRanks) publish(&flags.rank[threadIdx.x][flag::kArrive + rank], generation);
    const Pack8* src[kRanks];
#pragma unroll
    for (int r = 0; r < kRanks; ++r) src[r] = reinterpret_cast<const Pack8*>(input_dp->ptrs[r]);
    Pack8* tmp = aiter::get_tmp_buf<Pack8>(sg.signals[rank]);
    if ((int)blockIdx.x < owned * kSlices) {
        if (threadIdx.x < kRanks) wait_for(&local[flag::kArrive + threadIdx.x], generation);
        __syncthreads();
    }
    for (int task = blockIdx.x; task < owned * kSlices; task += gridDim.x)
        Reduce::run(src, tmp, stage, flags, flag::kReadyAttn, flag::kCountAttn, rank, owned, task, generation);
    BoundaryOperands op;
    for (int row = blockIdx.x; row < m; row += gridDim.x) {
        op.load(row, a.stream, a.alpha, a.beta, a.pre_weight);
        if (threadIdx.x == 0) wait_for(&local[flag::kReadyAttn + row], generation);
        __syncthreads();
        const int owner = row / owned;
        const Pack8* from = aiter::get_tmp_buf<Pack8>(sg.signals[owner]) + (row - owner * owned) * kPacks;
        Pack8 reduced[2];
#pragma unroll
        for (int p = 0; p < 2; ++p) reduced[p] = from[threadIdx.x + p * kThreads];
        attention_boundary(reduced, row, op, a, cells);
    }
    end_call(local, flag::kEpochAttn, flag::kDoneAttn, generation);
}

aiter::CustomAllreduce* state(int64_t fa) {
    auto* ca = reinterpret_cast<aiter::CustomAllreduce*>(fa);
    TORCH_CHECK(ca->world_size_ == kRanks, "fused collectives are built for 4 ranks");
    return ca;
}

Flags make_flags(const std::vector<int64_t>& ptrs) {
    TORCH_CHECK(ptrs.size() == kRanks);
    Flags flags;
    for (int r = 0; r < kRanks; ++r) flags.rank[r] = reinterpret_cast<uint32_t*>(ptrs[r]);
    return flags;
}

}  // namespace

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

void finalize_all_reduce(int64_t fa, const torch::Tensor& y, const torch::Tensor& gamma, const torch::Tensor& scores,
                         const torch::Tensor& ids, int64_t partial_ptr, int64_t partial_rows, torch::Tensor& out,
                         int64_t local_start, int64_t num_local, double eps, double gain, int64_t blocks,
                         bool two_stage, const std::vector<int64_t>& row_flags) {
    auto* ca = state(fa);
    const int m = ids.size(0);
    TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.is_contiguous() && y.size(0) == m * kTopK && y.size(1) == kMoeHidden);
    TORCH_CHECK(gamma.dtype() == torch::kBFloat16 && gamma.numel() == kMoeHidden);
    TORCH_CHECK(scores.dtype() == torch::kFloat32 && ids.dtype() == torch::kInt32 && ids.size(1) == kTopK);
    TORCH_CHECK(scores.is_contiguous() && ids.is_contiguous());
    TORCH_CHECK(partial_rows >= m && m <= kMaxRows);
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.size(0) == m);
    // The one-stage form uses AITER's per-block signals (at most kMaxBlocks blocks).
    TORCH_CHECK(m % (2 * kRanks) == 0 && 0 < blocks && (two_stage ? blocks <= 2 * kMaxRows : blocks <= std::min(m, aiter::kMaxBlocks)));
    hipStream_t stream = at::hip::getCurrentHIPStream();
    RankData* rd = ca->get_buffer_RD(stream, reinterpret_cast<void*>(partial_ptr));
    const FinalizeArgs args{reinterpret_cast<const uint16_t*>(y.data_ptr()), scores.data_ptr<float>(),
                            ids.data_ptr<int>(), (int)local_start, (int)num_local, (float)eps};
    const auto* g = reinterpret_cast<const uint16_t*>(gamma.data_ptr());
    auto* o = reinterpret_cast<uint16_t*>(out.data_ptr());
    if (two_stage) {
        // One thread group per slice (all four ranks' loads per thread) measured fastest at every width.
        finalize_all_reduce_2stage<1><<<blocks, kThreads, 0, stream>>>(args, g, (float)gain, rd, ca->sg_,
                                                                       make_flags(row_flags), o, ca->rank_, m);
    } else
        finalize_all_reduce_1stage<<<blocks, kThreads, 0, stream>>>(args, g, (float)gain, rd, ca->sg_, ca->self_sg_, o,
                                                                    ca->rank_, m);
}

void all_reduce_boundary(int64_t fa, const torch::Tensor& inp, const torch::Tensor& stream_in,
                         const torch::Tensor& alpha, const torch::Tensor& beta, const torch::Tensor& pre_weight,
                         torch::Tensor& ffn_input, torch::Tensor& new_stream, double post_eps, double pre_eps,
                         int64_t blocks, bool two_stage, const std::vector<int64_t>& row_flags) {
    auto* ca = state(fa);
    const int m = inp.size(0);
    TORCH_CHECK(inp.dtype() == torch::kBFloat16 && inp.is_contiguous() && inp.size(1) == kHidden);
    TORCH_CHECK(stream_in.dtype() == torch::kFloat32 && stream_in.is_contiguous() && stream_in.size(0) == m);
    TORCH_CHECK(alpha.dtype() == torch::kFloat32 && beta.dtype() == torch::kFloat32);
    TORCH_CHECK(alpha.numel() == kHidden && beta.numel() == kHidden && pre_weight.numel() == kHidden);
    TORCH_CHECK(pre_weight.dtype() == torch::kBFloat16);
    TORCH_CHECK(ffn_input.dtype() == torch::kBFloat16 && new_stream.dtype() == torch::kFloat32);
    TORCH_CHECK(m % (2 * kRanks) == 0 && m <= kMaxRows && 0 < blocks &&
                (two_stage ? blocks <= 2 * kMaxRows : blocks <= std::min(m, aiter::kMaxBlocks)));
    hipStream_t stream = at::hip::getCurrentHIPStream();
    RankData* rd = ca->get_buffer_RD(stream, inp.data_ptr());
    const BoundaryArgs args{stream_in.data_ptr<float>(), alpha.data_ptr<float>(), beta.data_ptr<float>(),
                            reinterpret_cast<const uint16_t*>(pre_weight.data_ptr()),
                            reinterpret_cast<uint16_t*>(ffn_input.data_ptr()), new_stream.data_ptr<float>(),
                            (float)post_eps, (float)pre_eps};
    if (two_stage) {
        // AITER's layout (four groups, one rank each) measured fastest at every width: 128-pack slices, so four
        // times as many blocks share the reduce-scatter as with one group.
        all_reduce_boundary_2stage<4><<<blocks, kThreads, 0, stream>>>(rd, ca->sg_, make_flags(row_flags), args,
                                                                       ca->rank_, m);
    } else
        all_reduce_boundary_1stage<<<blocks, kThreads, 0, stream>>>(rd, ca->sg_, ca->self_sg_, args, ca->rank_, m);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("open_peer_buffers", &open_peer_buffers);
    mod.def("finalize_all_reduce", &finalize_all_reduce);
    mod.def("all_reduce_boundary", &all_reduce_boundary);
}
