# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.types import e4m3_dtype

float8_info = torch.finfo(e4m3_dtype)


@triton.jit
def fast_exp(x):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    return tl.math.exp2(x * RCP_LN2)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.math.exp2(Sdiv)
    p2 = tl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def find_seq_idx_exact_q_blocks(query_start_len_ptr, q_block_idx, num_seqs, BLOCK_Q: tl.constexpr):
    # Request owning q-block q_block_idx when every request's query length is a multiple of BLOCK_Q, so request s
    # owns q-blocks [cu_q[s] // BLOCK_Q, cu_q[s + 1] // BLOCK_Q): largest s with cu_q[s] // BLOCK_Q <= q_block_idx.
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        mid_val = tl.load(query_start_len_ptr + mid) // BLOCK_Q
        if mid_val <= q_block_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale: tl.constexpr,  # float32
    q_descale_ptr,  # float32
    k_descale_ptr,  # float32
    v_descale_ptr,  # float32
    out_scale_ptr,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    ALL_DECODE: tl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: tl.constexpr = False,  # bool
    K_WIDTH: tl.constexpr = 0,  # int
    UNMASKED_INTERIOR: tl.constexpr = False,  # bool
):
    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    if ALL_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx: tl.int32 = 0
        cur_batch_query_len: tl.int32 = 1
        cur_batch_in_all_start_index: tl.int32 = q_block_global_idx
    else:
        seq_idx = find_seq_idx(
            query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
        )

        q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

        q_block_local_idx = q_block_global_idx - q_block_start_idx

        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

        if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
            return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    offs_shfl = None
    if SHUFFLED_KV_CACHE:
        offs_shfl = tl.arange(0, TILE_SIZE * HEAD_SIZE_PADDED)

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    if ALL_DECODE or BLOCK_M >= num_query_heads:
        Q_cache_modifier: tl.constexpr = ".cg"
    else:
        Q_cache_modifier: tl.constexpr = ""
    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
        cache_modifier=Q_cache_modifier,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        # Prescale with RCP_LN2, needed for exp2
        M = (
            tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
            * RCP_LN2
        )

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
    if q_descale_ptr is not None:
        q_descale = tl.load(q_descale_ptr)
        qk_scale = qk_scale * q_descale
    else:
        q_descale = None
    if k_descale_ptr is not None and v_descale_ptr is not None:
        k_descale = tl.load(k_descale_ptr)
        v_descale = tl.load(v_descale_ptr)
        qk_scale = qk_scale * k_descale
    else:
        k_descale = None
        v_descale = None
    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    # Long-context extend: tiles that every query row of this block sees in full need no causal,
    # padding or tile mask; only the diagonal tiles below run the masked loop. Same arithmetic per element.
    if (UNMASKED_INTERIOR and SLIDING_WINDOW == 0 and not ALL_DECODE and not SHUFFLED_KV_CACHE
            and not USE_ALIBI_SLOPES and not USE_QQ_BIAS and not USE_SOFTCAP and HEAD_SIZE == HEAD_SIZE_PADDED):
        full_end = tl.minimum(tile_end, (context_len + q_block_local_idx * BLOCK_Q + 1) // TILE_SIZE)
        for j in range(tile_start, full_end):
            seq_offset = j * TILE_SIZE + offs_t
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            K = tl.load(key_cache_ptr + k_offset).to(Q.dtype)
            V = tl.load(value_cache_ptr + v_offset).to(Q.dtype)
            # Positive qk_scale: max(round(x*s)) == round(max(x)*s).
            # Reduce before scaling; ordinary Triton expression (no inline ASM).
            raw_scores = tl.dot(Q, K)
            m_j = tl.maximum(M, tl.max(raw_scores, axis=1) * qk_scale)
            # Match the current native gfx950 32x32 MFMA contraction pattern.
            # Register order: 4 key columns per 8-column group, per lane half.
            # Native scaled-max reduction creates 8-element packs separated by
            # one reduction element; half are re-used rounded in subtraction.
            score_reg = (offs_t // 8) * 4 + offs_t % 4
            fused = ((score_reg < 8) | ((score_reg >= 18) & (score_reg < 26))
                     | ((score_reg >= 36) & (score_reg < 44)) | (score_reg >= 54))
            separate = tl.inline_asm_elementwise(
                "v_mul_f32 $0, $1, $2;\n v_sub_f32 $0, $0, $3;",
                constraints="=&v,v,v,v", args=[raw_scores, qk_scale, m_j[:, None]],
                dtype=tl.float32, is_pure=True, pack=1,
            )
            centered = tl.where(fused[None, :], tl.fma(raw_scores, qk_scale, -m_j[:, None]), separate)
            P = tl.math.exp2(centered)
            l_j = tl.sum(P, axis=1)
            alpha = tl.math.exp2(M - m_j)
            acc = acc * alpha[:, None]
            L = L * alpha + l_j
            M = m_j
            acc = tl.dot(P.to(V.dtype), V, acc=acc)
        tile_start = full_end
    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        # to reduce the masking effect when not needed
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len

        k_mask = None
        v_mask = None
        other = None
        if SHUFFLED_KV_CACHE:
            physical_block_idx_shfl = tl.load(
                block_tables_ptr + block_table_offset + j
            ).to(tl.int64)
            k_offset = (
                physical_block_idx_shfl * stride_k_cache_0
                + kv_head_idx * stride_k_cache_1
                + offs_shfl
            )

            v_offset = (
                physical_block_idx_shfl * stride_v_cache_0
                + kv_head_idx * stride_v_cache_1
                + offs_shfl
            )
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            v_mask = dim_mask[None, :] & tile_mask[:, None]

            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            k_mask = dim_mask[:, None] & tile_mask[None, :]
            other = 0.0

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=k_mask,
            other=other,
            cache_modifier=KV_cache_modifier,
        )

        K = K_load.to(Q.dtype)
        if SHUFFLED_KV_CACHE:
            K = (
                K.reshape(
                    HEAD_SIZE_PADDED // K_WIDTH,
                    TILE_SIZE,
                    K_WIDTH,
                )
                .permute(1, 0, 2)
                .reshape(TILE_SIZE, HEAD_SIZE_PADDED)
                .trans(1, 0)
            )

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=v_mask,
            other=other,
            cache_modifier=KV_cache_modifier,
        )

        V = V_load.to(Q.dtype)
        if SHUFFLED_KV_CACHE:
            V = (
                V.reshape(
                    TILE_SIZE // K_WIDTH,
                    HEAD_SIZE_PADDED,
                    K_WIDTH,
                )
                .permute(0, 2, 1)
                .reshape(TILE_SIZE, HEAD_SIZE_PADDED)
            )

        # S : (BLOCK_M, TILE_SIZE)
        # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later
        S = qk_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, softcap) * RCP_LN2
        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if SLIDING_WINDOW > 0:
            S = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            # prescale w. RCP_LN2 for later exp2
            S += alibi_slope[:, None] * (seq_offset - context_len) * RCP_LN2

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            # prescale w. RCP_LN2 for later exp2
            S += qq_bias * RCP_LN2

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = tl.dot(P.to(V.dtype), V, acc=acc)

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    if v_descale is not None:
        one_over_L = v_descale / L[:, None]
    else:
        one_over_L = 1.0 / L[:, None]
    acc = acc * one_over_L
    if out_scale_ptr is not None:
        acc = acc / tl.load(out_scale_ptr)

    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# The kernel above retains native tiles and KV accumulation order. Only its
# unmasked-interior max/scale dependency differs from the installed kernel.
import os
import types
_ENABLED = os.environ.get("SGLANG_MUSE_TP4_PREFILL_MAXSCALE", "0") == "1"
_FUNCTION = None
_NATIVE_COMPATIBLE = None

def _native_compatible():
    # The contraction layout is compiler-specific. Fail closed outside the
    # installed native implementation and compiler used for this optimization.
    global _NATIVE_COMPATIBLE
    if _NATIVE_COMPATIBLE is None:
        import hashlib
        import importlib
        from pathlib import Path
        import triton
        native = importlib.import_module('aiter.ops.triton._triton_kernels.attention.unified_attention')
        _NATIVE_COMPATIBLE = (triton.__version__ == '3.7.0'
            and hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
            == 'cd2f9cb29358b5582c98d299808a2eecbf643488cfcbc0fd8350bfdc1a494bfa')
    return _NATIVE_COMPATIBLE

def eligible(k):
    if not _ENABLED or not torch.version.hip:
        return False
    from sglang.srt.runtime_context import get_parallel
    parallel = get_parallel()
    q, kc, vc, out = k['q'], k['k'], k['v'], k['out']
    return (
        (parallel.tp_size, parallel.moe_ep_size) == (4, 4)
        and q.is_cuda and q.is_contiguous() and out.is_contiguous()
        and 4096 <= q.shape[0] <= 16384 and tuple(q.shape[1:]) == (32, 64)
        and out.shape == q.shape and out.dtype == torch.bfloat16
        and q.dtype == kc.dtype == vc.dtype == e4m3_dtype
        and kc.shape == vc.shape and tuple(kc.shape[1:]) == (16, 4, 64)
        and kc.is_contiguous() and vc.is_contiguous()
        and k.get('prefill_2d_config') == (128, 4, 2, 128, 2)
        and k.get('unmasked_interior') is True
        and k['window_size'] == (-1, -1) and k['causal'] is True
        and k['softmax_scale'] == .125 and k['softcap'] == 0
        and not k.get('shuffled_kv_cache', False)
        and all(k.get(n) is None for n in ('sinks','alibi_slopes','qq_bias','output_scale','q_scales'))
        and all(k.get(n) is not None and k[n].numel()==1 and k[n].dtype==torch.float32 for n in ('q_descale','k_descale','v_descale'))
        and torch.cuda.get_device_properties(q.device).gcnArchName.startswith('gfx950')
        and _native_compatible()
    )

def run(original, **kwargs):
    if not eligible(kwargs):
        return original(**kwargs)
    global _FUNCTION
    if _FUNCTION is None:
        from aiter.ops.triton.attention.unified_attention import unified_attention
        ns = dict(unified_attention.__globals__)
        ns['kernel_unified_attention_2d'] = kernel_unified_attention_2d
        _FUNCTION = types.FunctionType(unified_attention.__code__, ns,
                                      unified_attention.__name__, unified_attention.__defaults__, unified_attention.__closure__)
        _FUNCTION.__kwdefaults__ = unified_attention.__kwdefaults__
    return _FUNCTION(**kwargs)
