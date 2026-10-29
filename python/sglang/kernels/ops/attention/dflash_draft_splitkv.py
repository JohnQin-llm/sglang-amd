"""Split-KV attention for the DFlash draft block over a sliding window of cached K/V.

Every request brings a block of up to 8 query tokens. The block attends, non-causally,
to the request's window of cached K/V (``kv_indptr`` / ``kv_indices``) and to its own
K/V (``k_extend`` / ``v_extend``). The generic extend kernel runs one program per
(request, query head) and walks the whole window serially, so a small batch leaves most
of the GPU idle. Here one program owns one (request, KV head, split): the query heads
that share the KV head and the block's tokens form a single row tile (8 heads x 8 tokens
= 64 rows), each K/V tile is loaded once for all of them, and the window is divided into
splits. A second kernel merges the per-split softmax states.

The arithmetic follows ``extend_attention._fwd_kernel`` for the same inputs: on the
cached prefix, Q is cast to the cache dtype for Q.K and the probabilities are cast to
the cache dtype for P.V; the block's own K/V stays in the input dtype; the sliding-
window rule, tanh soft cap, k/v descale and softmax scale are the same.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    # Same definition as extend_attention.tanh.
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _softmax_update(acc, row_max, row_sum, scores, v, v_scale):
    """One online-softmax step; ``scores`` are in log2 units and masked with -inf."""
    tile_max = tl.max(scores, 1)
    tile_max = tl.where(tile_max == float("-inf"), -1e20, tile_max)
    new_max = tl.maximum(row_max, tile_max)
    rescale = tl.exp2(row_max - new_max)
    p = tl.exp2(scores - new_max[:, None])
    row_sum = row_sum * rescale + tl.sum(p, 1)
    acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v) * v_scale
    return acc, new_max, row_sum


@triton.jit
def _partials_kernel(
    Q,
    K_Extend,
    V_Extend,
    K_Cache,
    V_Cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    Partial_Out,
    Partial_Max,
    Partial_Sum,
    sm_scale,
    k_scale,
    v_scale,
    stride_q_tok,
    stride_q_head,
    stride_ke_tok,
    stride_ke_head,
    stride_ve_tok,
    stride_ve_head,
    stride_kc_slot,
    stride_kc_head,
    stride_vc_slot,
    stride_vc_head,
    NUM_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    EXTEND_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    LOGIT_CAP: tl.constexpr,
):
    LOG2E: tl.constexpr = 1.4426950408889634
    split = tl.program_id(0)
    kv_head = tl.program_id(1)
    req = tl.program_id(2)

    q_start = tl.load(qo_indptr + req)
    q_len = tl.load(qo_indptr + req + 1) - q_start
    kv_start = tl.load(kv_indptr + req)
    prefix_len = tl.load(kv_indptr + req + 1) - kv_start

    # Row r of the tile: query token r // GROUP of the block, and query head r % GROUP
    # of this KV head's group.
    rows = tl.arange(0, BLOCK_TOKENS * GROUP)
    tok = rows // GROUP
    head = kv_head * GROUP + rows % GROUP
    row_valid = tok < q_len
    d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q
        + (q_start + tok)[:, None] * stride_q_tok
        + head[:, None] * stride_q_head
        + d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    q_for_cache = q.to(K_Cache.dtype.element_ty)

    acc = tl.zeros([BLOCK_TOKENS * GROUP, HEAD_DIM], dtype=tl.float32)
    row_max = tl.full([BLOCK_TOKENS * GROUP], float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_TOKENS * GROUP], dtype=tl.float32)

    # Query position in window coordinates, as in _fwd_kernel: the block follows the
    # window prefix.
    q_pos = prefix_len + tok
    first_tile = 0
    if SLIDING_WINDOW > 0:
        # Tiles left of the first query's window are invisible to every row.
        first_tile = tl.maximum(prefix_len - SLIDING_WINDOW, 0) // BLOCK_N
    num_tiles = tl.cdiv(prefix_len, BLOCK_N)
    tiles_per_split = tl.cdiv(num_tiles - first_tile, NUM_SPLITS)
    tile_begin = first_tile + split * tiles_per_split
    tile_end = tl.minimum(tile_begin + tiles_per_split, num_tiles)

    cols = tl.arange(0, BLOCK_N)
    for tile in range(tile_begin, tile_end):
        pos = tile * BLOCK_N + cols
        kv_valid = pos < prefix_len
        slot = tl.load(kv_indices + kv_start + pos, mask=kv_valid, other=0).to(tl.int64)
        k = tl.load(
            K_Cache
            + slot[None, :] * stride_kc_slot
            + kv_head * stride_kc_head
            + d[:, None],
            mask=kv_valid[None, :],
            other=0.0,
        )
        scores = tl.dot(q_for_cache, k) * (sm_scale * k_scale)
        if LOGIT_CAP > 0:
            scores = LOGIT_CAP * _tanh(scores / LOGIT_CAP)
        visible = row_valid[:, None] & kv_valid[None, :]
        if SLIDING_WINDOW > 0:
            visible &= q_pos[:, None] <= pos[None, :] + SLIDING_WINDOW
        scores = tl.where(visible, scores * LOG2E, float("-inf"))
        v = tl.load(
            V_Cache
            + slot[:, None] * stride_vc_slot
            + kv_head * stride_vc_head
            + d[None, :],
            mask=kv_valid[:, None],
            other=0.0,
        )
        acc, row_max, row_sum = _softmax_update(
            acc, row_max, row_sum, scores, v, v_scale
        )

    # The block's own K/V (not descaled), handled by the last split.
    if split == NUM_SPLITS - 1:
        ext = tl.arange(0, EXTEND_N)
        ext_valid = ext < q_len
        k = tl.load(
            K_Extend
            + (q_start + ext)[None, :] * stride_ke_tok
            + kv_head * stride_ke_head
            + d[:, None],
            mask=ext_valid[None, :],
            other=0.0,
        )
        scores = tl.dot(q, k) * sm_scale
        if LOGIT_CAP > 0:
            scores = LOGIT_CAP * _tanh(scores / LOGIT_CAP)
        visible = row_valid[:, None] & ext_valid[None, :]
        if SLIDING_WINDOW > 0:
            visible &= (tok[:, None] <= ext[None, :] + SLIDING_WINDOW) & (
                ext[None, :] <= tok[:, None] + SLIDING_WINDOW
            )
        scores = tl.where(visible, scores * LOG2E, float("-inf"))
        v = tl.load(
            V_Extend
            + (q_start + ext)[:, None] * stride_ve_tok
            + kv_head * stride_ve_head
            + d[None, :],
            mask=ext_valid[:, None],
            other=0.0,
        )
        acc, row_max, row_sum = _softmax_update(acc, row_max, row_sum, scores, v, 1.0)

    state = ((q_start + tok) * NUM_HEADS + head) * NUM_SPLITS + split
    tl.store(Partial_Max + state, row_max, mask=row_valid)
    tl.store(Partial_Sum + state, row_sum, mask=row_valid)
    tl.store(
        Partial_Out + state[:, None] * HEAD_DIM + d[None, :],
        acc,
        mask=row_valid[:, None],
    )


@triton.jit
def _reduce_kernel(
    Partial_Out,
    Partial_Max,
    Partial_Sum,
    Out,
    stride_o_tok,
    stride_o_head,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    splits = tl.arange(0, NUM_SPLITS)
    d = tl.arange(0, HEAD_DIM)
    state = (token * NUM_HEADS + head) * NUM_SPLITS + splits
    split_max = tl.load(Partial_Max + state)
    split_sum = tl.load(Partial_Sum + state)
    total_max = tl.max(split_max, 0)
    weight = tl.exp2(split_max - total_max)
    total_sum = tl.sum(split_sum * weight, 0)
    partial = tl.load(Partial_Out + state[:, None] * HEAD_DIM + d[None, :])
    out = tl.sum(partial * weight[:, None], 0)
    # A row that saw no K/V stores 0, like _fwd_kernel.
    out = out / tl.where(total_sum == 0.0, 1.0, total_sum)
    tl.store(
        Out + token * stride_o_tok + head * stride_o_head + d,
        out.to(Out.dtype.element_ty),
    )


def _block_tokens(max_extend_len: int, group: int) -> int:
    # Query tokens per row tile, padded so the tile has at least 16 rows for the MFMA.
    return max(triton.next_power_of_2(max_extend_len), 16 // group, 1)


def launch_config(batch_size: int, window: int):
    """Split count and K/V tile size for a batch size.

    Up to 16 requests every split is a single K/V tile. A program's cost grows with the
    tokens it walks, so small batches use many short splits to put the whole window in
    flight at once; every split also writes a 64 KB fp32 partial per request, so larger
    batches use fewer, longer splits. Measured on MI355X with a 2048-token window
    (RESULTS.md, launch sweep).
    """
    if batch_size <= 2:
        tokens_per_split, block_n = 64, 64
    elif batch_size <= 8:
        tokens_per_split, block_n = 128, 128
    elif batch_size <= 16:
        tokens_per_split, block_n = 256, 256
    elif batch_size <= 32:
        tokens_per_split, block_n = 512, 256
    elif batch_size <= 64:
        tokens_per_split, block_n = 512, 64
    else:
        tokens_per_split, block_n = 1024, 64
    num_splits = triton.next_power_of_2(triton.cdiv(window, tokens_per_split))
    return num_splits, min(block_n, triton.next_power_of_2(window))


def dflash_draft_splitkv_supported(
    q: torch.Tensor,
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    max_extend_len: int,
    sliding_window_size: int,
) -> bool:
    """Shapes and dtypes this kernel serves; otherwise keep the generic kernel."""
    if q.ndim != 3 or k_extend.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        return False
    num_heads, head_dim = q.shape[1], q.shape[2]
    num_kv_heads = k_extend.shape[1]
    if num_kv_heads == 0 or num_heads % num_kv_heads != 0:
        return False
    group = num_heads // num_kv_heads
    rows = group * _block_tokens(max(max_extend_len, 1), group)
    return (
        head_dim == 128
        and k_extend.shape[2] == v_extend.shape[2] == head_dim
        and v_extend.shape[1] == num_kv_heads
        and k_cache.shape[1:] == v_cache.shape[1:] == (num_kv_heads, head_dim)
        and group & (group - 1) == 0
        and 16 <= rows <= 64
        and 1 <= max_extend_len <= sliding_window_size
        and q.dtype == k_extend.dtype == v_extend.dtype
        and q.dtype in (torch.bfloat16, torch.float16)
        and k_cache.dtype == v_cache.dtype
        and all(t.stride(-1) == 1 for t in (q, k_extend, v_extend, k_cache, v_cache))
    )


def dflash_draft_splitkv_fwd(
    q: torch.Tensor,
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    o: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    max_extend_len: int,
    sm_scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    sliding_window_size: int = -1,
    logit_cap: float = 0.0,
    num_splits: Optional[int] = None,
    block_n: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 1,
):
    """Non-causal attention of each request's block to its cached window and itself.

    q, o: [tokens, heads, head_dim]; k_extend, v_extend: [tokens, kv_heads, head_dim];
    k_cache, v_cache: [slots, kv_heads, head_dim]; ``kv_indptr`` / ``kv_indices`` list
    each request's window slots and ``qo_indptr`` its query tokens (at most
    ``max_extend_len`` per request).
    """
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k_extend.shape[1]
    batch_size = qo_indptr.shape[0] - 1
    group = num_heads // num_kv_heads
    block_tokens = _block_tokens(max_extend_len, group)
    assert sliding_window_size > 0, "the split count is sized by the window"
    default_splits, default_block_n = launch_config(batch_size, sliding_window_size)
    num_splits = num_splits or default_splits
    block_n = block_n or default_block_n

    states = num_tokens * num_heads * num_splits
    partial_out = torch.empty((states, head_dim), dtype=torch.float32, device=q.device)
    partial_max = torch.empty((states,), dtype=torch.float32, device=q.device)
    partial_sum = torch.empty((states,), dtype=torch.float32, device=q.device)

    _partials_kernel[(num_splits, num_kv_heads, batch_size)](
        q,
        k_extend,
        v_extend,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        partial_out,
        partial_max,
        partial_sum,
        sm_scale,
        k_scale,
        v_scale,
        q.stride(0),
        q.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        NUM_HEADS=num_heads,
        GROUP=group,
        BLOCK_TOKENS=block_tokens,
        EXTEND_N=max(16, block_tokens),
        HEAD_DIM=head_dim,
        BLOCK_N=block_n,
        NUM_SPLITS=num_splits,
        SLIDING_WINDOW=sliding_window_size,
        LOGIT_CAP=logit_cap,
        num_warps=num_warps,
        num_stages=num_stages,
        matrix_instr_nonkdim=16,
    )
    _reduce_kernel[(num_tokens, num_heads)](
        partial_out,
        partial_max,
        partial_sum,
        o,
        o.stride(0),
        o.stride(1),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        NUM_SPLITS=num_splits,
        num_warps=2,
    )
    return o
