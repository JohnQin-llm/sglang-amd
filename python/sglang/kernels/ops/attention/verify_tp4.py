"""Target-verify attention for Muse Spark at TP4 on gfx950 (MI355X).

Per rank: 32 Q heads over 4 KV heads, 8 verify tokens per request (causal among
themselves), head_dim 64, BF16 Q and output, FP8 e4m3 KV cache in 16-token pages.
Full-attention layers and sliding-window layers (2047 keys, the query's own
position included) share one split-KV kernel and a log-sum-exp merge of the splits.

One single-warp program owns a (request, KV split, KV head) and all 64 query rows
of that KV head (8 tokens x 8 Q heads), so each K/V element is loaded and upcast
to BF16 once. gqa_verify_splitkv_4kv spreads two KV heads over four warps, each of
which upcasts the whole tile; this layout alone took batch 16 at 120K tokens from
289 to 220 us.

Tiles that every query row sees in full skip the running maximum: p = exp2(score)
against a fixed reference of 0, and the row sums come from the MFMA as P x ones.
That cut the interior loop from about 420 to 185 VALU instructions per 64-key
tile, and batch 16 from 214 to 192 us. The result is exact while the row sums stay
inside [1e-30, 1e30]; a program whose sums leave that range recomputes its
interior tiles with an online maximum. Tiles on a mask edge (the causal end, the
end of the KV, the left edge of a window) always use the online maximum.

Given the attention's output gate (``OutputNormGate``), the merge also applies the
output norm and sigmoid gate, so the model skips its separate norm-gate kernel.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

TOKENS = 8  # verify tokens per request
HEAD_DIM = 64
PAGE = 16  # KV cache page size in tokens
TILE = 64  # keys per tile, a multiple of PAGE


@dataclass
class OutputNormGate:
    """The attention output's weightless per-head RMS norm and sigmoid gate, for the
    split merge to apply. ``gate`` is [tokens, heads x 64] BF16 with unit inner
    stride; the verify call sets ``applied`` when its merge applied both."""

    gate: torch.Tensor
    eps: float
    applied: bool = False


@triton.jit
def _online_softmax_tile(s, v, m, l, acc):
    """Add a tile of log2-unit scores ``s`` (-inf where masked), running row maximum."""
    n = tl.maximum(m, tl.max(s, axis=1))
    n = tl.where(n > float("-inf"), n, 0.0)  # rows that see no key yet
    p = tl.exp2(s - n[:, None])
    r = tl.exp2(m - n)
    l = l * r + tl.sum(p, axis=1)
    acc = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), acc=acc * r[:, None])
    return n, l, acc


@triton.jit
def _masked_tile(
    q,
    K,
    V,
    table_row,
    first,
    tile,
    kv_len,
    context,
    rows,
    keys,
    d,
    scale,
    kv_stride_0,
    kv_stride_1,
    kv_head,
    m,
    l,
    acc,
    WINDOW: tl.constexpr,
    BN: tl.constexpr,
    PAGE: tl.constexpr,
):
    position = first + tile * BN + keys
    # Keys past the end re-read the last key, so no table entry past the end is read.
    clamped = tl.minimum(position, kv_len - 1)
    kv = (
        tl.load(table_row + clamped // PAGE) * kv_stride_0
        + (clamped % PAGE) * kv_stride_1
        + kv_head * 64
    )
    k = tl.load(K + kv[None, :] + d[:, None])
    v = tl.load(V + kv[:, None] + d[None, :])
    s = tl.dot(q, k.to(tl.bfloat16)) * scale
    last = context + rows // 8  # last key each query row sees
    visible = (position[None, :] <= last[:, None]) & (position < kv_len)[None, :]
    if WINDOW > 0:
        visible &= position[None, :] > (last - WINDOW)[:, None]
    return _online_softmax_tile(tl.where(visible, s, float("-inf")), v, m, l, acc)


@triton.jit
def _split_kv_partials(
    Q,
    K,
    V,
    SEQ,
    TABLE,
    PARTIAL,
    LSE,
    K_SCALE,
    q_stride_0,
    q_stride_1,
    kv_stride_0: tl.constexpr,
    kv_stride_1: tl.constexpr,
    table_stride,
    num_work,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
    BN: tl.constexpr,
):
    work = tl.program_id(0)  # (request, split) pair
    kv_head = tl.program_id(1)
    if work >= num_work:
        return
    seq = work // SPLITS
    split = work % SPLITS
    kv_len = tl.load(SEQ + seq).to(tl.int32)
    context = kv_len - 8  # position of the first verify token
    rows = tl.arange(0, 64)  # 8 tokens x 8 Q heads of this KV head
    d = tl.arange(0, 64)
    keys = tl.arange(0, BN)
    token = seq * 8 + rows // 8
    head = kv_head * 8 + rows % 8
    scale = SCALE * 1.4426950408889634  # log2(e): scores in log2 units for exp2
    if K_SCALE is not None:
        scale *= tl.load(K_SCALE)
    q_rows = token[:, None] * q_stride_0 + head[:, None] * q_stride_1
    q = tl.load(Q + q_rows + d[None, :])

    # Tile t covers keys first + t * BN onward; a window's first tile holds the
    # rows' different left edges.
    if WINDOW > 0:
        first = tl.maximum(0, context - WINDOW + 1)
        edge_tiles = 1
    else:
        first = 0
        edge_tiles = 0
    tiles = tl.cdiv(kv_len - first, BN)
    per_split = tl.cdiv(tiles, SPLITS)
    begin = split * per_split
    end = tl.minimum(tiles, begin + per_split)
    # [inner_begin, inner_end): tiles that end before the first verify token, which
    # every row sees in full.
    inner_begin = tl.minimum(end, tl.maximum(begin, edge_tiles))
    inner_end = tl.maximum(inner_begin, tl.minimum(end, (context + 1 - first) // BN))
    table_row = TABLE + seq * table_stride
    in_page = ((first + keys) % PAGE) * kv_stride_1 + kv_head * 64

    m = tl.full((64,), float("-inf"), tl.float32)
    l = tl.zeros((64,), tl.float32)
    acc = tl.zeros((64, 64), tl.float32)
    if inner_begin < inner_end:
        ones = tl.full((BN, 16), 1.0, tl.bfloat16)
        row_sum = tl.zeros((64, 16), tl.float32)
        # Page numbers load one tile ahead (batch 8: 102 us, 132 us without).
        next_pages = tl.load(table_row + (first + inner_begin * BN + keys) // PAGE)
        for tile in range(inner_begin, inner_end):
            pages = next_pages
            next_pages = tl.load(
                table_row + (first + (tile + 1) * BN + keys) // PAGE,
                mask=tile + 1 < inner_end,
                other=0,
            )
            kv = pages * kv_stride_0 + in_page
            k = tl.load(K + kv[None, :] + d[:, None])
            v = tl.load(V + kv[:, None] + d[None, :])
            p = tl.exp2(tl.dot(q, k.to(tl.bfloat16)) * scale).to(tl.bfloat16)
            row_sum = tl.dot(p, ones, acc=row_sum)  # every column holds the row sum
            acc = tl.dot(p, v.to(tl.bfloat16), acc=acc)
        l = tl.max(row_sum, axis=1)
        m = tl.zeros((64,), tl.float32)
        if (tl.max(l, axis=0) > 1e30) | (tl.min(l, axis=0) < 1e-30):
            m = tl.full((64,), float("-inf"), tl.float32)
            l = tl.zeros((64,), tl.float32)
            acc = tl.zeros((64, 64), tl.float32)
            for tile in range(inner_begin, inner_end):
                pages = tl.load(table_row + (first + tile * BN + keys) // PAGE)
                kv = pages * kv_stride_0 + in_page
                k = tl.load(K + kv[None, :] + d[:, None])
                s = tl.dot(q, k.to(tl.bfloat16)) * scale
                v = tl.load(V + kv[:, None] + d[None, :])
                m, l, acc = _online_softmax_tile(s, v, m, l, acc)
    # At most three masked tiles per program; pipelining them: batch 1 25 -> 32 us.
    for tile in tl.range(inner_end, end, num_stages=1):
        m, l, acc = _masked_tile(
            q, K, V, table_row, first, tile, kv_len, context, rows, keys, d, scale,
            kv_stride_0, kv_stride_1, kv_head, m, l, acc, WINDOW, BN, PAGE,
        )  # fmt: skip
    for tile in tl.range(begin, inner_begin, num_stages=1):
        m, l, acc = _masked_tile(
            q, K, V, table_row, first, tile, kv_len, context, rows, keys, d, scale,
            kv_stride_0, kv_stride_1, kv_head, m, l, acc, WINDOW, BN, PAGE,
        )  # fmt: skip

    index = (token * HEADS + head) * SPLITS + split
    tl.store(LSE + index, tl.where(l > 0, m + tl.log2(l), float("-inf")))
    out = acc / tl.where(l > 0, l, 1.0)[:, None]
    # Normalized partials are bounded by max |V|, so FP16 holds them (batch 8: 102 us,
    # 107 us in FP32).
    tl.store(PARTIAL + index[:, None] * 64 + d[None, :], out.to(tl.float16))


@triton.jit
def _head_sum_of_squares(x, num_tokens):
    """sum(x * x) of one 64-wide head row in the order FusedRMSNormGate's Triton
    kernel adds it on gfx950, so the gated merge is bitwise equal to it: each lane
    adds its two columns' squares, then lanes xor 16, 8, 4, 2, 1. The always-true
    runtime mask keeps the two products unfused, as that kernel's row mask does."""
    squares = tl.where(tl.program_id(0) < num_tokens, x * x, 0.0)
    even, odd = tl.split(tl.reshape(squares, (32, 2)))
    t = even + odd
    t = tl.sum(tl.reshape(t, (2, 16)), axis=0)
    t = tl.sum(tl.reshape(t, (2, 8)), axis=0)
    t = tl.sum(tl.reshape(t, (2, 4)), axis=0)
    t = tl.sum(tl.reshape(t, (2, 2)), axis=0)
    return tl.sum(t, axis=0)


@triton.jit
def _merge_splits(
    PARTIAL,
    LSE,
    OUT,
    V_SCALE,
    GATE,
    o_stride_0,
    o_stride_1,
    gate_stride,
    eps,
    num_tokens,
    HEADS: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLITS_POW2: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    split = tl.arange(0, SPLITS_POW2)
    d = tl.arange(0, 64)
    index = (token * HEADS + head) * SPLITS + split
    lse = tl.load(LSE + index, mask=split < SPLITS, other=float("-inf"))
    partial = tl.load(
        PARTIAL + index[:, None] * 64 + d[None, :],
        mask=(split < SPLITS)[:, None],
        other=0.0,
    )
    weight = tl.exp2(lse - tl.max(lse, axis=0))
    out = tl.sum(partial.to(tl.float32) * weight[:, None], axis=0)
    out = out / tl.sum(weight, axis=0)
    if V_SCALE is not None:
        out *= tl.load(V_SCALE)
    if GATE is not None:
        # The output norm and gate on the BF16 merged row, as the model's
        # FusedRMSNormGate computes them.
        out = out.to(tl.bfloat16).to(tl.float32)
        rstd = 1.0 / tl.sqrt(_head_sum_of_squares(out, num_tokens) / 64 + eps)
        gate = tl.load(GATE + token * gate_stride + head * 64 + d).to(tl.float32)
        out = out * rstd * tl.sigmoid(gate)
    tl.store(OUT + token * o_stride_0 + head * o_stride_1 + d, out)


def _full_splits(num_seqs: int) -> int:
    # 256 (request, split) pairs = 1024 single-warp programs, one per SIMD. Any count
    # works (the merge masks to a power of two); at batch 10, 120K tokens, 25 splits
    # take 124 us and 32 take 198 us.
    return max(1, 256 // num_seqs)


def _window_splits(num_seqs: int) -> int:
    # A 2047-key window is 33 tiles: one tile per program up to batch 4, then fewer,
    # longer programs (measured best at batch 1..16).
    if num_seqs <= 4:
        return 33
    if num_seqs <= 8:
        return 8
    return 6 if num_seqs <= 10 else 4


def can_verify_tp4(q, k, v, out, seqused_k, block_table, k_descale, v_descale) -> bool:
    """Metadata-only layout guard for both layer types; never reads device data.

    Q / output: [tokens, 32, 64] BF16 with unit inner stride. K / V: [pages, 16, 4,
    64] FP8 e4m3 with equal strides and contiguous (head, dim). 8 verify tokens per
    request, int32 page table, scalar FP32 descales or None.
    """
    if not (
        q.ndim == out.ndim == 3
        and k.ndim == v.ndim == 4
        and seqused_k.ndim == 1
        and block_table.ndim == 2
        and q.is_cuda
        and all(x.device == q.device for x in (k, v, out, seqused_k, block_table))
        and q.dtype == out.dtype == torch.bfloat16
        and k.dtype == v.dtype == torch.float8_e4m3fn
        and q.shape == out.shape
        and q.shape[1:] == (32, HEAD_DIM)
        and k.shape == v.shape
        and k.shape[1:] == (PAGE, 4, HEAD_DIM)
        and k.stride() == v.stride()
        and k.stride(3) == 1
        and k.stride(2) == HEAD_DIM
        and k.shape[0] * k.stride(0) < 2**31  # 32-bit element offsets
        and seqused_k.numel() > 0
        and q.shape[0] == seqused_k.numel() * TOKENS
        and block_table.shape[0] == seqused_k.numel()
        and seqused_k.dtype in (torch.int32, torch.int64)
        and seqused_k.stride(0) == 1
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1
    ):
        return False
    if not all(x.stride(2) == 1 and x.stride(1) >= HEAD_DIM for x in (q, out)):
        return False
    return all(
        scale is None
        or (
            isinstance(scale, torch.Tensor)
            and scale.device == q.device
            and scale.dtype == torch.float32
            and scale.numel() == 1
        )
        for scale in (k_descale, v_descale)
    )


def _can_apply_output_gate(out, output_gate) -> bool:
    gate = output_gate.gate
    return (
        gate.ndim == 2
        and gate.dtype == torch.bfloat16
        and gate.device == out.device
        and gate.shape == (out.shape[0], out.shape[1] * HEAD_DIM)
        and gate.stride(1) == 1
    )


def _launch(
    q,
    k,
    v,
    out,
    seqused_k,
    block_table,
    softmax_scale,
    k_descale,
    v_descale,
    *,
    window,
    splits,
    num_warps,
    output_gate,
):
    heads, kv_heads = q.shape[1], k.shape[2]
    partial = torch.empty(
        (q.shape[0], heads, splits, HEAD_DIM), dtype=torch.float16, device=q.device
    )
    lse = torch.empty((q.shape[0], heads, splits), dtype=torch.float32, device=q.device)
    work = seqused_k.numel() * splits
    # Programs go round-robin to the 8 XCDs by linear id. Padding the work axis to a
    # multiple of 8 puts the four KV heads of a work item, which share each token's
    # 256-byte KV row, on one XCD and its L2 (batch 8: 102 us, 159 us on four XCDs).
    # Two pipeline stages; one takes batch 16 sliding-window from 15.7 to 22.8 us.
    grid = (triton.cdiv(work, 8) * 8, kv_heads)
    _split_kv_partials[grid](
        q, k, v, seqused_k, block_table, partial, lse, k_descale,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), block_table.stride(0),
        work, heads, PAGE, splits, float(softmax_scale), window, TILE,
        num_warps=num_warps, num_stages=2, matrix_instr_nonkdim=16,
    )  # fmt: skip
    gate, gate_stride, eps = None, 0, 0.0
    if output_gate is not None and _can_apply_output_gate(out, output_gate):
        gate, eps = output_gate.gate, output_gate.eps
        gate_stride = gate.stride(0)
        output_gate.applied = True
    _merge_splits[(q.shape[0], heads)](
        partial, lse, out, v_descale, gate, out.stride(0), out.stride(1), gate_stride,
        float(eps), q.shape[0], heads, splits, triton.next_power_of_2(splits), num_warps=1,
    )  # fmt: skip
    return out


def verify_full_tp4(
    q,
    k,
    v,
    out,
    seqused_k,
    block_table,
    softmax_scale,
    k_descale,
    v_descale,
    *,
    output_gate=None,
):
    """Full-attention verify into ``out``; the caller checks ``can_verify_tp4``.
    ``output_gate`` (``OutputNormGate``) is applied when its layout allows."""
    return _launch(
        q,
        k,
        v,
        out,
        seqused_k,
        block_table,
        softmax_scale,
        k_descale,
        v_descale,
        window=0,
        splits=_full_splits(seqused_k.numel()),
        num_warps=1,
        output_gate=output_gate,
    )


def verify_swa_tp4(
    q,
    k,
    v,
    out,
    seqused_k,
    block_table,
    softmax_scale,
    k_descale,
    v_descale,
    *,
    window=2047,
    output_gate=None,
):
    """Sliding-window verify into ``out``; ``window`` counts the keys a query sees,
    its own position included. The caller checks ``can_verify_tp4``.
    ``output_gate`` (``OutputNormGate``) is applied when its layout allows."""
    # Four warps split the 64 rows of the short window programs (batch 8: 13.5 us,
    # 21.2 us with one warp).
    return _launch(
        q,
        k,
        v,
        out,
        seqused_k,
        block_table,
        softmax_scale,
        k_descale,
        v_descale,
        window=window,
        splits=_window_splits(seqused_k.numel()),
        num_warps=4,
        output_gate=output_gate,
    )
