"""TP4 target-verify attention for Muse Spark: 32 Q / 4 KV heads per rank, 8 verify tokens per request (causal
among them), head_dim 64, FP8 e4m3 KV cache in 16-token pages, BF16 Q and output. Full-attention layers and
sliding-window layers (window 2047 keys including the query's own position).

Split-KV: one program per (request, split, KV head) owns the head's 64 query rows (8 tokens x 8 heads); with one
warp (full layers) each K/V element is read and upcast to BF16 once per row block. The four KV heads of a split run
as programs j, j + 8, j + 16, j + 24, which land on the same XCD, so a token's 256-byte KV line is fetched into one
L2. A second kernel merges the splits by their log-sum-exp; given the attention's output gate it also applies the
output norm and sigmoid gate (``_merge_splits_output_gate``), so the model's separate norm-gate kernel is skipped.

Softmax with a fixed reference: scores (FP32, scaled to log2 units) are exponentiated against 0 instead of a
running maximum, so tiles inside the visible range need no row max, no rescale of the accumulator and no
cross-lane reduction, and the row sums come from the MFMA as P x ones. This is exact in floating point while the
sums stay inside [1e-30, 1e30]; a program whose sums leave that range redoes those tiles with an online maximum.
Tiles that touch a mask boundary (the causal end, the end of the KV, the left edge of a window) run separately with
an online maximum.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

@dataclass
class OutputNormGate:
    """An attention output's norm and sigmoid gate for the verify merge to apply: ``gate`` [tokens, heads x 64]
    BF16, ``eps`` of the weightless head norm. The backend sets ``applied`` when its merge applied them."""

    gate: torch.Tensor
    eps: float
    applied: bool = False


TOKENS = 8  # verify tokens per request
HEAD_DIM = 64
PAGE = 16
TILE = 64  # keys per tile


@triton.jit
def _online_tile(s, v, m, l, acc):
    """One tile with a running per-row maximum (scores in log2 units, masked entries -inf)."""
    n = tl.maximum(m, tl.max(s, axis=1))
    n = tl.where(n > float("-inf"), n, 0.0)
    p = tl.exp2(s - n[:, None])
    r = tl.exp2(m - n)
    l = l * r + tl.sum(p, axis=1)
    acc = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), acc=acc * r[:, None])
    return n, l, acc


@triton.jit
def _boundary_tile(q, K, V, table_row, first, tile, kv_len, context, rows, keys, d, scale, kv_stride_0, kv_stride_1,
                   kv_head, m, l, acc, WINDOW: tl.constexpr, BN: tl.constexpr, PAGE: tl.constexpr):
    position = first + tile * BN + keys
    # keys past the end read the last key instead (finite data, masked out of the scores)
    clamped = tl.minimum(position, kv_len - 1)
    kv = tl.load(table_row + clamped // PAGE).to(tl.int64) * kv_stride_0 + (clamped % PAGE) * kv_stride_1 + kv_head * 64
    k = tl.load(K + kv[None, :] + d[:, None])
    v = tl.load(V + kv[:, None] + d[None, :])
    s = tl.dot(q, k.to(tl.bfloat16)) * scale
    last = context + rows // 8  # last key each row sees
    visible = (position[None, :] <= last[:, None]) & (position < kv_len)[None, :]
    if WINDOW > 0:
        visible &= position[None, :] > (last - WINDOW)[:, None]
    return _online_tile(tl.where(visible, s, float("-inf")), v, m, l, acc)


@triton.jit
def _verify_partials(
    Q, K, V, SEQ, TABLE, PARTIAL, LSE, K_SCALE,
    q_stride_0, q_stride_1, kv_stride_0: tl.constexpr, kv_stride_1: tl.constexpr, table_stride,
    num_work, HEADS: tl.constexpr, KV_HEADS: tl.constexpr, PAGE: tl.constexpr, SPLITS: tl.constexpr,
    SCALE: tl.constexpr, WINDOW: tl.constexpr, BN: tl.constexpr,
):
    slot = tl.program_id(0)
    kv_head = (slot // 8) % KV_HEADS
    work = (slot // (8 * KV_HEADS)) * 8 + slot % 8
    if work >= num_work:
        return
    seq = work // SPLITS
    split = work % SPLITS
    kv_len = tl.load(SEQ + seq).to(tl.int32)
    context = kv_len - 8
    rows = tl.arange(0, 64)
    d = tl.arange(0, 64)
    keys = tl.arange(0, BN)
    token = seq * 8 + rows // 8
    head = kv_head * 8 + rows % 8
    scale = SCALE * 1.4426950408889634
    if K_SCALE is not None:
        scale *= tl.load(K_SCALE)
    q = tl.load(Q + token[:, None] * q_stride_0 + head[:, None] * q_stride_1 + d[None, :])

    # tile t covers keys first + t * BN ...; a window's first tile holds the rows' different left edges
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
    # [inner_begin, inner_end): tiles every row sees in full (they end before the first verify token)
    inner_begin = tl.minimum(end, tl.maximum(begin, edge_tiles))
    inner_end = tl.maximum(inner_begin, tl.minimum(end, (context + 1 - first) // BN))
    table_row = TABLE + seq * table_stride
    in_page = ((first + keys) % PAGE) * kv_stride_1 + kv_head * 64  # BN is a multiple of PAGE

    m = tl.full((64,), float("-inf"), tl.float32)
    l = tl.zeros((64,), tl.float32)
    acc = tl.zeros((64, 64), tl.float32)
    if inner_begin < inner_end:
        ones = tl.full((BN, 16), 1.0, tl.bfloat16)
        row_sum = tl.zeros((64, 16), tl.float32)
        next_pages = tl.load(table_row + (first + inner_begin * BN + keys) // PAGE)
        for tile in range(inner_begin, inner_end):
            pages = next_pages
            next_pages = tl.load(table_row + (first + (tile + 1) * BN + keys) // PAGE, mask=tile + 1 < inner_end,
                                 other=0)
            kv = pages.to(tl.int64) * kv_stride_0 + in_page  # 64-bit: passes 2^31 above 8M tokens
            k = tl.load(K + kv[None, :] + d[:, None])
            v = tl.load(V + kv[:, None] + d[None, :])
            p = tl.exp2(tl.dot(q, k.to(tl.bfloat16)) * scale).to(tl.bfloat16)
            row_sum = tl.dot(p, ones, acc=row_sum)
            acc = tl.dot(p, v.to(tl.bfloat16), acc=acc)
        l = tl.max(row_sum, axis=1)
        m = tl.zeros((64,), tl.float32)
        if (tl.max(l, axis=0) > 1e30) | (tl.min(l, axis=0) < 1e-30):
            m = tl.full((64,), float("-inf"), tl.float32)
            l = tl.zeros((64,), tl.float32)
            acc = tl.zeros((64, 64), tl.float32)
            for tile in range(inner_begin, inner_end):
                kv = tl.load(table_row + (first + tile * BN + keys) // PAGE).to(tl.int64) * kv_stride_0 + in_page
                s = tl.dot(q, tl.load(K + kv[None, :] + d[:, None]).to(tl.bfloat16)) * scale
                m, l, acc = _online_tile(s, tl.load(V + kv[:, None] + d[None, :]), m, l, acc)
    for tile in tl.range(inner_end, end, num_stages=1):
        m, l, acc = _boundary_tile(q, K, V, table_row, first, tile, kv_len, context, rows, keys, d, scale,
                                   kv_stride_0, kv_stride_1, kv_head, m, l, acc, WINDOW, BN, PAGE)
    for tile in tl.range(begin, inner_begin, num_stages=1):
        m, l, acc = _boundary_tile(q, K, V, table_row, first, tile, kv_len, context, rows, keys, d, scale,
                                   kv_stride_0, kv_stride_1, kv_head, m, l, acc, WINDOW, BN, PAGE)

    index = (token * HEADS + head) * SPLITS + split
    tl.store(LSE + index, tl.where(l > 0, m + tl.log2(l), float("-inf")))
    out = acc / tl.where(l > 0, l, 1.0)[:, None]
    tl.store(PARTIAL + index[:, None] * 64 + d[None, :], out.to(tl.float16))


@triton.jit
def _merge_splits(
    PARTIAL, LSE, OUT, V_SCALE, o_stride_0, o_stride_1,
    HEADS: tl.constexpr, SPLITS: tl.constexpr, SPLITS_POW2: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    split = tl.arange(0, SPLITS_POW2)
    d = tl.arange(0, 64)
    index = (token * HEADS + head) * SPLITS + split
    lse = tl.load(LSE + index, mask=split < SPLITS, other=float("-inf"))
    partial = tl.load(PARTIAL + index[:, None] * 64 + d[None, :], mask=(split < SPLITS)[:, None], other=0.0)
    weight = tl.exp2(lse - tl.max(lse, axis=0))
    out = tl.sum(partial.to(tl.float32) * weight[:, None], axis=0) / tl.sum(weight, axis=0)
    if V_SCALE is not None:
        out *= tl.load(V_SCALE)
    tl.store(OUT + token * o_stride_0 + head * o_stride_1 + d, out)


@triton.jit
def _norm_gate_sum_squares(x, live):
    """sum(x^2) of a 64-wide head row as ``_rms_norm_gate_fwd_kernel`` adds it on gfx950 (block 8 x 64, four warps):
    a lane's two adjacent columns as x0 x0 + x1 x1 (unfused: its row mask sits between the products and the sum,
    as ``live`` does here), then the 32 lanes xor 16, 8, 4, 2, 1. Each two-element sum below is one add."""
    even, odd = tl.split(tl.reshape(tl.where(live, x * x, 0.0), (32, 2)))
    t = even + odd
    t = tl.sum(tl.reshape(t, (2, 16)), axis=0)
    t = tl.sum(tl.reshape(t, (2, 8)), axis=0)
    t = tl.sum(tl.reshape(t, (2, 4)), axis=0)
    t = tl.sum(tl.reshape(t, (2, 2)), axis=0)
    return tl.sum(t, axis=0)


@triton.jit
def _merge_splits_output_gate(
    PARTIAL, LSE, OUT, V_SCALE, GATE, o_stride_0, o_stride_1, gate_stride, eps, num_tokens,
    HEADS: tl.constexpr, SPLITS: tl.constexpr, SPLITS_POW2: tl.constexpr,
):
    """``_merge_splits``, then on the merged BF16 head row the output norm and gate of the model's
    ``FusedRMSNormGate`` (``_rms_norm_gate_fwd_kernel``: weightless RMS norm over the head, times sigmoid(gate)).
    Same arithmetic and summation order as the two kernels, BF16 rounding between them included."""
    token, head = tl.program_id(0), tl.program_id(1)
    split = tl.arange(0, SPLITS_POW2)
    d = tl.arange(0, 64)
    index = (token * HEADS + head) * SPLITS + split
    lse = tl.load(LSE + index, mask=split < SPLITS, other=float("-inf"))
    partial = tl.load(PARTIAL + index[:, None] * 64 + d[None, :], mask=(split < SPLITS)[:, None], other=0.0)
    weight = tl.exp2(lse - tl.max(lse, axis=0))
    out = tl.sum(partial.to(tl.float32) * weight[:, None], axis=0) / tl.sum(weight, axis=0)
    if V_SCALE is not None:
        out *= tl.load(V_SCALE)
    x = out.to(tl.bfloat16).to(tl.float32)
    var = _norm_gate_sum_squares(x, token < num_tokens) / 64
    rstd = 1.0 / tl.sqrt(var + eps)
    gate = tl.load(GATE + token * gate_stride + head * 64 + d).to(tl.float32)
    tl.store(OUT + token * o_stride_0 + head * o_stride_1 + d, x * rstd * tl.sigmoid(gate))


def full_splits(num_seqs: int) -> int:
    # 256 (request, split) pairs = 1024 single-warp programs, one per SIMD; also the best at batch 1..16 for
    # 120K contexts (any count is allowed: the merge masks up to the next power of two)
    return max(1, 256 // num_seqs)


def window_splits(num_seqs: int) -> int:
    # a 2047-key window is 33 tiles: one tile per program up to batch 4, then fewer, longer programs (measured
    # best at batch 1..16 with four warps per program)
    if num_seqs <= 4:
        return 33
    if num_seqs <= 8:
        return 8
    return 6 if num_seqs <= 10 else 4


def can_verify_tp4(q, k, v, out, seqused_k, block_table, k_descale, v_descale) -> bool:
    """Metadata-only guard for both layer types; never reads device data.

    Q / output: [tokens, 32, 64] BF16 with unit inner stride; K / V: [pages, 16, 4, 64] FP8 e4m3 with equal strides
    and contiguous (head, dim); 8 verify tokens per request; int32 page table; scalar FP32 descales or None.
    """
    if not (
        q.ndim == out.ndim == 3 and k.ndim == v.ndim == 4
        and seqused_k.ndim == 1 and block_table.ndim == 2
        and q.is_cuda and all(x.device == q.device for x in (k, v, out, seqused_k, block_table))
        and q.dtype == out.dtype == torch.bfloat16
        and k.dtype == v.dtype == torch.float8_e4m3fn
        and q.shape == out.shape and q.shape[1:] == (32, HEAD_DIM)
        and k.shape == v.shape and k.shape[1:] == (PAGE, 4, HEAD_DIM)
        and k.stride() == v.stride() and k.stride(3) == 1 and k.stride(2) == HEAD_DIM
        and seqused_k.numel() > 0 and q.shape[0] == seqused_k.numel() * TOKENS
        and block_table.shape[0] == seqused_k.numel()
        and seqused_k.dtype in (torch.int32, torch.int64) and seqused_k.stride(0) == 1
        and block_table.dtype == torch.int32 and block_table.stride(1) == 1
    ):
        return False
    if not all(x.stride(2) == 1 and x.stride(1) >= HEAD_DIM for x in (q, out)):
        return False
    return all(
        scale is None or (isinstance(scale, torch.Tensor) and scale.device == q.device
                          and scale.dtype == torch.float32 and scale.numel() == 1)
        for scale in (k_descale, v_descale)
    )


def can_apply_output_gate(out, gate) -> bool:
    """Whether ``_merge_splits_output_gate`` can apply this output gate ([tokens, 32 x 64] BF16, unit inner stride)."""
    return (
        gate is not None
        and gate.ndim == 2
        and gate.dtype == torch.bfloat16
        and gate.device == out.device
        and gate.shape == (out.shape[0], out.shape[1] * HEAD_DIM)
        and gate.stride(1) == 1
    )


def _verify(q, k, v, out, seqused_k, block_table, softmax_scale, k_descale, v_descale, window, splits, warps,
            stages, output_gate=None):
    num_seqs = seqused_k.numel()
    heads, kv_heads = q.shape[1], k.shape[2]
    partial = torch.empty((q.shape[0], heads, splits, HEAD_DIM), dtype=torch.float16, device=q.device)
    lse = torch.empty((q.shape[0], heads, splits), dtype=torch.float32, device=q.device)
    work = num_seqs * splits
    _verify_partials[(triton.cdiv(work, 8) * 8 * kv_heads,)](
        q, k, v, seqused_k, block_table, partial, lse, k_descale,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), block_table.stride(0),
        work, heads, kv_heads, k.shape[1], splits, float(softmax_scale), window, TILE,
        num_warps=warps, num_stages=stages, matrix_instr_nonkdim=16,
    )
    if output_gate is None:
        _merge_splits[(q.shape[0], heads)](
            partial, lse, out, v_descale, out.stride(0), out.stride(1),
            heads, splits, triton.next_power_of_2(splits), num_warps=1,
        )
        return out
    gate, eps = output_gate
    _merge_splits_output_gate[(q.shape[0], heads)](
        partial, lse, out, v_descale, gate, out.stride(0), out.stride(1), gate.stride(0), float(eps), q.shape[0],
        heads, splits, triton.next_power_of_2(splits), num_warps=1,
    )
    return out


def verify_full_tp4(q, k, v, out, seqused_k, block_table, softmax_scale, k_descale, v_descale, *, splits=None,
                    output_gate=None):
    """Full-attention verify into ``out``; ``output_gate`` = (gate, eps) also applies the output norm and gate."""
    return _verify(q, k, v, out, seqused_k, block_table, softmax_scale, k_descale, v_descale, 0,
                   splits or full_splits(seqused_k.numel()), 1, 3, output_gate)


def verify_swa_tp4(q, k, v, out, seqused_k, block_table, softmax_scale, k_descale, v_descale, *, window=2047,
                   splits=None, output_gate=None):
    """Sliding-window verify into ``out``; ``window`` counts the keys a query sees, its own position included;
    ``output_gate`` = (gate, eps) also applies the output norm and gate."""
    return _verify(q, k, v, out, seqused_k, block_table, softmax_scale, k_descale, v_descale, window,
                   splits or window_splits(seqused_k.numel()), 4, 2, output_gate)
