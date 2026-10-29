"""Per-head QK RMSNorm + interleaved RoPE + paged FP8 KV-cache store (Muse Spark, head_dim 64, page 16).

* ``qk_norm_rope_cache``: from projected Q/K/V. One program per (token, Q head) normalizes, scales and rotates
  Q; the first Q head of each GQA group also normalizes and rotates its K and stores K/V into the FP8 cache
  (optionally through the full -> SWA slot mapping).
* ``qkv_proj_qk_norm_rope_cache``: the same epilogue fed straight from the BF16 split-K partials of the fused
  QKV + output-gate projection (``bf16_splitk_gemm``), which also emits V and the gate. With ``tile_epilogue``
  (TP4, up to 48 rows) the last split-K program of each output tile runs the epilogue in the same launch.

The BF16 round trips of the unfused sequence (norm output, query scaling, RoPE products, projection store)
are explicit, so the results match the separate norm / RoPE / cache-write kernels.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.bf16_splitk_gemm import _dense_partial, _tile_partial


@triton.jit
def _norm_rope(
    x,
    COS,
    POS,
    eps,
    scale,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    Q_SCALE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
):
    d = tl.arange(0, D)
    f = x.to(tl.float32)
    inv = tl.rsqrt(tl.sum(f * f, axis=0) / D + eps)
    value = (f * inv).to(tl.bfloat16).to(tl.float32)
    if Q_SCALE:
        value = (value * scale).to(tl.bfloat16).to(tl.float32)
    if ROPE:
        pos = tl.load(POS + tl.program_id(0))
        # Day0 pos_enc.cu stores the FP32 cache loads in scalar_t (BF16),
        # and its BF16 operators round both products before the final add.
        cosine = tl.load(COS + pos * COS_STRIDE + d // 2).to(tl.bfloat16).to(tl.float32)
        sine = (
            tl.load(COS + pos * COS_STRIDE + D // 2 + d // 2)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        other = tl.reshape(tl.flip(tl.reshape(value, (D // 2, 2)), 1), (D,))
        rotated = tl.where(d % 2 == 0, -other, other)
        first = (value * cosine).to(tl.bfloat16).to(tl.float32)
        second = (rotated * sine).to(tl.bfloat16).to(tl.float32)
        value = first + second
    return value.to(tl.bfloat16)


@triton.jit
def _muse_qk_rope_cache(
    Q,
    K,
    V,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    QO,
    KO,
    KS,
    VS,
    QS: tl.constexpr,
    KSTR: tl.constexpr,
    VSTR: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_MAP: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    q = tl.load(Q + token * QS + head * D + d)
    q = _norm_rope(q, COS, POS, EPS, SCALE, D, ROPE, True, COS_STRIDE)
    tl.store(QO + token * HQ * D + head * D + d, q)
    if head % (HQ // HK) == 0:
        kh = head // (HQ // HK)
        k = tl.load(K + token * KSTR + kh * D + d)
        k = _norm_rope(k, COS, POS, EPS, 1.0, D, ROPE, False, COS_STRIDE)
        tl.store(KO + token * HK * D + kh * D + d, k)
        slot = tl.load(LOC + token).to(tl.int64)
        if slot >= 0:
            if HAS_MAP:
                slot = tl.load(MAP + slot).to(tl.int64)
            if slot >= 0:
                v = tl.load(V + token * VSTR + kh * D + d)
                if HAS_SCALE:
                    k_value = k.to(tl.float32) / tl.load(KS)
                    v_value = v.to(tl.float32) / tl.load(VS)
                else:
                    k_value = k.to(tl.float32)
                    v_value = v.to(tl.float32)
                offset = slot * HK * D + kh * D + d
                tl.store(KC + offset, k_value)
                tl.store(VC + offset, v_value)


@triton.jit
def _project_slice(
    P, TOKEN, FIRST, M: tl.constexpr, N: tl.constexpr, SK: tl.constexpr, D: tl.constexpr
):
    split = tl.arange(0, SK)
    col = FIRST + tl.arange(0, D)
    partial = tl.load(P + split[:, None] * M * N + TOKEN * N + col[None, :])
    return tl.sum(partial, axis=0).to(tl.bfloat16)


@triton.jit
def _qkv_epilogue(
    P,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    QO,
    KO,
    VO,
    GO,
    KS,
    VS,
    M: tl.constexpr,
    N: tl.constexpr,
    SK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_MAP: tl.constexpr,
    COS_STRIDE: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    gate = _project_slice(P, token, (HQ + 2 * HK) * D + head * D, M, N, SK, D)
    tl.store(GO + token * HQ * D + head * D + d, gate)
    q = _project_slice(P, token, head * D, M, N, SK, D)
    q = _norm_rope(q, COS, POS, EPS, SCALE, D, ROPE, True, COS_STRIDE)
    tl.store(QO + token * HQ * D + head * D + d, q)
    if head % (HQ // HK) == 0:
        kh = head // (HQ // HK)
        k = _project_slice(P, token, HQ * D + kh * D, M, N, SK, D)
        v = _project_slice(P, token, (HQ + HK) * D + kh * D, M, N, SK, D)
        k = _norm_rope(k, COS, POS, EPS, 1.0, D, ROPE, False, COS_STRIDE)
        tl.store(KO + token * HK * D + kh * D + d, k)
        tl.store(VO + token * HK * D + kh * D + d, v)
        slot = tl.load(LOC + token).to(tl.int64)
        if slot >= 0:
            if HAS_MAP:
                slot = tl.load(MAP + slot).to(tl.int64)
            if slot >= 0:
                offset = slot * HK * D + kh * D + d
                tl.store(KC + offset, k.to(tl.float32) / tl.load(KS))
                tl.store(VC + offset, v.to(tl.float32) / tl.load(VS))


@triton.jit
def _tile_split_sum(tile, stride, mask, SK: tl.constexpr):
    """The SK split-K partials of a tile (``stride`` apart), summed in the order
    _project_slice's tl.sum adds them on gfx950: split s with s + SK / 2, then
    with s + SK / 4, and so on."""
    p0 = tl.load(tile, mask=mask, other=0.0)
    p1 = tl.load(tile + stride, mask=mask, other=0.0)
    p2 = tl.load(tile + 2 * stride, mask=mask, other=0.0)
    p3 = tl.load(tile + 3 * stride, mask=mask, other=0.0)
    if SK == 4:
        total = (p0 + p2) + (p1 + p3)
    else:
        tl.static_assert(SK == 8)
        p4 = tl.load(tile + 4 * stride, mask=mask, other=0.0)
        p5 = tl.load(tile + 5 * stride, mask=mask, other=0.0)
        p6 = tl.load(tile + 6 * stride, mask=mask, other=0.0)
        p7 = tl.load(tile + 7 * stride, mask=mask, other=0.0)
        total = ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7))
    return total


@triton.jit
def _tile_head_rsqrt(f, eps, BM: tl.constexpr, D: tl.constexpr):
    """rsqrt(mean(f^2) + eps) of each row of f [BM, 64], summed as _norm_rope's
    tl.sum adds it on gfx950: per thread fma(c0, c0, c2 c2) + fma(c1, c1, c3 c3)
    over four adjacent columns, then 16 lanes xor 8, 4, 2, 1."""
    even, odd = tl.split(tl.reshape(f, (BM, D // 4, 2, 2)))
    c0, c2 = tl.split(even)
    c1, c3 = tl.split(odd)
    t = tl.fma(c0, c0, c2 * c2) + tl.fma(c1, c1, c3 * c3)
    t = tl.sum(tl.reshape(t, (BM, 2, 8)), axis=1)
    t = tl.sum(tl.reshape(t, (BM, 2, 4)), axis=1)
    t = tl.sum(tl.reshape(t, (BM, 2, 2)), axis=1)
    return tl.rsqrt(tl.sum(t, axis=1) / D + eps)


@triton.jit
def _tile_norm_rope(
    f,
    COS,
    pos,
    eps,
    scale,
    is_query,
    BM: tl.constexpr,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
):
    """_norm_rope on the rows of f [BM, 64], each token at its own position."""
    d = tl.arange(0, D)
    value = (
        (f * _tile_head_rsqrt(f, eps, BM, D)[:, None]).to(tl.bfloat16).to(tl.float32)
    )
    if is_query:
        value = (value * scale).to(tl.bfloat16).to(tl.float32)
    if ROPE:
        row = pos[:, None] * COS_STRIDE
        cosine = tl.load(COS + row + d[None, :] // 2).to(tl.bfloat16).to(tl.float32)
        sine = (
            tl.load(COS + row + D // 2 + d[None, :] // 2).to(tl.bfloat16).to(tl.float32)
        )
        other = tl.reshape(tl.flip(tl.reshape(value, (BM, D // 2, 2)), 2), (BM, D))
        rotated = tl.where((d % 2 == 0)[None, :], -other, other)
        first = (value * cosine).to(tl.bfloat16).to(tl.float32)
        second = (rotated * sine).to(tl.bfloat16).to(tl.float32)
        value = first + second
    return value.to(tl.bfloat16)


@triton.jit
def _tile_cache_store(
    CACHE, SCALE, slot, kv_head, value, HK: tl.constexpr, D: tl.constexpr
):
    """Rows of ``value`` [BM, 64] into the FP8 cache of KV head ``kv_head``; slot -1 is skipped."""
    d = tl.arange(0, D)
    tl.store(
        CACHE + slot[:, None] * HK * D + kv_head * D + d[None, :],
        value.to(tl.float32) / tl.load(SCALE),
        mask=(slot >= 0)[:, None],
    )


@triton.jit
def _qkv_tile_epilogue(
    X,
    W,
    P,
    TICKETS,
    POS,
    COS,
    LOC,
    MAP,
    KC,
    VC,
    QO,
    KO,
    VO,
    GO,
    KS,
    VS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_MAP: tl.constexpr,
    COS_STRIDE: tl.constexpr,
):
    """_dense_partial + _qkv_epilogue in one launch, bitwise equal to them. Each
    program stores one split-K partial of a BM x 64 tile, which is one Q, K, V or
    gate head of BM tokens; the tile's last program to finish sums the partials and
    runs the epilogue for that head.

    The partials are written through to memory and drained by the barrier before a
    relaxed ticket, and the last program's acquire invalidates its cached copies
    before it reads them. A first version with a release ticket (an L2 writeback in
    every program) and volatile reads took 35 instead of 18 us at 8 rows.
    """
    D: tl.constexpr = 64
    pn, pm, pk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    token = pm * BM + tl.arange(0, BM)
    live = token < M
    d = tl.arange(0, D)
    col = pn * D + d
    acc = _tile_partial(X, W, pn, pm, pk, M, K, BM, D, BK, SK, ".cg")
    tile = P + token[:, None] * N + col[None, :]
    tl.store(tile + pk * M * N, acc, mask=live[:, None], cache_modifier=".wt")
    tl.debug_barrier()
    ticket = TICKETS + pm * (N // D) + pn
    if tl.atomic_add(ticket, 1, sem="relaxed", scope="gpu") == SK - 1:
        tl.atomic_xchg(ticket, 0, sem="acquire", scope="gpu")  # for the next launch
        value = _tile_split_sum(tile, M * N, live[:, None], SK).to(tl.bfloat16)
        if pn < HQ + HK:  # Q or K: norm, query scale, RoPE
            pos = tl.load(POS + token, mask=live, other=0)
            is_query = pn < HQ
            value = value.to(tl.float32)
            value = _tile_norm_rope(
                value, COS, pos, EPS, SCALE, is_query, BM, D, ROPE, COS_STRIDE
            )
        if pn < HQ:
            out = QO + token[:, None] * HQ * D + pn * D + d[None, :]
            tl.store(out, value, mask=live[:, None])
        elif pn < HQ + 2 * HK:
            kv_head = (pn - HQ) % HK
            slot = tl.load(LOC + token, mask=live, other=-1).to(tl.int64)
            if HAS_MAP:
                mapped = tl.load(MAP + tl.maximum(slot, 0), mask=slot >= 0, other=-1)
                slot = tl.where(slot >= 0, mapped.to(tl.int64), -1)
            row = token[:, None] * HK * D + kv_head * D + d[None, :]
            if pn < HQ + HK:
                tl.store(KO + row, value, mask=live[:, None])
                _tile_cache_store(KC, KS, slot, kv_head, value, HK, D)
            else:
                tl.store(VO + row, value, mask=live[:, None])
                _tile_cache_store(VC, VS, slot, kv_head, value, HK, D)
        else:
            out = GO + token[:, None] * HQ * D + (pn - HQ - 2 * HK) * D + d[None, :]
            tl.store(out, value, mask=live[:, None])


@triton.jit
def _tp4_rope_rows(
    x,
    COS,
    POS,
    D: tl.constexpr,
    H: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    ROPE: tl.constexpr,
):
    value = x.to(tl.bfloat16).to(tl.float32)
    if ROPE:
        d = tl.arange(0, D)
        pos = tl.load(POS + tl.program_id(0))
        cosine = tl.load(COS + pos * COS_STRIDE + d // 2).to(tl.bfloat16).to(tl.float32)
        sine = (
            tl.load(COS + pos * COS_STRIDE + D // 2 + d // 2)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        other = tl.reshape(tl.flip(tl.reshape(value, (H, D // 2, 2)), 2), (H, D))
        rotated = tl.where((d % 2 == 0)[None, :], -other, other)
        first = (value * cosine[None, :]).to(tl.bfloat16).to(tl.float32)
        second = (rotated * sine[None, :]).to(tl.bfloat16).to(tl.float32)
        value = first + second
    return value.to(tl.bfloat16)


@triton.jit
def _muse_tp4_qk_rope_cache(
    Q,
    K,
    V,
    QO,
    KO,
    COS,
    POS,
    LOC,
    MAP,
    KC,
    VC,
    KS,
    VS,
    q_stride,
    k_stride,
    v_stride,
    eps,
    q_scale,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_MAP: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
):
    # Keep the native muse_qk_norm two-dimensional reduction layout. A separate
    # program per head changes the reduction order at BF16 rounding boundaries.
    t = tl.program_id(0)
    offs_q = tl.arange(0, HQ * D)
    q = tl.load(Q + t * q_stride + offs_q).to(tl.float32)
    q2 = tl.reshape(q, (HQ, D))
    rstd_q = tl.rsqrt(tl.sum(q2 * q2, axis=1) / D + eps)
    qn = (q2 * rstd_q[:, None]).to(tl.bfloat16).to(tl.float32) * q_scale
    qn = _tp4_rope_rows(qn, COS, POS, D, HQ, COS_STRIDE, ROPE)
    tl.store(QO + t * HQ * D + offs_q, tl.reshape(qn, (HQ * D,)))
    offs_k = tl.arange(0, HK * D)
    k = tl.load(K + t * k_stride + offs_k).to(tl.float32)
    k2 = tl.reshape(k, (HK, D))
    rstd_k = tl.rsqrt(tl.sum(k2 * k2, axis=1) / D + eps)
    kn = _tp4_rope_rows(k2 * rstd_k[:, None], COS, POS, D, HK, COS_STRIDE, ROPE)
    flat_k = tl.reshape(kn, (HK * D,))
    tl.store(KO + t * HK * D + offs_k, flat_k)
    slot = tl.load(LOC + t).to(tl.int64)
    if slot >= 0:
        if HAS_MAP:
            slot = tl.load(MAP + slot).to(tl.int64)
        if slot >= 0:
            v = tl.load(V + t * v_stride + offs_k).to(tl.float32)
            k_value = flat_k.to(tl.float32)
            if HAS_SCALE:
                k_value = k_value / tl.load(KS)
                v = v / tl.load(VS)
            tl.store(KC + slot * HK * D + offs_k, k_value)
            tl.store(VC + slot * HK * D + offs_k, v)


def qk_norm_rope_cache(
    q,
    k,
    v,
    positions,
    cos_sin,
    cache_k,
    cache_v,
    locations,
    swa_mapping,
    eps,
    q_scale,
    k_scale,
    v_scale,
    warps=1,
    fp_fusion=True,
):
    """Return (normalized + rotated Q [M, HQ*64], K [M, HK*64]); K/V of each token go to the FP8 cache."""
    m = q.shape[0]
    d = 64
    hq = q.shape[1] // d
    hk = k.shape[1] // d
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert cache_k.dtype == cache_v.dtype == torch.float8_e4m3fn
    assert cache_k.is_contiguous() and cache_v.is_contiguous()
    assert q.shape == (m, hq * d) and k.shape == v.shape == (m, hk * d)
    qo = torch.empty_like(q, memory_format=torch.contiguous_format)
    ko = torch.empty_like(k, memory_format=torch.contiguous_format)
    if m and (hq, hk) == (32, 4):
        _muse_tp4_qk_rope_cache[(m,)](
            q,
            k,
            v,
            qo,
            ko,
            cos_sin,
            positions,
            locations,
            swa_mapping,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            q.stride(0),
            k.stride(0),
            v.stride(0),
            float(eps),
            float(q_scale),
            hq,
            hk,
            d,
            cos_sin is not None,
            swa_mapping is not None,
            k_scale is not None,
            cos_sin.stride(0) if cos_sin is not None else d,
            num_warps=4,
        )
    elif m:
        _muse_qk_rope_cache[(m, hq)](
            q,
            k,
            v,
            positions,
            cos_sin,
            locations,
            swa_mapping,
            cache_k,
            cache_v,
            qo,
            ko,
            k_scale,
            v_scale,
            q.stride(0),
            k.stride(0),
            v.stride(0),
            hq,
            hk,
            d,
            float(eps),
            float(q_scale),
            cos_sin is not None,
            swa_mapping is not None,
            k_scale is not None,
            cos_sin.stride(0) if cos_sin is not None else d,
            num_warps=warps,
            enable_fp_fusion=fp_fusion,
        )
    return qo, ko


# Projection tiles (split-K, block N, block M, warps) by row bound. TP4: measured against hipBLASLt + the
# separate QK/cache kernel at every verify width (analysis/qkv_bench.py, cache-cold weights, MI355X); above
# 56 rows hipBLASLt wins. TP8 keeps its original tile up to 32 rows.
PROJECTION_TILES = {
    32: ((16, (8, 64, 16, 4)), (32, (8, 64, 32, 4)), (48, (4, 64, 64, 8)), (56, (4, 64, 64, 4))),
    16: ((32, (8, 32, 16, 4)),),
}


def projection_tile(m: int, num_q_heads: int):
    """(split-K, block N, block M, warps) of the fused projection at ``m`` rows, or None above its range."""
    for bound, tile in PROJECTION_TILES[num_q_heads]:
        if 0 < m <= bound:
            return tile
    return None


# TP4 rows the single-launch epilogue serves; at 56 rows it saves nothing.
TILE_EPILOGUE_MAX_ROWS = 48
# One ticket per output tile of a tile-epilogue launch (72 column tiles of the TP4
# projection, at most three row blocks), per device. Every launch leaves them at
# zero. The first call compiles the kernel, so it runs outside graph capture.
_TILE_TICKETS = 72 * 3
_tile_tickets: dict = {}


def _tile_epilogue_launch(m: int):
    """The projection tile at ``m`` rows, on 8 warps where block M is 32: the last
    program's 32 x 64 epilogue then takes 0.9 us less."""
    split_k, block_n, block_m, warps = projection_tile(m, 32)
    return split_k, block_n, block_m, 8 if block_m == 32 else warps


def qkv_proj_qk_norm_rope_cache(
    x,
    weight,
    positions,
    cos_sin,
    cache_k,
    cache_v,
    locations,
    mapping,
    eps,
    q_scale,
    k_scale,
    v_scale,
    *,
    num_q_heads=16,
    num_kv_heads=2,
    tile_epilogue=False,
):
    """Fused QKV + gate projection of ``x`` [M, 8192] (M <= max of ``PROJECTION_TILES``) with the QK-norm /
    RoPE / FP8-cache epilogue.

    Supports Muse TP8 (Q16/KV2) and TP4 (Q32/KV4). Returns BF16 Q, K, V
    and gate with the corresponding local head counts and head dimension 64.
    ``tile_epilogue`` (TP4, M <= ``TILE_EPILOGUE_MAX_ROWS``) runs the epilogue in
    the projection launch, with the same results.
    """
    m, k = x.shape
    hq, hk = num_q_heads, num_kv_heads
    assert (hq, hk) in ((16, 2), (32, 4))
    n = 2 * (hq + hk) * 64
    assert k == 8192 and weight.shape == (n, 8192)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous()
    assert cache_k.dtype == cache_v.dtype == torch.float8_e4m3fn
    assert cache_k.is_contiguous() and cache_v.is_contiguous()
    qo = torch.empty((m, hq * 64), device=x.device, dtype=torch.bfloat16)
    ko = torch.empty((m, hk * 64), device=x.device, dtype=torch.bfloat16)
    vo = torch.empty_like(ko)
    go = torch.empty_like(qo)
    if tile_epilogue:
        assert hq == 32 and m <= TILE_EPILOGUE_MAX_ROWS
        split_k, block_n, block_m, warps = _tile_epilogue_launch(m)
        assert block_n == 64
        grid = (n // 64, triton.cdiv(m, block_m), split_k)
        assert grid[0] * grid[1] <= _TILE_TICKETS
        tickets = _tile_tickets.get(x.device)
        if tickets is None:
            tickets = torch.zeros(_TILE_TICKETS, device=x.device, dtype=torch.int32)
            _tile_tickets[x.device] = tickets
        partials = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
        _qkv_tile_epilogue[grid](
            x,
            weight,
            partials,
            tickets,
            positions,
            cos_sin,
            locations,
            mapping,
            cache_k,
            cache_v,
            qo,
            ko,
            vo,
            go,
            k_scale,
            v_scale,
            m,
            n,
            k,
            block_m,
            128,
            split_k,
            hq,
            hk,
            float(eps),
            float(q_scale),
            cos_sin is not None,
            mapping is not None,
            cos_sin.stride(0) if cos_sin is not None else 64,
            num_warps=warps,
            num_stages=2,
            matrix_instr_nonkdim=16,
        )
        return qo, ko, vo, go
    split_k, block_n, block_m, warps = projection_tile(m, hq)
    partials = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
    _dense_partial[(n // block_n, triton.cdiv(m, block_m), split_k)](
        x,
        weight,
        partials,
        m,
        n,
        k,
        block_m,
        block_n,
        128,
        split_k,
        ".cg",
        num_warps=warps,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )
    _qkv_epilogue[(m, hq)](
        partials,
        positions,
        cos_sin,
        locations,
        mapping,
        cache_k,
        cache_v,
        qo,
        ko,
        vo,
        go,
        k_scale,
        v_scale,
        m,
        n,
        split_k,
        hq,
        hk,
        64,
        float(eps),
        float(q_scale),
        cos_sin is not None,
        mapping is not None,
        cos_sin.stride(0) if cos_sin is not None else 64,
        num_warps=1,
    )
    return qo, ko, vo, go
