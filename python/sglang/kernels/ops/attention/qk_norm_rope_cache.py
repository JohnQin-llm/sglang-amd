"""Per-head QK RMSNorm + interleaved RoPE + paged FP8 KV-cache store (Muse Spark, head_dim 64, page 16).

* ``qk_norm_rope_cache``: from projected Q/K/V. One program per (token, Q head) normalizes, scales and rotates
  Q; the first Q head of each GQA group also normalizes and rotates its K and stores K/V into the FP8 cache
  (optionally through the full -> SWA slot mapping).
* ``qkv_proj_qk_norm_rope_cache``: the same epilogue fed straight from the BF16 split-K partials of the fused
  QKV + output-gate projection (``bf16_splitk_gemm``), which also emits V and the gate.

The BF16 round trips of the unfused sequence (norm output, query scaling, RoPE products, projection store)
are explicit, so the results match the separate norm / RoPE / cache-write kernels.
"""

import logging
import os

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.bf16_splitk_gemm import _dense_partial

logger = logging.getLogger(__name__)

# TP4 projection at split-K 8 (<= 16 rows): FP32 partials from AITER's gfx1250 gluon TDM batched GEMM (falls back
# to the Triton partials at num_stages 4 when the gluon kernel is unavailable). Same K slices, same bytes.
QKV_TDM = os.environ.get("SGLANG_MUSE_QKV_TDM", "0") == "1"
# Epilogue grid (m, hq + hk): K / V / cache rows get their own programs instead of riding on Q heads.
QKV_SPLIT_EPI = os.environ.get("SGLANG_MUSE_QKV_SPLIT_EPI", "0") == "1"
_TDM_CFG = (32, 128, 3, 2)  # block N, block K, buffers, warps
_tdm_kernel = None  # (kernel, layouts) once built, False when the gluon kernel is unavailable
_live = set()


def _note_live(name: str) -> None:
    if name not in _live:
        _live.add(name)
        logger.info("Muse QKV projection: %s live", name)


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
def _qkv_epilogue_split(
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
    """``_qkv_epilogue`` on grid (M, HQ + HK), same per-element math: programs head < HQ do gate + Q, programs
    HQ + kh do K, V and the cache write of KV head kh."""
    token, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    if head < HQ:
        gate = _project_slice(P, token, (HQ + 2 * HK) * D + head * D, M, N, SK, D)
        tl.store(GO + token * HQ * D + head * D + d, gate)
        q = _project_slice(P, token, head * D, M, N, SK, D)
        q = _norm_rope(q, COS, POS, EPS, SCALE, D, ROPE, True, COS_STRIDE)
        tl.store(QO + token * HQ * D + head * D + d, q)
    else:
        kh = head - HQ
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


def _gluon_tdm_kernel():
    """AITER's gfx1250 gluon TDM batched GEMM and its layouts for ``_TDM_CFG``, or None if unavailable."""
    global _tdm_kernel
    if _tdm_kernel is None:
        try:
            from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
                create_shared_layouts,
                create_wmma_layouts,
            )
            from aiter.ops.triton._gluon_kernels.gfx1250.gemm.batched.batched_gemm_bf16 import (
                _KERNEL_MAP,
            )

            block_n, block_k, _, warps = _TDM_CFG
            _tdm_kernel = (
                _KERNEL_MAP["bandwidth_bound"],
                create_wmma_layouts(warps),
                create_shared_layouts(16, block_n, block_k, "TN", 16),
            )
        except Exception as ex:  # noqa: BLE001
            logger.warning("Muse QKV projection: gluon TDM GEMM unavailable (%s); Triton partials", ex)
            _tdm_kernel = False
    return _tdm_kernel or None


def _tdm_partials(x, weight, partials, m, n, k, split_k) -> bool:
    built = _gluon_tdm_kernel()
    if built is None:
        return False
    kern, (wmma, opa, opb), (sha, shb) = built
    block_n, block_k, buffers, warps = _TDM_CFG
    xq, wq = x.view(1, m, k), weight.view(1, n, k).transpose(1, 2)
    pv = partials.view(split_k, 1, m, n)
    split = k // split_k
    kern[(1, split_k * triton.cdiv(m, 16) * (n // block_n))](
        xq, wq, pv, None, m, n, k,
        xq.stride(0), xq.stride(1), xq.stride(2), wq.stride(0), wq.stride(1), wq.stride(2),
        pv.stride(1), pv.stride(2), pv.stride(3), pv.stride(0), 0,
        BLOCK_M=16, BLOCK_N=block_n, BLOCK_K=block_k, NUM_BUFFERS=min(buffers, split // block_k), LAYOUT="TN",
        SHARED_LAYOUT_A=sha, SHARED_LAYOUT_B=shb, WMMA_LAYOUT=wmma, OPERAND_LAYOUT_A=opa,
        OPERAND_LAYOUT_B=opb, ADD_BIAS=False, NUM_KSPLIT=split_k, SPLITK_BLOCK_SIZE=split, num_warps=warps,
        waves_per_eu=0, cache_modifier=".cg",
    )
    _note_live("gluon TDM split-K partials (BN 32, BK 128, split-K 8, 3 buffers, 2 warps)")
    return True


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
):
    """Fused QKV + gate projection of ``x`` [M, 8192] (M <= max of ``PROJECTION_TILES``) with the QK-norm /
    RoPE / FP8-cache epilogue.

    Supports Muse TP8 (Q16/KV2) and TP4 (Q32/KV4). Returns BF16 Q, K, V
    and gate with the corresponding local head counts and head dimension 64.
    """
    m, k = x.shape
    hq, hk = num_q_heads, num_kv_heads
    assert (hq, hk) in ((16, 2), (32, 4))
    n = 2 * (hq + hk) * 64
    split_k, block_n, block_m, warps = projection_tile(m, hq)
    assert k == 8192 and weight.shape == (n, 8192)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous()
    assert cache_k.dtype == cache_v.dtype == torch.float8_e4m3fn
    assert cache_k.is_contiguous() and cache_v.is_contiguous()
    partials = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
    tdm = QKV_TDM and hq == 32 and split_k == 8 and block_m == 16
    if not (tdm and _tdm_partials(x, weight, partials, m, n, k, split_k)):
        if tdm:
            _note_live("Triton split-K partials at num_stages 4")
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
            num_stages=4 if tdm else 2,
            matrix_instr_nonkdim=16,
        )
    qo = torch.empty((m, hq * 64), device=x.device, dtype=torch.bfloat16)
    ko = torch.empty((m, hk * 64), device=x.device, dtype=torch.bfloat16)
    vo = torch.empty_like(ko)
    go = torch.empty_like(qo)
    if QKV_SPLIT_EPI:
        _note_live("split epilogue (grid m x (hq + hk))")
        epilogue, grid = _qkv_epilogue_split, (m, hq + hk)
    else:
        epilogue, grid = _qkv_epilogue, (m, hq)
    epilogue[grid](
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
