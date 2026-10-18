"""Muse BF16 head norm + interleaved RoPE + FP8 cache write.

The caller owns eligibility and paged-cache layout. No quantization scale or
intermediate BF16 rounding is removed by this fusion.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _apply_rope(X, COS, POS, D: tl.constexpr):
    offsets = tl.arange(0, D)
    # HIP's interleaved RoPE rounds the cache and each product to BF16.
    cosine = tl.load(COS + POS * D + offsets // 2).to(tl.bfloat16).to(tl.float32)
    sine = tl.load(COS + POS * D + D // 2 + offsets // 2).to(tl.bfloat16).to(tl.float32)
    partner = tl.gather(X, tl.broadcast_to((offsets ^ 1)[None, :], X.shape), axis=1).to(
        tl.float32
    )
    cosine_product = (X.to(tl.float32) * cosine[None, :]).to(tl.bfloat16).to(tl.float32)
    sine_product = (partner * sine[None, :]).to(tl.bfloat16).to(tl.float32)
    return (
        cosine_product
        + tl.where(offsets[None, :] % 2 == 0, -sine_product, sine_product)
    ).to(tl.bfloat16)


@triton.jit
def _prepare(
    Q,
    K,
    V,
    QO,
    KO,
    POS,
    COS,
    KC,
    VC,
    LOC,
    SWA,
    KS,
    VS,
    QS: tl.constexpr,
    KSTR: tl.constexpr,
    VSTR: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    EPS,
    SCALE,
    ROPE: tl.constexpr,
    HAS_SWA: tl.constexpr,
    SCALED: tl.constexpr,
):
    token = tl.program_id(0)
    q_offsets = tl.arange(0, HQ * D)
    k_offsets = tl.arange(0, HK * D)
    # Keep the existing QK norm's token-wide CTA and reduction layout. A CTA
    # per head changes FP32 summation and can cross a BF16 rounding midpoint.
    q = tl.load(Q + token * QS + q_offsets).to(tl.float32).reshape(HQ, D)
    k = tl.load(K + token * KSTR + k_offsets).to(tl.float32).reshape(HK, D)
    q_rstd = tl.rsqrt(tl.sum(q * q, axis=1) / D + EPS)
    k_rstd = tl.rsqrt(tl.sum(k * k, axis=1) / D + EPS)
    q = ((q * q_rstd[:, None]).to(tl.bfloat16).to(tl.float32) * SCALE).to(tl.bfloat16)
    k = (k * k_rstd[:, None]).to(tl.bfloat16)
    if ROPE:
        position = tl.load(POS + token)
        q = _apply_rope(q, COS, position, D)
        k = _apply_rope(k, COS, position, D)
    tl.store(QO + token * HQ * D + q_offsets, q.reshape(HQ * D))
    tl.store(KO + token * HK * D + k_offsets, k.reshape(HK * D))
    slot = tl.load(LOC + token).to(tl.int64)
    if HAS_SWA:
        slot = tl.load(SWA + slot, mask=slot >= 0, other=-1).to(tl.int64)
    if slot >= 0:
        cache_k = k.reshape(HK * D).to(tl.float32)
        cache_v = tl.load(V + token * VSTR + k_offsets).to(tl.float32)
        if SCALED:
            cache_k = cache_k / tl.load(KS)
            cache_v = cache_v / tl.load(VS)
        tl.store(KC + slot * HK * D + k_offsets, cache_k)
        tl.store(VC + slot * HK * D + k_offsets, cache_v)


def fused_qkv_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor | None,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
    *,
    eps: float,
    q_scale: float,
    swa_mapping: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal launch; requires contiguous token-major cache and D=64."""
    qo, ko = (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
    )
    if q.shape[0]:
        _prepare[(q.shape[0],)](
            q,
            k,
            v,
            qo,
            ko,
            positions,
            cos_sin,
            key_cache,
            value_cache,
            slots,
            swa_mapping,
            k_scale,
            v_scale,
            QS=q.stride(0),
            KSTR=k.stride(0),
            VSTR=v.stride(0),
            HQ=q.shape[1] // 64,
            HK=k.shape[1] // 64,
            D=64,
            EPS=eps,
            SCALE=q_scale,
            ROPE=cos_sin is not None,
            HAS_SWA=swa_mapping is not None,
            SCALED=k_scale is not None,
            num_warps=4,
            enable_fp_fusion=True,
        )
    return qo, ko
