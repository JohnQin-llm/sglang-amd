# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MuseSpark post-expert finalize for ROCm.

One launch per chunk replaces: Triton post-expert RMSNorm -> route-weight
multiply -> sum over top-k -> copy. Each program owns one token and normalizes
its ``[top_k, H]`` routes as one tile (weight-first: ``rms_norm(x * (w + gc))``
or plain ``rms_norm(x) * (w + gc)``), summing the weighted routes in fp32.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _post_expert_finalize_kernel(
    x_ptr,
    w_ptr,
    tw_ptr,
    out_ptr,
    K,
    H,
    eps,
    gain_center,
    stride_xt,
    stride_xk,
    stride_twt,
    stride_out,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
):
    t = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    offs = tl.arange(0, BLOCK_H)
    k_mask = k < K
    h_mask = offs < H
    tw = tl.load(tw_ptr + t * stride_twt + k, mask=k_mask, other=0.0).to(tl.float32)
    x = tl.load(
        x_ptr + t * stride_xt + k[:, None] * stride_xk + offs[None, :],
        mask=k_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=h_mask, other=0.0).to(tl.float32) + gain_center
        if WEIGHT_FIRST:
            x = x * w[None, :]
    scale = tl.rsqrt(tl.sum(x * x, axis=1) / H + eps) * tw
    acc = tl.sum(x * scale[:, None], axis=0)
    if HAS_WEIGHT and not WEIGHT_FIRST:
        acc = acc * w
    tl.store(
        out_ptr + t * stride_out + offs, acc.to(out_ptr.dtype.element_ty), mask=h_mask
    )


def fused_post_expert_finalize(
    routed: torch.Tensor,
    weight: torch.Tensor | None,
    topk_weights: torch.Tensor,
    eps: float,
    gain_center: float,
    weight_first: bool,
    out: torch.Tensor,
) -> torch.Tensor:
    """``out[t] = sum_k tw[t, k] * norm(routed[t, k])`` written in ``out.dtype``.

    Args:
        routed: ``[T, K, H]`` per-route expert outputs (TP-reduced), any float dtype.
        weight: ``[H]`` norm weight or ``None``.
        topk_weights: ``[T, K]`` router weights.
        eps: Norm epsilon.
        gain_center: Added to ``weight`` (0.0 when unused).
        weight_first: ``rms_norm(x * w)`` when True, else ``rms_norm(x) * w``.
        out: ``[T, H]`` destination.
    """
    T, K, H = routed.shape
    assert routed.stride(2) == 1 and out.stride(1) == 1 and topk_weights.stride(1) == 1
    assert out.shape == (T, H) and topk_weights.shape == (T, K)
    if T == 0:
        return out
    block_k = triton.next_power_of_2(K)
    block_h = triton.next_power_of_2(H)
    assert block_k * block_h <= 65536, "routes too wide for the single-tile finalize"
    _post_expert_finalize_kernel[(T,)](
        routed,
        weight if weight is not None else routed,
        topk_weights,
        out,
        K,
        H,
        eps,
        gain_center,
        routed.stride(0),
        routed.stride(1),
        topk_weights.stride(0),
        out.stride(0),
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        HAS_WEIGHT=weight is not None,
        WEIGHT_FIRST=weight_first,
        num_warps=8 if block_k * block_h >= 4096 else 4,
    )
    return out
