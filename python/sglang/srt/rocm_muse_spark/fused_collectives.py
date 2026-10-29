"""TP4 target verify: the MoE finalize and the attention boundary fused with their
all-reduces (sglang.kernels.ops.muse_spark.tp4_fused_all_reduce_hip).

``SGLANG_MUSE_TP4_FUSED_FINALIZE_AR`` fuses the compact-A4 MoE finalize with the
all-reduce of its partial sums, ``SGLANG_MUSE_TP4_FUSED_AR_BOUNDARY`` the attention
all-reduce with the attention boundary. Both run in the TP4 target-verify CUDA
graphs (8..128 rows); eager calls keep the separate kernels.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.rocm_muse_spark import mxfp4_moe

logger = logging.getLogger(__name__)

FINALIZE_ALL_REDUCE = envs.SGLANG_MUSE_TP4_FUSED_FINALIZE_AR.get()
ALL_REDUCE_BOUNDARY = envs.SGLANG_MUSE_TP4_FUSED_AR_BOUNDARY.get()

_fused = None


def install(tp_group) -> None:
    """Register the fused kernels' workspace on every TP rank. Collective: call once,
    eagerly, after the weights are loaded and before graph capture."""
    global _fused
    if _fused is not None or not (FINALIZE_ALL_REDUCE or ALL_REDUCE_BOUNDARY):
        return
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from sglang.kernels.ops.muse_spark.tp4_fused_all_reduce_hip import (
        Tp4FusedAllReduce,
    )

    ca_comm = getattr(tp_group, "ca_comm", None)
    if (
        tp_group.world_size != 4
        or not isinstance(ca_comm, CustomAllreduce)
        or ca_comm.disabled
        or not ca_comm.enable_register_for_capturing
    ):
        logger.info("Muse fused TP4 all-reduces off: they need AITER's TP4 all-reduce")
        return
    _fused = Tp4FusedAllReduce(ca_comm)
    logger.info(
        "Muse fused TP4 all-reduces: finalize + all-reduce %s, "
        "all-reduce + attention boundary %s",
        FINALIZE_ALL_REDUCE,
        ALL_REDUCE_BOUNDARY,
    )


def _applies(rows: int) -> bool:
    return (
        _fused is not None
        and 0 < rows <= 128
        and rows % 8 == 0
        and torch.cuda.is_current_stream_capturing()
    )


def finalize_all_reduce(moe, x, scores, ids) -> torch.Tensor | None:
    """The all-reduced compact-A4 MoE output of ``x`` (before the unpool), or None to
    keep the separate finalize and all-reduce. The caller has checked that the
    compact-A4 experts serve these rows."""
    experts, norm = moe.experts, moe.post_expert_norm
    if not (
        FINALIZE_ALL_REDUCE
        and _applies(ids.shape[0])
        and norm is not None
        and norm.weight is not None
        and norm.weight.dtype == torch.bfloat16
    ):
        return None
    y = mxfp4_moe.compact_a4_experts(
        x,
        ids,
        experts.w13_weight,
        experts.w2_weight,
        experts.w13_weight_scale,
        experts.w2_weight_scale,
        local_start=moe.local_expert_start,
        single_layout=mxfp4_moe.single_layout_enabled(experts),
    )
    gain = float(norm.gain_center) if norm.zero_centered_gamma else 0.0
    return _fused.finalize_all_reduce(
        y,
        norm.weight,
        scores,
        ids,
        moe.local_expert_start,
        experts.w13_weight.shape[0],
        norm.eps,
        gain,
    )


def fuses_attention_boundary(boundary, attn_input, stream) -> bool:
    """Whether this layer's attention all-reduce and boundary run fused; the o_proj
    then skips its all-reduce."""
    return (
        ALL_REDUCE_BOUNDARY
        and _applies(attn_input.shape[0])
        and stream.dtype == torch.float32
        and stream.is_contiguous()
        and stream.shape == (attn_input.shape[0], 8192)
        and not boundary._post_norm_is_input_scale
        and not boundary._post_norm.has_weight
        and boundary._use_residual_gate
        and boundary._residual_add.gate.numel() == 8192
        and boundary._pre_norm.has_weight
        and not boundary._pre_norm_is_input_scale
    )


def all_reduce_attention_boundary(boundary, partial, stream):
    """(FFN input, new stream) from the o_proj partials; requires
    ``fuses_attention_boundary``."""
    boundary._ensure_cached_weights(torch.bfloat16)
    return _fused.all_reduce_boundary(
        partial,
        stream,
        boundary._cc_alpha,
        boundary._cc_beta,
        boundary._cc_pre_weight,
        boundary._post_norm.eps,
        boundary._pre_norm.eps,
    )
