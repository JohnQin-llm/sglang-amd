"""TP4 target verify at 48, 56, 64, 80 and 96 rows: the MoE pooler and unpool split
four ways across the TP ranks (sglang.kernels.ops.muse_spark.tp4_split_pool_unpool_hip),
under ``SGLANG_MUSE_TP4_SPLIT_POOL_UNPOOL``.

Each rank reads a quarter of the replicated pooler and unpool weights, 32 instead of
128 MiB a layer. The weights stay whole on every rank: prefill, eager steps and the
other widths keep the replicated projections, and the split ones read row ranges of
the same tensors. The split path runs in the TP4 target-verify CUDA graphs only.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

ENABLED = envs.SGLANG_MUSE_TP4_SPLIT_POOL_UNPOOL.get()

_split = None


def install(tp_group) -> None:
    """Register the exchange workspace on every TP rank. Collective: call once,
    eagerly, after the weights are loaded and before graph capture."""
    global _split
    if _split is not None or not ENABLED:
        return
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from sglang.kernels.ops.muse_spark.tp4_split_pool_unpool_hip import (
        Tp4SplitPoolUnpool,
    )

    ca_comm = getattr(tp_group, "ca_comm", None)
    if (
        tp_group.world_size != 4
        or not isinstance(ca_comm, CustomAllreduce)
        or ca_comm.disabled
    ):
        logger.info("Muse split pooler/unpool off: it needs AITER's TP4 all-reduce")
        return
    _split = Tp4SplitPoolUnpool(ca_comm)
    logger.info("Muse split pooler/unpool on")


def front_applies(moe, x) -> bool:
    """Whether ``moe.route(x)`` in target verify takes the split pooler."""
    if _split is None or not torch.cuda.is_current_stream_capturing():
        return False
    from sglang.kernels.ops.muse_spark.tp4_split_pool_unpool_hip import FRONT_LAUNCH
    from sglang.srt.rocm_muse_spark import router_pooler

    if x.shape[0] not in FRONT_LAUNCH:
        return False
    packed = getattr(moe, "router_pooler_packed_weight", None)
    return (
        moe.tp_size == moe.ep_size == 4
        and packed is not None
        and packed.shape == (4608, 8192)
        # The pooler is a view of the packed weight, so its router rows are current.
        and moe.pre_expert_proj.weight.data_ptr() == packed.data_ptr()
        and router_pooler.can_share_router_pooler(moe, x)
    )


def route(moe, x, num_real=None):
    """``MuseSpark1xMoE.route`` on the split pooler; requires ``front_applies``.
    ``num_real``: the graph's real-row count (device int32); later rows get expert
    id -1."""
    norm = moe.pre_expert_norm
    return _split.front(
        x,
        moe.router_pooler_packed_weight,
        moe.expert_bias,
        norm._get_effective_weight(),
        float(moe.router_multiplier),
        float(moe.route_scale),
        float(norm.eps),
        num_real,
    )


def tail_applies(moe, x, stream, boundary, pre_norm_dtype) -> bool:
    """Whether a target-verify MoE layer on ``x`` ends with ``unpool_boundary`` (the
    decoder layer then takes the MoE output before the unpool)."""
    if _split is None or boundary is None:
        return False
    if not torch.cuda.is_current_stream_capturing():
        return False
    from sglang.kernels.ops.muse_spark.tp4_split_pool_unpool_hip import TAIL_LAUNCH

    if x.shape[0] not in TAIL_LAUNCH:
        return False
    proj = moe.post_expert_proj
    return (
        moe.tp_size == moe.ep_size == 4
        and proj is not None
        and proj.bias is None
        and proj.weight.shape == (8192, 4096)
        and proj.weight.dtype == torch.bfloat16
        and proj.weight.is_contiguous()
        and stream.dtype == torch.float32
        and stream.is_contiguous()
        and stream.shape == (x.shape[0], 8192)
        and pre_norm_dtype in (None, torch.bfloat16)
        and boundary._post_norm_is_input_scale
        and boundary._post_norm.has_weight
        and boundary._use_residual_gate
        and boundary._residual_add.gate.numel() == 8192
        and boundary._pre_norm.has_weight
        and not boundary._pre_norm_is_input_scale
    )


def unpool_boundary(moe, expert_output, stream, boundary, pre_norm_dtype):
    """(next attention input, new stream) from the all-reduced expert output;
    requires ``tail_applies``."""
    boundary._ensure_cached_weights(torch.bfloat16)
    return _split.tail(
        expert_output,
        moe.post_expert_proj.weight,
        stream,
        boundary._cc_post_weight,
        boundary._cc_pre_weight,
        boundary._cc_alpha,
        boundary._cc_beta,
        boundary._post_norm.eps,
        boundary._pre_norm.eps,
        pre_norm_dtype is torch.bfloat16,
    )
