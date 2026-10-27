"""Muse Spark MoE routing front on ROCm: one kernel sequence for the FP32 router and the BF16 pooler.

At decode / verify widths (up to 64 rows) the router logits, the sigmoid top-8 routing, the pooler projection
and the pre-expert RMSNorm share one epilogue (``kernels/ops/moe/router_pooler_topk``) instead of separate
router GEMM, routing, pooler GEMM and norm launches (``SGLANG_MUSE_SHARED_ROUTER_POOLER``). Verify batches of
every multiple of 8 up to 128 rows run the router and the pooler as one GEMM over a packed weight prepared after
loading (``SGLANG_MUSE_JOINT_ROUTER_POOLER``).
"""

from __future__ import annotations

import logging

import torch

from sglang.kernels.ops.moe.router_pooler_topk import (
    JOINT_ROWS,
    joint_router_pooler_topk,
    pack_joint_router_pooler_weight,
    router_pooler_topk,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

SHARED_ROUTER_POOLER = envs.SGLANG_MUSE_SHARED_ROUTER_POOLER.get()
JOINT_ROUTER_POOLER = envs.SGLANG_MUSE_JOINT_ROUTER_POOLER.get()
_logged: set = set()


def can_share_router_pooler(moe, x: torch.Tensor) -> bool:
    """The Muse Spark TP8 routing front: 8192 hidden, 256 experts top-8 sigmoid, 4096-wide pooler + norm."""
    op = moe._routing_op()
    rows = x.shape[0]
    return (
        (0 < rows <= 64 or (JOINT_ROUTER_POOLER and rows in JOINT_ROWS))
        and x.shape[1] == 8192
        and x.is_contiguous()
        and x.dtype == torch.bfloat16
        and moe.router_in_fp32
        and moe.router.weight.dtype == torch.float32
        and moe.router.weight.shape == (8192, 256)
        and moe.router.weight.is_contiguous()
        and moe.top_k == 8
        and moe.num_experts == 256
        and moe.pre_expert_proj is not None
        and moe.pre_expert_proj.bias is None
        and moe.pre_expert_proj.weight.shape == (4096, 8192)
        and moe.pre_expert_proj.weight.dtype == torch.bfloat16
        and moe.pre_expert_proj.weight.is_contiguous()
        and moe.pre_expert_norm is not None
        and moe.pre_expert_norm.weight is not None
        and op.router_score_normalization == "sigmoid"
        and op.num_valid_tokens is None
        and op.local_expert_mapping is None
        and op.sol_topk_out is None
    )


def route_with_shared_pooler(moe, x: torch.Tensor, num_real=None):
    """(top-8 scores, expert ids, normalized expert input) for ``x``; requires ``can_share_router_pooler``.

    On the joint path, rows from ``num_real`` (device count of real rows) on get expert id -1."""
    m = x.shape[0]
    norm_weight = moe.pre_expert_norm._get_effective_weight()
    if JOINT_ROUTER_POOLER and m in JOINT_ROWS:
        packed = moe.router_pooler_packed_weight
        assert moe.pre_expert_proj.weight.data_ptr() == packed.data_ptr()
        key = ("joint", moe.layer_idx, m)
        if key not in _logged:
            _logged.add(key)
            logger.info(
                "Muse joint router/pooler active: layer=%d rows=%d", moe.layer_idx, m
            )
        return joint_router_pooler_topk(
            x,
            packed,
            moe.expert_bias,
            norm_weight,
            float(moe.router_multiplier),
            float(moe.route_scale),
            float(moe.pre_expert_norm.eps),
            num_real=num_real,
        )
    key = "shared" if m <= 16 else "shared_completed"
    if key not in _logged:
        _logged.add(key)
        logger.info(
            "Muse fast path: %s, rows=%d",
            (
                "shared router/preproject epilogue"
                if m <= 16
                else "shared router/norm with hipBLASLt pooler"
            ),
            m,
        )
    return router_pooler_topk(
        x,
        moe.router.weight,
        moe.pre_expert_proj.weight,
        moe.expert_bias,
        norm_weight,
        float(moe.router_multiplier),
        float(moe.route_scale),
        float(moe.pre_expert_norm.eps),
        pooler_tiled=getattr(moe.pre_expert_proj, "small_m_tiled_weight", None),
    )


def _create_hipblaslt_handle() -> None:
    """Create aiter's hipBLASLt handle for the joint GEMM's hipBLASLt widths before graph capture, once per
    process: aiter's tuned GEMM checks the same flag."""
    from aiter import hipb_create_extension, tuned_gemm

    if not tuned_gemm.extensions_created:
        hipb_create_extension()
        tuned_gemm.extensions_created = True


@torch.no_grad()
def prepare_joint_router_pooler(moe) -> None:
    """Pack ``[pooler; router high; router low]`` once after loading; the pooler weight aliases the prefix."""
    router, pooler = moe.router.weight, moe.pre_expert_proj.weight
    assert router.is_cuda and pooler.is_cuda
    assert router.shape == (8192, 256) and router.dtype == torch.float32
    assert pooler.shape == (4096, 8192) and pooler.dtype == torch.bfloat16
    packed = getattr(moe, "router_pooler_packed_weight", None)
    _create_hipblaslt_handle()
    if packed is None:
        # Runs in post_load_weights, before warmup or graph capture. The pooler prefix holds exactly the
        # original BF16 values and stays available to the plain nn.Linear path at larger M.
        packed = pack_joint_router_pooler_weight(router, pooler)
        pooler.data = packed[:4096]
        moe.register_buffer("router_pooler_packed_weight", packed, persistent=False)
    else:
        # Later checkpoint writes already land in the aliased prefix; refresh only the router parts,
        # without moving any captured addresses.
        assert pooler.data_ptr() == packed.data_ptr()
        high = router.T.to(torch.bfloat16)
        packed[4096:4352].copy_(high)
        packed[4352:].copy_((router.T - high.float()).to(torch.bfloat16))
    assert pooler.is_contiguous() and pooler.data_ptr() == packed.data_ptr()


def can_route_tp4_high_m(moe, x: torch.Tensor) -> bool:
    """Metadata-only eligibility; caller also gates opt-in, target verify and no SP.

    Keep the tuned router schedule and the original dense pooler. In particular,
    rows72/88/104/120, TP8, all old small-M paths, and unsupported routing modes
    retain their original dispatch.
    """
    from sglang.kernels.ops.gemm.router_gemm_fp32 import get_router_gemm_fp32_configs
    from sglang.kernels.ops.moe.router_pooler_topk import (
        TP4_HIGH_M_ROUTER_CONFIG,
        TP4_HIGH_M_ROWS,
    )

    if (
        moe.tp_size != 4
        or moe.ep_size != 4
        or x.dim() != 2
        or x.shape[0] not in TP4_HIGH_M_ROWS
        or x.shape[1] != 8192
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or not moe.router_in_fp32
        or moe.top_k != 8
        or moe.num_experts != 256
        or moe.route_eps != 1e-15
        or moe.pre_expert_proj is None
        or moe.pre_expert_proj.bias is not None
        or moe.pre_expert_norm is None
        or moe.pre_expert_norm.weight is None
    ):
        return False
    # Eligibility must not materialize weight+1 on the serving hot path.
    norm = getattr(moe.pre_expert_norm, "_effective_weight_cache", None)
    if norm is None:
        return False
    for tensor, shape, dtype in (
        (moe.router.weight, (8192, 256), torch.float32),
        (moe.pre_expert_proj.weight, (4096, 8192), torch.bfloat16),
        (moe.expert_bias, (256,), torch.float32),
        (norm, (4096,), torch.bfloat16),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            return False
    op = moe._routing_op()
    if not (
        op.router_score_normalization == "sigmoid"
        and op.expert_bias is moe.expert_bias
        and op.router_output_multiplier == moe.router_multiplier
        and op.route_scale == moe.route_scale
        and op.router_score_division_eps == moe.route_eps
        and op.num_valid_tokens is None
        and op.local_expert_mapping is None
        and op.sol_topk_out is None
    ):
        return False
    return (get_router_gemm_fp32_configs(256, 8192) or {}).get(
        x.shape[0]
    ) == TP4_HIGH_M_ROUTER_CONFIG


def route_with_tp4_high_m(moe, x: torch.Tensor, *, pooler_fn):
    """Requires can_route_tp4_high_m; pooler_fn is model._dense_linear unchanged."""
    from sglang.kernels.ops.moe.router_pooler_topk import router_pooler_topk_high_m

    return router_pooler_topk_high_m(
        x,
        moe.router.weight,
        moe.expert_bias,
        moe.pre_expert_norm._get_effective_weight(),
        float(moe.router_multiplier),
        float(moe.route_scale),
        float(moe.pre_expert_norm.eps),
        pooler=lambda value: pooler_fn(value, moe.pre_expert_proj, max_m=128),
    )
