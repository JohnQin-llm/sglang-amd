"""Muse Spark TP4 target verify at 48, 56, 64, 80 and 96 rows: the MoE pooler and
unpool split four ways across the TP ranks (tp4_split_pool_unpool_hip.cuh).

Each rank reads a quarter of the replicated pooler [4096, 8192] and unpool
[8192, 4096] weights instead of all of them, and the ranks exchange the projected
slices inside the kernel that needs whole rows:

* ``front``: split-K partials of this rank's 1024 pooler rows and of the 512 rows of
  the replicated router (its high and low BF16 parts, as packed for the joint
  router/pooler), then ``front_epilogue``: the router top-8 of every row and, once
  the four pooled slices of a row are exchanged, the pre-expert RMSNorm.
* ``tail``: split-K partials of the all-reduced expert output over this rank's 2048
  unpool rows, then ``tail_boundary``: the branch slices exchanged and the FFN
  boundary on whole rows.
"""

import torch
import triton
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.gemm.bf16_splitk_gemm import _dense_partial
from sglang.kernels.ops.muse_spark.tp4_peer_workspace import register_peer_workspace

RANKS, HIDDEN, POOLED, EXPERTS = 4, 8192, 4096, 256
POOL_SLICE, BRANCH_SLICE = POOLED // RANKS, HIDDEN // RANKS
ROUTER_ROW = (
    POOLED  # first router row of the packed [pooler; router high; router low] weight
)
# Two generations of whole pooled and branch rows (128 rows each, BF16), and the flag words.
_WORKSPACE_BYTES = 2 * 128 * (POOLED + HIDDEN) * 2 + 64 * 1024

# Split-K GEMM launch (block M, block N, block K, split-K, warps) per verify width, measured in the 4-rank layer
# (GEMMs and epilogues together): 8 splits for the front and 4 for the tail were 0.4-1.5 us faster than the
# GEMM-only optimum of 16 and 8, because the epilogues read every split's partials. The 80-row tail keeps 8.
FRONT_LAUNCH = {
    **{m: (32, 32, 128, 8, 4) for m in (48, 56)},
    64: (64, 64, 256, 8, 4),
    80: (64, 64, 128, 8, 4),
    96: (64, 64, 128, 8, 8),
}
TAIL_LAUNCH = {
    **{m: (32, 32, 256, 4, 4) for m in (48, 56)},
    64: (64, 32, 256, 4, 4),
    80: (64, 64, 128, 8, 8),
    96: (64, 64, 256, 4, 4),
}


@cache_once
def _module():
    return load_jit(
        "muse_tp4_split_pool_unpool_hip",
        cuda_files=["muse_spark/tp4_split_pool_unpool_hip.cuh"],
        cuda_wrappers=[
            (
                "front_epilogue",
                "muse_tp4_split_pool_unpool::SplitPoolUnpool::front_epilogue",
            ),
            (
                "tail_boundary",
                "muse_tp4_split_pool_unpool::SplitPoolUnpool::tail_boundary",
            ),
        ],
        # no FMA contraction: the replaced kernels round every multiply and add
        extra_cuda_cflags=["-ffp-contract=off"],
    )


def _partials(x: torch.Tensor, weight: torch.Tensor, launch) -> torch.Tensor:
    """FP32 split-K partials [SK, M, N] of ``x @ weight.T``; ``weight`` may be a row
    range of a larger weight."""
    m, k = x.shape
    n = weight.shape[0]
    block_m, block_n, block_k, split_k, warps = launch
    assert weight.stride() == (k, 1) and n % block_n == 0
    assert k % (split_k * block_k) == 0
    partial = torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
    _dense_partial[(n // block_n, triton.cdiv(m, block_m), split_k)](
        x,
        weight,
        partial,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        split_k,
        ".cg",
        num_warps=warps,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )
    return partial


class Tp4SplitPoolUnpool:
    """The split pooler and unpool of one TP4 group. Construction is collective (it
    registers a workspace on every rank)."""

    def __init__(self, communicator: CustomAllreduce):
        assert communicator.world_size == RANKS and not communicator.disabled
        _module()
        self.rank = communicator.rank
        _, self.workspace_bytes, self.peers = register_peer_workspace(
            communicator, "muse_tp4_split_pool_unpool", _WORKSPACE_BYTES
        )

    def front(self, x, packed, bias, norm, multiplier, scale, eps, num_real=None):
        """(top-8 scores FP32 [M, 8], expert ids int32 [M, 8], expert input BF16
        [M, 4096]) of ``x`` [M, 8192]: ``joint_router_pooler_topk`` with this rank's
        quarter of the pooler. ``packed`` is the joint [pooler; router high;
        router low] weight [4608, 8192]; ``num_real`` (device int32) gives the rows
        from it expert id -1."""
        m = x.shape[0]
        launch = FRONT_LAUNCH[m]
        start = self.rank * POOL_SLICE
        pool_partial = _partials(x, packed[start : start + POOL_SLICE], launch)
        router_partial = _partials(x, packed[ROUTER_ROW:], launch)
        scores = torch.empty((m, 8), dtype=torch.float32, device=x.device)
        ids = torch.empty((m, 8), dtype=torch.int32, device=x.device)
        expert_input = torch.empty((m, POOLED), dtype=torch.bfloat16, device=x.device)
        _module().front_epilogue(
            self.peers,
            self.workspace_bytes,
            self.rank,
            pool_partial,
            router_partial,
            bias,
            norm,
            num_real.view(1) if num_real is not None else None,
            scores,
            ids,
            expert_input,
            multiplier,
            scale,
            eps,
        )
        return scores, ids, expert_input

    def tail(
        self,
        expert_output,
        unpool_weight,
        stream,
        post_weight,
        pre_weight,
        alpha,
        beta,
        post_eps,
        pre_eps,
        pre_bf16,
    ):
        """(next attention input BF16 [M, 8192], new stream FP32 [M, 8192]): the
        unpool of the all-reduced expert output [M, 4096] with this rank's quarter
        of ``unpool_weight`` [8192, 4096], then the FFN boundary (``pre_bf16``: the
        last layer's BF16 pre-norm input)."""
        m = expert_output.shape[0]
        start = self.rank * BRANCH_SLICE
        partial = _partials(
            expert_output,
            unpool_weight[start : start + BRANCH_SLICE],
            TAIL_LAUNCH[m],
        )
        next_input = torch.empty(
            (m, HIDDEN), dtype=torch.bfloat16, device=expert_output.device
        )
        new_stream = torch.empty_like(stream)
        _module().tail_boundary(
            self.peers,
            self.workspace_bytes,
            self.rank,
            partial,
            stream,
            alpha,
            beta,
            post_weight,
            pre_weight,
            next_input,
            new_stream,
            post_eps,
            pre_eps,
            pre_bf16,
        )
        return next_input, new_stream
