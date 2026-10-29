"""Muse Spark TP4 target verify on AITER's custom all-reduce: the MoE finalize fused
with the all-reduce of its partial sums, and the attention all-reduce fused with
the attention boundary (tp4_fused_all_reduce_hip.cuh)."""

import torch
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from aiter.jit.core import AITER_CSRC_DIR

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.muse_spark.tp4_peer_workspace import register_peer_workspace

MOE_HIDDEN, HIDDEN, TOP_K, MAX_ROWS = 4096, 8192, 8, 128
# The finalized MoE partial rows (MAX_ROWS x MOE_HIDDEN BF16) and the flag words.
_WORKSPACE_BYTES = MAX_ROWS * MOE_HIDDEN * 2 + 64 * 1024


@cache_once
def _module():
    return load_jit(
        "muse_tp4_fused_all_reduce_hip",
        cuda_files=["muse_spark/tp4_fused_all_reduce_hip.cuh"],
        cuda_wrappers=[
            (
                "finalize_all_reduce",
                "muse_tp4_fused_all_reduce::FusedAllReduce::finalize_all_reduce",
            ),
            (
                "all_reduce_boundary",
                "muse_tp4_fused_all_reduce::FusedAllReduce::all_reduce_boundary",
            ),
        ],
        # AITER's CustomAllreduce, whose peer buffers, signals and tmp buffers the kernels run on
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        # no FMA contraction: the separate kernels round every multiply and add
        extra_cuda_cflags=["-ffp-contract=off"],
    )


class Tp4FusedAllReduce:
    """The fused all-reduces of one TP4 group. Construction is collective (it
    registers a workspace on every rank); the calls are for CUDA-graph capture,
    where AITER registers the attention input's peer addresses when the capture
    scope exits."""

    def __init__(self, communicator: CustomAllreduce):
        assert communicator.world_size == 4 and not communicator.disabled
        _module()
        self.communicator = communicator
        self.workspace, self.workspace_bytes, self.peers = register_peer_workspace(
            communicator, "muse_tp4_fused_all_reduce", _WORKSPACE_BYTES
        )

    def _check_capture(self) -> None:
        comm = self.communicator
        assert torch.cuda.is_current_stream_capturing() and comm._IS_CAPTURING
        assert comm.enable_register_for_capturing

    def finalize_all_reduce(
        self,
        y: torch.Tensor,
        weight: torch.Tensor,
        scores: torch.Tensor,
        ids: torch.Tensor,
        local_start: int,
        num_local: int,
        eps: float,
        gain: float,
    ) -> torch.Tensor:
        """All-reduced MoE output [M, 4096] from this rank's per-slot expert rows
        ``y`` [8 M, 4096]: per-slot RMSNorm (weight + gain, input scale), the
        route-weighted sum of the slots routed to experts local_start ..
        local_start + num_local - 1, then the sum over the ranks."""
        self._check_capture()
        out = torch.empty(
            (ids.shape[0], MOE_HIDDEN), dtype=torch.bfloat16, device=y.device
        )
        _module().finalize_all_reduce(
            self.communicator._ptr,
            self.workspace,
            self.workspace_bytes,
            self.peers,
            y,
            weight,
            scores,
            ids,
            out,
            local_start,
            num_local,
            eps,
            gain,
        )
        return out

    def all_reduce_boundary(
        self,
        partial: torch.Tensor,
        stream: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        pre_weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(FFN input, new stream) from this rank's o_proj partials [M, 8192]: the
        sum over the ranks, RMSNorm without weight, gated residual
        alpha * stream + beta * normed, then the weighted pre-FFN RMSNorm."""
        self._check_capture()
        ffn_input = torch.empty_like(partial)
        new_stream = torch.empty_like(stream)
        _module().all_reduce_boundary(
            self.communicator._ptr,
            self.workspace_bytes,
            self.peers,
            partial,
            stream,
            alpha,
            beta,
            pre_weight,
            ffn_input,
            new_stream,
            post_eps,
            pre_eps,
        )
        return ffn_input, new_stream
