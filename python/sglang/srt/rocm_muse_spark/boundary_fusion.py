"""Muse Spark TP4 target verify: the two per-layer all-reduces with their neighbouring kernel in one launch.

* ``finalize_all_reduce``: MoE finalize (per-slot RMSNorm with input scale, route-weighted sum of this rank's
  slots) and the all-reduce of the [M, 4096] partial sums. Bitwise equal to FlyDSL finalize + AITER all-reduce.
* ``all_reduce_boundary``: all-reduce of the attention o_proj partials [M, 8192] and the attention boundary
  (post-norm, gated residual, pre-FFN norm). The all-reduce is bitwise equal to AITER's; the boundary keeps the
  Triton kernel's formula with its two row sums in a different order.

Both kernels (``csrc/muse_ar_fusion.cu``) run on the TP group's AITER custom all-reduce state: its signals, tmp
buffers and IPC buffer registration, including the registration of CUDA-graph-captured inputs. They are used only
while a CUDA graph is being captured (the TP4 target-verify graphs, 8..128 rows in steps of 8); eager calls and
every other shape keep the separate kernels.

Switches: ``SGLANG_MUSE_TP4_FUSED_FINALIZE_AR=1`` (MoE) and ``SGLANG_MUSE_TP4_FUSED_AR_BOUNDARY=1`` (attention).
The model calls ``install`` once after loading (collective: registers buffers on every rank), then
``finalize_all_reduce`` / ``fuses_attention_boundary`` + ``all_reduce_attention_boundary`` per layer.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import torch

from sglang.srt.rocm_muse_spark.jit_build import stage_source

logger = logging.getLogger(__name__)

_SOURCE = Path(__file__).resolve().parent / "csrc" / "muse_ar_fusion.cu"
_SOURCE_SHA256 = "c7b456113dd3744295d72fe7c253198fbad5740adb083128fb6d5e595313efd1"
_MAX_ROWS = 128
_RANKS = 4
# AITER's two-stage threshold for 4 ranks: inputs of at least 160 KiB reduce-scatter, smaller ones read all ranks.
_TWO_STAGE_BYTES = 160 * 1024

FINALIZE_AR = os.environ.get("SGLANG_MUSE_TP4_FUSED_FINALIZE_AR", "0") == "1"
AR_BOUNDARY = os.environ.get("SGLANG_MUSE_TP4_FUSED_AR_BOUNDARY", "0") == "1"

_ext = None
_collectives = None


def _extension():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        source = _SOURCE.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if _SOURCE_SHA256 is not None and digest != _SOURCE_SHA256:
            raise ImportError(f"{_SOURCE} has sha256 {digest}, expected {_SOURCE_SHA256}")
        import aiter

        include = Path(aiter.__file__).resolve().parents[1] / "csrc" / "include"
        build, staged = stage_source(_SOURCE, source, digest, "MUSE_AR_FUSION_BUILD", "muse_ar_fusion")
        arch = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"  # the kernels are written for MI355X only
        try:
            _ext = load(
                name=f"muse_ar_fusion_{digest[:16]}",
                sources=[str(staged)],
                extra_include_paths=[str(include)],
                extra_cflags=["-O3", "-std=c++20"],
                extra_cuda_cflags=[
                    "-O3",
                    "-std=c++20",
                    "-ffp-contract=off",
                    "-DUSE_ROCM=1",
                    "-D__HIP_PLATFORM_AMD__=1",
                    "--offload-arch=gfx950",
                    "-Wno-unused-result",
                ],
                build_directory=str(build),
                verbose=False,
            )
        finally:
            if arch is None:
                os.environ.pop("PYTORCH_ROCM_ARCH")
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = arch
    return _ext


def _two_stage(rows: int, hidden: int) -> bool:
    return rows * hidden * 2 >= _TWO_STAGE_BYTES


class FusedCollectives:
    """Per TP group: the IPC-registered finalize partial buffer, the row flags and the two fused launches."""

    def __init__(self, ca_comm, device: torch.device, max_rows: int = _MAX_ROWS):
        assert ca_comm is not None and not ca_comm.disabled and ca_comm.world_size == _RANKS
        ext = _extension()
        self.ca_comm = ca_comm
        self.fa = ca_comm._ptr
        self.max_rows = max_rows
        pool = ca_comm._pool
        # Collective (eager, before any graph capture). Both buffers are uncached allocations of their own, like
        # AITER's signals: peers read them right after a relaxed barrier, and AITER exchanges a buffer's IPC handle
        # with offset 0. The row flags start at zero.
        partial_key, flags_key = "muse_finalize_partial", "muse_row_flags"
        if partial_key not in pool:
            pool.create(partial_key, max(max_rows * 4096 * 2, 2 * 1024 * 1024), uncached=True)
            pool.create(flags_key, 2 * 1024 * 1024, uncached=True)
        self.partial_ptr = pool[partial_key].data_ptr
        handles, offsets = pool.get_ipc_meta(partial_key)
        ca_comm._ops_register_input_buffer(self.fa, self.partial_ptr, [h.data_ptr() for h in handles], offsets)
        handles, offsets = pool.get_ipc_meta(flags_key)
        self.row_flags = ext.open_peer_buffers(pool[flags_key].data_ptr, list(handles), list(offsets),
                                               ca_comm.rank)
        # For diagnosing a stall from outside the process (diag/dump_flags.py): this rank's flag-buffer IPC handle.
        handle_dir = os.environ.get("MUSE_AR_FUSION_HANDLE_DIR")
        if handle_dir:
            path = Path(handle_dir) / f"flags-pid{os.getpid()}-rank{ca_comm.rank}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(handles[ca_comm.rank].numpy()) + int(offsets[ca_comm.rank]).to_bytes(8, "little"))

    def applies(self, rows: int) -> bool:
        return 0 < rows <= self.max_rows and rows % (2 * _RANKS) == 0

    def finalize_all_reduce(self, y, gamma, scores, ids, local_start, num_local, eps, gain, blocks=None,
                            two_stage=None):
        rows = ids.shape[0]
        two_stage = _two_stage(rows, 4096) if two_stage is None else two_stage
        out = torch.empty((rows, 4096), dtype=torch.bfloat16, device=y.device)
        blocks = blocks or min(80, rows // 2)
        _extension().finalize_all_reduce(
            self.fa, y, gamma, scores, ids, self.partial_ptr, self.max_rows, out, int(local_start), int(num_local),
            float(eps), float(gain), int(blocks), bool(two_stage), self.row_flags)
        return out

    def all_reduce_boundary(self, partial, stream, alpha, beta, pre_weight, post_eps, pre_eps, blocks=None,
                            two_stage=None):
        rows = partial.shape[0]
        two_stage = _two_stage(rows, 8192) if two_stage is None else two_stage
        ffn_input = torch.empty((rows, 8192), dtype=torch.bfloat16, device=partial.device)
        new_stream = torch.empty((rows, 8192), dtype=torch.float32, device=partial.device)
        blocks = blocks or min(80, rows)
        _extension().all_reduce_boundary(
            self.fa, partial, stream, alpha, beta, pre_weight, ffn_input, new_stream, float(post_eps),
            float(pre_eps), int(blocks), bool(two_stage), self.row_flags)
        return ffn_input, new_stream


# Launch per verify width, measured in 4-rank CUDA graphs (bench_collectives.py, results/collectives-final.json):
# one-stage below AITER's 160 KiB two-stage threshold, else two-stage with the fastest grid / rank grouping.
def _moe_launch(rows: int) -> dict:
    if not _two_stage(rows, 4096):
        return dict(two_stage=False, blocks=rows // 2)
    return dict(two_stage=True, blocks=rows)


def _attention_launch(rows: int) -> dict:
    if not _two_stage(rows, 8192):
        return dict(two_stage=False, blocks=rows)
    return dict(two_stage=True, blocks=2 * rows if rows <= 48 else rows)


def install(tp_group, device: torch.device) -> None:
    """Register the fused collectives' buffers on every TP rank; call once, eagerly, before graph capture."""
    global _collectives
    if _collectives is not None or not (FINALIZE_AR or AR_BOUNDARY):
        return
    ca_comm = getattr(tp_group, "ca_comm", None)
    if tp_group.world_size != _RANKS or ca_comm is None or ca_comm.disabled:
        logger.info("Muse fused collectives off: need TP4 with AITER custom all-reduce")
        return
    _collectives = FusedCollectives(ca_comm, device)
    logger.info("Muse fused collectives on: finalize+all-reduce=%s, all-reduce+boundary=%s", FINALIZE_AR,
                AR_BOUNDARY)


def _active(rows: int) -> bool:
    return _collectives is not None and _collectives.applies(rows) and torch.cuda.is_current_stream_capturing()


def finalize_all_reduce(moe, x, scores, ids):
    """All-reduced compact-A4 MoE output of ``x`` (before the unpooler), or None to take the separate kernels.

    Same expert call as ``mxfp4_moe.run_compact_a4``; the caller has checked that compact A4 applies.
    """
    from sglang.srt.rocm_muse_spark import mxfp4_moe

    rows = ids.shape[0]
    experts, norm = moe.experts, moe.post_expert_norm
    if not (
        FINALIZE_AR
        and _active(rows)
        and norm is not None
        and norm.weight is not None
        and norm.weight.dtype == torch.bfloat16
        and scores.dtype == torch.float32
        and ids.dtype == torch.int32
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
    return _collectives.finalize_all_reduce(
        y, norm.weight, scores, ids, moe.local_expert_start, experts.w13_weight.shape[0], norm.eps, gain,
        **_moe_launch(rows))


def fuses_attention_boundary(boundary, attn_input, stream) -> bool:
    """Whether this layer's attention all-reduce and boundary run fused (the o_proj then skips its all-reduce)."""
    return (
        AR_BOUNDARY
        and _active(attn_input.shape[0])
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
    """(ffn_input, new stream) from the o_proj partials; requires ``fuses_attention_boundary``."""
    boundary._ensure_cached_weights(torch.bfloat16)
    return _collectives.all_reduce_boundary(
        partial, stream, boundary._cc_alpha, boundary._cc_beta, boundary._cc_pre_weight, boundary._post_norm.eps,
        boundary._pre_norm.eps, **_attention_launch(partial.shape[0]))
