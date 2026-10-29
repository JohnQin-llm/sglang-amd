"""Muse Spark TP4 target verify at 48..128 rows: the MoE pooler and unpool split four ways across the TP ranks.

Above the TP4 K2's widths (8..40 rows) each rank used to read the whole replicated pooler [4096, 8192] and unpool
[8192, 4096] BF16 weights, 128 MiB a layer. Here each rank reads a quarter of each (32 MiB) and the ranks exchange
the projected slices inside the kernel that consumes them (``csrc/muse_split_gather.cu``):

* ``route`` (the MoE front): one split-K GEMM over this rank's 1024 pooler rows and the 512 rows of the replicated
  router (its high and low BF16 parts, as packed for the joint router/pooler), then ``front_epilogue``: the router
  top-8 of every row and, once the four pooled slices of a row are exchanged, the pre-expert RMSNorm.
* ``unpool_boundary`` (the MoE tail): one split-K GEMM of the all-reduced expert output over this rank's 2048 unpool
  rows, then ``tail_boundary``: the branch slices exchanged and the FFN boundary (input-scale post-norm, gated
  residual, the next layer's pre-attention norm) on whole rows.

The fused finalize + all-reduce between the two (``boundary_fusion``) is unchanged. The weights stay whole on
every rank, because prefill, the K2 widths, eager steps and the other widths still read them; the slices used here
are row ranges of the joint router/pooler weight and of the unpool weight, so no copy is made.

Switch: ``SGLANG_MUSE_TP4_SPLIT_POOL_UNPOOL=1`` (default off). Like ``boundary_fusion``, the split path is used only
while a CUDA graph is captured (the TP4 target-verify graphs), at the widths of ``_FRONT_GEMM`` / ``_TAIL_GEMM``;
eager calls and every other width keep the replicated path. ``install`` runs once after loading (collective).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang.srt.rocm_muse_spark.jit_build import stage_source

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("SGLANG_MUSE_TP4_SPLIT_POOL_UNPOOL", "0") == "1"

_SOURCE = Path(__file__).resolve().parent / "csrc" / "muse_split_gather.cu"
_SOURCE_SHA256 = "21eadfbd8984fdc67f46ae53517e218c8d58998952f99424d922df88fc2cf45c"
_RANKS = 4
_HIDDEN, _POOLED, _EXPERTS = 8192, 4096, 256
_POOL_SLICE, _BRANCH_SLICE = _POOLED // _RANKS, _HIDDEN // _RANKS
_FRONT_COLS = _POOL_SLICE + 2 * _EXPERTS  # pooler slice, router high, router low
_ROUTER_ROW = _POOLED  # first router row in the packed [pooler; router high; router low] weight

# Split-K GEMM launch (block M, block N, block K, split-K, warps) per verify width. The GEMM-only sweep
# (f1-native/bench_gemm.py, cold weights; results/bench-gemm.json) favours 16 splits for the front at 48..64 rows and
# 8 for the tail; the epilogues read every split's partials, and in the 4-rank layer bench (results/bench-layer-lowsk.log
# against bench-layer-gemm-tables.log) the best launch of 8 front / 4 tail splits made front + tail 0.4..1.5 us faster
# at 48, 56, 64, 96 and 112 rows. At 80 rows the 8-split tail stayed 0.1 us faster. Front: 9.0..12.9 us against
# 22.0..28.4 us for the replicated joint GEMM; tail: 6.7..8.4 us against 18.6..22.2 us for the unpool (GEMM alone).
_FRONT_GEMM = {
    **{m: (32, 32, 128, 8, 4) for m in (48, 56)},
    64: (64, 64, 256, 8, 4),
    80: (64, 64, 128, 8, 4),
    **{m: (64, 64, 128, 8, 8) for m in (96, 112, 128)},
}
_TAIL_GEMM = {
    **{m: (32, 32, 256, 4, 4) for m in (48, 56)},
    64: (64, 32, 256, 4, 4),
    80: (64, 64, 128, 8, 8),
    **{m: (64, 64, 256, 4, 4) for m in (96, 112, 128)},
}

_ext = None
_state = None


@triton.jit
def _slice_partial(
    X,
    W,
    P,
    row0,
    row1,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    N0: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    """``P[s] = X @ W[rows].T`` over K slice ``s`` (FP32 [SK, M, N]), with the weight rows taken from two row
    ranges of ``W``: output column n < N0 reads row ``row0 + n``, the others row ``row1 + n - N0``.
    (``bf16_splitk_gemm._dense_partial`` with a row map; BN divides N0 so a tile stays in one range.)"""
    pn, pm, pk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    rows = tl.where(n < N0, row0 + n, row1 + n - N0)
    k = pk * (K // SK) + tl.arange(0, BK)
    ap = X + m[:, None] * K + k[None, :]
    bp = W + rows[None, :] * K + k[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(K // SK // BK):
        a = tl.load(ap, m[:, None] < M, other=0)
        b = tl.load(bp, cache_modifier=".cg")
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK
    tl.store(P + pk * M * N + m[:, None] * N + n[None, :], acc, m[:, None] < M)


def slice_partials(x, weight, n, n0, row0, row1, launch):
    """FP32 split-K partials [SK, M, n] of ``x @ weight[rows].T`` (rows as in ``_slice_partial``)."""
    m, k = x.shape
    bm, bn, bk, sk, warps = launch
    assert x.dtype == weight.dtype == torch.bfloat16 and x.is_contiguous() and weight.is_contiguous()
    assert weight.shape[1] == k and n0 % bn == 0 and n % bn == 0 and k % (sk * bk) == 0
    partial = torch.empty((sk, m, n), dtype=torch.float32, device=x.device)
    _slice_partial[(n // bn, triton.cdiv(m, bm), sk)](
        x, weight, partial, row0, row1, m, n, k, n0, bm, bn, bk, sk,
        num_warps=warps, num_stages=2, matrix_instr_nonkdim=16,
    )
    return partial


def _extension():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        source = _SOURCE.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if digest != _SOURCE_SHA256:
            raise ImportError(f"{_SOURCE} has sha256 {digest}, expected {_SOURCE_SHA256}")
        build, staged = stage_source(_SOURCE, source, digest, "MUSE_SPLIT_PROJ_BUILD", "muse_split_gather")
        arch = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"  # the kernels are written for MI355X only
        try:
            _ext = load(
                name=f"muse_split_gather_{digest[:16]}",
                sources=[str(staged)],
                extra_cflags=["-O3", "-std=c++20"],
                extra_cuda_cflags=["-O3", "-std=c++20", "-ffp-contract=off", "--offload-arch=gfx950"],
                build_directory=str(build),
                verbose=False,
            )
        finally:
            if arch is None:
                os.environ.pop("PYTORCH_ROCM_ARCH")
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = arch
    return _ext


class SplitExchange:
    """Per TP group: every rank's exchange and flag buffers (uncached, IPC-mapped) and the two launches."""

    def __init__(self, ca_comm):
        assert ca_comm is not None and not ca_comm.disabled and ca_comm.world_size == _RANKS
        ext = _extension()
        self.rank = ca_comm.rank
        pool = ca_comm._pool
        data_key, flags_key = "muse_split_data", "muse_split_flags"
        # Collective (eager, before any graph capture). The flags start at zero on every rank before any rank can
        # map them: get_ipc_meta returns only after all ranks posted their handles.
        if data_key not in pool:
            pool.create(data_key, ext.data_bytes(), uncached=True)
            pool.create(flags_key, ext.flag_bytes(), uncached=True)
            ext.zero_flags(pool[flags_key].data_ptr)
        self.data = self._open(ext, pool, data_key)
        self.flags = self._open(ext, pool, flags_key)

    def _open(self, ext, pool, key):
        handles, offsets = pool.get_ipc_meta(key)
        return ext.open_peer_buffers(pool[key].data_ptr, list(handles), list(offsets), self.rank)

    def front(self, x, packed, bias, norm_weight, multiplier, scale, eps, num_real=None):
        """(top-8 scores FP32 [M, 8], expert ids int32 [M, 8], expert input BF16 [M, 4096]) of ``x`` [M, 8192]:
        ``joint_router_pooler_topk`` with this rank's quarter of the pooler. ``packed`` is the joint
        [pooler; router high; router low] weight [4608, 8192]."""
        m = x.shape[0]
        assert packed.shape == (_ROUTER_ROW + 2 * _EXPERTS, _HIDDEN)
        partial = slice_partials(x, packed, _FRONT_COLS, _POOL_SLICE, self.rank * _POOL_SLICE, _ROUTER_ROW,
                                 _FRONT_GEMM[m])
        scores = torch.empty((m, 8), dtype=torch.float32, device=x.device)
        ids = torch.empty((m, 8), dtype=torch.int32, device=x.device)
        xe = torch.empty((m, _POOLED), dtype=torch.bfloat16, device=x.device)
        _extension().front_epilogue(partial, bias, norm_weight, scores, ids, xe, float(multiplier), float(scale),
                                    float(eps), num_real, self.data, self.flags, self.rank)
        return scores, ids, xe

    def tail(self, expert_output, unpool_weight, stream, post_weight, pre_weight, alpha, beta, post_eps, pre_eps,
             pre_bf16):
        """(next attention input BF16 [M, 8192], new stream FP32 [M, 8192]): the unpool of the all-reduced expert
        output [M, 4096] with this rank's quarter of ``unpool_weight`` [8192, 4096], then the FFN boundary."""
        m = expert_output.shape[0]
        assert unpool_weight.shape == (_HIDDEN, _POOLED)
        row0 = self.rank * _BRANCH_SLICE
        partial = slice_partials(expert_output, unpool_weight, _BRANCH_SLICE, _BRANCH_SLICE, row0, row0,
                                 _TAIL_GEMM[m])
        next_input = torch.empty((m, _HIDDEN), dtype=torch.bfloat16, device=expert_output.device)
        new_stream = torch.empty((m, _HIDDEN), dtype=torch.float32, device=expert_output.device)
        _extension().tail_boundary(partial, stream, alpha, beta, post_weight, pre_weight, float(post_eps),
                                   float(pre_eps), bool(pre_bf16), next_input, new_stream, self.data, self.flags,
                                   self.rank)
        return next_input, new_stream


def install(tp_group) -> None:
    """Map the exchange buffers on every TP rank; call once, eagerly, before graph capture (collective)."""
    global _state
    if _state is not None or not ENABLED:
        return
    ca_comm = getattr(tp_group, "ca_comm", None)
    if getattr(tp_group, "world_size", None) != _RANKS or ca_comm is None or ca_comm.disabled:
        logger.info("Muse split pooler/unpool off: needs TP4 with AITER custom all-reduce")
        return
    _state = SplitExchange(ca_comm)
    logger.info("Muse split pooler/unpool on: verify widths %s, rank %d", sorted(_FRONT_GEMM), _state.rank)


def _active(rows: int, table) -> bool:
    return _state is not None and rows in table and torch.cuda.is_current_stream_capturing()


def front_applies(moe, x) -> bool:
    """Whether ``moe.route(x)`` (a target-verify step) takes the split front."""
    if not (ENABLED and _active(x.shape[0], _FRONT_GEMM)):
        return False
    from sglang.srt.rocm_muse_spark import router_pooler

    packed = getattr(moe, "router_pooler_packed_weight", None)
    return (
        moe.tp_size == moe.ep_size == _RANKS
        and packed is not None
        and packed.shape == (_ROUTER_ROW + 2 * _EXPERTS, _HIDDEN)
        and moe.pre_expert_proj.weight.data_ptr() == packed.data_ptr()  # the router parts are current
        and router_pooler.can_share_router_pooler(moe, x)
    )


def route(moe, x, num_real=None):
    """``MuseSpark1xMoE.route`` on the split front; requires ``front_applies``. ``num_real``: the graph's real-row
    count (device int32); the rows after it get expert id -1."""
    return _state.front(x, moe.router_pooler_packed_weight, moe.expert_bias,
                        moe.pre_expert_norm._get_effective_weight(), moe.router_multiplier, moe.route_scale,
                        moe.pre_expert_norm.eps, num_real)


def tail_applies(moe, x, stream, boundary, pre_norm_dtype) -> bool:
    """Whether a target-verify MoE layer on ``x`` ends with ``unpool_boundary`` (the decoder layer then asks the
    MoE for its output before the unpool)."""
    if not (ENABLED and _active(x.shape[0], _TAIL_GEMM)) or boundary is None:
        return False
    proj = moe.post_expert_proj
    return (
        moe.tp_size == moe.ep_size == _RANKS
        and proj is not None
        and proj.bias is None
        and proj.weight.shape == (_HIDDEN, _POOLED)
        and proj.weight.dtype == torch.bfloat16
        and proj.weight.is_contiguous()
        and stream.dtype == torch.float32
        and stream.is_contiguous()
        and stream.shape == (x.shape[0], _HIDDEN)
        and pre_norm_dtype in (None, torch.bfloat16)
        and boundary._post_norm_is_input_scale
        and boundary._post_norm.has_weight
        and boundary._use_residual_gate
        and boundary._residual_add.gate.numel() == _HIDDEN
        and boundary._pre_norm.has_weight
        and not boundary._pre_norm_is_input_scale
    )


def unpool_boundary(moe, expert_output, stream, boundary, pre_norm_dtype):
    """(next attention input, new stream) from the all-reduced expert output; requires ``tail_applies``."""
    boundary._ensure_cached_weights(torch.bfloat16)
    return _state.tail(expert_output, moe.post_expert_proj.weight, stream, boundary._cc_post_weight,
                       boundary._cc_pre_weight, boundary._cc_alpha, boundary._cc_beta, boundary._post_norm.eps,
                       boundary._pre_norm.eps, pre_norm_dtype is torch.bfloat16)
