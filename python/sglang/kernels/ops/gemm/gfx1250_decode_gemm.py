# SPDX-License-Identifier: Apache-2.0
"""Decode-width BF16 GEMMs on gfx1250 with per-shape tuned AITER kernels.

At 1..16 rows these GEMMs only stream their weights, and the default picks run far from the HBM rate on gfx1250:
hipBLASLt's MT64x16x256 runs the Muse unpooler at 3.5 TB/s, its FP32-output lm_head tile takes 467 us for 828 MB,
and the AITER gluon a16w16 kernel has no gfx1250 config for these shapes (its DEFAULT tile, BLOCK_M 32 / BLOCK_N 64 /
BLOCK_K 64, runs the target QKV at 3 TB/s). The table below holds the measured best kernel per weight shape (AITER
FlyDSL split-K ``gemm_a16w16_gfx1250`` or gluon ``gemm_a16w16``), swept at M = 1, 2, 4, 8, 16 over cold weights in
graph replay. Within a shape the winner does not change with M up to 16.

Only exact (N, K, output dtype) matches at 1..16 rows take this path; ``mm`` returns None otherwise and the caller
keeps its own GEMM. The output equals the BF16 x BF16 product with FP32 accumulation (summation order differs
from hipBLASLt). ``SGLANG_GFX1250_DECODE_GEMM=0`` turns it off.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import torch

ENABLED = os.environ.get("SGLANG_GFX1250_DECODE_GEMM", "1") == "1"
MAX_M = 16


def _flydsl(tile_n, tile_k, n_warp, num_buffers, split_k):
    return (
        "flydsl",
        dict(
            tile_m=16,
            tile_n=tile_n,
            tile_k=tile_k,
            m_warp=1,
            n_warp=n_warp,
            num_buffers=num_buffers,
            split_k=split_k,
        ),
    )


def _gluon(block_n, block_k, num_warps):
    return (
        "gluon",
        dict(
            BLOCK_M=16,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            NUM_BUFFERS=4,
            kernel_type="bandwidth_bound",
            num_warps=num_warps,
        ),
    )


_BF16, _FP32 = torch.bfloat16, torch.float32

# (N, K, out dtype) -> kernel; us at M = 8 (cold weights) vs the path it replaces.
_TABLE = {
    # Muse Spark target, TP4 shards.
    (4608, 8192, _BF16): _flydsl(64, 128, 4, 4, 8),  # qkv_proj: 7.8 vs 25.5 (gluon DEFAULT)
    (8192, 2048, _BF16): _gluon(64, 256, 2),  # o_proj: 4.8 vs 8.1 (gluon DEFAULT)
    (8192, 4096, _BF16): _gluon(32, 512, 4),  # post_expert_proj (replicated): 6.9 vs 19.1 (hipBLASLt)
    (4096, 8192, _BF16): _flydsl(32, 256, 2, 3, 8),  # pre_expert_proj (replicated): 7.3 vs 13.2 (hipBLASLt)
    (50512, 8192, _FP32): _flydsl(32, 128, 2, 3, 1),  # lm_head shard, FP32 logits: 51.8 vs 467 (hipBLASLt)
    # DFlash draft (TP4).
    (50512, 8192, _BF16): _flydsl(128, 128, 4, 4, 1),  # candidate logits: 51.4 vs 69.0 (hipBLASLt)
    (2560, 8192, _BF16): _flydsl(32, 128, 2, 3, 16),  # qkv_proj: 5.8 vs 24.8 (gluon DEFAULT)
    (14336, 8192, _BF16): _gluon(64, 256, 2),  # gate_up_proj: 16.5 vs 27.8 (gluon DEFAULT)
    (8192, 7168, _BF16): _gluon(32, 512, 2),  # down_proj: 9.7 vs 22.9 (gluon DEFAULT)
    (2048, 8192, _BF16): _flydsl(32, 256, 2, 3, 16),  # conv kernel_projection: 5.2 vs 11.7 (hipBLASLt)
    (256, 8192, _BF16): _flydsl(16, 256, 1, 3, 16),  # selector hidden_projection: 3.5 vs 8.2 (hipBLASLt)
    (8192, 40960, _BF16): _gluon(16, 512, 1),  # fc over 5 target layers, M ~ 4: 42.7 vs 55.7 (hipBLASLt)
}


@functools.cache
def _on_gfx1250() -> bool:
    return (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    )


@functools.cache
def _kernels():
    from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16 as flydsl_mm
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as gluon_mm

    return flydsl_mm, gluon_mm


def applies(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype) -> bool:
    return (
        ENABLED
        and x.dim() == 2
        and 0 < x.shape[0] <= MAX_M
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and (weight.shape[0], weight.shape[1], out_dtype) in _TABLE
        and _on_gfx1250()
    )


def mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """``x @ weight.T`` (no bias) in ``out_dtype`` on the tuned kernel, or None when the shape is not covered."""
    if not applies(x, weight, out_dtype):
        return None
    if out is not None and (
        out.shape != (x.shape[0], weight.shape[0])
        or out.dtype != out_dtype
        or out.stride(1) != 1
    ):
        out = None
    backend, cfg = _TABLE[(weight.shape[0], weight.shape[1], out_dtype)]
    flydsl_mm, gluon_mm = _kernels()
    if backend == "flydsl":
        return flydsl_mm(x, weight, None, out_dtype, out, **cfg)
    return gluon_mm(x, weight, None, out_dtype, out, config=dict(cfg), backend="gluon")
