# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""q8 o_proj owner reduction and tagged reduced-AR broadcast."""

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    bf16_pair,
    bf16_round,
    bf2_f32,
    row_sum,
    traced,
)
from sglang.srt.rocm_muse_spark.mono.ops.stages import gemv, oproj


@traced
def stage_oproj(c, bid):
    """CTA ``bid`` computes both q8 o_proj tiles for their column owner."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    oproj.load_core(c)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    if wave < 2:
        rg = bid + 256 * wave
        acc = gemv.mfmas_full(
            lane, c["xl"], oproj.CORE_ROW, oproj.CORE, s, oproj.o_loads_full(c, rg)
        )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    for j in range_constexpr(2):
        if tid < s * (oproj.ROWS // 4):
            tt = tid // (oproj.ROWS // 4)
            rq = tid % (oproj.ROWS // 4)
            vs = [
                bf16_round(row_sum(red, 4 * rq + d, tt, waves=(j,)))
                for d in range(4)
            ]
            column = (bid + 256 * j) * 16 + 4 * rq
            owner = column // 1024
            at = (c["rank"] * s + tt) * 8192 + column
            for peer in range_constexpr(8):
                if owner == peer:
                    c["put_bf"](c["ar_peer"][peer], at, vs)
    gpu.barrier()


@traced
def stage_reduce(c, j):
    """Owner CTA ``j`` reduces its rank's columns and broadcasts tagged BF16."""
    t = c["wave"]
    col = c["rank"] * 1024 + j * 128 + c["lane"] * 2
    ar = c["poll"](
        [
            (c["ar_own"], ((src * 8 + t) * 8192 + col) // 2, 1)
            for src in range(8)
        ],
        batch=8,
    )
    lo, hi = bf2_f32(ar[0][0])
    for src in range_constexpr(1, 8):
        ranklo, rankhi = bf2_f32(ar[src][0])
        lo = lo + ranklo
        hi = hi + rankhi
    out = bf16_pair(lo, hi).bitcast(fx.Int32)
    for p in range_constexpr(8):
        c["put_words"](
            c["peer"]["ar_reduced"][p], (t * 8192 + col) // 2, [out]
        )
    gpu.barrier()
