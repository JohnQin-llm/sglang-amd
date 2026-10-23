# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in q8 K2 seam: four exact K-quarter CTAs per token.

CTA ``4 * token + quarter`` owns the quarter's sixteen original 128-column
slices.  Each wave keeps two slices' AR results and coefficients live across
the scalar rendezvous; no whole row is staged.  ``red[0]`` is reused only
after o_proj has finished and before any later stage uses that existing LDS
union/lifetime.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    bf_hi,
    bf_lo,
    hw_rsq,
    rsrc,
    traced,
    wave_sum,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.sync import publish
from sglang.srt.rocm_muse_spark.mono.ops.stages.seam_b import COLS, TP, H, _fma


def f32_of(w):
    return w.bitcast(fx.Float32)


@traced
def _load_scalar(c, region, token):
    """One tagged scalar read per CTA, then a uniform LDS broadcast."""
    if c["tid"] == 0:
        word = c["poll"]([(region, token, 1)])[0][0]
        fx.ptr_store(f32_of(word), c["red"])
    gpu.barrier()
    return fx.ptr_load(c["red"])


@traced
def stage_seam(c, bid):
    lane, wave = c["lane"], c["wave"]
    token, quarter = bid // 4, bid % 4
    j0 = quarter * 16 + wave * 2
    j1 = j0 + 1
    col0, col1 = j0 * COLS + 2 * lane, j1 * COLS + 2 * lane
    e0, e1 = token * H + col0, token * H + col1
    a = c["args"]

    # Keep both slices' source values in wave-local registers. Accumulate each
    # slice in the original rank order and only then round to bf16.
    sv0 = fx.Vector(bo.buffer_load(rsrc(a["stream_a"]), e0, vec_width=2, dtype=T.f32))
    sv1 = fx.Vector(bo.buffer_load(rsrc(a["stream_a"]), e1, vec_width=2, dtype=T.f32))
    al0 = fx.Vector(bo.buffer_load(rsrc(a["alpha_a"]), col0, vec_width=2, dtype=T.f32))
    al1 = fx.Vector(bo.buffer_load(rsrc(a["alpha_a"]), col1, vec_width=2, dtype=T.f32))
    be0 = fx.Vector(bo.buffer_load(rsrc(a["beta_a"]), col0, vec_width=2, dtype=T.f32))
    be1 = fx.Vector(bo.buffer_load(rsrc(a["beta_a"]), col1, vec_width=2, dtype=T.f32))
    wf0 = fx.Int32(bo.buffer_load(rsrc(a["w_fn"]), col0 // 2, vec_width=1, dtype=T.i32))
    wf1 = fx.Int32(bo.buffer_load(rsrc(a["w_fn"]), col1 // 2, vec_width=1, dtype=T.i32))
    ar = c["poll"](
        [
            (c["ar_own"], ((src * 8 + token) * H + col) // 2, 1)
            for col in (col0, col1)
            for src in range(TP)
        ],
        batch=16,
    )
    lo0, hi0 = bf2_f32(ar[0][0])
    lo1, hi1 = bf2_f32(ar[TP][0])
    for src in range_constexpr(1, TP):
        rank_lo0, rank_hi0 = bf2_f32(ar[src][0])
        rank_lo1, rank_hi1 = bf2_f32(ar[TP + src][0])
        lo0 = lo0 + rank_lo0
        hi0 = hi0 + rank_hi0
        lo1 = lo1 + rank_lo1
        hi1 = hi1 + rank_hi1
    x00, x01 = bf16_round(lo0), bf16_round(hi0)
    x10, x11 = bf16_round(lo1), bf16_round(hi1)
    c["stamp"](12)

    sq0 = wave_sum(x00 * x00 + x01 * x01)
    sq1 = wave_sum(x10 * x10 + x11 * x11)
    if lane == 0:
        c["put"](c["msq1"], j0 * 8 + token, sq0)
        c["put"](c["msq1"], j1 * 8 + token, sq1)
    gpu.barrier()
    if (quarter == 0) & (wave == 0):
        total = wave_sum(f32_of(c["poll"]([(c["msq1"], lane * 8 + token, 1)])[0][0]))
        if lane == 0:
            c["put"](
                c["rstd1"],
                token,
                hw_rsq(total * fx.Float32(1.0 / H) + fx.Float32(c["eps_attn"])),
            )
    r1 = _load_scalar(c, c["rstd1"], token)
    c["stamp"](13)

    n00 = _fma(al0[0], sv0[0], be0[0] * (x00 * r1))
    n01 = _fma(al0[1], sv0[1], be0[1] * (x01 * r1))
    n10 = _fma(al1[0], sv1[0], be1[0] * (x10 * r1))
    n11 = _fma(al1[1], sv1[1], be1[1] * (x11 * r1))
    bo.buffer_store(
        fx.Vector.from_elements([n00, n01], fx.Float32), rsrc(a["stream_out"]), e0
    )
    bo.buffer_store(
        fx.Vector.from_elements([n10, n11], fx.Float32), rsrc(a["stream_out"]), e1
    )

    sq20 = wave_sum(n00 * n00 + n01 * n01)
    sq21 = wave_sum(n10 * n10 + n11 * n11)
    if lane == 0:
        c["put"](c["msq2"], j0 * 8 + token, sq20)
        c["put"](c["msq2"], j1 * 8 + token, sq21)
    gpu.barrier()
    if (quarter == 0) & (wave == 0):
        total2 = wave_sum(f32_of(c["poll"]([(c["msq2"], lane * 8 + token, 1)])[0][0]))
        if lane == 0:
            c["put"](
                c["rstd2"],
                token,
                hw_rsq(total2 * fx.Float32(1.0 / H) + fx.Float32(c["eps_ffn"])),
            )
    r2 = _load_scalar(c, c["rstd2"], token)

    y00 = (n00 * r2) * (bf_lo(wf0) + fx.Float32(1.0))
    y01 = (n01 * r2) * (bf_hi(wf0) + fx.Float32(1.0))
    y10 = (n10 * r2) * (bf_lo(wf1) + fx.Float32(1.0))
    y11 = (n11 * r2) * (bf_hi(wf1) + fx.Float32(1.0))
    xw0 = (
        fx.Vector.from_elements([y00, y01], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)
    )
    xw1 = (
        fx.Vector.from_elements([y10, y11], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)
    )
    bo.buffer_store(xw0[0], rsrc(c["xbuf"]), e0 // 2, cache_modifier=CM_DEV)
    bo.buffer_store(xw1[0], rsrc(c["xbuf"]), e1 // 2, cache_modifier=CM_DEV)
    publish(c["put"], c["xrdy"], token * 4 + quarter, fx.Int32(1), c["tid"] == 0)
    gpu.barrier()
