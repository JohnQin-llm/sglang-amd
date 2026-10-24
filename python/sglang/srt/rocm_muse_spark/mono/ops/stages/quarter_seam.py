# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in q8 K2 seam.

AR#1 runs as reduce-scatter + all-gather: CTA ``4 * token + quarter`` sums
this rank's 256 columns of the quarter over the eight ranks in rank order and
all-gathers the bf16 x and its per-slice sums of squares.  The two norms of a
token then run in one CTA with the original 64-slice lane layout and
butterflies, so neither norm needs a cross-CTA rendezvous.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    CM_DEV,
    bf2_f32,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    hw_rsq,
    rsrc,
    traced,
    wave_sum,
)
from sglang.srt.rocm_muse_spark.mono.ops.stages.seam_b import COLS, TP, H, _fma


def f32_of(w):
    return w.bitcast(fx.Float32)


OWNER_COLS = H // TP  # 1024 columns a rank sums
OWNER_PART = OWNER_COLS // 4  # 256 columns a CTA (four CTAs a token)


@traced
def stage_ar_owner_sum(c, bid):
    """This rank's columns of token ``bid // 4``, quarter ``bid % 4``: the eight
    ranks' bf16 partials summed in rank order and rounded to bf16 (the seam's
    arithmetic), then all-gathered as bf16 pairs to every rank's XG."""
    tid = c["tid"]
    token, part = bid // 4, bid % 4
    if tid < OWNER_PART // 2:
        col = c["rank"] * OWNER_COLS + part * OWNER_PART + 2 * tid
        ar = c["poll"](
            [(c["ar_own"], ((src * 8 + token) * H + col) // 2, 1) for src in range(TP)]
        )
        lo, hi = bf2_f32(ar[0][0])
        for src in range_constexpr(1, TP):
            rank_lo, rank_hi = bf2_f32(ar[src][0])
            lo = lo + rank_lo
            hi = hi + rank_hi
        x0, x1 = bf16_round(lo), bf16_round(hi)
        word = bf16_pair(x0, x1).bitcast(fx.Int32)
        for p in range_constexpr(TP):
            c["put_words"](c["peer"]["xg"][p], (token * H + col) // 2, [word])
        # This wave's 128-column slice: the seam's per-slice sum of squares.
        sq = wave_sum(x0 * x0 + x1 * x1)
        if c["lane"] == 0:
            j = (
                c["rank"] * (OWNER_COLS // COLS)
                + (part * OWNER_PART) // COLS
                + c["wave"]
            )
            for p in range_constexpr(TP):
                c["put"](c["peer"]["xsq"][p], token * (H // COLS) + j, sq)


SEAM_SLICES = 8  # 128-column slices a wave (eight waves cover the row)
NSL_ROW = H // COLS  # 64


@traced
def stage_seam_token(c, token):
    """The q8 seam for one token in one CTA.

    Wave w, slice j = 8 w + i, lane l: columns 128 j + 2 l and + 1, exactly
    the quarter seam's per-slice layout.  rstd1 comes from the owners' slice
    sums (XSQ); the slice sums of n^2 are reduced through LDS in lane = slice
    order with the same butterfly as the MSQ2 rendezvous.
    """
    lane, wave, red = c["lane"], c["wave"], c["red"]
    a = c["args"]
    js = [wave * SEAM_SLICES + i for i in range(SEAM_SLICES)]
    cols = [j * COLS + 2 * lane for j in js]
    es = [token * H + col for col in cols]
    sv = [
        fx.Vector(bo.buffer_load(rsrc(a["stream_a"]), e, vec_width=2, dtype=T.f32))
        for e in es
    ]
    al = [
        fx.Vector(bo.buffer_load(rsrc(a["alpha_a"]), col, vec_width=2, dtype=T.f32))
        for col in cols
    ]
    be = [
        fx.Vector(bo.buffer_load(rsrc(a["beta_a"]), col, vec_width=2, dtype=T.f32))
        for col in cols
    ]
    wf = [
        fx.Int32(bo.buffer_load(rsrc(a["w_fn"]), col // 2, vec_width=1, dtype=T.i32))
        for col in cols
    ]
    xs = c["poll"]([(c["peer"]["xg_own"], (token * H + col) // 2, 1) for col in cols])
    c["stamp"](12)
    if wave == 0:
        total = wave_sum(
            f32_of(c["poll"]([(c["peer"]["xsq_own"], token * NSL_ROW + lane, 1)])[0][0])
        )
        if lane == 0:
            fx.ptr_store(
                hw_rsq(total * fx.Float32(1.0 / H) + fx.Float32(c["eps_attn"])), red
            )
    gpu.barrier()
    r1 = fx.ptr_load(red)
    c["stamp"](13)
    ns = []
    for i in range_constexpr(SEAM_SLICES):
        x0, x1 = bf2_f32(xs[i][0])
        n0 = _fma(al[i][0], sv[i][0], be[i][0] * (x0 * r1))
        n1 = _fma(al[i][1], sv[i][1], be[i][1] * (x1 * r1))
        bo.buffer_store(
            fx.Vector.from_elements([n0, n1], fx.Float32), rsrc(a["stream_out"]), es[i]
        )
        sq2 = wave_sum(n0 * n0 + n1 * n1)
        if lane == 0:
            fx.ptr_store(sq2, red + (NSL_ROW + js[i]))
        ns.append((n0, n1))
    gpu.barrier()
    if wave == 0:
        total2 = wave_sum(fx.ptr_load(red + (NSL_ROW + lane)))
        if lane == 0:
            fx.ptr_store(
                hw_rsq(total2 * fx.Float32(1.0 / H) + fx.Float32(c["eps_ffn"])),
                red + 1,
            )
    gpu.barrier()
    r2 = fx.ptr_load(red + 1)
    for i in range_constexpr(SEAM_SLICES):
        n0, n1 = ns[i]
        y0 = (n0 * r2) * (bf_lo(wf[i]) + fx.Float32(1.0))
        y1 = (n1 * r2) * (bf_hi(wf[i]) + fx.Float32(1.0))
        xw = (
            fx.Vector.from_elements([y0, y1], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)
        )
        bo.buffer_store(xw[0], rsrc(c["xbuf"]), es[i] // 2, cache_modifier=CM_DEV)
    # Drain the x rows, then this token's four K-quarter flags.
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] < 4:
        c["put"](c["xrdy"], token * 4 + c["tid"], fx.Int32(1))
    gpu.barrier()
