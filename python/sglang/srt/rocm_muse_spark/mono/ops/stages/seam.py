# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 stage 1a, seam A: the previous layer's FFN boundary closed and this
layer's pre-QKV norm, as ``fused_norm_residual_norm_fwd`` computes them::

    xh       = branch * (w_pf + 1)                      (weight-first post norm)
    stream_a = alpha * stream + beta * xh * rsqrt(mean(xh^2) + post_eps)
    attn_in  = bf16(stream_a * rsqrt(mean(stream_a^2) + eps) * (w_qn + 1))

Each of the 64 slice tasks owns 128 contiguous columns.  For the post norm, the
lower 32 tasks also load their corresponding upper-half slice and reproduce
Triton's 16-wave ownership: eight adjacent values per lane, two 32-lane halves
per wave, then 16 wave partials.  ``stream_a`` is stored (fp32, the layer's
residual_state input to K2); the bf16 ``attn_in`` columns go to the scratch x
rows at device scope, followed by the slice's XRDY flag.

Rounding follows the Triton kernel elementwise (fp32 math, the weight added in
fp32, ``alpha * stream + beta * y`` as one fma, rsqrt then the weight, one bf16
rounding), including the two 8192-wide sums of squares.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    butterfly,
    hw_rsq,
    rsrc,
    traced,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.sync import publish

HIDDEN = 8192
NS = 64
COLS = HIDDEN // NS
QC = COLS // 4


def ld_bf4(ptr, i):
    """bf16 elements i .. i + 4 (i a multiple of 4) as four f32."""
    w = fx.Vector(bo.buffer_load(rsrc(ptr), i // 2, vec_width=2, dtype=T.i32))
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


def ld_f4(ptr, i):
    """fp32 elements i .. i + 4 (i a multiple of 4)."""
    v = fx.Vector(bo.buffer_load(rsrc(ptr), i, vec_width=4, dtype=T.f32))
    return [v[k] for k in range(4)]


def _fma(a, b, c):
    return fx.Float32(fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


@traced
def _row_rstd(c, region, eps, red):
    s_rows, tid, lane = c["T"], c["tid"], c["lane"]
    if tid < 64 * s_rows:
        tt = tid // 64
        v = c["poll"]([(region, lane * s_rows + tt, 1)])[0][0].bitcast(
            fx.Float32
        )
        tot = butterfly(v, (32, 16, 8, 4, 2, 1))
        if lane == 0:
            fx.ptr_store(tot, red + tt)
    gpu.barrier()
    t = fx.min(tid // QC, s_rows - 1)
    return hw_rsq(
        fx.ptr_load(red + t) * fx.Float32(1.0 / HIDDEN) + fx.Float32(eps)
    )


@traced
def _post_row_rstd(c, region, eps, red):
    """Reduce the 16 production wave partials for the post-norm sum."""
    s_rows, tid = c["T"], c["tid"]
    if tid < 16 * s_rows:
        tt = tid // 16
        slot = tid % 16
        vals = c["poll"](
            [
                (region, (2 * slot) * s_rows + tt, 1),
                (region, (2 * slot + 1) * s_rows + tt, 1),
            ]
        )
        v = vals[0][0].bitcast(fx.Float32) + vals[1][0].bitcast(
            fx.Float32
        )
        tot = butterfly(v, (8, 4, 2, 1))
        if tid % 16 == 0:
            fx.ptr_store(tot, red + tt)
    gpu.barrier()
    t = fx.min(tid // QC, s_rows - 1)
    return hw_rsq(
        fx.ptr_load(red + t) * fx.Float32(1.0 / HIDDEN) + fx.Float32(eps)
    )


@traced
def stage_seam(c, s):
    """Post norm, residual mix, pre norm, then XRDY[s]."""
    s_rows, tid, lane = c["T"], c["tid"], c["lane"]
    a = c["args"]
    red = c["red"]
    mine = tid < s_rows * QC
    t = fx.min(tid // QC, s_rows - 1)
    col = s * COLS + (tid % QC) * 4
    e = t * HIDDEN + col
    br = ld_bf4(a["branch_in"], e)
    st = ld_f4(a["stream_in"], e)
    wpf = ld_bf4(a["w_pf"], col)
    al = ld_f4(a["alpha_f"], col)
    be = ld_f4(a["beta_f"], col)
    wqn = ld_bf4(a["w_qn"], col)
    xh = [br[k] * (wpf[k] + fx.Float32(1.0)) for k in range(4)]
    if s < NS // 2:
        br_hi = ld_bf4(a["branch_in"], e + HIDDEN // 2)
        wpf_hi = ld_bf4(a["w_pf"], col + HIDDEN // 2)
        xh_hi = [
            br_hi[k] * (wpf_hi[k] + fx.Float32(1.0)) for k in range(4)
        ]
        # Match Triton's per-lane accumulator and DPP row-reduction order.
        sq = xh[0] * xh[0]
        sq = _fma(xh[1], xh[1], sq)
        sq = _fma(xh[2], xh[2], sq)
        sq = _fma(xh[3], xh[3], sq)
        sq = _fma(xh_hi[0], xh_hi[0], sq)
        sq = _fma(xh_hi[1], xh_hi[1], sq)
        sq = _fma(xh_hi[2], xh_hi[2], sq)
        sq = _fma(xh_hi[3], xh_hi[3], sq)
        # Triton's tl.sum lowers to SWAP16, bidirectional row shifts at 8/4,
        # then quad swaps at 2/1.  ``butterfly`` emits that exact sequence.
        sq = butterfly(sq, (16, 8, 4, 2, 1))
        if mine & (lane % 32 == 0):
            c["put"](c["p1"], s * s_rows + t, sq)
    r1 = _post_row_rstd(c, c["p1"], c["post_eps"], red)
    sv = [_fma(be[k], xh[k] * r1, al[k] * st[k]) for k in range(4)]
    if mine:
        bo.buffer_store(
            fx.Vector.from_elements(sv, fx.Float32), rsrc(a["stream_a"]), e
        )
    sq2 = (sv[0] * sv[0] + sv[1] * sv[1]) + (
        sv[2] * sv[2] + sv[3] * sv[3]
    )
    sq2 = butterfly(sq2, (16, 8, 4, 2, 1))
    if mine & (lane % 32 == 0):
        c["put"](c["p2"], s * s_rows + t, sq2)
    r2 = _row_rstd(c, c["p2"], c["eps"], red + 16)
    y = [(sv[k] * r2) * (wqn[k] + fx.Float32(1.0)) for k in range(4)]
    words = (
        fx.Vector.from_elements(y, fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)
    )
    if mine:
        bo.buffer_store(words, rsrc(c["xbuf"]), e // 2, cache_modifier=CM_DEV)
    publish(c["put"], c["xrdy"], s, fx.Int32(1), tid == 0)
    gpu.barrier()
