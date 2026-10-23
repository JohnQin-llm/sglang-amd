# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2's seam B: the o_proj all-reduce and production's
``norm_residual_then_norm`` between the attention and the MoE, per 128-column
slice. Adapted from Kimi-K3's K2a reduce (``attention/back.py``
``stage_reduce``) and AttnRes slice / mix (``attention/kda.py`` of PR #60683):
partial sums of squares a slice through a mailbox, every slice summing all 64
in the same order, so every slice applies the same rstd.

Slice j (CTA j < 64), thread (token = wave, column pair = lane):

    attn  = bf16(sum over ranks 0..7, in order, of the bf16 o_proj partials)
    s'    = fma(alpha_a, s, beta_a (attn rsqrt(mean(attn^2) + eps_attn)))
            -> ``stream_out`` (fp32)
    x     = bf16((s' rsqrt(mean(s'^2) + eps_ffn)) (w_fn + 1))
            -> the plain x rows (device scope), then the slice's XRDY flag

as ``fused_norm_residual_norm_fwd`` (post norm without weight) rounds them
elementwise; only the order of the two 8192-wide sums of squares differs. The
means: each slice's (token) partial -> MSQ1 / MSQ2, then a lane a slice polls
and the wave sums them (one fixed butterfly).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    CM_DEV,
    bf16_round,
    bf_hi,
    bf_lo,
    hw_rsq,
    rsrc,
    traced,
    wave_sum,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.ranks import sum_partials
from sglang.srt.rocm_muse_spark.mono.ops.common.sync import publish

TP = 8
H = 8192
COLS = 128
NSL = H // COLS  # 64 slices: one a lane of a gather


def f32_of(w):
    return w.bitcast(fx.Float32)


def _fma(a, b, c):
    return fx.Float32(fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


@traced
def _stage_seam_wave(c, j, t, col, live):
    s, lane = c["S"], c["lane"]
    a = c["args"]
    e = t * H + col
    # the inputs that need no hand-off, issued before the first poll
    sv = fx.Vector(bo.buffer_load(rsrc(a["stream_a"]), e, vec_width=2, dtype=T.f32))
    al = fx.Vector(bo.buffer_load(rsrc(a["alpha_a"]), col, vec_width=2, dtype=T.f32))
    be = fx.Vector(bo.buffer_load(rsrc(a["beta_a"]), col, vec_width=2, dtype=T.f32))
    wf = fx.Int32(bo.buffer_load(rsrc(a["w_fn"]), col // 2, vec_width=1, dtype=T.i32))
    # AR#1: the ranks' partials in rank order, fp32 -> bf16
    lo, hi = sum_partials(
        c["poll"], c["ar_own"], lambda src: ((src * s + t) * H + col) // 2, TP
    )
    x0 = bf16_round(lo)
    x1 = bf16_round(hi)
    c["stamp"](12)
    # post_attn_norm (no weight), then the gated residual
    sq = wave_sum(x0 * x0 + x1 * x1)
    if live & (lane == 0):
        c["put"](c["msq1"], j * s + t, sq)
    tot = wave_sum(f32_of(c["poll"]([(c["msq1"], lane * s + t, 1)])[0][0]))
    c["stamp"](13)
    r1 = hw_rsq(tot * fx.Float32(1.0 / H) + fx.Float32(c["eps_attn"]))
    n0 = _fma(al[0], sv[0], be[0] * (x0 * r1))
    n1 = _fma(al[1], sv[1], be[1] * (x1 * r1))
    if live:
        bo.buffer_store(
            fx.Vector.from_elements([n0, n1], fx.Float32),
            rsrc(a["stream_out"]),
            e,
        )
    # pre_ffn_norm: (s' rstd) (w + 1)
    sq2 = wave_sum(n0 * n0 + n1 * n1)
    if live & (lane == 0):
        c["put"](c["msq2"], j * s + t, sq2)
    tot2 = wave_sum(f32_of(c["poll"]([(c["msq2"], lane * s + t, 1)])[0][0]))
    r2 = hw_rsq(tot2 * fx.Float32(1.0 / H) + fx.Float32(c["eps_ffn"]))
    y0 = (n0 * r2) * (bf_lo(wf) + fx.Float32(1.0))
    y1 = (n1 * r2) * (bf_hi(wf) + fx.Float32(1.0))
    xw = fx.Vector.from_elements([y0, y1], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    if live:
        bo.buffer_store(xw[0], rsrc(c["xbuf"]), e // 2, cache_modifier=CM_DEV)


@traced
def stage_seam(c, j):
    s, lane, wave = c["S"], c["lane"], c["wave"]
    live = wave < s
    t = fx.min(wave, s - 1)
    col = j * COLS + 2 * lane
    if const_expr(s == 2):
        if live:
            _stage_seam_wave(c, j, t, col, live)
    else:
        _stage_seam_wave(c, j, t, col, live)
    publish(c["put"], c["xrdy"], j, fx.Int32(1), c["tid"] == 0)
    gpu.barrier()
