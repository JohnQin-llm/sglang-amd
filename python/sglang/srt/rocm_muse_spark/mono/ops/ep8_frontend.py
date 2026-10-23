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

This local native frontend differs from the original Mono at three BF16
boundaries: post-attention normalization, effective FFN weight, and effective
pre-expert weight.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    CM_DEV,
    butterfly,
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
    n0 = _fma(al[0], sv[0], be[0] * bf16_round(x0 * r1))
    n1 = _fma(al[1], sv[1], be[1] * bf16_round(x1 * r1))
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
    y0 = (n0 * r2) * bf16_round(bf_lo(wf) + fx.Float32(1.0))
    y1 = (n1 * r2) * bf16_round(bf_hi(wf) + fx.Float32(1.0))
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


from sglang.srt.rocm_muse_spark.mono.ops.stages import mxq
from sglang.srt.rocm_muse_spark.mono.ops.stages.moe import MH, WAVES, XQ_W, XQ_G

@traced
def stage_xq(c, t):
    """Token t (CTA XQ0 + t): its pooled row (every rank's 512 columns from
    POOL) -> pre_expert_norm -> bf16 xe -> MXFP4 (``mxq.act_code``) -> the
    plain XQ words / codes (device scope), then XQRDY[t]. A thread 8 columns
    (one E2M1 word), 4 threads a 32-group."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    c0 = tid * 8
    wv = fx.Vector(bo.buffer_load(rsrc(a["w_pre"]), c0 // 2, vec_width=4, dtype=T.i32))
    base = (t * MH + c0) // 2
    got = c["poll"](
        [(c["peer"]["pool_own"], base, 2), (c["peer"]["pool_own"], base + 2, 2)]
    )
    xs = []
    for g in got:
        for w_ in g:
            xs.append(bf_lo(w_))
            xs.append(bf_hi(w_))
    sq = fx.Float32(0.0)
    for x in xs:
        sq = sq + x * x
    sq = wave_sum(sq)
    if lane == 0:
        fx.ptr_store(sq, red + wave)
    gpu.barrier()
    tot = fx.ptr_load(red + 0)
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(red + w)
    r = hw_rsq(tot * fx.Float32(1.0 / MH) + fx.Float32(c["eps_pre"]))
    ws = []
    for d in range_constexpr(4):
        ws.append(bf_lo(wv[d]))
        ws.append(bf_hi(wv[d]))
    xe = [
        bf16_round((xs[k] * r) * bf16_round(ws[k] + fx.Float32(1.0)))
        for k in range(8)
    ]
    am = fx.Float32(0.0)
    for v in xe:
        am = fx.max(am, abs(v))
    am = butterfly(am, (2, 1), fx.max)
    code = mxq.act_code(am)
    pk = mxq.pack8(xe, mxq.code_scale(code))
    bo.buffer_store(pk, rsrc(c["xqb"]), t * XQ_W + tid, cache_modifier=CM_DEV)
    if lane % 4 == 0:
        bo.buffer_store(
            code, rsrc(c["xscb"]), t * XQ_G + tid // 4, cache_modifier=CM_DEV
        )
    publish(c["put"], c["xqrdy"], t, fx.Int32(1), tid == 0)
    gpu.barrier()

