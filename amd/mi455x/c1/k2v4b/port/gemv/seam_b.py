"""K2's seam B on gfx1250 at TP4 (port of ``k2bundle/ops/stages/seam_b.py``): the o_proj all-reduce and
production's ``norm_residual_then_norm`` between the attention and the MoE, per 128-column slice.  C1 (q8) runs
``quarter_seam`` instead; this default-build path and the constants ``quarter_seam`` imports are kept.

Slice j (CTA j < 64), wave = token, lane = 4 columns (two bf16 pairs):

    attn  = bf16(sum over ranks 0..3, in order, of the bf16 o_proj partials)
    s'    = fma(alpha_a, s, beta_a (attn rsqrt(mean(attn^2) + eps_attn)))   -> ``stream_out`` (fp32)
    x     = bf16((s' rsqrt(mean(s'^2) + eps_ffn)) (w_fn + 1))                -> x rows (device scope), XRDY[j]

The 8192-wide sums of squares: a slice's per-token partial (lane: (c0^2 + c1^2) + (c2^2 + c3^2), then the wave32
butterfly) -> MSQ; every slice reads all 64 (lane l: slices l and l + 32, added, then the butterfly), so every
slice applies the same rstd.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from . import common as C
from .common import CM_DEV_ST, bf16_round, bf_hi, bf_lo, hw_rsq, publish, rsrc, traced, wave_sum
from .common import fma as _fma

TP = C.TP
H = C.H
COLS = 128
NSL = H // COLS  # 64 slices
LANE_COLS = COLS // C.WAVE  # 4


def f32_of(w):
    return w.bitcast(fx.Float32)


def sq4(v):
    return (v[0] * v[0] + v[1] * v[1]) + (v[2] * v[2] + v[3] * v[3])


def slice_total(c, region, s, t):
    """Sum of the 64 slices' partials of token t at ``region`` [slice][t] (lane l: l and l + 32)."""
    lane = c["lane"]
    got = c["poll"]([(region, lane * s + t, 1), (region, (lane + C.WAVE) * s + t, 1)])
    return wave_sum(f32_of(got[0][0]) + f32_of(got[1][0]))


@traced
def stage_seam(c, j):
    s, lane, wave = c["S"], c["lane"], c["wave"]
    a = c["args"]
    live = wave < s
    t = fx.min(wave, s - 1)
    col = j * COLS + LANE_COLS * lane
    e = t * H + col
    sv = fx.Vector(bo.buffer_load(rsrc(a["stream_a"]), e, vec_width=4, dtype=T.f32))
    al = fx.Vector(bo.buffer_load(rsrc(a["alpha_a"]), col, vec_width=4, dtype=T.f32))
    be = fx.Vector(bo.buffer_load(rsrc(a["beta_a"]), col, vec_width=4, dtype=T.f32))
    wf = fx.Vector(bo.buffer_load(rsrc(a["w_fn"]), col // 2, vec_width=2, dtype=T.i32))
    ws = c["poll"]([(c["ar_own"], ((src * s + t) * H + col) // 2, 2) for src in range(TP)])
    xs = []
    for p in range_constexpr(2):
        lo, hi = bf_lo(ws[0][p]), bf_hi(ws[0][p])
        for src in range_constexpr(1, TP):
            lo = lo + bf_lo(ws[src][p])
            hi = hi + bf_hi(ws[src][p])
        xs += [bf16_round(lo), bf16_round(hi)]
    sq = wave_sum(sq4(xs))
    if live & (lane == 0):
        c["put"](c["msq1"], j * s + t, sq)
    r1 = hw_rsq(slice_total(c, c["msq1"], s, t) * fx.Float32(1.0 / H) + fx.Float32(c["eps_attn"]))
    n = [_fma(al[k], sv[k], be[k] * (xs[k] * r1)) for k in range(4)]
    if live:
        bo.buffer_store(fx.Vector.from_elements(n, fx.Float32), rsrc(a["stream_out"]), e)
    sq2 = wave_sum(sq4(n))
    if live & (lane == 0):
        c["put"](c["msq2"], j * s + t, sq2)
    r2 = hw_rsq(slice_total(c, c["msq2"], s, t) * fx.Float32(1.0 / H) + fx.Float32(c["eps_ffn"]))
    wv = [bf_lo(wf[0]), bf_hi(wf[0]), bf_lo(wf[1]), bf_hi(wf[1])]
    y = [(n[k] * r2) * (wv[k] + fx.Float32(1.0)) for k in range(4)]
    xw = fx.Vector.from_elements(y, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    if live:
        bo.buffer_store(xw, rsrc(c["xbuf"]), e // 2, cache_modifier=CM_DEV_ST)
    publish(c["put"], c["xrdy"], j, fx.Int32(1), c["tid"] == 0)
    gpu.barrier()
