"""K1 stage 1a, seam A, on gfx1250 (port of ``k2bundle/ops/stages/seam.py``): the previous layer's FFN boundary
closed and this layer's pre-QKV norm, as ``fused_norm_residual_norm_fwd`` computes them::

    xh       = branch * (w_pf + 1)                      (weight-first post norm)
    stream_a = alpha * stream + beta * xh * rsqrt(mean(xh^2) + post_eps)
    attn_in  = bf16(stream_a * rsqrt(mean(stream_a^2) + eps) * (w_qn + 1))

Each of the 64 slice tasks owns 128 columns; wave = token, lane = 4 columns.  Rounding follows the Triton kernel
elementwise (fp32 math, the weight added in fp32, ``beta * y + alpha * stream`` as one fma, rsqrt then the
weight, one bf16 rounding).  gfx950's reproduction of Triton's 16-wave ownership of the post-norm sum is dropped
(no gfx950 reference on this GPU): both 8192-wide sums are a slice's wave32 partial (lane (c0^2 + c1^2) +
(c2^2 + c3^2), butterfly) through a mailbox, then every slice sums the 64 partials in one fixed order.
``stream_a`` is stored fp32 (K2's residual_state input); the bf16 ``attn_in`` columns go to the x rows at
device scope, then the slice's XRDY flag.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu
from flydsl.expr.typing import T

from . import common as C
from .common import CM_DEV_ST, bf_hi, bf_lo, hw_rsq, publish, rsrc, traced, wave_sum
from .common import fma as _fma
from .seam_b import sq4

HIDDEN = C.H
NS = 64
COLS = HIDDEN // NS
LANE_COLS = COLS // C.WAVE  # 4


def ld_bf4(ptr, i):
    """bf16 elements i .. i + 4 (i a multiple of 4) as four f32."""
    w = fx.Vector(bo.buffer_load(rsrc(ptr), i // 2, vec_width=2, dtype=T.i32))
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


def ld_f4(ptr, i):
    v = fx.Vector(bo.buffer_load(rsrc(ptr), i, vec_width=4, dtype=T.f32))
    return [v[k] for k in range(4)]


@traced
def _row_rstd(c, region, eps, t):
    """rsqrt(mean + eps) of token t from the 64 slices' partials at ``region`` [slice][t]."""
    s_rows, lane = c["T"], c["lane"]
    got = c["poll"]([(region, lane * s_rows + t, 1), (region, (lane + C.WAVE) * s_rows + t, 1)])
    tot = wave_sum(got[0][0].bitcast(fx.Float32) + got[1][0].bitcast(fx.Float32))
    return hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(eps))


@traced
def stage_seam(c, s):
    """Post norm, residual mix, pre norm, then XRDY[s]."""
    s_rows, lane, wave = c["T"], c["lane"], c["wave"]
    a = c["args"]
    mine = wave < s_rows
    t = fx.min(wave, s_rows - 1)
    col = s * COLS + LANE_COLS * lane
    e = t * HIDDEN + col
    br = ld_bf4(a["branch_in"], e)
    st = ld_f4(a["stream_in"], e)
    wpf = ld_bf4(a["w_pf"], col)
    al = ld_f4(a["alpha_f"], col)
    be = ld_f4(a["beta_f"], col)
    wqn = ld_bf4(a["w_qn"], col)
    xh = [br[k] * (wpf[k] + fx.Float32(1.0)) for k in range(4)]
    sq = wave_sum(sq4(xh))
    if mine & (lane == 0):
        c["put"](c["p1"], s * s_rows + t, sq)
    r1 = _row_rstd(c, c["p1"], c["post_eps"], t)
    sv = [_fma(be[k], xh[k] * r1, al[k] * st[k]) for k in range(4)]
    if mine:
        bo.buffer_store(fx.Vector.from_elements(sv, fx.Float32), rsrc(a["stream_a"]), e)
    sq2 = wave_sum(sq4(sv))
    if mine & (lane == 0):
        c["put"](c["p2"], s * s_rows + t, sq2)
    r2 = _row_rstd(c, c["p2"], c["eps"], t)
    y = [(sv[k] * r2) * (wqn[k] + fx.Float32(1.0)) for k in range(4)]
    words = fx.Vector.from_elements(y, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    if mine:
        bo.buffer_store(words, rsrc(c["xbuf"]), e // 2, cache_modifier=CM_DEV_ST)
    publish(c["put"], c["xrdy"], s, fx.Int32(1), c["tid"] == 0)
    gpu.barrier()
