"""C1's (q8) K2 seam on gfx1250 at TP4 (port of ``k2bundle/ops/stages/quarter_seam.py``).

AR#1 runs as reduce-scatter + all-gather: CTA ``4 * token + quarter`` sums this rank's 512 columns of the quarter
over the four ranks in rank order, rounds to bf16 and all-gathers the bf16 x and its per-slice (128-column) sums
of squares.  The two norms of a token then run in one CTA, so neither needs a cross-CTA rendezvous.

Lane layout: a wave32 owns a 128-column slice, a lane 4 columns (two bf16 pairs, one 16 B mailbox load).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from . import common as C
from .common import CM_DEV_ST, bf16_pair, bf16_round, bf_hi, bf_lo, drain_stores, hw_rsq, rsrc, traced, wave_sum
from .common import fma as _fma
from .seam_b import COLS, LANE_COLS, H, TP, f32_of, sq4
from port.sync.sync import cta_barrier, stamp

OWNER_COLS = H // TP  # 2048 columns a rank sums
OWNER_PART = OWNER_COLS // 4  # 512 columns a CTA (four CTAs a token)
OWNER_WAVES = OWNER_PART // COLS  # 4 waves a CTA, a slice each
SEAM_SLICES = H // COLS // C.WAVES  # 4 slices a wave (16 waves cover the row)
NSL_ROW = H // COLS  # 64
_SEAM_POLL1 = __import__("os").environ.get("K2_SEAM_POLL1", "1") == "1"  # seam: one poll batch for its 4 slices


@traced
def stage_ar_owner_sum(c, bid):
    """This rank's columns of token ``bid // 4``, quarter ``bid % 4``: the four ranks' bf16 partials summed in rank
    order and rounded to bf16, all-gathered to every rank's XG, with the slice's sum of squares to every XSQ."""
    tid, s = c["tid"], c["S"]
    token, part = bid // 4, bid % 4
    if tid < OWNER_PART // LANE_COLS:
        col = c["rank"] * OWNER_COLS + part * OWNER_PART + LANE_COLS * tid
        ar = c["poll"]([(c["ar_own"], ((src * s + token) * H + col) // 2, 2) for src in range(TP)])
        stamp(c, 23)
        xs, words = [], []
        for p in range_constexpr(2):
            lo, hi = bf_lo(ar[0][p]), bf_hi(ar[0][p])
            for src in range_constexpr(1, TP):
                lo = lo + bf_lo(ar[src][p])
                hi = hi + bf_hi(ar[src][p])
            x0, x1 = bf16_round(lo), bf16_round(hi)
            xs += [x0, x1]
            words.append(bf16_pair(x0, x1).bitcast(fx.Int32))
        for p in range_constexpr(TP):
            c["put_words"](c["peer"]["xg"][p], (token * H + col) // 2, words)
        sq = wave_sum(sq4(xs))
        if c["lane"] == 0:
            j = c["rank"] * (OWNER_COLS // COLS) + (part * OWNER_PART) // COLS + c["wave"]
            for p in range_constexpr(TP):
                c["put"](c["peer"]["xsq"][p], token * NSL_ROW + j, sq)


@traced
def _direct_rows(c, token, cols):
    """K2_SEAM_DIRECT: this lane's columns of token ``token`` summed over the four ranks' AR#1 partials in rank
    order and rounded to bf16 (the owner sum's arithmetic), a slice's sum of squares -> LDS ``red`` + 2 NSL_ROW + j
    (the owner sum's XSQ value); returns the x words (bf16 pairs, as XG holds them)."""
    s, lane, wave, red = c["S"], c["lane"], c["wave"], c["red"]
    words = []
    if const_expr(_SEAM_POLL1):
        specs = [(c["ar_own"], ((src * s + token) * H + cols[i]) // 2, 2) for i in range(SEAM_SLICES)
                 for src in range(TP)]
        allar = c["poll"](specs, len(specs))  # one round trip for the 4 slices x 4 ranks
    for i in range_constexpr(SEAM_SLICES):
        col = cols[i]
        if const_expr(_SEAM_POLL1):
            ar = allar[i * TP:(i + 1) * TP]
        else:
            ar = c["poll"]([(c["ar_own"], ((src * s + token) * H + col) // 2, 2) for src in range(TP)])
        xs, ws = [], []
        for p in range_constexpr(2):
            lo, hi = bf_lo(ar[0][p]), bf_hi(ar[0][p])
            for src in range_constexpr(1, TP):
                lo = lo + bf_lo(ar[src][p])
                hi = hi + bf_hi(ar[src][p])
            x0, x1 = bf16_round(lo), bf16_round(hi)
            xs += [x0, x1]
            ws.append(bf16_pair(x0, x1).bitcast(fx.Int32))
        sq = wave_sum(sq4(xs))
        if lane == 0:
            fx.ptr_store(sq, red + (2 * NSL_ROW + wave * SEAM_SLICES + i))
        words.append(ws)
    return words


@traced
def stage_seam_token(c, token, direct=False):
    """The q8 seam for one token in one CTA.  Wave w, slice j = 4 w + i, lane l: columns 128 j + 4 l .. + 3.
    rstd1 from the owners' slice sums (XSQ); the slice sums of n^2 go through LDS, both totals reduced the same
    way (lane l: slices l and l + 32, added, then the wave32 butterfly)."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    a = c["args"]
    js = [wave * SEAM_SLICES + i for i in range(SEAM_SLICES)]
    cols = [j * COLS + LANE_COLS * lane for j in js]
    es = [token * H + col for col in cols]
    sv = [fx.Vector(bo.buffer_load(rsrc(a["stream_a"]), e, vec_width=4, dtype=T.f32)) for e in es]
    al = [fx.Vector(bo.buffer_load(rsrc(a["alpha_a"]), col, vec_width=4, dtype=T.f32)) for col in cols]
    be = [fx.Vector(bo.buffer_load(rsrc(a["beta_a"]), col, vec_width=4, dtype=T.f32)) for col in cols]
    wf = [fx.Vector(bo.buffer_load(rsrc(a["w_fn"]), col // 2, vec_width=2, dtype=T.i32)) for col in cols]
    if const_expr(direct):
        xs = _direct_rows(c, token, cols)
        cta_barrier()
        if wave == 0:
            total = wave_sum(fx.ptr_load(red + (2 * NSL_ROW + lane)) + fx.ptr_load(red + (2 * NSL_ROW + 32 + lane)))
            if lane == 0:
                fx.ptr_store(hw_rsq(total * fx.Float32(1.0 / H) + fx.Float32(c["eps_attn"])), red)
    else:
        xs = c["poll"]([(c["peer"]["xg_own"], (token * H + col) // 2, 2) for col in cols])
        if wave == 0:
            got = c["poll"](
                [(c["peer"]["xsq_own"], token * NSL_ROW + lane, 1),
                 (c["peer"]["xsq_own"], token * NSL_ROW + lane + 32, 1)]
            )
            total = wave_sum(f32_of(got[0][0]) + f32_of(got[1][0]))
            if lane == 0:
                fx.ptr_store(hw_rsq(total * fx.Float32(1.0 / H) + fx.Float32(c["eps_attn"])), red)
    cta_barrier()
    stamp(c, 22)
    r1 = fx.ptr_load(red)
    ns = []
    for i in range_constexpr(SEAM_SLICES):
        x = [bf_lo(xs[i][0]), bf_hi(xs[i][0]), bf_lo(xs[i][1]), bf_hi(xs[i][1])]
        n = [_fma(al[i][k], sv[i][k], be[i][k] * (x[k] * r1)) for k in range(4)]
        bo.buffer_store(fx.Vector.from_elements(n, fx.Float32), rsrc(a["stream_out"]), es[i])
        sq2 = wave_sum(sq4(n))
        if lane == 0:
            fx.ptr_store(sq2, red + (NSL_ROW + js[i]))
        ns.append(n)
    cta_barrier()
    if wave == 0:
        total2 = wave_sum(fx.ptr_load(red + (NSL_ROW + lane)) + fx.ptr_load(red + (NSL_ROW + 32 + lane)))
        if lane == 0:
            fx.ptr_store(hw_rsq(total2 * fx.Float32(1.0 / H) + fx.Float32(c["eps_ffn"])), red + 1)
    cta_barrier()
    r2 = fx.ptr_load(red + 1)
    for i in range_constexpr(SEAM_SLICES):
        n = ns[i]
        wv = [bf_lo(wf[i][0]), bf_hi(wf[i][0]), bf_lo(wf[i][1]), bf_hi(wf[i][1])]
        y = [(n[k] * r2) * (wv[k] + fx.Float32(1.0)) for k in range(4)]
        xw = fx.Vector.from_elements(y, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        bo.buffer_store(xw, rsrc(c["xbuf"]), es[i] // 2, cache_modifier=CM_DEV_ST)
    # Drain the x rows, then this token's four K-quarter flags.
    drain_stores()
    gpu.barrier()
    if c["tid"] < 4:
        c["put"](c["xrdy"], token * 4 + c["tid"], fx.Int32(1))
    cta_barrier()
    stamp(c, 44)
