"""The dense parts of K2's MoE on gfx1250 at TP4 / EP1 (port of the router, pooler, route, unpooler and gather of
``k2bundle/ops/stages/moe.py``; the names and signatures are the ones ``ops/k2.py`` calls as ``moe.*``).

The router (fp32 [256, 8192]), pooler (bf16 [4096, 8192]) and unpooler (bf16 [8192, 4096]) are replicated on
every GPU; each rank computes its quarter and the results are exchanged over the peer buffer, as on MI355X:

    router  (4 x 16 experts of this rank's 64, K quarters, q8: 2 tokens a task): fp32 VALU GEMV, a wave an
            expert, a lane 4 fp32 every 128 (16 b128 loads, coalesced); per-lane fma chain, wave32 butterfly ->
            the K quarter's partial to every rank's RLOG_PARTS (q8) or summed in part order through RPART -> RLOG
    pooler  (64 x 16 rows of this rank's 1024, K quarters): bf16 WMMA GEMV, one 128-K chunk a wave; parts
            summed in part order through PPART -> bf16 -> every rank's POOL
    route   (token t, wave 0): the 4 parts of each logit summed in part order, x router_output_multiplier,
            sigmoid 1 / (1 + exp2(-x log2 e)), + bias, top-8 by a radix select over 32-bit ballots (equal
            keys: the lower experts), slot order by biased score, weights s_k * (scale / (sum + 1e-15)) -> ROUTE
    unpool  (128 x 16 rows of this rank's 2048 branch columns, K 4096): bf16 WMMA GEMV of FIN, two chunks a
            wave -> bf16 -> every rank's BR (AG#4)
    gather  (64 x 128 columns): BR -> ``branch_out``

``w_pool`` / ``w_unpool`` are read in ``gemv.SHUFFLED`` byte order (preshuffled once at load time,
``gemv.shuffle_rows`` of the full replicated tensor); the router stays row-major (its loads are coalesced).
"""

import math

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from port.sync.sync import cta_barrier, gate, stamp

from . import common as C
from . import gemv
from .common import (
    CM_NT,
    ballot,
    bf16_round,
    bf_hi,
    bf_lo,
    hw_exp2,
    lanes_below,
    popcount,
    row_sum,
    rsrc,
    traced,
    wave_sum,
)

TP = C.TP
H = C.H  # hidden
MH = 4096  # MoE hidden (pooled)
E = 256
TOPK = 8
EPR = E // TP  # 64 router rows a rank
ROWS = 16
LOG2E = 1.0 / math.log(2.0)
ROUTE_EPS = 1e-15  # router_score_division_eps
ROUTE_SCALE = 1.0  # routed_scaling_factor
ROUTE_ARGMAX = __import__("os").environ.get("K2_ROUTE_ARGMAX", "1") == "1"  # top-8 by 8 argmax rounds (0: radix)

R_GROUPS = EPR // C.WAVES  # 4: 16 experts a task, one a wave
R_PARTS = 4  # K quarters
R_K = H // R_PARTS  # 2048
R_LD = R_K // (4 * C.WAVE)  # 16 b128 loads a lane
R_TASKS = R_GROUPS * R_PARTS  # 16
ROUTER_PAIR = 2
R_TOKEN_GROUPS_Q8 = 8 // ROUTER_PAIR  # 4
R_TASKS_Q8 = R_TASKS * R_TOKEN_GROUPS_Q8  # 64
POOL_N = MH // TP  # 1024 pooled columns a rank
P_GROUPS = POOL_N // ROWS  # 64
P_PARTS = 4
P_K = H // P_PARTS  # 2048
P_TASKS = P_GROUPS * P_PARTS  # 256
EARLY_K = R_K  # the x part an early task stages (router and pooler alike)
assert P_K == R_K
FIN_COLS = 128
FIN_TASKS = MH // FIN_COLS  # 32 FRDY flags
UP_N = H // TP  # 2048 branch columns a rank
UP_TASKS = UP_N // ROWS  # 128
GATHER_COLS = 128
GATHER_TASKS = H // GATHER_COLS  # 64
NSL = 64  # 128-column x slices (non-quarter seam XRDY)


def peer_pairs(s):
    """This module's peer regions, pairs: name -> count."""
    return {
        "rlog": TP * s * EPR,  # [src][t][64] fp32
        "rlog_parts": TP * R_PARTS * s * EPR,  # [src][q][t][64] fp32 (q8)
        "pool": s * MH // 2,  # [t][4096] bf16 pairs
        "br": s * H // 2,  # [t][8192] bf16 pairs
    }


def scratch_pairs(s):
    return {
        "rpart": R_TASKS * s * C.WAVES,
        "ppart": P_TASKS * s * ROWS,
        "route": s * 2 * TOPK,
        "frdy": FIN_TASKS,
    }


def f32_of(w):
    return w.bitcast(fx.Float32)


def bo_ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


# ---------------------------------------------------------------- early
@traced
def early_x(c, q):
    """The x rows' K part q -> LDS after that part's slices are XRDY."""
    tid = c["tid"]
    if const_expr(c["quarter_seam"]):
        if tid < c["S"]:
            c["poll"]([(c["xrdy"], tid * R_PARTS + q, 1)], site="xrdy")
    else:
        if tid < NSL // R_PARTS:
            c["poll"]([(c["xrdy"], q * (NSL // R_PARTS) + tid, 1)])
    cta_barrier()
    stamp(c, 27)
    gemv.rows_to_lds(tid, c["xbuf"], R_K, c["S"], c["xl"], R_K + gemv.LDS_PAD, C.CM_XROWS, ldk=H, k0=q * R_K)
    stamp(c, 32)


def router_loads(c, i, q, mask=None, local=False):
    """Router task (i, q): expert rank 64 + 16 i + wave, K quarter q: a lane 4 fp32 every 128 (16 b128).
    ``mask``: a CTA predicate (a masked lane's offset falls outside the sized descriptor: no traffic).
    ``local`` (K2_RLOCAL): expert 16 i + wave of all 256 (i < 16)."""
    lane, wave = c["lane"], c["wave"]
    e = i * C.WAVES + wave if local else c["rank"] * EPR + i * C.WAVES + wave
    r_w = rsrc(c["args"]["w_router"], E * H * 4)
    base = e * H + q * R_K + lane * 4
    return [
        fx.Vector(bo.buffer_load(r_w, base + 4 * C.WAVE * j, vec_width=4, dtype=T.f32, cache_modifier=CM_NT,
                                 mask=mask))
        for j in range(R_LD)
    ]


@traced
def _router_dots(c, ops, t_rows, row0=0, lds_bar=False):
    """Per LDS token row ``row0 + t``: the lane's fp32 fma chain over its 64 K, then the wave's sum ->
    av[t * WAVES + wave]."""
    lane, wave = c["lane"], c["wave"]
    xrow = EARLY_K + gemv.LDS_PAD
    for t in range_constexpr(t_rows):
        acc = fx.Float32(0.0)
        for j in range_constexpr(R_LD):
            xv = fx.Vector(
                fx.ptr_load(
                    c["xl"] + ((row0 + t) * xrow + 4 * C.WAVE * j + lane * 4) // 2,
                    result_type=fx.Vector.make_type(2, fx.Int32),
                )
            )
            w = ops[j]
            acc = acc + bf_lo(xv[0]) * w[0]
            acc = acc + bf_hi(xv[0]) * w[1]
            acc = acc + bf_lo(xv[1]) * w[2]
            acc = acc + bf_hi(xv[1]) * w[3]
        v = wave_sum(acc)
        if lane == 0:
            fx.ptr_store(v, c["av"] + (t * C.WAVES + wave))
    if const_expr(lds_bar):
        cta_barrier()  # LDS only: no wait for this wave's outstanding peer stores (the POOLP pushes)
    else:
        gpu.barrier()


@traced
def router_compute(c, i, q, ops):
    """Router task (i, q), every token: parts q > 0 -> RPART, part 0 adds them in part order -> every rank's
    RLOG."""
    s, tid = c["S"], c["tid"]
    early_x(c, q)
    _router_dots(c, ops, s)
    if tid < s * C.WAVES:
        v = fx.ptr_load(c["av"] + tid)
        at = (i * R_PARTS + q) * s * C.WAVES + tid
        if q > 0:
            c["put"](c["rpart"], at, v)
        if q == 0:
            got = c["poll"]([(c["rpart"], at + qq * s * C.WAVES, 1) for qq in range(1, R_PARTS)])
            for qq in range_constexpr(R_PARTS - 1):
                v = v + f32_of(got[qq][0])
            t = tid // C.WAVES
            w = tid % C.WAVES
            for p in range_constexpr(TP):
                c["put"](c["peer"]["rlog"][p], (c["rank"] * s + t) * EPR + i * C.WAVES + w, v)
    gpu.barrier()


@traced
def router_compute_pair(c, i, q, tg, ops, x8=False):
    """Router task (i, q) for tokens 2 tg, 2 tg + 1 (C1 / q8): each K quarter's partial -> every rank's
    RLOG_PARTS; the route sums the four parts in part order. ``x8``: stage every token's K part q
    (``early_x``), the LDS rows the CTA's pooler task (K quarter q too) reads next (``pool_compute(x_ready)``)."""
    s, tid = c["S"], c["tid"]
    t0 = tg * ROUTER_PAIR
    if const_expr(x8 == 2):  # the CTA's pooler task (K quarter q) staged the 8 rows already
        stamp(c, 41)
        _router_dots(c, ops, ROUTER_PAIR, t0)
    elif const_expr(x8):
        early_x(c, q)
        stamp(c, 41)
        _router_dots(c, ops, ROUTER_PAIR, t0)
    else:
        if tid < ROUTER_PAIR:
            c["poll"]([(c["xrdy"], (t0 + tid) * R_PARTS + q, 1)])
        gpu.barrier()
        stamp(c, 41)
        gemv.rows_to_lds(
            tid, c["xbuf"] + fx.Int64(t0 * (H * 2)), R_K, ROUTER_PAIR, c["xl"], R_K + gemv.LDS_PAD, C.CM_XROWS,
            ldk=H, k0=q * R_K,
        )
        _router_dots(c, ops, ROUTER_PAIR)
    if tid < ROUTER_PAIR * C.WAVES:
        v = fx.ptr_load(c["av"] + tid)
        t = t0 + tid // C.WAVES
        w = tid % C.WAVES
        for p in range_constexpr(TP):
            c["put"](c["peer"]["rlog_parts"][p], ((c["rank"] * R_PARTS + q) * s + t) * EPR + i * C.WAVES + w, v)
    cta_barrier()
    stamp(c, 42)


@traced
def router_local_pair(c, i, q, tg, ops):
    """K2_RLOCAL: router task (i, q, tg) of all 256 experts on every rank: experts 16 i + wave, K quarter q,
    tokens 2 tg, 2 tg + 1 from the CTA's x8 staging (LDS rows already there) -> this rank's RLOGL [q][t][256]
    (device scope): the route's parts need no cross-GPU hop. The per-part dot is the q8 task's."""
    s, tid = c["S"], c["tid"]
    t0 = tg * ROUTER_PAIR
    stamp(c, 41)
    _router_dots(c, ops, ROUTER_PAIR, t0, True)
    if tid < ROUTER_PAIR * C.WAVES:
        v = fx.ptr_load(c["av"] + tid)
        t = t0 + tid // C.WAVES
        w = tid % C.WAVES
        c["put"](c["rlogl"], (q * s + t) * E + i * C.WAVES + w, v)
    cta_barrier()
    stamp(c, 42)


def pool_loads_rep(c, rr, i, q):
    """(K2_POOL_REP) Pooler task (i, q) of rank ``rr``'s column block (rows rr 1024 + 16 i ..), on this rank."""
    return gemv.loads(
        c["lane"], c["wave"], c["args"]["w_pool"], rr * POOL_N + i * ROWS, P_K, ldk=H, k0=q * P_K,
        nbytes=MH * H * 2,
    )


@traced
def pool_partial_rep(c, rr, i, q, ops):
    """(K2_POOL_REP) ``pool_partial``'s POOL_DIRECT arithmetic for rank ``rr``'s column block, the fp32 part
    into this rank's own POOLP only (every rank computes all 4096 pooled columns: no POOLP all-gather)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    acc = gemv.wmmas(lane, wave, c["xl"], P_K + gemv.LDS_PAD, P_K, s, ops)
    gemv.store_acc(red, wave, lane, acc)
    cta_barrier()
    stamp(c, 33)
    r2 = 2 * (tid % (ROWS // 2))
    t = fx.min(tid // (ROWS // 2), s - 1)
    v0 = row_sum(red, r2, t)
    v1 = row_sum(red, r2 + 1, t)
    if tid < (ROWS // 2) * s:
        at = (t * P_PARTS + q) * MH + rr * POOL_N + i * ROWS + r2
        c["put_words"](c["peer"]["poolp_own"], at, [v0.bitcast(fx.Int32), v1.bitcast(fx.Int32)])
    cta_barrier()
    stamp(c, 35)


@traced
def pool_partial_rep4(c, i, q, opss):
    """(K2_POOL_REP_MERGE) ``pool_partial_rep`` of all 4 ranks' blocks in one pass: one B-operand read for the 4
    tiles (``wmmas_multi``: each accumulator sees its own ``wmmas`` sequence), the 4 tiles' sums in ``red4``, one
    barrier, threads 64 j .. 64 j + 63 put block j's parts."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    accs = gemv.wmmas_multi(lane, wave, c["xl"], P_K + gemv.LDS_PAD, P_K, s, opss)
    for j in range_constexpr(TP):
        gemv.store_acc(c["red4"] + j * (C.WAVES * 256), wave, lane, accs[j])
    cta_barrier()
    stamp(c, 33)
    n = (ROWS // 2) * s
    j = fx.min(tid // n, TP - 1)
    tt = tid % n
    red = c["red4"] + j * (C.WAVES * 256)
    r2 = 2 * (tt % (ROWS // 2))
    t = fx.min(tt // (ROWS // 2), s - 1)
    v0 = row_sum(red, r2, t)
    v1 = row_sum(red, r2 + 1, t)
    if tid < TP * n:
        at = (t * P_PARTS + q) * MH + j * POOL_N + i * ROWS + r2
        c["put_words"](c["peer"]["poolp_own"], at, [v0.bitcast(fx.Int32), v1.bitcast(fx.Int32)])
    cta_barrier()
    stamp(c, 35)


@traced
def pool_partial(c, i, q, ops):
    """``pool_compute``'s first half (x part already in LDS): WMMA GEMV, the part's row sums; parts q > 0 ->
    PPART. Returns this thread's row sum (tid < 16 S), which ``pool_finish`` completes."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    acc = gemv.wmmas(lane, wave, c["xl"], P_K + gemv.LDS_PAD, P_K, s, ops)
    gemv.store_acc(red, wave, lane, acc)
    cta_barrier()
    stamp(c, 33)
    if const_expr(c.get("pool_direct", False)):
        # K2_POOL_DIRECT: the part's fp32 row sums (2 rows a thread, one 16 B pair store) to every rank's POOLP
        # [t][q][4096]; xq adds the 4 parts in part order and rounds to bf16 (part 0's arithmetic): no PPART
        r2 = 2 * (tid % (ROWS // 2))
        t = fx.min(tid // (ROWS // 2), s - 1)
        v0 = row_sum(red, r2, t)
        v1 = row_sum(red, r2 + 1, t)
        if tid < (ROWS // 2) * s:
            at = (t * P_PARTS + q) * MH + c["rank"] * POOL_N + i * ROWS + r2
            for p in range_constexpr(TP):
                c["put_words"](c["peer"]["poolp"][p], at, [v0.bitcast(fx.Int32), v1.bitcast(fx.Int32)])
        cta_barrier()
        stamp(c, 35)
        return v0
    v = row_sum(red, tid % ROWS, fx.min(tid // ROWS, s - 1))
    if (q > 0) & (tid < ROWS * s):
        c["put"](c["ppart"], (i * P_PARTS + q) * s * ROWS + tid, v)
    return v


@traced
def pool_finish(c, i, q, v):
    """``pool_compute``'s second half: part 0 adds the PPART parts in part order -> bf16 -> every rank's POOL."""
    s, tid = c["S"], c["tid"]
    if (q == 0) & (tid < ROWS * s):
        at = (i * P_PARTS + q) * s * ROWS + tid
        got = c["poll"]([(c["ppart"], at + qq * s * ROWS, 1) for qq in range(1, P_PARTS)], site="ppart")
        vv = v
        for qq in range_constexpr(P_PARTS - 1):
            vv = vv + f32_of(got[qq][0])
        fx.ptr_store(vv, c["av"] + tid)
    cta_barrier()
    stamp(c, 34)
    if (q == 0) & (tid < (ROWS // 4) * s):
        rq = tid % (ROWS // 4)
        t = tid // (ROWS // 4)
        vs = [bf16_round(fx.ptr_load(c["av"] + (t * ROWS + 4 * rq + d))) for d in range(4)]
        col = c["rank"] * POOL_N + i * ROWS + 4 * rq
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["pool"][p], t * MH + col, vs)
    cta_barrier()
    stamp(c, 35)


def pool_loads(c, i, q, mask=None):
    """Pooler task (i, q): rows rank 1024 + 16 i .., K quarter q (one 128-K chunk a wave)."""
    return gemv.loads(
        c["lane"], c["wave"], c["args"]["w_pool"], c["rank"] * POOL_N + i * ROWS, P_K, ldk=H, k0=q * P_K,
        nbytes=MH * H * 2, mask=mask,
    )


@traced
def pool_compute(c, i, q, ops, x_ready=None):
    """Pooler task (i, q): bf16 WMMA GEMV of the x part; parts q > 0 -> PPART, part 0 adds them in part order
    -> bf16 -> every rank's POOL. ``x_ready`` (traced, CTA-uniform) != 0: the x part is in LDS already."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    if const_expr(x_ready is None):
        early_x(c, q)
    elif const_expr(x_ready is True):
        pass
    else:
        if x_ready == 0:
            early_x(c, q)
    acc = gemv.wmmas(lane, wave, c["xl"], P_K + gemv.LDS_PAD, P_K, s, ops)
    gemv.store_acc(red, wave, lane, acc)
    cta_barrier()
    stamp(c, 33)
    if tid < ROWS * s:
        r = tid % ROWS
        t = tid // ROWS
        v = row_sum(red, r, t)
        at = (i * P_PARTS + q) * s * ROWS + tid
        if q > 0:
            c["put"](c["ppart"], at, v)
        if q == 0:
            got = c["poll"]([(c["ppart"], at + qq * s * ROWS, 1) for qq in range(1, P_PARTS)])
            for qq in range_constexpr(P_PARTS - 1):
                v = v + f32_of(got[qq][0])
            fx.ptr_store(v, c["av"] + tid)
    cta_barrier()
    stamp(c, 34)
    if (q == 0) & (tid < (ROWS // 4) * s):
        rq = tid % (ROWS // 4)
        t = tid // (ROWS // 4)
        vs = [bf16_round(fx.ptr_load(c["av"] + (t * ROWS + 4 * rq + d))) for d in range(4)]
        col = c["rank"] * POOL_N + i * ROWS + 4 * rq
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["pool"][p], t * MH + col, vs)
    cta_barrier()
    stamp(c, 35)


# ---------------------------------------------------------------- route
def _okey(v):
    """An order-preserving uint32 key of an f32 (larger float, larger key)."""
    b = v.bitcast(fx.Int32)
    neg = b < fx.Int32(0)
    return fx.Uint32(neg.select(~b, b ^ fx.Int32(-2147483648)))


@traced
def stage_route(c, t):
    """Token t (wave 0): 256 logits, 8 a lane (expert lane + 32 i) -> x multiplier -> sigmoid (IEEE
    1 / (1 + exp2(-x log2 e))) + bias -> the top-8 by a radix select (equal keys: the lower experts) -> ranked
    by biased score, descending -> the unbiased sigmoids times scale / (their slot-order sum + eps) -> ROUTE."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    a = c["args"]
    use_parts = c.get("rlog_parts", False)
    use_local = c.get("rlog_local", False)
    mult = c["router_output_multiplier"]
    if wave == 0:
        s = c["S"]
        per = E // C.WAVE  # 8
        bias = [bo_ld_f32(a["expert_bias"], lane + C.WAVE * i) for i in range(per)]
        specs = []
        if const_expr(use_local):
            for i in range_constexpr(per):
                e = lane + C.WAVE * i
                for qq in range_constexpr(R_PARTS):
                    specs.append((c["rlogl"], (qq * s + t) * E + e, 1))
            parts = c["poll"](specs, batch=len(specs), site="rlog")
            stamp(c, 31)
            got = []
            for i in range_constexpr(per):
                v = f32_of(parts[i * R_PARTS][0])
                for qq in range_constexpr(1, R_PARTS):
                    v = v + f32_of(parts[i * R_PARTS + qq][0])
                got.append([v.bitcast(fx.Int32)])
        elif const_expr(use_parts):
            for i in range_constexpr(per):
                e = lane + C.WAVE * i
                for qq in range_constexpr(R_PARTS):
                    specs.append(
                        (c["peer"]["rlog_parts_own"], (((e // EPR) * R_PARTS + qq) * s + t) * EPR + e % EPR, 1)
                    )
            parts = c["poll"](specs, batch=len(specs))  # one round trip (64 VGPRs of pairs)
            stamp(c, 31)
            got = []
            for i in range_constexpr(per):
                v = f32_of(parts[i * R_PARTS][0])
                for qq in range_constexpr(1, R_PARTS):
                    v = v + f32_of(parts[i * R_PARTS + qq][0])
                got.append([v.bitcast(fx.Int32)])
        else:
            for i in range_constexpr(per):
                e = lane + C.WAVE * i
                specs.append((c["peer"]["rlog_own"], ((e // EPR) * s + t) * EPR + e % EPR, 1))
            got = c["poll"](specs)
        keys, sigs = [], []
        for i in range_constexpr(per):
            lg = f32_of(got[i][0])
            if const_expr(mult != 1.0):
                lg = lg * fx.Float32(mult)
            sg = fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(lg * fx.Float32(-LOG2E)))
            sc = sg + bias[i]
            keys.append(_okey((sc == sc).select(sc, fx.Float32(-3.0e38))))  # noqa: PLR0124
            sigs.append((sg == sg).select(sg, fx.Float32(0.0)))  # noqa: PLR0124
        if const_expr(ROUTE_ARGMAX):
            _route_argmax(c, t, keys, sigs)
        else:
            _route_radix(c, t, keys, sigs)
    cta_barrier()
    stamp(c, 45)


@traced
def _route_argmax(c, t, keys, sigs):
    """Top-8 as 8 wave argmax rounds over (key desc, expert asc): round r's winner is slot r, the order the
    radix path ranks by; the weights' sum runs in slot order as there."""
    lane, red = c["lane"], c["red"]
    per = len(keys)
    removed = fx.Int32(-2147483648)
    sk = [fx.Int32(keys[i] ^ fx.Uint32(0x80000000)) for i in range(per)]
    for r in range_constexpr(TOPK):
        bk = sk[0]
        bi = fx.Int32(0)
        bs = sigs[0]
        for i in range_constexpr(1, per):
            better = sk[i] > bk
            bk = better.select(sk[i], bk)
            bi = better.select(fx.Int32(i), bi)
            bs = better.select(sigs[i], bs)
        wk = C.butterfly(bk, C.WAVE_OFFSETS, fx.max)
        me = lane + C.WAVE * bi
        we = C.butterfly((bk == wk).select(me, fx.Int32(1 << 20)), C.WAVE_OFFSETS, fx.min)
        win = me == we
        if win:
            fx.ptr_store(fx.Int32(me).bitcast(fx.Float32), red + 2 * r)
            fx.ptr_store(bs, red + (2 * r + 1))
        for i in range_constexpr(per):
            sk[i] = (win & (bi == i)).select(removed, sk[i])
    q = fx.min(lane, TOPK - 1)
    e = fx.ptr_load(red + 2 * q).bitcast(fx.Int32)
    ms = fx.ptr_load(red + (2 * q + 1))
    total = fx.Float32(0.0)
    for j in range_constexpr(TOPK):
        total = total + fx.ptr_load(red + (2 * j + 1))
    inv = fx.Float32(ROUTE_SCALE) / (total + fx.Float32(ROUTE_EPS))
    if lane < TOPK:
        c["put"](c["route"], t * 2 * TOPK + lane, fx.min(fx.max(e, 0), fx.Int32(E - 1)))
        c["put"](c["route"], t * 2 * TOPK + TOPK + lane, ms * inv)


@traced
def _route_radix(c, t, keys, sigs):
    lane, red = c["lane"], c["red"]
    per = len(keys)

    def count_ge(th):
        n = fx.Int32(0)
        for i in range_constexpr(per):
            n = n + popcount(ballot(keys[i] >= th))
        return n

    th = fx.Uint32(0)
    for bit in range_constexpr(31, -1, -1):
        cand = th | fx.Uint32(1 << bit)
        th = (count_ge(cand) >= TOPK).select(cand, th)
    n_gt = fx.Int32(0)
    for i in range_constexpr(per):
        n_gt = n_gt + popcount(ballot(keys[i] > th))
    need = fx.Int32(TOPK) - n_gt
    eq_before = fx.Int32(0)
    sel_before = fx.Int32(0)
    for i in range_constexpr(per):
        eqb = ballot(keys[i] == th)
        eq_rank = eq_before + lanes_below(eqb, lane)
        sel = (keys[i] > th) | ((keys[i] == th) & (eq_rank < need))
        selb = ballot(sel)
        pos = sel_before + lanes_below(selb, lane)
        if sel:
            fx.ptr_store(keys[i].bitcast(fx.Float32), red + pos * 3)
            fx.ptr_store(fx.Int32(lane + C.WAVE * i).bitcast(fx.Float32), red + (pos * 3 + 1))
            fx.ptr_store(sigs[i], red + (pos * 3 + 2))
        eq_before = eq_before + popcount(eqb)
        sel_before = sel_before + popcount(selb)
    q = fx.min(lane, TOPK - 1)
    mk = fx.Uint32(fx.ptr_load(red + q * 3).bitcast(fx.Int32))
    me = fx.ptr_load(red + (q * 3 + 1)).bitcast(fx.Int32)
    ms = fx.ptr_load(red + (q * 3 + 2))
    rank = fx.Int32(0)
    for j in range_constexpr(TOPK):
        ok = fx.Uint32(fx.ptr_load(red + j * 3).bitcast(fx.Int32))
        oe = fx.ptr_load(red + (j * 3 + 1)).bitcast(fx.Int32)
        before = (ok > mk) | ((ok == mk) & (oe < me))
        rank = rank + before.select(fx.Int32(1), fx.Int32(0))
    if lane < TOPK:
        fx.ptr_store(ms, red + (64 + rank))
    total = fx.Float32(0.0)
    for j in range_constexpr(TOPK):
        total = total + fx.ptr_load(red + (64 + j))
    inv = fx.Float32(ROUTE_SCALE) / (total + fx.Float32(ROUTE_EPS))
    if lane < TOPK:
        c["put"](c["route"], t * 2 * TOPK + rank, fx.min(fx.max(me, 0), fx.Int32(E - 1)))
        c["put"](c["route"], t * 2 * TOPK + TOPK + rank, ms * inv)


# ---------------------------------------------------------------- unpooler / gather
def up_loads(c, j):
    """Unpooler task j: rows (branch columns) rank 2048 + 16 j .., K 4096 (two chunks a wave)."""
    return gemv.loads(c["lane"], c["wave"], c["args"]["w_unpool"], c["rank"] * UP_N + j * ROWS, MH, nbytes=H * MH * 2)


def up_loads_masked(c, j, live):
    """``up_loads`` with a CTA predicate, for issue ahead of FRDY: the same addresses (clamped task) and cache
    policy; a masked lane's offset falls outside the sized descriptor."""
    jj = fx.max(fx.min(j, UP_TASKS - 1), 0)
    return gemv.loads(
        c["lane"], c["wave"], c["args"]["w_unpool"], c["rank"] * UP_N + jj * ROWS, MH, mask=live, nbytes=H * MH * 2
    )


def _fin_pairs_to_lds(c, s, lrow):
    """FIN rows (tagged bf16 pairs, [t][MH]) -> LDS ``xl`` (``lrow`` a row): 8 bf16 (4 pairs) a thread a round."""
    tid = c["tid"]
    rounds = s * MH // 8 // C.THREADS
    specs = []
    for i in range(rounds):
        e = tid + C.THREADS * i
        p0 = (e // (MH // 8) * MH + e % (MH // 8) * 8) // 2
        specs += [(c["finp"], p0, 2), (c["finp"], p0 + 2, 2)]
    got = c["poll"](specs, len(specs), "finld")
    for i in range(rounds):
        e = tid + C.THREADS * i
        t = e // (MH // 8)
        kk = e % (MH // 8) * 8
        fx.ptr_store(fx.Vector.from_elements(got[2 * i] + got[2 * i + 1], fx.Int32), c["xl"] + (t * lrow + kk) // 2)
    gpu.barrier()


@traced
def stage_up(c, j, ops):
    """Branch columns rank 2048 + 16 j ..: FIN (after every FRDY) GEMV -> bf16 -> every rank's BR."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    lrow = MH + gemv.LDS_PAD
    if const_expr(c.get("fin_pairs", False)):
        # gate: wave 0, lane l = finalize slice l (128 columns) of every token; then every thread its FIN pairs
        gate(c, [(c["finp"], (m * MH + FIN_COLS * lane) // 2, 1) for m in range(s)], wave == 0)
        stamp(c, 29)
        _fin_pairs_to_lds(c, s, lrow)
    else:
        if tid < c.get("frdy_flags", FIN_TASKS):
            c["poll"]([(c["frdy"], tid, 1)])
        gpu.barrier()
        stamp(c, 29)
        gemv.dev_rows_to_lds(tid, c["fin"], MH, s, c["xl"], lrow)
    stamp(c, 39)
    acc = gemv.wmmas(lane, wave, c["xl"], lrow, MH, s, ops)
    gemv.store_acc(red, wave, lane, acc)
    gpu.barrier()
    if tid < (ROWS // 4) * s:
        rq = tid % (ROWS // 4)
        tt = tid // (ROWS // 4)
        vs = [bf16_round(row_sum(red, 4 * rq + d, tt)) for d in range(4)]
        col = c["rank"] * UP_N + j * ROWS + 4 * rq
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["br"][p], tt * H + col, vs)
    cta_barrier()
    stamp(c, 40)


UP_ALL_TASKS = H // ROWS  # 512: every branch column on every rank (K2_UP_ALL)
UP_MERGE = __import__("os").environ.get("K2_UP_MERGE", "0") == "1"  # K2_UP_ALL: both tasks' WMMAs, one barrier
# k2v7: each wave reads only the FIN K slices its WMMAs use (2 x 128 columns of the 8 rows) straight into its B operands:
# no 128 KB FIN -> LDS staging and no CTA barrier before the GEMV. "gate": wave 0's FIN gate first (then every pair is
# there at the first load); "spin": no gate, every wave polls its own slices (site "finwp"), so a wave starts as soon
# as its slices landed. Same WMMA sequence per task as ``gemv.wmmas``: bit-identical.
UP_WAVEPOLL = __import__("os").environ.get("K2_UP_WAVEPOLL", "0")


def up_loads_all(c, g):
    """K2_UP_ALL task g: branch columns 16 g .. of all 8192 (rank r's task j of ``up_loads`` is g = 128 r + j:
    the same weight rows, chunks and wave assignment)."""
    return gemv.loads(c["lane"], c["wave"], c["args"]["w_unpool"], g * ROWS, MH, nbytes=H * MH * 2)


UP_WP_QS = int(__import__("os").environ.get("K2_UP_WP_QS", "2"))  # B operands (q) a poll batch: 4 = one batch a chunk


def _fin_chunk_operands(c, i, site):
    """This wave's B operands (q = 0..3) of its i-th K chunk of the unpool GEMV, polled from FINP: the values
    ``gemv.x_chunk_operand`` would read from the LDS staging."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    per, n = gemv.chunks(MH, C.WAVES)
    assert per * C.WAVES == n
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    kb = (wave + C.WAVES * i) * gemv.KCH
    specs = []
    for q in range_constexpr(4):
        if const_expr(gemv.INTERLEAVE):
            pieces = [kb + 32 * q + 8 * g, kb + 32 * q + 8 * g + 16]
        else:
            pieces = [kb + 64 * g + 16 * q, kb + 64 * g + 16 * q + 8]
        for p in pieces:
            specs += [(c["finp"], (t * MH + p) // 2, 2), (c["finp"], (t * MH + p) // 2 + 2, 2)]
    got = []
    for b in range_constexpr(4 // UP_WP_QS):
        got += c["poll"](specs[16 * b // (4 // UP_WP_QS) * 0 + 4 * UP_WP_QS * b: 4 * UP_WP_QS * (b + 1)], 4 * UP_WP_QS, site)
    return [fx.Vector.from_elements([w for j in range(4) for w in got[4 * q + j]], fx.Int32) for q in range(4)]


@traced
def stage_up_all_wp(c, gs, opss):
    """``stage_up_all`` with the FIN rows read a wave at a time (``K2_UP_WAVEPOLL``); epilogues unchanged."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    if const_expr(UP_WAVEPOLL == "gate"):
        gate(c, [(c["finp"], (m * MH + FIN_COLS * lane) // 2, 1) for m in range(s)], wave == 0, "fin")
    stamp(c, 29)
    per, _ = gemv.chunks(MH, C.WAVES)
    accs = [fx.Vector.filled(8, 0.0, fx.Float32) for _ in gs]
    for i in range_constexpr(per):
        xbs = _fin_chunk_operands(c, i, "finld" if UP_WAVEPOLL == "gate" else "finwp")
        for q in range_constexpr(4):
            accs = [C.wmma_bf16(opss[j][i][q], xbs[q], accs[j]) for j in range(len(gs))]
    stamp(c, 39)
    bout = rsrc(c["args"]["branch_out"])
    for i in range_constexpr(len(gs)):
        gemv.store_acc(red, wave, lane, accs[i])
        cta_barrier()
        if tid < (ROWS // 4) * s:
            rq = tid % (ROWS // 4)
            tt = tid // (ROWS // 4)
            vs = [bf16_round(row_sum(red, 4 * rq + d, tt)) for d in range(4)]
            col = gs[i] * ROWS + 4 * rq
            words = [C.bf16_pair(vs[0], vs[1]).bitcast(fx.Int32), C.bf16_pair(vs[2], vs[3]).bitcast(fx.Int32)]
            bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), bout, (tt * H + col) // 2)
        cta_barrier()
    stamp(c, 40)


@traced
def stage_up_all(c, gs, opss):
    """K2_UP_ALL: this CTA's tasks ``gs`` (branch columns 16 g ..) of every rank's columns from the all-gathered
    FIN rows (one LDS staging for all tasks), each task's sums as in ``stage_up``, the bf16 words stored straight
    to ``branch_out`` (the words ``stage_gather`` copied from BR): no BR exchange, no gather."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    lrow = MH + gemv.LDS_PAD
    gate(c, [(c["finp"], (m * MH + FIN_COLS * lane) // 2, 1) for m in range(s)], wave == 0, "fin")
    stamp(c, 29)
    _fin_pairs_to_lds(c, s, lrow)
    stamp(c, 39)
    bout = rsrc(c["args"]["branch_out"])
    if const_expr(UP_MERGE):
        # one B-operand read and one barrier for all tasks; task i's row sums on wave i
        reds = [red, c["red2"]]
        accs = gemv.wmmas_multi(lane, wave, c["xl"], lrow, MH, s, opss)
        for i in range_constexpr(len(gs)):
            gemv.store_acc(reds[i], wave, lane, accs[i])
        cta_barrier()
        for i in range_constexpr(len(gs)):
            ti = tid - C.WAVE * i
            if (ti >= 0) & (ti < (ROWS // 4) * s):
                rq = ti % (ROWS // 4)
                tt = ti // (ROWS // 4)
                vs = [bf16_round(row_sum(reds[i], 4 * rq + d, tt)) for d in range(4)]
                col = gs[i] * ROWS + 4 * rq
                words = [C.bf16_pair(vs[0], vs[1]).bitcast(fx.Int32), C.bf16_pair(vs[2], vs[3]).bitcast(fx.Int32)]
                bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), bout, (tt * H + col) // 2)
        cta_barrier()
        stamp(c, 40)
        return
    for i in range_constexpr(len(gs)):
        g = gs[i]
        acc = gemv.wmmas(lane, wave, c["xl"], lrow, MH, s, opss[i])
        gemv.store_acc(red, wave, lane, acc)
        cta_barrier()
        if tid < (ROWS // 4) * s:
            rq = tid % (ROWS // 4)
            tt = tid // (ROWS // 4)
            vs = [bf16_round(row_sum(red, 4 * rq + d, tt)) for d in range(4)]
            col = g * ROWS + 4 * rq
            words = [C.bf16_pair(vs[0], vs[1]).bitcast(fx.Int32), C.bf16_pair(vs[2], vs[3]).bitcast(fx.Int32)]
            bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), bout, (tt * H + col) // 2)
        cta_barrier()
    stamp(c, 40)


@traced
def stage_gather(c, j):
    """Branch columns 128 j ..: BR (every rank's columns) -> ``branch_out``; wave = token, 4 columns a lane."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    live = wave < s
    t = fx.min(wave, s - 1)
    col = j * GATHER_COLS + 4 * lane
    w = c["poll"]([(c["peer"]["br_own"], (t * H + col) // 2, 2)])[0]
    stamp(c, 30)
    if live:
        bo.buffer_store(fx.Vector.from_elements(w, fx.Int32), rsrc(c["args"]["branch_out"]), (t * H + col) // 2)
    cta_barrier()
