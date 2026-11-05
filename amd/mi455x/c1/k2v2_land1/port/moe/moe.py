"""gfx1250 (MI455X) port of K2's MoE stage (``k2bundle/ops/stages/moe.py``) at TP4 / EP1, 8 verify rows.

Scope: the routed-expert half of the MoE block, from the replicated pooled rows (POOL) and the route table
(ROUTE) to the finalized MoE rows replicated on every rank (FIN), the input of the unpooler:

    xq      (token t on CTA t)  POOL row -> pre_expert_norm -> bf16 xe -> MX activations (``mxq``) -> XQ / XSC
    route   (every expert CTA)  ROUTE ids -> the union U in LDS: expert_of[slot] (ascending id), kof[slot][t]
                                (token t's top-k index of that expert, -1 if not picked), |U|
    ug      (slot, 32-col group) a wave a 16-col tile of this rank's 1024-wide slice: gate and up chains
                                over K = 4096 (V_WMMA_SCALE_F32_16X16X128_F8F6F4, weights FP4, acts FP4/FP8);
                                silu(g) u -> bf16 -> MX requant per 32-group -> INTER / ISC of each pick
    down    (slot, 16-row MH tiles) a wave a tile: one chain over the rank's K = 1024 -> bf16 partial per
                                route (unweighted) -> YR[src = rank][t][k] on row t's owner (rank t // 2)
    fin     (owned token, 128-col slice; a wave a route): the 4 ranks' partials in rank order (fp32) -> bf16 y;
                                v = bf16(y bf16(w_post + 1)); per-slice sum of v^2 -> FMSQ; the 32 slices'
                                sums -> sc_k = w_k / sqrt(mean + eps); fin = bf16(sum_k sc_k v_k), k in order
    fin_gather                  the owner's 2 FIN rows -> every rank's FIN (the all-gather)

Not in this module (other helpers of the parallel port): router / pooler GEMVs and the route stage (their
outputs ROUTE / POOL are this stage's inputs), the unpooler and the branch gather.

Hand-offs. In the persistent K2 every arrow above is a tagged mailbox put / poll (device scope inside a GPU,
system scope for YR and FIN). The sync primitives (``ep1/port/sync``) did not exist when this was written, so
each stage is its own launch here and a launch boundary stands in for each poll; the stage functions take the
same context dict ``c`` as the MI355X stages and touch every hand-off through ``c`` entries named after the
MI355X regions (``xqb``, ``xscb``, ``interb``, ``iscb``, ``yr_owner``, ``fmsq``, ``fin``), so the mailbox can be
substituted without changing the arithmetic.

Layers 0 and 61 have BF16 experts: this stage is MXFP4-only and the host refuses them (``check_layer``); they
stay on the served BF16 path, as on MI355X.
"""

import functools
import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mxq as Q  # noqa: E402
from port.common.ops import traced  # noqa: E402
from port.sync.sync import cta_barrier, gate, stamp  # noqa: E402
from port.sync.memmodel import CM_SYS as _CM_SYS, bf16_pair as _bf16_pair, rsrc as _prsrc  # noqa: E402
from port.sync.sync import _noted  # noqa: E402

TP = 4
MAX_TOKENS = 8
H = 8192
MH = 4096  # pooled expert hidden
E = 256
TOPK = 8
I_FULL = 4096
RI = I_FULL // TP  # 1024: the rank's slice of every expert's intermediate (EP1)
ROWS = 16  # WMMA tile rows / columns
W13_STEPS = MH // 128  # 32
W2_STEPS = RI // 128  # 8
W13_TILES = RI // ROWS  # 64 gate (and 64 up) tiles a slice
DOWN_TILES = MH // ROWS  # 256
UG_GROUPS = RI // 32  # 32 32-column groups a slice
MAX_U = MAX_TOKENS * TOPK  # 64
OWN = MAX_TOKENS // TP  # 2 rows a rank owns
FIN_COLS = 128
FIN_SLICES = MH // FIN_COLS  # 32
XQ_THREADS = MH // 8  # 512
FIN_THREADS = 32 * TOPK  # 256: a wave a route
BF16_LAYERS = (0, 61)
REGIONS = ("xqb", "xscb", "interb", "iscb", "yr", "fmsq", "fin")


def check_layer(layer):
    if layer in BF16_LAYERS:
        raise ValueError(f"layer {layer} has BF16 experts: not an MXFP4 K2 layer (served BF16 path)")


def xq_words(act):
    """dwords an activation row of MH holds."""
    return MH // 8 * Q.words_per8(act)


def inter_words(act):
    return RI // 8 * Q.words_per8(act)


ISC_PS_L2 = 256  # K2 with L2-cached activation reads: a pick's codes on their own cache lines


def isc_ps(c):
    """Bytes between two picks' INTER codes."""
    return c.get("isc_ps", RI // 32)


def picks(c):
    """INTER / ISC rows: the picks, plus (L2-cached reads) a trash row the unpicked lanes of a down task read,
    so no lane pulls another slot's not-yet-written line into its L2."""
    return c["S"] * TOPK + (1 if c.get("trash_pick") else 0)


def scratch_plain(s, act, l2=False):
    """Per-rank plain buffers (bytes), the MI355X ``scratch_plain`` names."""
    n = s * TOPK + (1 if l2 else 0)
    return {
        "xqb": s * xq_words(act) * 4,  # xe activations [t][MH] (E2M1 pairs or E4M3)
        "xscb": s * MH // 32,  # xe E8M0 codes [t][MH / 32], a byte each
        "interb": n * inter_words(act) * 4,  # a2 [pick][RI]
        "iscb": n * (ISC_PS_L2 if l2 else RI // 32),  # a2 codes [pick][RI / 32]
        "fmsq": s * TOPK * FIN_SLICES * 4,  # per-slice sums of v^2 [t][k][slice] fp32 (owned rows)
    }


def peer_plain(s):
    """Per-rank peer-visible buffers (bytes): what other ranks write into."""
    return {
        "yr": TP * s * TOPK * MH * 2,  # [src][t][k][MH] bf16 partials (owned rows only are written)
        "fin": s * MH * 2,  # [t][MH] bf16, every row (the all-gather)
    }


def _rsrc(addr, nbytes):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def _ld(ptr, nbytes, i, n=1, cm=0):
    v = bo.buffer_load(_rsrc(ptr, nbytes), i, vec_width=n, dtype=T.i32, cache_modifier=cm)
    return fx.Int32(v) if n == 1 else fx.Vector(v)


def _k2(c):
    """Inside the persistent K2 (``c`` carries the mailbox): hand-offs are tagged polls and published flags;
    otherwise (this module's one-stage launchers) a launch boundary stands in for each."""
    return "poll" in c


_DST = 0 if __import__("os").environ.get("K2_PUB_FENCE", "1") == "1" else (2 << 3)  # write-through to device scope when publish skips the L2 write-back

# K2 xq: XQ / XSC stored write-through to device scope, then XQRDY without the L2 write-back (K2_XQ_WT=1)
_XQ_WT = os.environ.get("K2_XQ_WT", "0") == "1"
_XQ_DST = (2 << 3) | 3 if _XQ_WT else _DST
_DEBUG_XQ_DELAY = int(os.environ.get("K2_DEBUG_XQ_DELAY", "0"))  # experiment: XQRDY published this many 10 ns later


@traced
def _debug_delay(ticks):
    from port.sync.memmodel import memrealtime

    t0 = memrealtime()
    now = memrealtime()
    while now - t0 < ticks:
        rocdl.s_sleep(2)
        now = memrealtime()


def _publish(c, a, i):
    from port.sync.sync import publish

    publish(c["put"], a, i, 1, c["tid"] == 0)


@traced
def _publish_wt(c, a, i):
    """``publish`` for data stored write-through at the flag's scope: the store drain and the CTA barrier order
    it before the flag, no L2 write-back."""
    rocdl.s_wait_storecnt(0)
    gpu.barrier()
    if c["tid"] == 0:
        c["put"](a, i, 1)


def ug_store_cm(c):
    """Cache policy of the INTER / ISC stores: write-through to device scope when UGRDY skips the L2 write-back."""
    return 2 << 3 if c.get("ug_wt") else _DST


@traced
def _put_if(c, cond, a, i, v):
    if cond:
        c["put"](a, i, v)


@traced
def _put_bf_if(c, cond, peers, i, vs):
    if cond:
        for p in range_constexpr(len(peers)):
            c["put_bf"](peers[p], i, vs)


@traced
def _poll_if(c, cond, a, i):
    if cond:
        c["poll"]([(a, i, 1)])


def wait_xq(c):
    """K2: every token's XQ / XSC announced (XQRDY), before the first ``stage_ug``."""
    _poll_if(c, c["tid"] < c["S"], c["xqrdy"], fx.min(c["tid"], c["S"] - 1))
    gpu.barrier()


@traced
def _yr_put(c, picked, owner, el, acc):
    """K2 down: route (t, k)'s 8 bf16 of this lane as tagged pairs into row t's owner's YR."""
    for p in range_constexpr(TP):
        if picked & (owner == p):
            for h in range_constexpr(2):
                c["put_bf"](c["yr_owner"][p], el + 4 * h, [acc[4 * h + v] for v in range(4)])


# ---------------------------------------------------------------- xq
def stage_xq(c, t):
    """Token t (512 threads, a thread 8 columns): POOL row -> pre_expert_norm (sum of squares: a thread's 8 in
    order, the wave butterfly, the 16 waves in order; r = 1 / sqrt(mean + eps); xe = bf16((x r) bf16(w + 1)))
    -> MX activations (4 threads a 32-group) -> XQ words, XSC codes."""
    s, tid, lane, wave, red, act = c["S"], c["tid"], c["lane"], c["wave"], c["red"], c["act"]
    a = c["args"]
    wp = Q.words_per8(act)
    c0 = tid * 8
    xs, ws = [], []
    if const_expr(_k2(c) and c.get("pool_direct", False)):
        # K2_POOL_DIRECT: the 4 pool parts' fp32 sums of columns c0 .. c0 + 7 (POOLP [t][q][4096] pairs), added
        # in part order and rounded to bf16 as the pooler's part 0 did
        specs = [(c["peer"]["poolp_own"], (t * 4 + q) * MH + c0 + 2 * h, 2) for q in range(4) for h in range(4)]
        pp = c["poll"](specs, len(specs))
        stamp(c, 28)
        for k in range(8):
            v = pp[k // 2][k % 2].bitcast(fx.Float32)
            for q in range(1, 4):
                v = v + pp[4 * q + k // 2][k % 2].bitcast(fx.Float32)
            xs.append(Q.bf16_round(v))
    elif _k2(c):
        pw = c["poll"]([(c["peer"]["pool_own"], (t * MH + c0) // 2, 2), (c["peer"]["pool_own"], (t * MH + c0) // 2 + 2, 2)])
        stamp(c, 28)
        xw = pw[0] + pw[1]
        for d in range(4):
            xs += [Q.bf_lo(xw[d]), Q.bf_hi(xw[d])]
    else:
        xw = _ld(a["pool"], s * MH * 2, (t * MH + c0) // 2, 4)
        for d in range(4):
            xs += [Q.bf_lo(xw[d]), Q.bf_hi(xw[d])]
    ww = _ld(a["w_pre"], MH * 2, c0 // 2, 4)
    for d in range(4):
        ws += [Q.bf_lo(ww[d]), Q.bf_hi(ww[d])]
    sq = fx.Float32(0.0)
    for x in xs:
        sq = sq + Q.opaque(x * x)
    sq = Q.xsum(sq)
    fx.ptr_store(sq, red + wave)
    gpu.barrier()
    stamp(c, 46)
    tot = fx.ptr_load(red + 0)
    for w in range(1, XQ_THREADS // 32):
        tot = tot + fx.ptr_load(red + w)
    r = Q.inv_sqrt_ieee(tot * fx.Float32(1.0 / MH) + fx.Float32(c["eps_pre"]))
    xe = [Q.bf16_round((xs[i] * r) * Q.bf16_round(ws[i] + fx.Float32(1.0))) for i in range(8)]
    am = abs(xe[0])
    for v in xe[1:]:
        am = fx.max(am, abs(v))
    am = Q.xmax(am, (2, 1))
    code = Q.code(am, act)
    words = Q.pack8(xe, code, act)
    out = words[0] if wp == 1 else fx.Vector.from_elements(words, fx.Int32)
    dst = _XQ_DST if _k2(c) else _DST
    bo.buffer_store(out, _rsrc(c["xqb"], s * xq_words(act) * 4), t * xq_words(act) + tid * wp, cache_modifier=dst)
    bo.buffer_store(fx.Int8(code), _rsrc(c["xscb"], s * MH // 32), t * (MH // 32) + tid // 4, mask=(lane % 4) == 0, cache_modifier=dst)
    if _k2(c):
        stamp(c, 47)
        if const_expr(_DEBUG_XQ_DELAY > 0):
            _debug_delay(_DEBUG_XQ_DELAY)
        if const_expr(_XQ_WT):
            from port.sync.sync import publish_wt

            publish_wt(c["put"], c["xqrdy"], t, 1, c["tid"] == 0)
        else:
            _publish(c, c["xqrdy"], t)
        stamp(c, 43)


# ---------------------------------------------------------------- route table
def load_route(c):
    """ROUTE ids -> the union U in LDS (``c['rt']``): ``expert`` [slot] (distinct experts, ascending id; slot
    MAX_U is a trash slot), ``kof`` [slot][t] (-1: t did not pick it); returns |U| (wave-uniform).
    Needs >= S * TOPK threads; branch-free (spare threads repeat thread S TOPK - 1's identical stores)."""
    s, tid, rt = c["S"], c["tid"], c["rt"]
    n = s * TOPK
    i = fx.min(tid, n - 1)
    if _k2(c):
        e = c["poll"]([(c["route"], (i // TOPK) * (2 * TOPK) + i % TOPK, 1)])[0][0]
        stamp(c, 50)
    else:
        e = _ld(c["args"]["route"], s * 2 * TOPK * 4, (i // TOPK) * (2 * TOPK) + i % TOPK)
    e = fx.max(fx.min(e, fx.Int32(E - 1)), fx.Int32(0))
    fx.ptr_store(e, rt["ids"] + i)
    nk = (MAX_U + 1) * 8
    for q in range((nk + c["threads"] - 1) // c["threads"]):
        fx.ptr_store(fx.Int32(-1), rt["kof"] + fx.min(tid + c["threads"] * q, nk - 1))
    gpu.barrier()
    first = fx.Int32(1)
    for j in range(n):
        dup = (fx.ptr_load(rt["ids"] + j) == e) & (fx.Int32(j) < i)
        first = dup.select(fx.Int32(0), first)
    fx.ptr_store(first, rt["fl"] + i)
    gpu.barrier()
    slot = fx.Int32(0)
    nu = fx.Int32(0)
    for j in range(n):
        fj = fx.ptr_load(rt["fl"] + j)
        nu = nu + fj
        slot = slot + (fx.ptr_load(rt["ids"] + j) < e).select(fj, fx.Int32(0))
    fx.ptr_store(e, rt["expert"] + (first == 1).select(slot, fx.Int32(MAX_U)))
    fx.ptr_store(i % TOPK, rt["kof"] + (slot * 8 + i // TOPK))
    gpu.barrier()
    return Q.uniform(nu)


def load_route_fast(c):
    """``load_route``'s table (same stores) without the 2 x 64 serial LDS reads a thread (K2_RT_FAST): the 64
    entries mark a 256-bit expert bitmap and their expert's first entry (LDS atomics); slot = the distinct
    experts below e (popcounts), |U| = all bits, first = the entry is its expert's lowest."""
    s, tid, rt = c["S"], c["tid"], c["rt"]
    n = s * TOPK
    i = fx.min(tid, n - 1)
    e = c["poll"]([(c["route"], (i // TOPK) * (2 * TOPK) + i % TOPK, 1)])[0][0]
    stamp(c, 50)
    e = fx.max(fx.min(e, fx.Int32(E - 1)), fx.Int32(0))
    nk = (MAX_U + 1) * 8
    for q in range((nk + c["threads"] - 1) // c["threads"]):
        fx.ptr_store(fx.Int32(-1), rt["kof"] + fx.min(tid + c["threads"] * q, nk - 1))
    fx.ptr_store(fx.Int32(n), rt["own"] + fx.min(tid, E - 1))
    fx.ptr_store(fx.Int32(0), rt["bm"] + fx.min(tid, E // 32 - 1))
    cta_barrier()
    _rt_mark(rt, tid < n, e, i)
    cta_barrier()
    first = fx.ptr_load(rt["own"] + e) == i
    wsel = e >> 5
    nu = fx.Int32(0)
    below = fx.Int32(0)
    cur = fx.Int32(0)
    for k in range(E // 32):
        wk = fx.ptr_load(rt["bm"] + k)
        pk = fx.Int32(fx.ctpop(wk))
        nu = nu + pk
        below = below + (fx.Int32(k) < wsel).select(pk, fx.Int32(0))
        cur = (fx.Int32(k) == wsel).select(wk, cur)
    slot = below + fx.Int32(fx.ctpop(cur & ((fx.Int32(1) << (e & 31)) - fx.Int32(1))))
    fx.ptr_store(e, rt["expert"] + first.select(slot, fx.Int32(MAX_U)))
    fx.ptr_store(i % TOPK, rt["kof"] + (slot * 8 + i // TOPK))
    gpu.barrier()
    return Q.uniform(nu)


@traced
def _rt_mark(rt, live, e, i):
    if live:
        fx.atomic_min(rt["own"] + e, i, syncscope=VR_WS)
        fx.atomic_or(rt["bm"] + (e >> 5), fx.Int32(1) << (e & 31), syncscope=VR_WS)


# ---------------------------------------------------------------- the scaled-WMMA chain
def _chain(c, wbases, sbases, rw, rws, rx, rxs, xbase, xsbase, steps, st0, prefetch):
    """One wave: ``len(wbases)`` fp32 chains (16 weight rows x 16 activation rows) over steps st0 .. st0 + steps,
    weights FP4 (two b128 a step), activations FP4 (two b128) / FP8 (four b128), scales a dword a step pair
    (opsel = step parity), every operand from global (weights NT). ``st0`` may be traced (a wave's K part) and
    must be even; step parity and the ring are over the static i = st - st0."""
    act, nt = c["act"], c["nt"]
    xcm = c.get("xcm", 0)
    a8 = act == "a8"
    assert steps % 2 == 0 or (isinstance(st0, int) and st0 % 2 == 0)

    def load(i):
        ops = _wload(c, rw, rws, wbases, sbases, st0, i)
        ops.update(_xload(c, rx, rxs, xbase, xsbase, st0, i))
        return ops

    return _mma_steps(c, len(wbases), load, steps, prefetch)


def _wload(c, rw, rws, wbases, sbases, st0, i):
    """Step ``st0 + i`` of the weight operands (and the step pair's weight scales on even i)."""
    st, nt = st0 + i, c["nt"]
    ops = {"a": [fx.Vector(bo.buffer_load(rw, wb + st * 256, vec_width=4, dtype=T.i32, cache_modifier=nt))
                 .shuffle(fx.Vector(bo.buffer_load(rw, wb + st * 256 + 128, vec_width=4, dtype=T.i32,
                                                   cache_modifier=nt)), list(range(8))) for wb in wbases]}
    if i % 2 == 0:
        ops["sa"] = [bo.buffer_load(rws, sb + (st0 // 2 + i // 2) * 32, vec_width=1, dtype=T.i32) for sb in sbases]
    return ops


def _xload(c, rx, rxs, xbase, xsbase, st0, i):
    """Step ``st0 + i`` of the activation operand (and its scale codes on even i)."""
    st, xcm = st0 + i, c.get("xcm", 0)
    ops = {}
    if c["act"] == "a8":
        bq = [fx.Vector(bo.buffer_load(rx, xbase + st * 32 + 8 * q, vec_width=4, dtype=T.i32, cache_modifier=xcm))
              for q in range(4)]
        ops["b"] = bq[0].shuffle(bq[1], list(range(8))).shuffle(bq[2].shuffle(bq[3], list(range(8))),
                                                                list(range(16)))
    else:
        ops["b"] = fx.Vector(bo.buffer_load(rx, xbase + st * 16, vec_width=4, dtype=T.i32, cache_modifier=xcm)).shuffle(
            fx.Vector(bo.buffer_load(rx, xbase + st * 16 + 8, vec_width=4, dtype=T.i32, cache_modifier=xcm)),
            list(range(8)))
    if i % 2 == 0:
        ops["sb"] = bo.buffer_load(rxs, xsbase + st, vec_width=1, dtype=T.i32, cache_modifier=xcm)
    return ops


def _mma_steps(c, n, load, steps, prefetch):
    """``n`` fp32 chains over ``steps`` steps; ``load(i)`` issues step i's operands, ``prefetch`` steps ahead."""
    a8 = c["act"] == "a8"
    accs = [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(n)]
    ring = {i: load(i) for i in range(min(prefetch, steps))}
    sa = sb = None
    for i in range(steps):
        if i + prefetch < steps:
            ring[i + prefetch] = load(i + prefetch)
        ops = ring.pop(i)
        if i % 2 == 0:
            sa, sb = ops["sa"], ops["sb"]
        for j in range_constexpr(len(accs)):
            accs[j] = fx.Vector(rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                T.vec(8, T.f32), ops["a"][j], ops["b"], accs[j], sa[j], sb, fmtA=4, fmtB=0 if a8 else 4,
                scaleAType=i % 2, scaleBType=i % 2))
    return accs


def _ksplit_reduce(c, accs, part, nparts, slot_base):
    """In-CTA split-K: parts 1.. park their chains in LDS, part 0 adds them in part order."""
    lane, red = c["lane"], c["kred"]
    n = len(accs)
    if nparts == 1:
        return accs
    for j in range(n):
        for v in range(8):
            fx.ptr_store(accs[j][v], red + (((slot_base + part) * n + j) * 8 + v) * 32 + lane)
    gpu.barrier()
    out = []
    for j in range(n):
        vals = []
        for v in range(8):
            x = fx.ptr_load(red + (((slot_base + 0) * n + j) * 8 + v) * 32 + lane)
            for p in range(1, nparts):
                x = x + fx.ptr_load(red + (((slot_base + p) * n + j) * 8 + v) * 32 + lane)
            vals.append(x)
        out.append(vals)
    return out


# ---------------------------------------------------------------- ug
def stage_ug(c, slot, g):
    """Slot ``slot`` (expert ``expert[slot]``), 32-column group g of the rank's slice: wave w = tile 2 g + w % 2,
    K part w // 2 (``ksplit`` parts); gate and up chains vs the 8 tokens' xe; silu(g) u -> bf16; the 32-group
    amax over the two tiles (LDS); MX requant -> INTER words / ISC code of each picking token."""
    s, lane, wave, act, rt = c["S"], c["lane"], c["wave"], c["act"], c["rt"]
    ks = c["ksplit"]
    a = c["args"]
    wp = Q.words_per8(act)
    tile = wave % 2
    part = wave // 2
    rg = 2 * g + tile
    e = Q.uniform(fx.ptr_load(rt["expert"] + slot))
    gq = lane // 16
    w_dw_e = 2 * RI * MH // 8
    rw = _rsrc(a["w13"], E * w_dw_e * 4)
    rws = _rsrc(a["w13s"], E * w_dw_e // 4)
    rx = _rsrc(c["xqb"], s * xq_words(act) * 4)
    rxs = _rsrc(c["xscb"], s * MH // 32)
    wbases = [e * w_dw_e + (rg + j * W13_TILES) * (2 * MH) + gq * 64 + (lane % 16) * 4 for j in range(2)]
    sbases = [(e * (2 * RI // 16) + rg + j * W13_TILES) * (MH // 256) * 32 + lane for j in range(2)]
    arow = fx.min(lane % 16, s - 1)
    xbase = arow * xq_words(act) + gq * 4
    xsbase = arow * (MH // 128) + gq
    steps = W13_STEPS // ks
    accs = _chain(c, wbases, sbases, rw, rws, rx, rxs, xbase, xsbase, steps, part * steps, c["prefetch"])
    _ug_tail(c, slot, g, accs)


def _ug_geom(c, slot, g):
    """A UG task's per-wave geometry: (tile, K part, rg, expert, gq, steps)."""
    wave, lane, rt = c["wave"], c["lane"], c["rt"]
    tile = wave % 2
    part = wave // 2
    e = Q.uniform(fx.ptr_load(rt["expert"] + fx.min(slot, fx.Int32(MAX_U))))
    return tile, part, 2 * g + tile, e, lane // 16, W13_STEPS // c["ksplit"]


_OOB_W, _OOB_S = 0x10000000, 0x1000000  # dword offsets past the w13 / w13s records: dead loads, no traffic


def ug_weights(c, slot, g, live):
    """This wave's w13 operands (gate and up, its K part) of UG task (slot, g), issued now; ``live`` false: the
    offsets point past the buffers (the loads return 0 without a memory access)."""
    lane, a = c["lane"], c["args"]
    tile, part, rg, e, gq, steps = _ug_geom(c, slot, g)
    w_dw_e = 2 * RI * MH // 8
    rw = _rsrc(a["w13"], E * w_dw_e * 4)
    rws = _rsrc(a["w13s"], E * w_dw_e // 4)
    ow = live.select(fx.Int32(0), fx.Int32(_OOB_W))
    os_ = live.select(fx.Int32(0), fx.Int32(_OOB_S))
    wbases = [e * w_dw_e + (rg + j * W13_TILES) * (2 * MH) + gq * 64 + (lane % 16) * 4 + ow for j in range(2)]
    sbases = [(e * (2 * RI // 16) + rg + j * W13_TILES) * (MH // 256) * 32 + lane + os_ for j in range(2)]
    return [_wload(c, rw, rws, wbases, sbases, part * steps, i) for i in range(steps)]


def stage_ug_pipe(c, slot, g, w, nslot, nlive):
    """``stage_ug`` with this task's weights ``w`` (``ug_weights``) already in flight: issues the activation loads,
    then the next task's (nslot, g) weights, so they stream during this task's WMMAs and epilogue. Returns them."""
    s, lane, act = c["S"], c["lane"], c["act"]
    tile, part, rg, e, gq, steps = _ug_geom(c, slot, g)
    rx = _rsrc(c["xqb"], s * xq_words(act) * 4)
    rxs = _rsrc(c["xscb"], s * MH // 32)
    arow = fx.min(lane % 16, s - 1)
    xbase = arow * xq_words(act) + gq * 4
    xsbase = arow * (MH // 128) + gq
    xs = [_xload(c, rx, rxs, xbase, xsbase, part * steps, i) for i in range(steps)]
    wn = ug_weights(c, nslot, g, nlive)
    accs = _mma_steps(c, 2, lambda i: dict(w[i], **xs[i]), steps, steps)
    _ug_tail(c, slot, g, accs)
    return wn


def _ug_tail(c, slot, g, accs):
    """UG epilogue: split-K reduce, silu(g) u, the 32-group amax, MX requant, INTER / ISC stores, UGRDY."""
    s, lane, act, rt = c["S"], c["lane"], c["act"], c["rt"]
    ks = c["ksplit"]
    wp = Q.words_per8(act)
    tile, part, rg, e, gq, steps = _ug_geom(c, slot, g)
    accs = _ksplit_reduce(c, accs, part, ks, tile * ks)
    t = lane % 16
    tt = fx.min(t, s - 1)
    kk = fx.ptr_load(rt["kof"] + (slot * 8 + tt))
    picked = (t < s) & (kk >= 0) & (part == 0)
    vals = [Q.bf16_round(Q.silu_mul(accs[0][v], accs[1][v])) for v in range(8)]
    am = abs(vals[0])
    for v in vals[1:]:
        am = fx.max(am, abs(v))
    am = Q.xmax(am, (16,))
    fx.ptr_store(am, c["amx"] + (tile * 16 + t))
    gpu.barrier()
    am = fx.max(fx.ptr_load(c["amx"] + t), fx.ptr_load(c["amx"] + (16 + t)))
    code = Q.code(am, act, inter=True)
    words = Q.pack8(vals, code, act)
    pick = tt * TOPK + fx.max(kk, fx.Int32(0))
    ips, npk, scm = isc_ps(c), picks(c), ug_store_cm(c)
    out = words[0] if wp == 1 else fx.Vector.from_elements(words, fx.Int32)
    bo.buffer_store(out, _rsrc(c["interb"], npk * inter_words(act) * 4),
                    pick * inter_words(act) + (16 * rg + 8 * gq) // 8 * wp, mask=picked, cache_modifier=scm)
    bo.buffer_store(fx.Int8(code), _rsrc(c["iscb"], npk * ips), pick * ips + g,
                    mask=picked & (tile == 0) & (gq == 0), cache_modifier=scm)
    if _k2(c):
        if c.get("ug_wt"):
            _publish_wt(c, c["ugrdy"], slot * UG_GROUPS + g)
        else:
            _publish(c, c["ugrdy"], slot * UG_GROUPS + g)



# ---------------------------------------------------------------- down
def _read_pick(c, picked, pick):
    """The INTER / ISC row a down lane reads: unpicked lanes read the trash row when there is one."""
    if c.get("trash_pick"):
        return picked.select(pick, fx.Int32(c["S"] * TOPK))
    return pick


def stage_down(c, slot, rgd):
    """Slot ``slot``, MH tile ``rgd`` (this wave): one chain over K = RI (8 steps) of w2 vs the picks' INTER
    -> bf16 per route (unweighted) -> YR[rank][t][k] on row t's owner."""
    s, lane, act, rt = c["S"], c["lane"], c["act"], c["rt"]
    a = c["args"]
    e = Q.uniform(fx.ptr_load(rt["expert"] + slot))
    gq = lane // 16
    t = lane % 16
    tt = fx.min(t, s - 1)
    kk = fx.ptr_load(rt["kof"] + (slot * 8 + tt))
    picked = (t < s) & (kk >= 0)
    pick = tt * TOPK + fx.max(kk, fx.Int32(0))
    ips, npk = isc_ps(c), picks(c)
    rpick = _read_pick(c, picked, pick)
    w_dw_e = MH * RI // 8
    rw = _rsrc(a["w2"], E * w_dw_e * 4)
    rws = _rsrc(a["w2s"], E * w_dw_e // 4)
    rx = _rsrc(c["interb"], npk * inter_words(act) * 4)
    rxs = _rsrc(c["iscb"], npk * ips)
    wbases = [e * w_dw_e + rgd * (2 * RI) + gq * 64 + (lane % 16) * 4]
    sbases = [(e * (MH // 16) + rgd) * (RI // 256) * 32 + lane]
    xbase = rpick * inter_words(act) + gq * 4
    xsbase = rpick * (ips // 4) + gq
    wl = [_wload(c, rw, rws, wbases, sbases, 0, i) for i in range(W2_STEPS)] if c.get("down_wfirst") else None
    if _k2(c):
        c["poll"]([(c["ugrdy"], slot * UG_GROUPS + lane, 1)])
    if const_expr(bool(c.get("tl_first"))):
        c["tl_hook"](14)
    if c.get("down_wfirst"):
        acc = _mma_steps(c, 1, lambda i: dict(wl[i], **_xload(c, rx, rxs, xbase, xsbase, 0, i)), W2_STEPS,
                         c["dprefetch"])[0]
    else:
        acc = _chain(c, wbases, sbases, rw, rws, rx, rxs, xbase, xsbase, W2_STEPS, 0, c["dprefetch"])[0]
    if const_expr(bool(c.get("tl_first"))):
        rocdl.s_wait_loadcnt(0)
        c["tl_hook"](15)
    if _k2(c):
        el = ((c["rank"] * s + tt) * TOPK + fx.max(kk, fx.Int32(0))) * MH + rgd * ROWS + 8 * gq
        _yr_put(c, picked, tt // (s // TP), el, acc)
        if const_expr(bool(c.get("tl_first"))):
            rocdl.s_wait_storecnt(0)
            c["tl_hook"](26)
        return
    packed = fx.Vector.from_elements([acc[v] for v in range(8)], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    off = (((c["rank"] * s + tt) * TOPK + kk) * MH + rgd * ROWS + 8 * gq) // 2
    for p in range(TP):
        bo.buffer_store(packed, _rsrc(c["yr_owner"][p], TP * s * TOPK * MH * 2), off,
                        mask=picked & (tt // (s // TP) == p))


# ---------------------------------------------------------------- vres2 expert phase (K2: K2_VRES2)
# K split over the CTA's waves with each wave's activations resident in VGPRs; a task's K parts are
# ds_add_f32'ed into an LDS sum and the last wave to arrive (an LDS counter) runs the epilogue, so no CTA
# barrier sits inside the weight stream.
#   UG   task (slot, 32-col group) = 4 chains (gate / up x 2 tiles); wave w: K steps 2w, 2w + 1 of all four
#   down a CTA's contiguous run of (slot, 16-row MH tile); wave w: K half w % 2, tiles j, j + 8, .. (j = w // 2)
VR_CTAS = 256
VR_UG_LOCAL = (MAX_U * UG_GROUPS + VR_CTAS - 1) // VR_CTAS  # 8: UG tasks a CTA at most
VR_YS = 33  # LDS row (32 columns + 1: the 16 adding lanes hit 16 banks)
VR_UG_WORDS = 2 * MAX_TOKENS * VR_YS  # a task's sums [gate, up][token][column]
VR_DN_LOCAL = 80  # down tiles a CTA at most (vr_down_range: nu + 2 (nu % 8) <= 77)
VR_DN_WORDS = ROWS * MAX_TOKENS  # a tile's sums [v][gq][token]
VR_WS = "workgroup"


def vr_lds_words():
    """(UG sums, down sums, counters) words of LDS the vres2 phase needs."""
    return VR_UG_LOCAL * VR_UG_WORDS, VR_DN_LOCAL * VR_DN_WORDS, VR_UG_LOCAL + VR_DN_LOCAL


def _lds_fence():
    """s_wait_dscnt 0, also a compiler barrier (no memory access is moved across the asm)."""
    from flydsl._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(T.i32, [], "s_wait_dscnt 0x0\n s_mov_b32 $0, 0", "=s", has_side_effects=True)


@traced
def _st0_if(p, i, n):
    if i < n:
        fx.ptr_store(fx.Float32(0.0), p + i)


@traced
def _st0i_if(p, i, n):
    if i < n:
        fx.ptr_store(fx.Int32(0), p + i)


def vr_init(c):
    """Zero the LDS sums and counters (before a CTA barrier)."""
    tid, nt = c["tid"], c["threads"]
    nu_w, nd_w, nc_w = vr_lds_words()
    for p, n in ((c["vr_uy"], nu_w), (c["vr_dy"], nd_w)):
        for q in range((n + nt - 1) // nt):
            _st0_if(p, tid + q * nt, n)
    for q in range((nc_w + nt - 1) // nt):
        _st0i_if(c["vr_cnt"], tid + q * nt, nc_w)


@traced
def _arrive(c, i):
    """This wave's adds done, then one lane counts the wave in at counter i: the count before (uniform)."""
    _lds_fence()
    old = fx.Int32(0)
    if c["lane"] == 0:
        old = fx.Int32(fx.atomic_add(c["vr_cnt"] + i, fx.Int32(1), syncscope=VR_WS))
    return Q.uniform(old)


def _expert(c, slot):
    e = Q.uniform(fx.ptr_load(c["rt"]["expert"] + fx.min(slot, fx.Int32(MAX_U))))
    return fx.max(fx.min(e, fx.Int32(E - 1)), fx.Int32(0))


def vr_ug_weights(c, slot, g, live):
    """This wave's w13 operands of UG task (slot, g): K steps 2 w, 2 w + 1 of the 4 chains (tile 0 gate,
    tile 0 up, tile 1 gate, tile 1 up), issued now; dead (no traffic) unless ``live``."""
    lane, a = c["lane"], c["args"]
    e = _expert(c, slot)
    gq = lane // 16
    w_dw_e = 2 * RI * MH // 8
    rw = _rsrc(a["w13"], E * w_dw_e * 4)
    rws = _rsrc(a["w13s"], E * w_dw_e // 4)
    ow = live.select(fx.Int32(0), fx.Int32(_OOB_W))
    os_ = live.select(fx.Int32(0), fx.Int32(_OOB_S))
    rows = [2 * g + tile + j * W13_TILES for tile in range(2) for j in range(2)]
    wbases = [e * w_dw_e + r * (2 * MH) + gq * 64 + (lane % 16) * 4 + ow for r in rows]
    sbases = [(e * (2 * RI // 16) + r) * (MH // 256) * 32 + lane + os_ for r in rows]
    return [_wload(c, rw, rws, wbases, sbases, 2 * c["wave"], i) for i in range(2)]


def vr_ug_acts(c):
    """This wave's resident activation operands: K steps 2 w, 2 w + 1 of the 8 tokens' XQ (after the XQ acquire)."""
    s, lane, act = c["S"], c["lane"], c["act"]
    rx = _rsrc(c["xqb"], s * xq_words(act) * 4)
    rxs = _rsrc(c["xscb"], s * MH // 32)
    arow = fx.min(lane % 16, s - 1)
    return [_xload(c, rx, rxs, arow * xq_words(act) + (lane // 16) * 4, arow * (MH // 128) + lane // 16,
                   2 * c["wave"], i) for i in range(2)]


@traced
def _vr_ug_add(base, t, gq, accs):
    if t < MAX_TOKENS:
        for ch in range_constexpr(4):
            for v in range_constexpr(8):
                fx.atomic_add(base + (((ch % 2) * MAX_TOKENS + t) * VR_YS + 16 * (ch // 2) + 8 * gq + v),
                              accs[ch][v], syncscope=VR_WS)


@traced
def _vr_ug_epi(c, slot, g, yb):
    """The last wave of UG task (slot, g): lane = (token t, column octet q); silu(g) u -> bf16, the 32-group amax
    (4 lanes), MX requant -> INTER / ISC (write-through), drained, then UGRDY (slot, g)."""
    s, lane, act, rt = c["S"], c["lane"], c["act"], c["rt"]
    wp = Q.words_per8(act)
    t = lane // 4
    q = lane % 4
    gv = [fx.ptr_load(yb + (t * VR_YS + 8 * q + v)) for v in range(8)]
    uv = [fx.ptr_load(yb + ((MAX_TOKENS + t) * VR_YS + 8 * q + v)) for v in range(8)]
    vals = [Q.bf16_round(Q.silu_mul(gv[v], uv[v])) for v in range(8)]
    am = abs(vals[0])
    for v in vals[1:]:
        am = fx.max(am, abs(v))
    am = Q.xmax(am, (2, 1))
    code = Q.code(am, act, inter=True)
    words = Q.pack8(vals, code, act)
    kk = fx.ptr_load(rt["kof"] + (slot * 8 + t))
    picked = kk >= 0
    pick = t * TOPK + fx.max(kk, fx.Int32(0))
    ips, npk, scm = isc_ps(c), picks(c), ug_store_cm(c)
    out = words[0] if wp == 1 else fx.Vector.from_elements(words, fx.Int32)
    bo.buffer_store(out, _rsrc(c["interb"], npk * inter_words(act) * 4), pick * inter_words(act) + (4 * g + q) * wp,
                    mask=picked, cache_modifier=scm)
    bo.buffer_store(fx.Int8(code), _rsrc(c["iscb"], npk * ips), pick * ips + g, mask=picked & (q == 0),
                    cache_modifier=scm)


@traced
def vr_flush(c, pend):
    """Publish UGRDY ``pend`` (>= 0) of this wave's last epilogue: its write-through stores drained first."""
    if pend >= 0:
        rocdl.s_wait_storecnt(0)
        _put_if(c, c["lane"] == 0, c["ugrdy"], pend, fx.Int32(1))


@traced
def _vr_ug_last(c, last, slot, g, yb):
    """The epilogue if this wave arrived last; the UGRDY index it leaves to publish (-1: none)."""
    pend = fx.Int32(-1)
    if last:
        _vr_ug_epi(c, slot, g, yb)
        pend = slot * UG_GROUPS + g
    return pend


def _cnt_read(c, i):
    """Counter i, read by every lane (one address: a broadcast; no lane-0 branch, which would make a spin loop
    around it divergent)."""
    return Q.uniform(fx.generic_load(c["vr_cnt"] + i, dtype=fx.Int32, memory_order=fx.AtomicOrdering.Monotonic,
                                     syncscope=VR_WS, volatile=True))


VR_SPIN_MAX = 1 << 22  # ~0.3 s of s_sleep 1: a bound, never reached unless a wave is lost


@traced
def _vr_ug_mine(c, mine, i, n, slot, g, yb):
    """(``vr_epi_rot``) The epilogue if this wave owns task i (wave i % 16), once all n waves added: the
    epilogues rotate over the waves instead of piling onto the slowest one."""
    pend = fx.Int32(-1)
    if mine:
        v = _cnt_read(c, i)
        it = fx.Int32(0)
        while (v < n) & (it < VR_SPIN_MAX):
            rocdl.s_sleep(1)
            v = _cnt_read(c, i)
            it = it + 1
        _vr_ug_epi(c, slot, g, yb)
        pend = slot * UG_GROUPS + g
    return pend


def vr_ug_task(c, slot, g, i, w, xs, nslot, ng, nlive, pend):
    """UG task (slot, g), this CTA's i-th, with its weights ``w`` in flight: issues the next task's (nslot, ng),
    4 chains of 2 WMMAs, the partial sums into LDS, then the last wave runs the epilogue. ``pend``: the UGRDY of
    this wave's previous epilogue, published here (``vr_lazy``: its stores have drained by now) or at once.
    Returns (the next weights, the UGRDY still to publish)."""
    lane = c["lane"]
    wn = vr_ug_weights(c, nslot, ng, nlive)
    rocdl.sched_barrier(0)  # the next task's loads stay ahead of this task's waits
    accs = _mma_steps(c, 4, lambda k: dict(w[k], **xs[k]), 2, 2)
    vr_flush(c, pend)
    yb = c["vr_uy"] + i * VR_UG_WORDS
    _vr_ug_add(yb, lane % 16, lane // 16, accs)
    old = _arrive(c, i)
    nw = c["threads"] // 32
    if c.get("vr_epi_rot"):
        pend = _vr_ug_mine(c, c["wave"] == i % nw, i, nw, slot, g, yb)
    else:
        pend = _vr_ug_last(c, old == nw - 1, slot, g, yb)
    if not c.get("vr_lazy"):
        vr_flush(c, pend)
        pend = fx.Int32(-1)
    return wn, pend


def vr_down_range(nu, ub, ugw=16):
    """The run [d0, d1) of the slot-major down tiles of the CTA with UG index ``ub`` (UG task u runs on index
    u % 256). The r = 32 (nu % 8) lowest indices have one UG task (16 down tiles of bytes) more: they get
    ``ugw`` tiles fewer and the run's tail (the late UG slots, which they produce last), the others the head
    (an even split of the rest)."""
    m = nu % 8
    r = 32 * m
    dh = fx.max((8 * nu + m * ugw - 8 * ugw) // 8, fx.Int32(0))
    light = nu * DOWN_TILES - r * dh
    heavy = ub < r
    nl = fx.Int32(VR_CTAS) - r
    d0 = heavy.select(light + ub * dh, ((ub - r) * light) // nl)
    d1 = heavy.select(light + (ub + 1) * dh, ((ub - r + 1) * light) // nl)
    return Q.uniform(d0), Q.uniform(d1)


def vr_dn_weights(c, d0, g, k):
    """This wave's w2 operands (its K half) of its k-th tile (``vr_down_geom`` g); dead unless k < g's count."""
    lane, a, wave = c["lane"], c["args"], c["wave"]
    l0, stride, nk, _ = g
    v = d0 + l0 + stride * k
    live = k < nk
    e = _expert(c, v // DOWN_TILES)
    rgd = v % DOWN_TILES
    gq = lane // 16
    w_dw_e = MH * RI // 8
    rw = _rsrc(a["w2"], E * w_dw_e * 4)
    rws = _rsrc(a["w2s"], E * w_dw_e // 4)
    ow = live.select(fx.Int32(0), fx.Int32(_OOB_W))
    os_ = live.select(fx.Int32(0), fx.Int32(_OOB_S))
    wb = [e * w_dw_e + rgd * (2 * RI) + gq * 64 + (lane % 16) * 4 + ow]
    sb = [(e * (MH // 16) + rgd) * (RI // 256) * 32 + lane + os_]
    return [_wload(c, rw, rws, wb, sb, 4 * (wave % 2), i) for i in range(W2_STEPS // 2)]


def vr_down_geom(c, d0, d1):
    """This wave's down tiles in the run [d0, d1) (at most two slots): local tiles l0 + stride k, k < nk, all of
    one slot. The run's b tiles of its first slot go to the first m0 = round(8 b / n) tile streams (wave pairs),
    the rest to the others, so no wave switches slots (no second poll / activation reload mid-stream)."""
    j = c["wave"] // 2
    n = d1 - d0
    s0 = d0 // DOWN_TILES
    b = fx.min(fx.max((s0 + 1) * DOWN_TILES - d0, fx.Int32(0)), n)
    m0 = fx.min(fx.max((16 * b + n) // (2 * fx.max(n, fx.Int32(1))), fx.Int32(1)), fx.Int32(7))
    m0 = (b == 0).select(fx.Int32(0), m0)
    m0 = (b == n).select(fx.Int32(8), m0)
    first = j < m0
    l0 = first.select(j, b + j - m0)
    stride = fx.max(first.select(m0, 8 - m0), fx.Int32(1))
    end = first.select(b, n)
    nk = fx.max((end - l0 + stride - 1) // stride, fx.Int32(0))
    return Q.uniform(l0), Q.uniform(stride), Q.uniform(nk), Q.uniform(first.select(s0, s0 + 1))


def vr_down_prime(c, d0, d1):
    """The first two tiles' weights of this wave (before the UG -> down acquire)."""
    g = vr_down_geom(c, d0, d1)
    return [vr_dn_weights(c, d0, g, fx.Int32(k)) for k in range(2)]


def _vr_dn_acts(c, slot, live):
    """Resident operands of slot ``slot`` (this wave's K half): picks' INTER rows; unpicked lanes and a dead
    segment read the trash pick row (never a not-yet-announced line)."""
    s, lane, act, rt = c["S"], c["lane"], c["act"], c["rt"]
    t = lane % 16
    tt = fx.min(t, s - 1)
    kk = fx.ptr_load(rt["kof"] + (fx.min(slot, fx.Int32(MAX_U)) * 8 + tt))
    picked = (t < s) & (kk >= 0) & live
    pick = tt * TOPK + fx.max(kk, fx.Int32(0))
    rpick = picked.select(pick, fx.Int32(s * TOPK))
    ips, npk = isc_ps(c), picks(c)
    rx = _rsrc(c["interb"], npk * inter_words(act) * 4)
    rxs = _rsrc(c["iscb"], npk * ips)
    xb = rpick * inter_words(act) + (lane // 16) * 4
    xsb = rpick * (ips // 4) + lane // 16
    acts = [_xload(c, rx, rxs, xb, xsb, 4 * (c["wave"] % 2), i) for i in range(W2_STEPS // 2)]
    return acts, picked, kk


@traced
def _vr_dn_add(base, t, gq, acc):
    if t < MAX_TOKENS:
        for v in range_constexpr(8):
            fx.atomic_add(base + (v * 16 + gq * 8 + t), acc[v], syncscope=VR_WS)


@traced
def _vr_dn_last(c, last, yb, picked, kk, v):
    """(``vr_yr_tile``) The second K half of down tile v: the sum -> bf16 YR puts of the picked routes."""
    if last:
        s, lane = c["S"], c["lane"]
        t = lane % 16
        tt = fx.min(t, s - 1)
        gq = lane // 16
        acc = [fx.ptr_load(yb + (vv * 16 + gq * 8 + tt)) for vv in range(8)]
        el = ((c["rank"] * s + tt) * TOPK + fx.max(kk, fx.Int32(0))) * MH + (v % DOWN_TILES) * ROWS + 8 * gq
        _yr_put(c, picked, tt // (s // TP), el, acc)


VR_YR_ITERS = (MAX_TOKENS * ((VR_DN_LOCAL + 7) // 8) + 15) // 16  # 5


def vr_yr_out(c, d0, d1):
    """After the down run and a CTA barrier: the run's sums (LDS) -> YR. A unit (token t, 8 consecutive local
    tiles) is a wave, lane = (tile i, row quad q): 4 bf16 as 2 tagged pairs, 512 contiguous bytes of route
    (t, k) when the 8 tiles share a slot. Row t's owner is wave-uniform (one resource, no owner branches);
    unpicked lanes are masked by an out-of-range offset (no EXEC change: on gfx1250 restoring EXEC waits until
    the in-flight system-scope stores are address-translated). All units' words are built first, then stored."""
    s, lane, wave, rt, tag = c["S"], c["lane"], c["wave"], c["rt"], c["tag"]
    n = d1 - d0
    nunits = s * ((n + 7) // 8)
    i = lane // 4
    q = lane % 4
    yrs = c["yr_owner"]
    for a in yrs:
        _noted(a, "put")
    outs = []
    for it in range(VR_YR_ITERS):
        unit = wave + 16 * it
        t = unit % s
        l = 8 * (unit // s) + i
        v = d0 + l
        kk = fx.ptr_load(rt["kof"] + (fx.min(v // DOWN_TILES, fx.Int32(MAX_U)) * 8 + t))
        picked = (unit < nunits) & (l < n) & (kk >= 0)
        lc = fx.min(l, fx.Int32(VR_DN_LOCAL - 1))
        ys = [fx.ptr_load(c["vr_dy"] + (lc * VR_DN_WORDS + (4 * (q % 2) + e) * 16 + (q // 2) * 8 + t))
              for e in range(4)]
        words = fx.Vector.from_elements([_bf16_pair(ys[0], ys[1]).bitcast(fx.Int32), tag,
                                         _bf16_pair(ys[2], ys[3]).bitcast(fx.Int32), tag], fx.Int32)
        el = ((c["rank"] * s + t) * TOPK + fx.max(kk, fx.Int32(0))) * MH + (v % DOWN_TILES) * ROWS + 4 * q
        owner = t // (s // TP)
        base = yrs[0].value
        for p in range(1, TP):
            base = (owner == p).select(yrs[p].value, base)
        outs.append((words, base, el, picked))
    rocdl.sched_barrier(0)
    for words, base, el, picked in outs:
        bo.buffer_store(words, _prsrc(base, yrs[0].extent), el, mask=picked, cache_modifier=_CM_SYS)


@traced
def _vr_poll_slot(c, cond, slot):
    if cond:
        c["poll"]([(c["ugrdy"], slot * UG_GROUPS + c["lane"], 1)])


def vr_down(c, d0, d1, ring):
    """This wave's tiles of the down run [d0, d1), their weights 2 ahead (``ring``)."""
    g = vr_down_geom(c, d0, d1)
    live = g[2] > 0
    _vr_poll_slot(c, live, g[3])
    if c.get("tl_on"):
        c["tl_hook"](14)
    acts, picked, kk = _vr_dn_acts(c, g[3], live)
    ring = _vr_dn_loop(c, d0, g, ring, acts, picked, kk)
    if c.get("tl_on"):
        c["tl_hook"](15)
    return ring


def vr_dn_tile(c, d0, g, k, ring, acts, picked, kk):
    """This wave's k-th down tile: the weights 2 ahead issued, 4 WMMAs, the partial into LDS (``vr_yr_tile``: the
    second K half to arrive puts the routes). Returns the shifted ring."""
    lane = c["lane"]
    wn = vr_dn_weights(c, d0, g, k + 2)
    rocdl.sched_barrier(0)
    w = ring[0]
    acc = _mma_steps(c, 1, lambda i: dict(w[i], **acts[i]), W2_STEPS // 2, W2_STEPS // 2)[0]
    loc = g[0] + g[1] * k
    yb = c["vr_dy"] + loc * VR_DN_WORDS
    _vr_dn_add(yb, lane % 16, lane // 16, acc)
    if c.get("vr_yr_tile"):
        old = _arrive(c, VR_UG_LOCAL + loc)
        _vr_dn_last(c, old == 1, yb, picked, kk, d0 + loc)
    return [ring[1], wn]


@traced
def _vr_dn_loop(c, d0, g, ring, acts, picked, kk):
    for k in range(g[2]):
        ring = vr_dn_tile(c, d0, g, k, ring, acts, picked, kk)
    return ring


# ---------------------------------------------------------------- finalize
def _fin_out(c, t, j, k, col, vs, tot, wk):
    """K2 tail of ``stage_fin`` (``upto`` = 2 arithmetic): FIN plain into every rank's peer copy, then FRDY
    (row t, slice j) published at system scope on every rank."""
    from port.sync.memmodel import CM_SYS

    s, lane, wave = c["S"], c["lane"], c["wave"]
    sc = wk * Q.inv_sqrt_ieee(tot * fx.Float32(1.0 / MH) + fx.Float32(c["eps_post"]))
    for i in range(4):
        fx.ptr_store(Q.opaque(sc * vs[i]), c["red"] + ((k * 4 + i) * 32 + lane))
    gpu.barrier()
    acc = []
    for i in range(4):
        x = fx.Float32(0.0)
        for kq in range(TOPK):
            x = x + fx.ptr_load(c["red"] + ((kq * 4 + i) * 32 + lane))
        acc.append(x)
    if c.get("fin_pairs", False):
        # FIN as tagged bf16 pairs into every rank's FINP: the unpoolers poll the rows; no drain, no FRDY
        _put_bf_if(c, wave == 0, c["finp_peer"], t * MH + col, acc)
        cta_barrier()
        return
    fw = fx.Vector.from_elements(acc, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    for p in range(TP):
        bo.buffer_store(fw, _rsrc(c["fin_peer"][p], s * MH * 2), (t * MH + col) // 2, mask=wave == 0,
                        cache_modifier=CM_SYS)
    _publish(c, c["frdy_peer"][0], t * FIN_SLICES + j)
    for p in range(1, TP):
        _put_if(c, c["tid"] == 0, c["frdy_peer"][p], t * FIN_SLICES + j, fx.Int32(1))
    gpu.barrier()


def stage_fin(c, tl, j, upto):
    """Owned token ``rank * OWN + tl``, columns 128 j .. (wave k = route k, a lane 4 columns): YR from the 4
    ranks in rank order (fp32) -> bf16 y; v = bf16(y bf16(w_post + 1)); the slice's sum of v^2 (lane's 4 in
    order, butterfly) -> FMSQ[t][k][j]; [``upto`` = 1 stops here: the launch boundary is the FMSQ poll]; the 32
    slices' sums (butterfly over slices) -> sc_k = w_k / sqrt(sum / MH + eps); fin = bf16(sum_k sc_k v_k) in
    k order (products rounded) -> FIN rows of every rank."""
    s, lane, wave, rank = c["S"], c["lane"], c["wave"], c["rank"]
    a = c["args"]
    t = rank * (s // TP) + tl
    k = fx.min(wave, TOPK - 1) if _k2(c) else wave  # K2: 16 waves; waves 8.. repeat route 7's identical work
    col = j * FIN_COLS + 4 * lane
    ww = _ld(a["w_post"], MH * 2, col // 2, 2)
    w1 = [Q.bf16_round(Q.bf_lo(ww[0]) + fx.Float32(1.0)), Q.bf16_round(Q.bf_hi(ww[0]) + fx.Float32(1.0)),
          Q.bf16_round(Q.bf_lo(ww[1]) + fx.Float32(1.0)), Q.bf16_round(Q.bf_hi(ww[1]) + fx.Float32(1.0))]
    if _k2(c):
        parts = c["poll"]([(c["yr"], (((src * s + t) * TOPK + k) * MH + col) // 2, 2) for src in range(TP)])
        stamp(c, 36)
    else:
        ry = _rsrc(c["yr"], TP * s * TOPK * MH * 2)
        parts = [fx.Vector(bo.buffer_load(ry, (((src * s + t) * TOPK + k) * MH + col) // 2, vec_width=2,
                                          dtype=T.i32)) for src in range(TP)]
    ys = []
    for h in range(2):
        for f in (Q.bf_lo, Q.bf_hi):
            y = f(parts[0][h])
            for src in range(1, TP):
                y = y + f(parts[src][h])
            ys.append(Q.bf16_round(y))
    vs = [Q.bf16_round(ys[i] * w1[i]) for i in range(4)]
    ssl = fx.Float32(0.0)
    for v in vs:
        ssl = ssl + Q.opaque(v * v)
    ss = Q.xsum(ssl)
    if _k2(c):
        _put_if(c, lane == 0, c["fmsq"], (t * TOPK + k) * FIN_SLICES + j, ss)
        got = c["poll"]([(c["fmsq"], (t * TOPK + k) * FIN_SLICES + lane, 1),
                         (c["route"], t * 2 * TOPK + TOPK + k, 1)])
        stamp(c, 37)
        tot = Q.xsum(got[0][0].bitcast(fx.Float32))
        wk = got[1][0].bitcast(fx.Float32)
        _fin_out(c, t, j, k, col, vs, tot, wk)
        stamp(c, 38)
        return
    rf = _rsrc(c["fmsq"], s * TOPK * FIN_SLICES * 4)
    if upto == 1:
        bo.buffer_store(ss, rf, (t * TOPK + k) * FIN_SLICES + j, mask=lane == 0)
        return
    tot = Q.xsum(fx.Float32(bo.buffer_load(rf, (t * TOPK + k) * FIN_SLICES + lane, vec_width=1, dtype=T.f32)))
    wk = fx.Float32(bo.buffer_load(_rsrc(a["route"], s * 2 * TOPK * 4), t * 2 * TOPK + TOPK + k, vec_width=1,
                                   dtype=T.f32))
    sc = wk * Q.inv_sqrt_ieee(tot * fx.Float32(1.0 / MH) + fx.Float32(c["eps_post"]))
    for i in range(4):
        fx.ptr_store(Q.opaque(sc * vs[i]), c["red"] + ((k * 4 + i) * 32 + lane))
    gpu.barrier()
    if upto == 2:
        acc = []
        for i in range(4):
            x = fx.Float32(0.0)
            for kq in range(TOPK):
                x = x + fx.ptr_load(c["red"] + ((kq * 4 + i) * 32 + lane))
            acc.append(x)
        fw = fx.Vector.from_elements(acc, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        for p in range(TP):  # fin_gather: the owner's rows to every rank's FIN
            bo.buffer_store(fw, _rsrc(c["fin_peer"][p], s * MH * 2), (t * MH + col) // 2, mask=wave == 0)


# ---------------------------------------------------------------- launchers (one stage a launch)
_STREAM = fx.Stream(None)


@functools.cache
def make_xq(s, act, eps_pre=1e-5):
    @flyc.kernel(name=f"k2g_moe_xq_s{s}_{act}", known_block_size=[XQ_THREADS, 1, 1])
    def kern(pool: Int64, w_pre: Int64, xqb: Int64, xscb: Int64):
        tid = fx.thread_idx.x

        @fx.struct
        class L:
            red: fx.Array[fx.Float32, XQ_THREADS // 32, 16]

        lds = fx.SharedAllocator().allocate(L).peek()
        c = dict(S=s, act=act, tid=tid, lane=tid % 32, wave=tid // 32, red=lds.red.ptr, eps_pre=eps_pre,
                 args=dict(pool=pool, w_pre=w_pre), xqb=xqb, xscb=xscb)
        stage_xq(c, fx.block_idx.x)

    @flyc.jit
    def launch(pool: Int64, w_pre: Int64, xqb: Int64, xscb: Int64, stream: fx.Stream = _STREAM):
        kern(pool, w_pre, xqb, xscb).launch(grid=(s,), block=(XQ_THREADS,), stream=stream)

    return launch


def _route_lds():
    @fx.struct
    class RouteLds:
        ids: fx.Array[fx.Int32, MAX_U, 16]
        fl: fx.Array[fx.Int32, MAX_U, 16]
        expert: fx.Array[fx.Int32, MAX_U + 1, 16]
        kof: fx.Array[fx.Int32, (MAX_U + 1) * 8, 16]

    return RouteLds


def _rt(r):
    return {"ids": r.ids.ptr, "fl": r.fl.ptr, "expert": r.expert.ptr, "kof": r.kof.ptr}


@functools.cache
def make_ug(s, act, ksplit=1, prefetch=16, nt=1):
    waves = 2 * ksplit
    threads = 32 * waves
    assert W13_STEPS % (2 * ksplit) == 0

    @flyc.kernel(name=f"k2g_moe_ug_s{s}_{act}_k{ksplit}_p{prefetch}_nt{nt}", known_block_size=[threads, 1, 1])
    def kern(route: Int64, xqb: Int64, xscb: Int64, w13: Int64, w13s: Int64, interb: Int64, iscb: Int64):
        tid = fx.thread_idx.x
        alloc = fx.SharedAllocator()
        rl = alloc.allocate(_route_lds()).peek()

        @fx.struct
        class L:
            amx: fx.Array[fx.Float32, 32, 16]
            kred: fx.Array[fx.Float32, max(1, waves * 2 * 8 * 32 if ksplit > 1 else 1), 16]

        lds = alloc.allocate(L).peek()
        c = dict(S=s, act=act, tid=tid, lane=tid % 32, wave=tid // 32, threads=threads, ksplit=ksplit,
                 prefetch=prefetch, nt=nt, rt=_rt(rl), amx=lds.amx.ptr, kred=lds.kred.ptr,
                 args=dict(route=route, w13=w13, w13s=w13s), xqb=xqb, xscb=xscb, interb=interb, iscb=iscb)
        nu = load_route(c)
        slot = fx.block_idx.x
        if slot < nu:
            stage_ug(c, slot, fx.block_idx.y)

    @flyc.jit
    def launch(route: Int64, xqb: Int64, xscb: Int64, w13: Int64, w13s: Int64, interb: Int64, iscb: Int64,
               stream: fx.Stream = _STREAM):
        kern(route, xqb, xscb, w13, w13s, interb, iscb).launch(grid=(MAX_U, UG_GROUPS), block=(threads,),
                                                                stream=stream)

    return launch


@functools.cache
def make_down(s, act, waves=4, prefetch=8, nt=1):
    threads = 32 * waves
    assert DOWN_TILES % waves == 0

    @flyc.kernel(name=f"k2g_moe_down_s{s}_{act}_w{waves}_p{prefetch}_nt{nt}", known_block_size=[threads, 1, 1])
    def kern(route: Int64, interb: Int64, iscb: Int64, w2: Int64, w2s: Int64, rank: fx.Int32, yr0: Int64,
             yr1: Int64, yr2: Int64, yr3: Int64):
        tid = fx.thread_idx.x
        rl = fx.SharedAllocator().allocate(_route_lds()).peek()
        c = dict(S=s, act=act, tid=tid, lane=tid % 32, wave=tid // 32, threads=threads, dprefetch=prefetch,
                 nt=nt, rt=_rt(rl), rank=rank, args=dict(route=route, w2=w2, w2s=w2s), interb=interb, iscb=iscb,
                 yr_owner=[yr0, yr1, yr2, yr3])
        nu = load_route(c)
        slot = fx.block_idx.x
        if slot < nu:
            stage_down(c, slot, fx.block_idx.y * waves + Q.uniform(tid // 32))

    @flyc.jit
    def launch(route: Int64, interb: Int64, iscb: Int64, w2: Int64, w2s: Int64, rank: fx.Int32, yr0: Int64,
               yr1: Int64, yr2: Int64, yr3: Int64, stream: fx.Stream = _STREAM):
        kern(route, interb, iscb, w2, w2s, rank, yr0, yr1, yr2, yr3).launch(
            grid=(MAX_U, DOWN_TILES // waves), block=(threads,), stream=stream)

    return launch


@functools.cache
def make_fin(s, upto, eps_post=1e-8):
    @flyc.kernel(name=f"k2g_moe_fin_s{s}_u{upto}", known_block_size=[FIN_THREADS, 1, 1])
    def kern(route: Int64, w_post: Int64, yr: Int64, fmsq: Int64, rank: fx.Int32, fin0: Int64, fin1: Int64,
             fin2: Int64, fin3: Int64):
        tid = fx.thread_idx.x

        @fx.struct
        class L:
            red: fx.Array[fx.Float32, TOPK * 4 * 32, 16]

        lds = fx.SharedAllocator().allocate(L).peek()
        c = dict(S=s, tid=tid, lane=tid % 32, wave=Q.uniform(tid // 32), rank=rank, red=lds.red.ptr,
                 eps_post=eps_post, args=dict(route=route, w_post=w_post), yr=yr, fmsq=fmsq,
                 fin_peer=[fin0, fin1, fin2, fin3])
        b = fx.block_idx.x
        stage_fin(c, b // FIN_SLICES, b % FIN_SLICES, upto)

    @flyc.jit
    def launch(route: Int64, w_post: Int64, yr: Int64, fmsq: Int64, rank: fx.Int32, fin0: Int64, fin1: Int64,
               fin2: Int64, fin3: Int64, stream: fx.Stream = _STREAM):
        kern(route, w_post, yr, fmsq, rank, fin0, fin1, fin2, fin3).launch(
            grid=((s // TP) * FIN_SLICES,), block=(FIN_THREADS,), stream=stream)

    return launch
