# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2's MoE: the Muse Spark pooled MXFP4 MoE from the FFN input row to the
layer's branch output, every TP collective inside the kernel. Adapted from
Kimi-K3's K2b (``stages/moe.py`` of PR #60683: early kinds, route, load_route,
ug, load_inter, down, the rank-ordered reductions).

vLLM replicates the router (fp32 [256, 8192]), the pooler (bf16 [4096, 8192])
and the unpooler (bf16 [8192, 4096]) on every rank; here each rank computes
its eighth of each and the results are all-gathered over the peer buffer.

    router  (4 x 8 experts of this rank's 32, K quarters): fp32 VALU GEMV ->
            parts summed in order -> fp32 logits to every rank's RLOG (AG#2)
    pooler  (32 x 16 rows of this rank's 512, K quarters): bf16 MFMA GEMV ->
            parts summed in order -> bf16 to every rank's POOL (AG#2)
    route   (token t on CTA t): sigmoid, + bias, top-8 (ties: lower id), slot
            order by biased score, weights s_k / (sum + 1e-15) -> ROUTE
    xq      (every CTA): POOL -> pre_expert_norm -> bf16 xe -> MXFP4 (``mxq``)
    ug      (|U| x 16 groups of 32 intermediate columns, one a CTA at q8 when
            |U| <= 32 and two otherwise): a wave a 16-row gate / up tile, one
            scaled-MFMA chain over K = 4096;
            silu(g) u; MXFP4 a 32-group -> INTER / ISC of each picking token
    down    (256 x 16 MH rows, one a CTA): every expert of U against its
            picks' INTER, one fp32 chain over the rank's K = 512 -> bf16 per
            route (unweighted) -> column owner's YR at q8 (otherwise push-all)
    fin     (32 x 128 MH columns, 4 owned slices x 8 token CTAs per rank at q8):
            YR summed over ranks in order -> bf16 y;
            x = y (w_post + 1); per-route sums of squares through FMSQ;
            fin = bf16(sum_k x_k rsqrt(mean(x_k^2) + eps) w_k) -> FIN rows
    unpool  (64 x 16 rows of this rank's 1024 branch columns): bf16 GEMV of
            FIN -> every rank's BR (AG#4)
    gather  (64 x 128 columns): BR -> ``branch_out``

Peer regions are double-buffered by the epoch's parity.
"""

import math

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from sglang.srt.rocm_muse_spark.mono.ops.common.mx import (
    FP4,
    UNIT_SCALE,
    fp4_tile_load,
    mfma_scaled,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    CM_DEV,
    CM_NT,
    ballot,
    bf16_round,
    bf_hi,
    bf_lo,
    block_excl_scan,
    butterfly,
    hw_exp2,
    hw_rsq,
    lanes_below,
    popcount,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_sum,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    pair_layout,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.sync import publish
from sglang.srt.rocm_muse_spark.mono.ops.stages import gemv, mxq

REGIONS = (
    "rpart",
    "ppart",
    "route",
    "xqrdy",
    "ugrdy",
    "fmsq",
    "frdy",
    "rlog",
    "pool",
    "yr",
    "br",
    "owned_fmsq",
    "owned_fin",
)

TP = 8
H = 8192  # hidden
MH = 4096  # MoE hidden (pooled)
E = 256
TOPK = 8
EPR = E // TP  # 32 router rows a rank
RI = 512  # routed intermediate a rank
ROWS = 16
LOG2E = 1.0 / math.log(2.0)
ROUTE_EPS = 1e-15  # router_score_division_eps
ROUTE_SCALE = 1.0  # routed_scaling_factor

R_GROUPS = EPR // WAVES  # 4: 8 experts a task, one a wave
R_PARTS = 4  # K quarters
R_K = H // R_PARTS  # 2048
R_TASKS = R_GROUPS * R_PARTS  # 16
POOL_N = MH // TP  # 512 pooled columns a rank
P_GROUPS = POOL_N // ROWS  # 32
P_PARTS = 4
P_K = H // P_PARTS  # 2048
P_TASKS = P_GROUPS * P_PARTS  # 128
EARLY_K = R_K  # the x part an early task stages (router and pooler alike)
assert P_K == R_K

W13_K = MH  # 4096
W13_STEPS = W13_K // 128  # 32
W13_SCALE_G = W13_K // 32  # 128 scale columns a w13 row
W2_STEPS = RI // 128  # 4
W2_SCALE_G = RI // 32  # 16
UG_GROUPS = RI // 32  # 16 32-column groups a picked expert
UG_UNITS = 2  # 32-column groups a CTA task (4 waves each)
UG_SINGLE_GROUP_ROWS = (2,)
UG_Q8_ONE_GROUP_MAX_U = 32
INTER_W = RI // 8  # 64 E2M1 words a route
MAX_U = 8 * TOPK  # 64: the union of a step's picks
DOWN_TASKS = MH // ROWS  # 256
DOWN_BATCH = 4  # maximum experts a wave keeps in flight together
FIN_COLS = 128
FIN_TASKS = MH // FIN_COLS  # 32
OWNED_FIN_TASKS = FIN_TASKS // TP  # q8: four 128-column slices per rank
OWNER_FIN_CTAS = OWNED_FIN_TASKS * 8  # q8: one CTA per owned slice and token
UP_N = H // TP  # 1024 branch columns a rank
UP_TASKS = UP_N // ROWS  # 64
GATHER_COLS = 128
GATHER_TASKS = H // GATHER_COLS  # 64
XQ_W = MH // 8  # 512 E2M1 words an xe row
XQ_G = MH // 32  # 128 scale codes an xe row
XQ0 = 8  # the xe quantizing CTAs: XQ0 + t for token t
UG_TASKS_BASE = MAX_U * UG_GROUPS // UG_UNITS  # 512
UG_TASKS_MAX = UG_TASKS_BASE  # both adaptive q8 branches are bounded by 512
FIN_BATCH = 32  # YR pairs a fin poll round
assert UG_TASKS_MAX <= THREADS


def ug_units(s):
    """Number of 32-column groups assigned to one UG task at width ``s``."""
    return 1 if s in UG_SINGLE_GROUP_ROWS else UG_UNITS


def ug_task_capacity(s):
    """UGRDY capacity; the adaptive q8 threshold keeps every path <= 512."""
    return UG_TASKS_BASE


def peer_pairs(s):
    """One parity's MoE peer regions, pairs: name -> count."""
    return {
        "rlog": TP * s * EPR,  # [src][t][32] fp32
        "pool": s * MH // 2,  # [t][4096] bf16 pairs
        "yr": TP * s * TOPK * MH // 2,  # [src][t][k][4096] bf16 pairs
        "br": s * H // 2,  # [t][8192] bf16 pairs
        "owned_fmsq": FIN_TASKS * s * TOPK,  # [slice][t][k] fp32
        "owned_fin": s * MH // 2,  # [t][4096] bf16 pairs; distinct from POOL
    }


def scratch_pairs(s):
    return {
        "rpart": R_TASKS * s * WAVES,
        "ppart": P_TASKS * s * ROWS,
        "route": s * 2 * TOPK,
        "xqrdy": s,
        "ugrdy": ug_task_capacity(s),
        "fmsq": FIN_TASKS * s * TOPK,
        "frdy": FIN_TASKS,
    }


# plain (device-scope) data a flag announces: name -> bytes
def scratch_plain(s):
    return {
        "xqb": s * XQ_W * 4,  # xe E2M1 words [t][512] (XQRDY)
        "xscb": s * XQ_G * 4,  # xe scale codes [t][128], a word each
        "interb": s * TOPK * INTER_W * 4,  # a2 E2M1 words [pick][64] (UGRDY)
        "iscb": s * TOPK * UG_GROUPS * 4,  # a2 codes [pick][16], a word each
        "fin": s * MH * 2,  # fin rows bf16 [t][4096] (FRDY)
    }


def scratch_layout(s: int, start: int = 0) -> dict:
    lay = pair_layout(scratch_pairs(s).items(), start=start)
    for name, n in scratch_plain(s).items():
        end = max(o + m for o, m in lay.values())
        lay[name] = ((end + 15) // 16 * 16, n)
    return lay


@traced
def dev_copy(tid, src, dst, words):
    """``words`` (a multiple of 4) words of plain device-scope data at
    ``src`` -> LDS ``dst``, 16 B a lane."""
    n = words // 4
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, n - 1)
        v = fx.Vector(
            bo.buffer_load(
                rsrc(src), e * 4, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
            )
        )
        if tid + THREADS * i < n:
            fx.ptr_store(v, dst + e * 4)


def f32_of(w):
    return w.bitcast(fx.Float32)


def bo_ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def scale_dword(m, g, cols):
    """The dword of ``e8m0_shuffle``'d scales [rows, cols] holding (row m,
    column g); its byte is ``scale_byte(m, g)``."""
    return (m >> 5) * (8 * cols) + (g >> 3) * 64 + (g & 3) * 16 + (m & 15)


def scale_shift(m, g):
    """The bit shift of (row m, column g)'s byte in its ``scale_dword``."""
    return (((g >> 2) & 1) * 2 + ((m >> 4) & 1)) * 8


# ---------------------------------------------------------------- early
@traced
def early_x(c, q):
    """The x rows' K part q -> LDS after that part's slices are XRDY."""
    tid = c["tid"]
    if tid < c["nsl"] // R_PARTS:
        c["poll"]([(c["xrdy"], q * (c["nsl"] // R_PARTS) + tid, 1)])
    gpu.barrier()
    gemv.rows_to_lds(
        tid,
        c["xbuf"],
        R_K,
        c["S"],
        c["xl"],
        R_K + gemv.LDS_PAD,
        CM_DEV,
        ldk=H,
        k0=q * R_K,
    )


def router_loads(c, i, q):
    """Router task (i, q): expert rank 32 + 8 i + wave, K quarter q: a lane 4
    fp32 every 256 (8 dwordx4)."""
    lane, wave = c["lane"], c["wave"]
    e = c["rank"] * EPR + i * WAVES + wave
    r_w = rsrc(c["args"]["w_router"])
    base = e * H + q * R_K + lane * 4
    return [
        fx.Vector(
            bo.buffer_load(
                r_w, base + 256 * j, vec_width=4, dtype=T.f32, cache_modifier=CM_NT
            )
        )
        for j in range(R_K // 256)
    ]


@traced
def router_compute(c, i, q, ops):
    """Router task (i, q): fp32 dots of the x part (bf16 -> fp32) with the
    wave's expert row; a wave's sum per token; parts q > 0 -> RPART, part 0
    adds them in part order -> every rank's RLOG."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    early_x(c, q)
    xrow = EARLY_K + gemv.LDS_PAD
    for t in range_constexpr(s):
        acc = fx.Float32(0.0)
        for j in range_constexpr(R_K // 256):
            xv = fx.Vector(
                fx.ptr_load(
                    c["xl"] + (t * xrow + 256 * j + lane * 4) // 2,
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
            fx.ptr_store(v, c["av"] + (t * WAVES + wave))
    gpu.barrier()
    if tid < s * WAVES:
        v = fx.ptr_load(c["av"] + tid)
        at = (i * R_PARTS + q) * s * WAVES + tid
        if q > 0:
            c["put"](c["rpart"], at, v)
        if q == 0:
            got = c["poll"](
                [(c["rpart"], at + qq * s * WAVES, 1) for qq in range(1, R_PARTS)]
            )
            for qq in range_constexpr(R_PARTS - 1):
                v = v + f32_of(got[qq][0])
            t = tid // WAVES
            w = tid % WAVES
            for p in range_constexpr(TP):
                c["put"](
                    c["peer"]["rlog"][p], (c["rank"] * s + t) * EPR + i * WAVES + w, v
                )
    gpu.barrier()


def pool_loads(c, i, q):
    """Pooler task (i, q): rows rank 512 + 16 i .., K quarter q."""
    return gemv.loads(
        c["lane"],
        c["wave"],
        c["args"]["w_pool"],
        c["rank"] * POOL_N + i * ROWS,
        P_K,
        ldk=H,
        k0=q * P_K,
    )


@traced
def pool_compute(c, i, q, ops):
    """Pooler task (i, q): bf16 MFMA GEMV of the x part; parts q > 0 -> PPART,
    part 0 adds them in part order -> bf16 -> every rank's POOL."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    early_x(c, q)
    xrow = P_K + gemv.LDS_PAD
    acc = gemv.mfmas(lane, wave, c["xl"], xrow, P_K, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * s:
        r = tid % ROWS
        t = tid // ROWS
        v = row_sum(red, r, t)
        at = (i * P_PARTS + q) * s * ROWS + tid
        if q > 0:
            c["put"](c["ppart"], at, v)
        if q == 0:
            got = c["poll"](
                [(c["ppart"], at + qq * s * ROWS, 1) for qq in range(1, P_PARTS)]
            )
            for qq in range_constexpr(P_PARTS - 1):
                v = v + f32_of(got[qq][0])
            fx.ptr_store(v, c["av"] + tid)
    gpu.barrier()
    if (q == 0) & (tid < (ROWS // 4) * s):
        rq = tid % (ROWS // 4)
        t = tid // (ROWS // 4)
        vs = [
            bf16_round(fx.ptr_load(c["av"] + (t * ROWS + 4 * rq + d))) for d in range(4)
        ]
        col = c["rank"] * POOL_N + i * ROWS + 4 * rq
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["pool"][p], t * MH + col, vs)
    gpu.barrier()


# ---------------------------------------------------------------- route
def _okey(v):
    """An order-preserving uint32 key of an f32 (larger float, larger key)."""
    b = v.bitcast(fx.Int32)
    neg = b < fx.Int32(0)
    return fx.Uint32(neg.select(~b, b ^ fx.Int32(-2147483648)))


@traced
def stage_route(c, t):
    """Token t (wave 0): 256 logits, 4 a lane (expert lane + 64 i) -> sigmoid
    (IEEE 1 / (1 + exp2(-x log2 e))) + bias -> the top-8 by a radix select
    (equal keys: the lower experts) -> ranked by biased score, descending (the
    routing kernel's slot order) -> the unbiased sigmoids over their sum + eps
    (summed in slot order, IEEE 1 / sum) -> ROUTE."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    a = c["args"]
    if wave == 0:
        s = c["S"]
        per = E // 64
        bias = [bo_ld_f32(a["expert_bias"], lane + 64 * i) for i in range(per)]
        specs = []
        for i in range_constexpr(per):
            e = lane + 64 * i
            specs.append(
                (c["peer"]["rlog_own"], ((e // EPR) * s + t) * EPR + e % EPR, 1)
            )
        got = c["poll"](specs)
        keys, sigs = [], []
        for i in range_constexpr(per):
            # Current SGLang applies the model's output multiplier to the raw
            # router logits before sigmoid.  It is a compile-time scalar here;
            # frozen original-vLLM fixtures use the default 1.0.
            lg = f32_of(got[i][0]) * fx.Float32(c["router_output_multiplier"])
            sg = fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(lg * fx.Float32(-LOG2E)))
            sc = sg + bias[i]
            # a NaN logit (a warmup's garbage row) ranks last
            keys.append(_okey((sc == sc).select(sc, fx.Float32(-3.0e38))))  # noqa: PLR0124
            sigs.append((sg == sg).select(sg, fx.Float32(0.0)))  # noqa: PLR0124

        def count_ge(th):
            n = fx.Int32(0)
            for i in range_constexpr(per):
                n = n + popcount(ballot(keys[i] >= th))
            return n

        th = fx.Uint32(0)
        for bit in range_constexpr(31, -1, -1):
            cand = th | fx.Uint32(1 << bit)
            th = (count_ge(cand) >= TOPK).select(cand, th)
        # selected: key > th, then the lowest experts with key == th
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
                fx.ptr_store(
                    fx.Int32(lane + 64 * i).bitcast(fx.Float32), red + (pos * 3 + 1)
                )
                fx.ptr_store(sigs[i], red + (pos * 3 + 2))
            eq_before = eq_before + popcount(eqb)
            sel_before = sel_before + popcount(selb)
        # rank the 8 by key descending, the lower expert first on a tie
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
        # the sigmoids in slot order, summed serially (the routing kernel)
        if lane < TOPK:
            fx.ptr_store(ms, red + (64 + rank))
        total = fx.Float32(0.0)
        for j in range_constexpr(TOPK):
            total = total + fx.ptr_load(red + (64 + j))
        inv = fx.Float32(ROUTE_SCALE) / (total + fx.Float32(ROUTE_EPS))
        if lane < TOPK:
            c["put"](
                c["route"], t * 2 * TOPK + rank, fx.min(fx.max(me, 0), fx.Int32(E - 1))
            )
            c["put"](c["route"], t * 2 * TOPK + TOPK + rank, ms * inv)
    gpu.barrier()


@traced
def load_route(c):
    """ROUTE -> LDS (ids, weights); the union U: slot_of[e], expert_of[slot],
    kof[slot][t] (the token's top-k index, -1 if not picked); returns |U|."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    n = s * 2 * TOPK
    if tid < n:
        v = c["poll"]([(c["route"], tid, 1)])[0][0]
        fx.ptr_store(v, rt["route"] + tid)
    for i in range_constexpr((E + THREADS - 1) // THREADS):
        e = tid + THREADS * i
        if e < E:
            fx.ptr_store(fx.Int32(0), rt["flag"] + e)
    for i in range_constexpr((MAX_U * 8 + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < MAX_U * 8:
            fx.ptr_store(fx.Int32(-1), rt["kof"] + j)
    gpu.barrier()
    if tid < s * TOPK:
        t = tid // TOPK
        k = tid % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * TOPK + k)), E - 1), 0)
        fx.ptr_store(fx.Int32(1), rt["flag"] + e)
    gpu.barrier()
    e0 = fx.min(2 * tid, E - 1)
    f0 = (2 * tid < E).select(fx.ptr_load(rt["flag"] + e0), fx.Int32(0))
    f1 = (2 * tid + 1 < E).select(
        fx.ptr_load(rt["flag"] + fx.min(2 * tid + 1, E - 1)), fx.Int32(0)
    )
    off, tot = block_excl_scan(f0 + f1, lane, wave, c["scan"])
    gpu.barrier()
    if f0 == 1:
        fx.ptr_store(off, rt["flag"] + 2 * tid)
        fx.ptr_store(fx.Int32(2 * tid), rt["expert"] + off)
    if f1 == 1:
        fx.ptr_store(off + f0, rt["flag"] + 2 * tid + 1)
        fx.ptr_store(fx.Int32(2 * tid + 1), rt["expert"] + off + f0)
    gpu.barrier()
    if tid < s * TOPK:
        t = tid // TOPK
        k = tid % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * TOPK + k)), E - 1), 0)
        slot = fx.ptr_load(rt["flag"] + e)
        fx.ptr_store(fx.Int32(k), rt["kof"] + (slot * 8 + t))
    gpu.barrier()
    return uniform(tot)


# ---------------------------------------------------------------- xq
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
    xe = [bf16_round((xs[k] * r) * (ws[k] + fx.Float32(1.0))) for k in range(8)]
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


@traced
def load_xq(c):
    """Every token's XQ rows (after XQRDY) -> LDS xq words, xsc codes."""
    s, tid = c["S"], c["tid"]
    if tid < s:
        c["poll"]([(c["xqrdy"], tid, 1)])
    gpu.barrier()
    dev_copy(tid, c["xqb"], c["xq"], s * XQ_W)
    dev_copy(tid, c["xscb"], c["xsc"], s * XQ_G)
    gpu.barrier()


# ---------------------------------------------------------------- ug
def ug_tile(c, task, units=UG_UNITS):
    """This wave's (slot, group, 16-row tile) of ug task ``task``: waves 4 u ..
    4 u + 3 take unit 2 task + u, wave 4 u + w its tile w (gate 2 g, gate 2 g +
    1, up 2 g, up 2 g + 1 of the 32-column group g)."""
    # A narrow-width build may schedule one 32-column group per CTA.  Clamp
    # the otherwise-idle waves onto the live unit so every derived address is
    # in range; their global loads are masked and their MFMA chain is skipped.
    wave = fx.min(c["wave"], 4 * units - 1)
    unit = task * units + wave // 4
    slot = unit // UG_GROUPS
    g = unit % UG_GROUPS
    wl = wave % 4
    rg = (wl // 2) * (RI // ROWS) + 2 * g + wl % 2
    return slot, g, rg


def ug_loads(c, task, units=UG_UNITS):
    """Task ``task``'s w13 tile for this wave (all 32 K steps) and its scale
    dwords (one a step pair)."""
    lane = c["lane"]
    a, rt = c["args"], c["rt"]
    slot, _g, rg = ug_tile(c, task, units)
    active = c["wave"] < 4 * units
    e = uniform(fx.ptr_load(rt["expert"] + slot))
    r_w = rsrc(a["w13"])
    base = e * (2 * RI * W13_K // 8)
    av = [
        fx.Vector(
            bo.buffer_load(
                r_w,
                base + rg * (2 * W13_K) + st * 256 + lane * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
                mask=active,
            )
        )
        for st in range(W13_STEPS)
    ]
    r_s = rsrc(a["w13s"])
    m = e * (2 * RI) + rg * ROWS + lane % 16
    sw = [
        fx.Int32(
            bo.buffer_load(
                r_s,
                scale_dword(m, sp * 8 + lane // 16, W13_SCALE_G),
                vec_width=1,
                dtype=T.i32,
                mask=active,
            )
        )
        for sp in range(W13_STEPS // 2)
    ]
    return av + sw


@traced
def stage_ug(c, task, ops, units=UG_UNITS):
    """Task ``task``: one or two 32-column groups, a wave a 16-row tile against
    the MXFP4 xe (one MFMA chain over K); silu(g) u; MXFP4 a 32-group -> INTER /
    ISC of each picking token."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt = c["rt"]
    _slot, _g, rg = ug_tile(c, task, units)
    av = ops[:W13_STEPS]
    sw = ops[W13_STEPS:]
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    if wave < 4 * units:
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for st in range_constexpr(W13_STEPS):
            bw = fx.Vector(
                fx.ptr_load(
                    c["xq"] + (t * XQ_W + st * 16 + kb * 4),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            sb = fx.ptr_load(c["xsc"] + (t * XQ_G + st * 4 + kb))
            sa = (sw[st // 2] >> ((st % 2) * 16 + (rg % 2) * 8)) & 0xFF
            acc = mfma_scaled(av[st], bw, acc, sa, sb, FP4, FP4, 0, 0)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    # thread (unit, token, column): g / u of its tile row, silu(g) u
    active_thread = tid < 256 * units
    u = fx.min(tid // 256, units - 1)
    tt = (tid % 256) // 32
    col = tid % 32
    live = active_thread & (tt < s)
    tt = fx.min(tt, s - 1)
    half = col // 16
    r = col % 16
    gv = fx.ptr_load(red + (((4 * u + half) * 64 + 16 * (r // 4) + tt) * 4 + r % 4))
    uv = fx.ptr_load(red + (((4 * u + 2 + half) * 64 + 16 * (r // 4) + tt) * 4 + r % 4))
    v = mxq.silu_mul(gv, uv)
    # Current serving AITER uses a two-stage q8 expert path: stage 1 stores
    # SiLU(gate) * up as BF16, then a separate kernel derives the MXFP4
    # scale/codes from that BF16 tensor.  The frozen original flat kernel
    # quantizes its FP32 epilogue directly; keep that arithmetic for one-row
    # decode.  This width guard intentionally also changes a q1C8 batch.
    if s == 8:
        v = bf16_round(v)
    amax = butterfly(abs(v), (16, 8, 4, 2, 1), fx.max)
    code = mxq.inter_code(amax)
    gpu.barrier()
    if active_thread:
        fx.ptr_store(v, c["av"] + tid)
    gpu.barrier()
    uslot = (task * units + u) // UG_GROUPS
    ug = (task * units + u) % UG_GROUPS
    k = fx.ptr_load(rt["kof"] + (uslot * 8 + tt))
    if live & (col < 4) & (k >= 0):
        scale = mxq.code_scale(code)
        vals = [fx.ptr_load(c["av"] + (tid - col + col * 8 + j)) for j in range(8)]
        pk = mxq.pack8(vals, scale)
        pick = tt * TOPK + k
        bo.buffer_store(
            pk, rsrc(c["interb"]), pick * INTER_W + ug * 4 + col, cache_modifier=CM_DEV
        )
        if col == 0:
            bo.buffer_store(
                code, rsrc(c["iscb"]), pick * UG_GROUPS + ug, cache_modifier=CM_DEV
            )
    publish(c["put"], c["ugrdy"], task, fx.Int32(1), tid == 0)
    gpu.barrier()


@traced
def run_ug(c, nu, units=UG_UNITS):
    """Every ug task of this CTA: tasks 255 - bid, + 256, ...: the leftover
    tasks of a round go to the high CTAs, whose early work ends first."""
    total = nu * (UG_GROUPS // units)
    task = BLOCKS - 1 - c["bid"]
    while task < total:
        ops = ug_loads(c, task, units)
        stage_ug(c, task, ops, units)
        task = task + BLOCKS


@traced
def run_ug_q8(c, nu):
    """Use one group per task only for the profiled low-union q8 regime."""
    if nu <= UG_Q8_ONE_GROUP_MAX_U:
        run_ug(c, nu, 1)
    else:
        run_ug(c, nu, UG_UNITS)


# ---------------------------------------------------------------- down
@traced
def load_inter(c, nu, units=UG_UNITS):
    """Every ug task's UGRDY flag (a thread a task), then the picks' plain
    INTER words and ISC codes -> LDS."""
    s, tid = c["S"], c["tid"]
    if tid < nu * (UG_GROUPS // units):
        c["poll"]([(c["ugrdy"], tid, 1)])
    gpu.barrier()
    dev_copy(tid, c["interb"], c["interq"], s * TOPK * INTER_W)
    dev_copy(tid, c["iscb"], c["intsc"], s * TOPK * UG_GROUPS)
    gpu.barrier()


@traced
def load_inter_q8(c, nu):
    """Wait for the same route-count-selected q8 UG schedule as ``run_ug_q8``."""
    if nu <= UG_Q8_ONE_GROUP_MAX_U:
        load_inter(c, nu, 1)
    else:
        load_inter(c, nu, UG_UNITS)


def down_loads(c, rgd, nu, b0, batch=DOWN_BATCH):
    """The w2 tiles and scale dwords of a wave's batch from slot b0 (slots b0,
    b0 + 8, ...; past |U| the last slot's): flat, ``stage_down`` unpacks."""
    lane = c["lane"]
    r_w = rsrc(c["args"]["w2"])
    r_s = rsrc(c["args"]["w2s"])
    rt = c["rt"]
    slots = [fx.min(b0 + WAVES * j, nu - 1) for j in range(batch)]
    es = [uniform(fx.ptr_load(rt["expert"] + sl)) for sl in slots]
    av = [
        fp4_tile_load(r_w, es[j] * (MH * RI // 8), rgd, RI, st, lane * 4)
        for j in range(batch)
        for st in range(W2_STEPS)
    ]
    sw = [
        fx.Int32(
            bo.buffer_load(
                r_s,
                scale_dword(
                    es[j] * MH + rgd * ROWS + lane % 16, sp * 8 + lane // 16, W2_SCALE_G
                ),
                vec_width=1,
                dtype=T.i32,
            )
        )
        for j in range(batch)
        for sp in range(W2_STEPS // 2)
    ]
    return av + sw


@traced
def stage_down(c, rgd, nu, ops0, batch=DOWN_BATCH, prefetch_next=True):
    """MH rows 16 rgd ..: every slot's w2 against its picks (wave w the slots
    w, w + 8, ...; a batch of ``batch`` in flight, the next issued under this
    one's MFMAs) -> bf16(acc) a (token, k) in LDS -> YR (only the column
    owner at q8, every rank otherwise)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    nbs = batch * W2_STEPS
    b0 = wave
    ops = ops0
    while b0 < nu:
        nb = b0 + WAVES * batch
        # rows=2 has nu <= 16 and WAVES * batch == 16, so this loop can
        # execute only once. Avoid keeping an unused next-batch load live in
        # that specialization; wider shapes retain the existing prefetch.
        if const_expr(prefetch_next):
            ops_n = down_loads(c, rgd, nu, fx.min(nb, nu - 1), batch)
        else:
            ops_n = ops
        slots = [fx.min(b0 + WAVES * j, nu - 1) for j in range(batch)]
        for j in range_constexpr(batch):
            live = (b0 + WAVES * j) < nu
            k = fx.ptr_load(rt["kof"] + (slots[j] * 8 + t))
            picked = (k >= 0) & (lane % 16 < s)
            pick = t * TOPK + fx.max(k, 0)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(W2_STEPS):
                bw = fx.Vector(
                    fx.ptr_load(
                        c["interq"] + (pick * INTER_W + st * 16 + kb * 4),
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                scode = fx.ptr_load(c["intsc"] + (pick * UG_GROUPS + st * 4 + kb))
                bw = fx.Vector.from_elements(
                    [picked.select(bw[d], fx.Int32(0)) for d in range(4)],
                    fx.Int32,
                )
                sb = picked.select(scode, fx.Int32(UNIT_SCALE))
                sw = ops[nbs + j * (W2_STEPS // 2) + st // 2]
                sa = (sw >> ((st % 2) * 16 + (rgd % 2) * 8)) & 0xFF
                acc = mfma_scaled(
                    ops[j * W2_STEPS + st], bw, acc, sa, sb, FP4, FP4, 0, 0
                )
            # lane: rows 4 (lane / 16) + i of token lane % 16
            if live & picked:
                for i in range_constexpr(4):
                    row = 4 * (lane // 16) + i
                    fx.ptr_store(
                        bf16_round(acc[i]),
                        c["contrib"] + ((t * TOPK + k) * ROWS + row),
                    )
        b0 = nb
        ops = ops_n
    gpu.barrier()
    if tid < s * TOPK * (ROWS // 4):
        rq = tid % (ROWS // 4)
        route = tid // (ROWS // 4)  # t TOPK + k
        ys = [fx.ptr_load(c["contrib"] + (route * ROWS + 4 * rq + d)) for d in range(4)]
        if const_expr(s == 8):
            # This CTA's 16 columns belong to rank rgd // 32. The owner
            # address is loaded before the expert work; no replicated sends.
            c["put_bf"](
                c["yr_owner"],
                (c["rank"] * s * TOPK + route) * MH + rgd * ROWS + 4 * rq,
                ys,
            )
        else:
            for p in range_constexpr(TP):
                c["put_bf"](
                    c["peer"]["yr"][p],
                    (c["rank"] * s * TOPK + route) * MH + rgd * ROWS + 4 * rq,
                    ys,
                )
    gpu.barrier()


# ---------------------------------------------------------------- finalize
@traced
def _stage_fin_wave(c, j, t, col, live):
    """Collect and finalize one token wave; the caller owns CTA publication."""
    s, lane = c["S"], c["lane"]
    a, rt = c["args"], c["rt"]
    wp = fx.Int32(bo.buffer_load(rsrc(a["w_post"]), col // 2, vec_width=1, dtype=T.i32))
    w0 = bf_lo(wp) + fx.Float32(1.0)
    w1 = bf_hi(wp) + fx.Float32(1.0)
    specs = [
        (c["peer"]["yr_own"], (((src * s + t) * TOPK + k) * MH + col) // 2, 1)
        for k in range(TOPK)
        for src in range(TP)
    ]
    got = c["poll"](specs, batch=FIN_BATCH)
    c["stamp"](14)
    xs = []
    for k in range_constexpr(TOPK):
        lo, hi = bf_lo(got[k * TP][0]), bf_hi(got[k * TP][0])
        for src in range_constexpr(1, TP):
            lo = lo + bf_lo(got[k * TP + src][0])
            hi = hi + bf_hi(got[k * TP + src][0])
        x0 = bf16_round(lo) * w0
        x1 = bf16_round(hi) * w1
        xs.append((x0, x1))
        sq = wave_sum(x0 * x0 + x1 * x1)
        if live & (lane == 0):
            if const_expr(s == 8):
                for p in range_constexpr(TP):
                    c["put"](c["peer"]["owned_fmsq"][p], (j * s + t) * TOPK + k, sq)
            else:
                c["put"](c["fmsq"], (j * s + t) * TOPK + k, sq)
    q = lane % FIN_TASKS
    means = c["peer"]["owned_fmsq_own"] if s == 8 else c["fmsq"]
    got = c["poll"]([(means, (q * s + t) * TOPK + k, 1) for k in range(TOPK)])
    c["stamp"](15)
    f0 = fx.Float32(0.0)
    f1 = fx.Float32(0.0)
    for k in range_constexpr(TOPK):
        v = (lane < FIN_TASKS).select(f32_of(got[k][0]), fx.Float32(0.0))
        tot = wave_sum(v)
        tw = fx.ptr_load(rt["route"] + (t * 2 * TOPK + TOPK + k)).bitcast(fx.Float32)
        sc = hw_rsq(tot * fx.Float32(1.0 / MH) + fx.Float32(c["eps_post"])) * tw
        f0 = f0 + xs[k][0] * sc
        f1 = f1 + xs[k][1] * sc
    fw = fx.Vector.from_elements([f0, f1], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    if live:
        if const_expr(s == 8):
            for p in range_constexpr(TP):
                c["put_words"](c["peer"]["owned_fin"][p], (t * MH + col) // 2, [fw[0]])
        else:
            bo.buffer_store(
                fw[0], rsrc(c["fin"]), (t * MH + col) // 2, cache_modifier=CM_DEV
            )


@traced
def stage_fin(c, j):
    """MH columns 128 j ..: each route's 8 ranks' YR in rank order (fp32 ->
    bf16 y), x = y (w_post + 1), its sum of squares -> FMSQ; every task's ->
    rstd; fin = bf16(sum_k x_k rstd_k w_k).

    At eight token rows (independent of query length), only the column owner
    runs this stage. FMSQ and FIN go to distinct tagged peer regions; the local
    receive stage publishes FRDY. Other widths use local FMSQ and FIN/FRDY.

    Rows=2 has only two live token waves. Keep its other six waves out of the
    tagged YR/FMSQ polls: clamping them to token 1 issued four copies of every
    readiness read and contended with the two consumers that produce output.
    """
    s, lane, wave = c["S"], c["lane"], c["wave"]
    live = wave < s
    t = fx.min(wave, s - 1)
    col = j * FIN_COLS + 2 * lane
    if const_expr(s == 2):
        if live:
            _stage_fin_wave(c, j, t, col, live)
    else:
        _stage_fin_wave(c, j, t, col, live)
    if const_expr(s != 8):
        publish(c["put"], c["frdy"], j, fx.Int32(1), c["tid"] == 0)
    gpu.barrier()


@traced
def stage_fin_token(c, j, t):
    """Finalize one q8 ``(128-column slice, token)`` on this CTA's wave 0.

    The other waves issue no tagged polls. All threads still reach the CTA
    barrier before this CTA proceeds to the unchanged FIN gather stage.
    """
    lane, wave = c["lane"], c["wave"]
    if wave == 0:
        _stage_fin_wave(c, j, t, j * FIN_COLS + 2 * lane, True)
    gpu.barrier()


@traced
def stage_fin_gather(c, j):
    """Materialize one tagged all-gather slice into local FIN, then FRDY.

    Only 32 CTAs poll the peer payload, rather than every unpool CTA loading
    every tagged word. Device-scope publication retains the original unpool
    loads and MFMA arithmetic. No plain remote store/flag hand-off is used.
    """
    t, col = c["wave"], j * FIN_COLS + 2 * c["lane"]
    at = (t * MH + col) // 2
    word = c["poll"]([(c["peer"]["owned_fin_own"], at, 1)])[0][0]
    bo.buffer_store(word, rsrc(c["fin"]), at, cache_modifier=CM_DEV)
    publish(c["put"], c["frdy"], j, fx.Int32(1), c["tid"] == 0)
    gpu.barrier()


# ---------------------------------------------------------------- unpooler / gather
def up_loads(c, j):
    """Unpooler task j: rows (branch columns) rank 1024 + 16 j .., K 4096."""
    return gemv.loads(
        c["lane"], c["wave"], c["args"]["w_unpool"], c["rank"] * UP_N + j * ROWS, MH
    )


@traced
def stage_up(c, j, ops):
    """Branch columns rank 1024 + 16 j ..: FIN (after every FRDY) GEMV -> bf16
    -> every rank's BR."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    if tid < FIN_TASKS:
        c["poll"]([(c["frdy"], tid, 1)])
    gpu.barrier()
    lrow = MH + gemv.LDS_PAD
    gemv.dev_rows_to_lds(tid, c["fin"], MH, s, c["xl"], lrow)
    acc = gemv.mfmas(lane, wave, c["xl"], lrow, MH, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 4) * s:
        rq = tid % (ROWS // 4)
        tt = tid // (ROWS // 4)
        vs = [bf16_round(row_sum(red, 4 * rq + d, tt)) for d in range(4)]
        col = c["rank"] * UP_N + j * ROWS + 4 * rq
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["br"][p], tt * H + col, vs)
    gpu.barrier()


@traced
def stage_gather(c, j):
    """Branch columns 128 j ..: BR (every rank's columns) -> ``branch_out``."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    live = wave < s
    t = fx.min(wave, s - 1)
    col = j * GATHER_COLS + 2 * lane
    w = c["poll"]([(c["peer"]["br_own"], (t * H + col) // 2, 1)])[0][0]
    if live:
        bo.buffer_store(w, rsrc(c["args"]["branch_out"]), (t * H + col) // 2)
    gpu.barrier()
