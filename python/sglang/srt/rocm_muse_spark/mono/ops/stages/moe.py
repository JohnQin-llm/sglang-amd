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
    fresh,
    hw_exp2,
    hw_rsq,
    lanes_below,
    popcount,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_incl_scan,
    wave_sum,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    enter_stage,
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
# q8 two-CTA build: each router task computes two tokens; four token groups
# give 64 tasks (the default builds keep 16 tasks of every token).
ROUTER_PAIR = 2
R_TOKEN_GROUPS_Q8 = 8 // ROUTER_PAIR  # 4
R_TASKS_Q8 = R_TASKS * R_TOKEN_GROUPS_Q8  # 64
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
W13_BYTES = E * (2 * RI) * W13_K // 2  # packed MXFP4 [E, 2RI, K]
W13_SCALE_BYTES = E * (2 * RI) * W13_SCALE_G  # E8M0 [E, 2RI, K/32]
W2_STEPS = RI // 128  # 4
W2_SCALE_G = RI // 32  # 16
W2_BYTES = E * MH * RI // 2  # packed MXFP4 [E, MH, RI]
W2_SCALE_BYTES = E * MH * W2_SCALE_G  # E8M0 [E, MH, RI/32]
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
UP_PARALLEL_SHARDS = 4
UP_PARALLEL_TASKS = UP_PARALLEL_SHARDS * UP_TASKS  # 256
UP_PARTIAL_BYTES = UP_TASKS * WAVES * 8 * ROWS * 4  # 262144
UP_PARTIAL_READY_PAIRS = UP_PARALLEL_TASKS
GATHER_COLS = 128
GATHER_TASKS = H // GATHER_COLS  # 64
XQ_W = MH // 8  # 512 E2M1 words an xe row
XQ_G = MH // 32  # 128 scale codes an xe row
XQ0 = 8  # the xe quantizing CTAs: XQ0 + t for token t
UG_TASKS_BASE = MAX_U * UG_GROUPS // UG_UNITS  # 512
UG_TASKS_MAX = UG_TASKS_BASE  # default builds retain the inherited 512 flags
UG_TASKS_WAVE_SPLITK = MAX_U * UG_GROUPS  # opt-in q8: one group per task
FIN_BATCH = 32  # YR pairs a fin poll round
# K2 keeps points 0 .. 16 unchanged.  Its q8 expert diagnostic reserves one
# nine-point range for each of the team's possible slots (team, team + 32).
EXPERT_TL_BASE = 17
EXPERT_TL_POINTS = 9
assert UG_TASKS_MAX <= THREADS
assert UG_TASKS_WAVE_SPLITK == 2 * THREADS


def ug_units(s):
    """Number of 32-column groups assigned to one UG task at width ``s``."""
    return 1 if s in UG_SINGLE_GROUP_ROWS else UG_UNITS


def ug_task_capacity(s, *, wave_splitk=False):
    """UGRDY capacity; only the opt-in q8 split-K build needs 1024 flags."""
    if wave_splitk:
        if s != 8:
            raise ValueError("wave_splitk is restricted to tokens == 8")
        return UG_TASKS_WAVE_SPLITK
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


def scratch_pairs(s, *, wave_splitk=False):
    return {
        "rpart": R_TASKS * s * WAVES,
        "ppart": P_TASKS * s * ROWS,
        "route": s * 2 * TOPK,
        "xqrdy": s,
        "ugrdy": ug_task_capacity(s, wave_splitk=wave_splitk),
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


def scratch_layout(s: int, start: int = 0, *, wave_splitk=False) -> dict:
    lay = pair_layout(scratch_pairs(s, wave_splitk=wave_splitk).items(), start=start)
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
    if const_expr(c["quarter_seam"]):
        if tid < c["S"]:
            c["poll"]([(c["xrdy"], tid * R_PARTS + q, 1)])
    else:
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


@traced
def router_compute_pair(c, i, q, tg, ops):
    """Router task (i, q) for tokens 2 tg and 2 tg + 1 (q8 two-CTA build).

    Exactly ``router_compute``'s arithmetic for those two tokens: the same
    lane chains over the K quarter, wave sums, RPART slots and part order
    into RLOG.  Only the token rows staged and computed per CTA shrink.
    """
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    t0 = tg * ROUTER_PAIR
    if tid < ROUTER_PAIR:
        c["poll"]([(c["xrdy"], (t0 + tid) * R_PARTS + q, 1)])
    gpu.barrier()
    gemv.rows_to_lds(
        tid,
        c["xbuf"] + fx.Int64(t0 * (H * 2)),
        R_K,
        ROUTER_PAIR,
        c["xl"],
        R_K + gemv.LDS_PAD,
        CM_DEV,
        ldk=H,
        k0=q * R_K,
    )
    xrow = EARLY_K + gemv.LDS_PAD
    for t in range_constexpr(ROUTER_PAIR):
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
    if tid < ROUTER_PAIR * WAVES:
        # Each K quarter's partial goes to every rank as is; the route CTA adds
        # the four parts in part order, as part 0 did through RPART.
        v = fx.ptr_load(c["av"] + tid)
        t = t0 + tid // WAVES
        w = tid % WAVES
        for p in range_constexpr(TP):
            c["put"](
                c["peer"]["rlog_parts"][p],
                ((c["rank"] * R_PARTS + q) * s + t) * EPR + i * WAVES + w,
                v,
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
    # Read outside the traced branch: a method call on ``c`` inside it would
    # make the rewriter carry ``c`` as branch state.
    use_parts = c.get("rlog_parts", False)
    if wave == 0:
        s = c["S"]
        per = E // 64
        bias = [bo_ld_f32(a["expert_bias"], lane + 64 * i) for i in range(per)]
        specs = []
        if const_expr(use_parts):
            # q8 two-CTA: the four K-quarter partials of each logit, summed in
            # part order exactly as the router's part 0 did.
            for i in range_constexpr(per):
                e = lane + 64 * i
                for qq in range_constexpr(R_PARTS):
                    specs.append(
                        (
                            c["peer"]["rlog_parts_own"],
                            (((e // EPR) * R_PARTS + qq) * s + t) * EPR + e % EPR,
                            1,
                        )
                    )
            parts = c["poll"](specs, batch=per * R_PARTS)
            got = []
            for i in range_constexpr(per):
                v = f32_of(parts[i * R_PARTS][0])
                for qq in range_constexpr(1, R_PARTS):
                    v = v + f32_of(parts[i * R_PARTS + qq][0])
                got.append([v.bitcast(fx.Int32)])
        else:
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
def _load_route_q8(c):
    """``load_route`` for the q8 two-CTA build: the same tables, built by wave 0.

    The table resets do not depend on ROUTE and precede its wait.  Wave 0's 64
    lanes are the 64 picks: they mark ``flag``; each lane then scans four
    flags in expert order (``wave_incl_scan``), so slots are assigned in
    ascending expert id exactly as the CTA scan did.  LDS operations of one
    wave complete in order; two CTA barriers bracket the wave's work.
    """
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    for i in range_constexpr((MAX_U * 8 + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < MAX_U * 8:
            fx.ptr_store(fx.Int32(-1), rt["kof"] + j)
    if tid < MAX_U:
        fx.ptr_store(fx.Int32(0), rt["expert"] + tid)
    for i in range_constexpr((E + THREADS - 1) // THREADS):
        e = tid + THREADS * i
        if e < E:
            fx.ptr_store(fx.Int32(0), rt["flag"] + e)
    n = s * 2 * TOPK
    if tid < n:
        v = c["poll"]([(c["route"], tid, 1)])[0][0]
        fx.ptr_store(v, rt["route"] + tid)
    gpu.barrier()
    if wave == 0:
        t = lane // TOPK
        k = lane % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * TOPK + k)), E - 1), 0)
        fx.ptr_store(fx.Int32(1), rt["flag"] + e)
        per = E // 64
        fs = [fx.ptr_load(rt["flag"] + (lane * per + d)) for d in range(per)]
        cnt = fs[0]
        for d in range_constexpr(1, per):
            cnt = cnt + fs[d]
        incl = wave_incl_scan(cnt, lane)
        off = incl - cnt
        for d in range_constexpr(per):
            if fs[d] == 1:
                fx.ptr_store(off, rt["flag"] + (lane * per + d))
                fx.ptr_store(fx.Int32(lane * per + d), rt["expert"] + off)
            off = off + fs[d]
        slot = fx.ptr_load(rt["flag"] + e)
        fx.ptr_store(fx.Int32(k), rt["kof"] + (slot * 8 + t))
        if lane == 63:
            fx.ptr_store(incl, c["scan"])
    gpu.barrier()
    return uniform(fx.ptr_load(c["scan"]))


@traced
def load_route(c):
    """ROUTE -> LDS (ids, weights); the union U: slot_of[e], expert_of[slot],
    kof[slot][t] (the token's top-k index, -1 if not picked); returns |U|."""
    if const_expr(c["experimental_two_cta"] and c["S"] == 8):
        return _load_route_q8(c)
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
    if const_expr(c["experimental_two_cta"]):  # noqa: SIM102
        if tid < MAX_U:
            fx.ptr_store(fx.Int32(0), rt["expert"] + tid)
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
    r_w = rsrc(a["w13"], W13_BYTES)
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
    r_s = rsrc(a["w13s"], W13_SCALE_BYTES)
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


def ug_wave_splitk_loads(c, task):
    """Candidate W13 operands with wave-uniform terms in scalar offsets."""
    lane, wave = c["lane"], c["wave"]
    a, rt = c["args"], c["rt"]
    slot = task // UG_GROUPS
    group = task % UG_GROUPS
    tile = wave % 4
    half = wave // 4
    rg = (tile // 2) * (RI // ROWS) + 2 * group + tile % 2
    e = uniform(fx.ptr_load(rt["expert"] + slot))
    r_w = rsrc(a["w13"], W13_BYTES)
    av = [
        fx.Vector(
            bo.buffer_load(
                r_w,
                lane * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
                soffset_bytes=uniform(
                    (
                        e * (2 * RI * W13_K // 8)
                        + rg * (2 * W13_K)
                        + (half * 16 + i) * 256
                    )
                    * 4
                ).ir_value(),
            )
        )
        for i in range(16)
    ]
    r_s = rsrc(a["w13s"], W13_SCALE_BYTES)
    m = e * (2 * RI) + rg * ROWS + lane % 16
    sw = [
        fx.Int32(
            bo.buffer_load(
                r_s,
                scale_dword(m, (half * 8 + sp) * 8 + lane // 16, W13_SCALE_G),
                vec_width=1,
                dtype=T.i32,
            )
        )
        for sp in range(8)
    ]
    return av + sw


@traced
def _stage_ug_epilogue(c, task, units=UG_UNITS):
    """Shared, bit-identical SiLU/BF16/MXFP4 epilogue for both W13 paths."""
    s, tid, red = c["S"], c["tid"], c["red"]
    rt = c["rt"]
    gpu.barrier()
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
    # Match the native two-stage q8 BF16 boundary before MXFP4 quantization.
    # Narrower widths preserve the original monokernel arithmetic.
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
def stage_ug(c, task, ops, units=UG_UNITS):
    """Task ``task``: one or two 32-column groups, a wave a 16-row tile against
    the MXFP4 xe (one MFMA chain over K); silu(g) u; MXFP4 a 32-group -> INTER /
    ISC of each picking token."""
    s, lane, wave, red = c["S"], c["lane"], c["wave"], c["red"]
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
    _stage_ug_epilogue(c, task, units)


@traced
def stage_ug_wave_splitk(c, task, ops):
    """One group with two native-FP4 K halves combined in FP32 in LDS."""
    s, lane, wave, red = c["S"], c["lane"], c["wave"], c["red"]
    group = task % UG_GROUPS
    tile = wave % 4
    half = wave // 4
    rg = (tile // 2) * (RI // ROWS) + 2 * group + tile % 2
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    av = ops[:16]
    sw = ops[16:]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(16):
        st = half * 16 + i
        bw = fx.Vector(
            fx.ptr_load(
                c["xq"] + (t * XQ_W + st * 16 + kb * 4),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        sb = fx.ptr_load(c["xsc"] + (t * XQ_G + st * 4 + kb))
        sa = (sw[i // 2] >> ((st % 2) * 16 + (rg % 2) * 8)) & 0xFF
        acc = mfma_scaled(av[i], bw, acc, sa, sb, FP4, FP4, 0, 0)
    roff = (wave * 64 + lane) * 4
    fx.ptr_store(acc, red + roff)
    gpu.barrier()
    if wave < 4:
        other = fx.Vector(
            fx.ptr_load(
                red + ((wave + 4) * 64 + lane) * 4,
                result_type=fx.Vector.make_type(4, fx.Float32),
            )
        )
        fx.ptr_store(acc + other, red + roff)
    # The epilogue's entry barrier completes the required post-combine fence.
    _stage_ug_epilogue(c, task, 1)


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
def load_expert_inter(c, slot, debug_base):
    """Wait for this expert's 16 one-group publications and stage its rows.

    Only the current expert's picked-token INTER/ISC rows enter LDS.  Unpicked
    rows use zero data and unit scale; bounded descriptors and masked loads
    prevent reads from a different route or a future slot.
    """
    s, tid = c["S"], c["tid"]
    if tid < UG_GROUPS:
        c["poll"]([(c["ugrdy"], slot * UG_GROUPS + tid, 1)])
    gpu.barrier()
    # The polls and their existing CTA barrier are complete.  This stamp is
    # before either INTER or ISC starts copying into expert-local LDS.
    c["stamp"](debug_base + 3)

    t = tid // INTER_W
    w = tid % INTER_W
    k = fx.ptr_load(c["rt"]["kof"] + (slot * 8 + t))
    picked = k >= 0
    pick = t * TOPK + fx.max(k, 0)
    iv = fx.Int32(
        bo.buffer_load(
            rsrc(c["interb"], s * TOPK * INTER_W * 4),
            pick * INTER_W + w,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=CM_DEV,
            mask=picked,
        )
    )
    fx.ptr_store(picked.select(iv, fx.Int32(0)), c["interq"] + t * INTER_W + w)

    if tid < s * UG_GROUPS:
        st = tid // UG_GROUPS
        g = tid % UG_GROUPS
        sk = fx.ptr_load(c["rt"]["kof"] + (slot * 8 + st))
        scaled = sk >= 0
        spick = st * TOPK + fx.max(sk, 0)
        sv = fx.Int32(
            bo.buffer_load(
                rsrc(c["iscb"], s * TOPK * UG_GROUPS * 4),
                spick * UG_GROUPS + g,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_DEV,
                mask=scaled,
            )
        )
        fx.ptr_store(
            scaled.select(sv, fx.Int32(UNIT_SCALE)),
            c["intsc"] + st * UG_GROUPS + g,
        )
    gpu.barrier()


def expert_down_loads(c, slot, rgd):
    """Candidate W2 operands with wave-uniform terms in scalar offsets."""
    lane = c["lane"]
    r_w = rsrc(c["args"]["w2"], W2_BYTES)
    r_s = rsrc(c["args"]["w2s"], W2_SCALE_BYTES)
    e = uniform(fx.ptr_load(c["rt"]["expert"] + slot))
    av = [
        fx.Vector(
            bo.buffer_load(
                r_w,
                lane * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
                soffset_bytes=uniform(
                    (e * (MH * RI // 8) + rgd * (2 * RI) + st * 256) * 4
                ).ir_value(),
            )
        )
        for st in range(W2_STEPS)
    ]
    sw = [
        fx.Int32(
            bo.buffer_load(
                r_s,
                scale_dword(
                    e * MH + rgd * ROWS + lane % 16,
                    sp * 8 + lane // 16,
                    W2_SCALE_G,
                ),
                vec_width=1,
                dtype=T.i32,
            )
        )
        for sp in range(W2_STEPS // 2)
    ]
    return av + sw


def expert_stage_context(c):
    """Rematerialize only thread-derived addresses at one pipeline stage."""
    stage_c = c.copy()
    tid = fresh(c["tid"])
    stage_c["tid"] = tid
    stage_c["lane"] = tid % 64
    stage_c["wave"] = uniform(tid // 64)
    return stage_c


@traced
def stage_down_expert(c, slot, rgd, ops):
    """One expert and one wave-owned W2 tile -> direct bf16 YR partials."""
    s, lane = c["S"], c["lane"]
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    k = fx.ptr_load(c["rt"]["kof"] + (slot * 8 + t))
    picked = (k >= 0) & (lane % 16 < s)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for st in range_constexpr(W2_STEPS):
        bw = fx.Vector(
            fx.ptr_load(
                c["interq"] + (t * INTER_W + st * 16 + kb * 4),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        sb = fx.ptr_load(c["intsc"] + (t * UG_GROUPS + st * 4 + kb))
        sw = ops[W2_STEPS + st // 2]
        sa = (sw >> ((st % 2) * 16 + (rgd % 2) * 8)) & 0xFF
        acc = mfma_scaled(ops[st], bw, acc, sa, sb, FP4, FP4, 0, 0)
    if picked:
        c["put_bf"](
            c["yr_owner"],
            (c["rank"] * s * TOPK + t * TOPK + k) * MH + rgd * ROWS + 4 * (lane // 16),
            [bf16_round(acc[i]) for i in range(4)],
        )
    gpu.barrier()


@traced
def run_expert_wave_pipeline(c, nu):
    """Joint q8 schedule: 32 teams of 16 CTAs, one expert at a time.

    Each member publishes one native-FP4 wave-K UG group.  The team waits only
    for those 16 groups, then each member computes 16 distinct W2 tiles.  A CTA
    completes down before reusing expert-local LDS for its next ``slot + 32``.
    """
    team = c["bid"] // 16
    member = c["bid"] % 16
    slot = team
    while slot < nu:
        debug_base = EXPERT_TL_BASE + EXPERT_TL_POINTS * (slot // 32)
        task = slot * UG_GROUPS + member
        ug_c = expert_stage_context(c)
        c["stamp"](debug_base)
        ug_ops = ug_wave_splitk_loads(ug_c, task)
        c["stamp"](debug_base + 1)
        stage_ug_wave_splitk(ug_c, task, ug_ops)
        c["stamp"](debug_base + 2)
        enter_stage("inter_wait")
        inter_c = expert_stage_context(c)
        load_expert_inter(inter_c, slot, debug_base)
        c["stamp"](debug_base + 4)
        c["stamp"](16)
        enter_stage("down")
        for it in range_constexpr(2):
            down_c = expert_stage_context(c)
            rgd = member * 16 + down_c["wave"] + 8 * it
            down_ops = expert_down_loads(down_c, slot, rgd)
            c["stamp"](debug_base + 5 + 2 * it)
            stage_down_expert(down_c, slot, rgd, down_ops)
            c["stamp"](debug_base + 6 + 2 * it)
        slot = slot + 32


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
def stage_fin_routes(c, j, t):
    """Finalize one q8 ``(128-column slice, token)`` with one route/wave.

    Each wave consumes one route. Wave 0 performs the original ordered top-k
    accumulation after all eight routes stage ``(x0, x1, scale)`` in ``red``.
    """
    s, lane, wave, red = c["S"], c["lane"], c["wave"], c["red"]
    a, rt = c["args"], c["rt"]
    col = j * FIN_COLS + 2 * lane
    wp = fx.Int32(bo.buffer_load(rsrc(a["w_post"]), col // 2, vec_width=1, dtype=T.i32))
    w0 = bf_lo(wp) + fx.Float32(1.0)
    w1 = bf_hi(wp) + fx.Float32(1.0)

    got = c["poll"](
        [
            (
                c["peer"]["yr_own"],
                (((src * s + t) * TOPK + wave) * MH + col) // 2,
                1,
            )
            for src in range_constexpr(TP)
        ],
        batch=FIN_BATCH,
    )
    c["stamp"](14)
    lo, hi = bf_lo(got[0][0]), bf_hi(got[0][0])
    for src in range_constexpr(1, TP):
        lo = lo + bf_lo(got[src][0])
        hi = hi + bf_hi(got[src][0])
    x0 = bf16_round(lo) * w0
    x1 = bf16_round(hi) * w1
    sq = wave_sum(x0 * x0 + x1 * x1)
    if lane == 0:
        for p in range_constexpr(TP):
            c["put"](
                c["peer"]["owned_fmsq"][p],
                (j * s + t) * TOPK + wave,
                sq,
            )

    q = lane % FIN_TASKS
    norms = c["poll"](
        [
            (
                c["peer"]["owned_fmsq_own"],
                (q * s + t) * TOPK + wave,
                1,
            )
        ]
    )
    c["stamp"](15)
    v = (lane < FIN_TASKS).select(f32_of(norms[0][0]), fx.Float32(0.0))
    total = wave_sum(v)
    tw = fx.ptr_load(rt["route"] + (t * 2 * TOPK + TOPK + wave)).bitcast(fx.Float32)
    scale = hw_rsq(total * fx.Float32(1.0 / MH) + fx.Float32(c["eps_post"])) * tw
    roff = (wave * 64 + lane) * 3
    fx.ptr_store(x0, red + roff)
    fx.ptr_store(x1, red + roff + 1)
    fx.ptr_store(scale, red + roff + 2)
    gpu.barrier()

    if wave == 0:
        f0 = fx.Float32(0.0)
        f1 = fx.Float32(0.0)
        for k in range_constexpr(TOPK):
            koff = (k * 64 + lane) * 3
            kx0 = fx.ptr_load(red + koff)
            kx1 = fx.ptr_load(red + koff + 1)
            kscale = fx.ptr_load(red + koff + 2)
            f0 = f0 + kx0 * kscale
            f1 = f1 + kx1 * kscale
        fw = (
            fx.Vector.from_elements([f0, f1], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)
        )
        for p in range_constexpr(TP):
            c["put_words"](c["peer"]["owned_fin"][p], (t * MH + col) // 2, [fw[0]])
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


def up_loads_masked(c, j, live):
    """``up_loads`` with a CTA predicate, for issue at the top level ahead of
    FRDY: the same addresses (every K chunk is in range at K = 4096) and cache
    policy; a masked lane's offset falls outside the bounded descriptor."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_unpool"], H * MH * 2)
    row = c["rank"] * UP_N + fx.max(fx.min(j, UP_TASKS - 1), 0) * ROWS + lane % ROWS
    g = lane // ROWS
    per, n = gemv.chunks(MH)
    assert per * WAVES == n
    out = []
    for i in range(per):
        base = (row * MH + (wave + WAVES * i) * gemv.KCH + g * 8) // 2
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        base + 16 * q,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                        mask=live,
                    )
                )
                for q in range(4)
            ]
        )
    return out


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
