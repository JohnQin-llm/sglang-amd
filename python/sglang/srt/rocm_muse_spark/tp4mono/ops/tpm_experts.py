# SPDX-License-Identifier: Apache-2.0
"""TP4/EP4 K2 expert half for M verify rows (one build per M): 64 complete local
experts per rank (64 r .. 64 r + 63), adaptive teams, A4 gate/up and down on the
single-layout MXFP4 weights over each expert's fan (the tokens that picked it, in
token order, 16-row tiles), per-route post-expert norm, local route sum, the MoE
all-reduce in the custom AR's order, the unpool slice (2048 branch columns a
rank) and the gather.

Expert numerics follow production ``mxfp4_compact_a4_moe._expert_a4`` (one
scaled-MFMA chain over K ascending, ``acc * sigmoid(acc) * up`` with an IEEE
division, bf16, aiter MXFP4 requant) and ``muse_moe_finalize``.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

from sglang.srt.rocm_muse_spark.tp4mono.ops.common.mx import FP4, mfma_scaled
from sglang.srt.rocm_muse_spark.tp4mono.ops.common.ops import (
    CM_DEV,
    fresh,
    CM_NT,
    ballot,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    lanes_below,
    mfma_bf16,
    popcount,
    row_sum,
    rsrc,
    traced,
    uniform,
)
from sglang.srt.rocm_muse_spark.tp4mono.ops.common.plan import THREADS
from sglang.srt.rocm_muse_spark.tp4mono.ops.common.sync import publish
from sglang.srt.rocm_muse_spark.tp4mono.ops.stages import gemv, mxq
from sglang.srt.rocm_muse_spark.tp4mono.ops.tpm_front import (
    AR_ONE_STAGE_BYTES,
    RS_MIN_M,
    XROW,
    ar_start,
    br_index,
    owner_sum_start,
    stage_rows,
    sum_from,
)

TP = 4
E = 64  # local experts
RI = 4096
MH = 4096
H = 8192
TOPK = 8
ROWS = 16
FT = 16  # fan rows a tile
UG_GROUPS = RI // 32  # 128 32-column groups an expert
INTER_W = RI // 8  # 512 E2M1 words a route
W13_K = MH
W13_STEPS = W13_K // 128
W13_SCALE_G = W13_K // 32
W13_BYTES = E * 2 * RI * W13_K // 2
W13_SCALE_BYTES = E * 2 * RI * W13_SCALE_G
W2_STEPS = RI // 128
W2_SCALE_G = RI // 32
W2_BYTES = E * MH * RI // 2
W2_SCALE_BYTES = E * MH * W2_SCALE_G
XQ_W = MH // 8
XQ_G = MH // 32
DOWN_TILES = MH // ROWS  # 256
UP_N = H // TP  # 2048 branch columns a rank
UP_TASKS = UP_N // ROWS  # 128
FIN_COLS = 128
FIN_SLICES = MH // FIN_COLS  # 32
GATHER_SLICES = H // 128  # 64
BLOCKS = 256


def moe_rs(m, first=RS_MIN_M):
    """Whether the MoE all-reduce runs as reduce-scatter to the row-block owner + all-gather
    of the owner's bf16 sum: from ``first`` rows (K2Build.moe_rs_from; at the default, 32, every
    such width is a 2-stage one)."""
    return m >= first and (m * MH * 2 >= AR_ONE_STAGE_BYTES or first < RS_MIN_M)


def groups8(m):
    return (m + 7) // 8


def fin_units(m):
    return FIN_SLICES * groups8(m)


def scale_dword(m, g, cols):
    return (m >> 5) * (8 * cols) + (g >> 3) * 64 + (g & 3) * 16 + (m & 15)


# ---------------------------------------------------------------- fan list
@traced
def build_fan(c, expert):
    """Wave 0: the tokens that picked local expert ``expert`` (token order) and their
    top-k slot -> LDS fan_t / fan_k; returns the fan size (uniform)."""
    m, lane, wave = c["S"], c["lane"], c["wave"]
    rt = c["rt"]
    start = c["rank"] * E
    if wave == 0:
        base = fx.Int32(0)
        for i in range_constexpr((m + 63) // 64):
            t = 64 * i + lane
            tc = fx.min(t, m - 1)
            k = fx.Int32(-1)
            for j in range_constexpr(TOPK):
                e = fx.ptr_load(rt["route"] + tc * 16 + j)
                k = (e == start + expert).select(fx.Int32(j), k)
            hit = (k >= 0) & (t < m)
            b = ballot(hit)
            pos = base + lanes_below(b, lane)
            if hit:
                fx.ptr_store(t, rt["fan_t"] + pos)
                fx.ptr_store(k, rt["fan_k"] + pos)
            base = base + popcount(b)
        if lane == 0:
            fx.ptr_store(base, rt["fan_n"])
    gpu.barrier()
    return uniform(fx.ptr_load(rt["fan_n"]))


# ---------------------------------------------------------------- work units
UG_PAIRS = UG_GROUPS // 2  # gate/up tasks a unit: groups 2 p (waves 0..3) and 2 p + 1 (waves 4..7)


def max_units(m):
    """Bound on a rank's work units (expert, 16-row fan tile): sum of ceil(fan / 16) <= 64 + 8 m / 16."""
    return E + m // 2


@traced
def build_units(c):
    """Every CTA: the local experts' fan sizes (wave w counts experts 8 w .. 8 w + 7 by ballots
    over the route table) -> the work units (expert, fan tile), expert-major, in LDS unit_e /
    unit_f; returns the unit count (uniform). An expert with a fan above 16 is one unit a fan
    tile, each streaming the expert's weights: no team runs more than one tile."""
    m, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    start = c["rank"] * E
    n = [fx.Int32(0) for _ in range(8)]
    for i in range_constexpr((m * TOPK + 63) // 64):
        p = 64 * i + lane
        pc = fx.min(p, m * TOPK - 1)
        e = fx.ptr_load(rt["route"] + (pc // 8) * 16 + pc % 8)
        live = p < m * TOPK
        for j in range_constexpr(8):
            n[j] = n[j] + popcount(ballot(live & (e == start + wave * 8 + j)))
    if lane < 8:
        v = n[7]
        for j in range_constexpr(6, -1, -1):
            v = (lane == j).select(n[j], v)
        fx.ptr_store(v, rt["ecnt"] + (wave * 8 + lane))
    gpu.barrier()
    if tid == 0:
        total = fx.Int32(0)
        for e in range_constexpr(E):
            fx.ptr_store(total, rt["ustart"] + e)
            total = total + (fx.ptr_load(rt["ecnt"] + e) + (FT - 1)) // FT
        fx.ptr_store(total, rt["count"])
    gpu.barrier()
    if tid < E:
        cnt = fx.ptr_load(rt["ecnt"] + tid)
        s0 = fx.ptr_load(rt["ustart"] + tid)
        for f in range_constexpr((m + FT - 1) // FT):
            if f * FT < cnt:
                fx.ptr_store(tid, rt["unit_e"] + (s0 + f))
                fx.ptr_store(fx.Int32(f), rt["unit_f"] + (s0 + f))
    gpu.barrier()
    return uniform(fx.ptr_load(rt["count"]))


# ---------------------------------------------------------------- gate / up
def ug_row_tile(c, p):
    """Wave w's W13 16-row tile in task p: group g = 2 p + w // 4; w % 4 = 0, 1: gate rows
    32 g, 32 g + 16; w % 4 = 2, 3: the up rows 4096 + 32 g, 4096 + 32 g + 16."""
    w4 = c["wave"] % 4
    g = 2 * p + c["wave"] // 4
    return (w4 // 2) * (RI // ROWS) + 2 * g + w4 % 2


def ug_loads(c, expert, p, live):
    """Task p's W13 tile of this wave and its scales; masked off (no traffic) unless ``live``."""
    lane = c["lane"]
    rg = ug_row_tile(c, p)
    r_w = rsrc(c["args"]["w13"], W13_BYTES)
    base = expert * (2 * RI * W13_K // 8)
    av = [fx.Vector(bo.buffer_load(r_w, base + rg * (2 * W13_K) + st * 256 + lane * 4, vec_width=4, dtype=T.i32,
                                   cache_modifier=CM_NT, mask=live)) for st in range(W13_STEPS)]
    r_s = rsrc(c["args"]["w13s"], W13_SCALE_BYTES)
    mrow = expert * (2 * RI) + rg * ROWS + lane % 16
    sw = [fx.Int32(bo.buffer_load(r_s, scale_dword(mrow, sp * 8 + lane // 16, W13_SCALE_G), vec_width=1, dtype=T.i32,
                                  mask=live)) for sp in range(W13_STEPS // 2)]
    return av + sw


@traced
def load_xq_tile(c, f0, nf):
    """The MXFP4 xe rows of fan entries f0 .. f0 + 16 (those < nf) -> LDS xq / xsc."""
    tid = c["tid"]
    rt = c["rt"]
    for i in range_constexpr(4):
        e = tid + 512 * i  # 16 rows x 128 dwordx4
        row = e // 128
        w = (e % 128) * 4
        live = f0 + row < nf
        tok = fx.ptr_load(rt["fan_t"] + fx.min(f0 + row, nf - 1))
        v = fx.Vector(bo.buffer_load(rsrc(c["xqb"]), tok * XQ_W + w, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV, mask=live))
        if live:
            fx.ptr_store(v, c["xq"] + row * XQ_W + w)
    row = tid // 32
    w = (tid % 32) * 4
    live = f0 + row < nf
    tok = fx.ptr_load(rt["fan_t"] + fx.min(f0 + row, nf - 1))
    v = fx.Vector(bo.buffer_load(rsrc(c["xscb"]), tok * XQ_G + w, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV, mask=live))
    if live:
        fx.ptr_store(v, c["xsc"] + row * XQ_G + w)
    gpu.barrier()


def fan_tile_loads(c, f0, nf, parts):
    """Issue a fan tile's loads (store them with ``store_fan_tile``): fan entries f0 .. f0 + 16 (those < nf) of
    row-major dword arrays, every load before any LDS store, so the tile costs one memory round trip
    (load_xq_tile and load_inter_tile store each load under its row's condition; that branch makes their five
    loads wait for one another, five round trips). ``parts``: (source, its bytes, words a row, LDS
    destination, the source row of a fan entry). A row at or past nf loads past the end of its sized resource
    (zeros, no traffic) into an LDS row no MFMA reads."""
    tid = c["tid"]
    staged = []
    for src, nbytes, row_words, dst, source_row in parts:
        quads = row_words // 4
        for i in range_constexpr(FT * quads // THREADS):
            e = tid + THREADS * i
            row = e // quads
            w = (e % quads) * 4
            r = source_row(fx.min(f0 + row, nf - 1))
            v = fx.Vector(bo.buffer_load(rsrc(src, nbytes), r * row_words + w, vec_width=4, dtype=T.i32,
                                         cache_modifier=CM_DEV, mask=f0 + row < nf))
            staged.append((dst + row * row_words + w, v))
    return staged


@traced
def store_fan_tile(staged):
    for dst, v in staged:
        fx.ptr_store(v, dst)
    gpu.barrier()


def xq_tile_parts(c):
    """fan_tile_loads parts of the xq tile: a fan entry's row is its token's."""
    m, rt = c["S"], c["rt"]

    def token(fi):
        return fx.ptr_load(rt["fan_t"] + fi)

    return [(c["xqb"], m * XQ_W * 4, XQ_W, c["xq"], token), (c["xscb"], m * XQ_G * 4, XQ_G, c["xsc"], token)]


def inter_tile_parts(c):
    """fan_tile_loads parts of the intermediate tile: a fan entry's row is its route's (token, top-k slot)."""
    m, rt = c["S"], c["rt"]

    def route(fi):
        return fx.ptr_load(rt["fan_t"] + fi) * TOPK + fx.ptr_load(rt["fan_k"] + fi)

    return [(c["interb"], m * TOPK * INTER_W * 4, INTER_W, c["interq"], route),
            (c["iscb"], m * TOPK * UG_GROUPS * 4, UG_GROUPS, c["intsc"], route)]


@traced
def ug_mfma(c, p, ops, f0, nf):
    """Task p (groups 2 p, 2 p + 1) on the staged fan tile: a wave's 16 W13 rows, one
    scaled-MFMA chain over K ascending -> the wave's accumulator."""
    lane = c["lane"]
    rg = ug_row_tile(c, p)
    av = ops[:W13_STEPS]
    sw = ops[W13_STEPS:]
    nrow = fx.min(nf - f0, FT)
    t = fx.min(lane % 16, nrow - 1)
    kb = lane // 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for st in range_constexpr(W13_STEPS):
        bw = fx.Vector(fx.ptr_load(c["xq"] + (t * XQ_W + st * 16 + kb * 4), result_type=fx.Vector.make_type(4, fx.Int32)))
        sb = fx.ptr_load(c["xsc"] + (t * XQ_G + st * 4 + kb))
        sa = (sw[st // 2] >> ((st % 2) * 16 + (rg % 2) * 8)) & 0xFF
        acc = mfma_scaled(av[st], bw, acc, sa, sb, FP4, FP4, 0, 0)
    return acc


@traced
def ug_epilogue(c, p, acc, f0, nf):
    """silu(gate) up -> bf16 -> MXFP4 per 32 columns of task p -> INTERB / ISCB."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    rt = c["rt"]
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    tt = tid // 32  # fan row of the tile
    col = tid % 32
    half = col // 16
    r = col % 16
    codes = []
    for h in range_constexpr(2):
        gv = fx.ptr_load(red + (((4 * h + half) * 64 + 16 * (r // 4) + tt) * 4 + r % 4))
        uv = fx.ptr_load(red + (((4 * h + 2 + half) * 64 + 16 * (r // 4) + tt) * 4 + r % 4))
        v = bf16_round(mxq.silu_mul_ieee(gv, uv))
        amax = butterfly(abs(v), (16, 8, 4, 2, 1), fx.max)
        codes.append(mxq.a4_code(amax))
        fx.ptr_store(v, c["av"] + (h * 512 + tid))
    gpu.barrier()
    live = f0 + tt < nf
    fi = fx.min(f0 + tt, nf - 1)
    pick = fx.ptr_load(rt["fan_t"] + fi) * TOPK + fx.ptr_load(rt["fan_k"] + fi)
    for h in range_constexpr(2):
        ug = 2 * p + h
        if live & (col < 4):
            vals = [fx.ptr_load(c["av"] + (h * 512 + tid - col + col * 8 + j)) for j in range(8)]
            pk = mxq.pack8_a4(vals, codes[h])
            bo.buffer_store(pk, rsrc(c["interb"]), pick * INTER_W + ug * 4 + col, cache_modifier=CM_DEV)
            if col == 0:
                bo.buffer_store(codes[h], rsrc(c["iscb"]), pick * UG_GROUPS + ug, cache_modifier=CM_DEV)
    gpu.barrier()


def ug_task(c, p, ops, f0, nf):
    """Gate/up task p on its loaded weights: the MFMA chain, then the epilogue."""
    acc = ug_mfma(c, p, ops, f0, nf)
    rocdl.sched_barrier(0)
    ug_epilogue(c, p, acc, f0, nf)


# ---------------------------------------------------------------- down
def down_loads(c, e, rgd, live):
    """W2 16-row tile ``rgd`` of expert e (this wave's) and its scales; masked off unless ``live``."""
    lane = c["lane"]
    r_w = rsrc(c["args"]["w2"], W2_BYTES)
    r_s = rsrc(c["args"]["w2s"], W2_SCALE_BYTES)
    base = e * (MH * RI // 8) + rgd * (2 * RI) + lane * 4
    av = [fx.Vector(bo.buffer_load(r_w, base + st * 256, vec_width=4, dtype=T.i32, cache_modifier=CM_NT, mask=live))
          for st in range(W2_STEPS)]
    sw = [fx.Int32(bo.buffer_load(r_s, scale_dword(e * MH + rgd * ROWS + lane % 16, sp * 8 + lane // 16, W2_SCALE_G),
                                  vec_width=1, dtype=T.i32, mask=live)) for sp in range(W2_STEPS // 2)]
    return av + sw


@traced
def put_bf_local(dst, offset, values):
    out = fx.Vector.from_elements([bf16_pair(values[0], values[1]).bitcast(fx.Int32),
                                   bf16_pair(values[2], values[3]).bitcast(fx.Int32)], fx.Int32)
    bo.buffer_store(out, rsrc(dst), offset // 2, cache_modifier=CM_DEV)


@traced
def load_inter_tile(c, f0, nf):
    """The intermediate (MXFP4) rows of fan entries f0 .. f0 + 16 -> LDS interq / intsc."""
    tid = c["tid"]
    rt = c["rt"]
    for i in range_constexpr(4):
        e = tid + 512 * i
        row = e // 128
        w = (e % 128) * 4
        live = f0 + row < nf
        fi = fx.min(f0 + row, nf - 1)
        pick = fx.ptr_load(rt["fan_t"] + fi) * TOPK + fx.ptr_load(rt["fan_k"] + fi)
        v = fx.Vector(bo.buffer_load(rsrc(c["interb"]), pick * INTER_W + w, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV, mask=live))
        if live:
            fx.ptr_store(v, c["interq"] + row * INTER_W + w)
    row = tid // 32
    w = (tid % 32) * 4
    live = f0 + row < nf
    fi = fx.min(f0 + row, nf - 1)
    pick = fx.ptr_load(rt["fan_t"] + fi) * TOPK + fx.ptr_load(rt["fan_k"] + fi)
    v = fx.Vector(bo.buffer_load(rsrc(c["iscb"]), pick * UG_GROUPS + w, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV, mask=live))
    if live:
        fx.ptr_store(v, c["intsc"] + row * UG_GROUPS + w)
    gpu.barrier()


@traced
def down_mfma(c, rgd, ops, f0, nf):
    """One wave-owned W2 tile against fan tile f0: one scaled-MFMA chain over K ascending."""
    lane = c["lane"]
    nrow = fx.min(nf - f0, FT)
    t = fx.min(lane % 16, nrow - 1)
    kb = lane // 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for st in range_constexpr(W2_STEPS):
        bw = fx.Vector(fx.ptr_load(c["interq"] + (t * INTER_W + st * 16 + kb * 4), result_type=fx.Vector.make_type(4, fx.Int32)))
        sb = fx.ptr_load(c["intsc"] + (t * UG_GROUPS + st * 4 + kb))
        sw = ops[W2_STEPS + st // 2]
        sa = (sw >> ((st % 2) * 16 + (rgd % 2) * 8)) & 0xFF
        acc = mfma_scaled(ops[st], bw, acc, sa, sb, FP4, FP4, 0, 0)
    return acc


@traced
def down_store(c, rgd, acc, f0, nf):
    """The tile's unweighted bf16 per-route rows -> EXPERT_Y."""
    lane = c["lane"]
    rt = c["rt"]
    nrow = fx.min(nf - f0, FT)
    t = fx.min(lane % 16, nrow - 1)
    fi = f0 + t
    pick = fx.ptr_load(rt["fan_t"] + fi) * TOPK + fx.ptr_load(rt["fan_k"] + fi)
    if lane % 16 < nrow:
        put_bf_local(c["expert_y"], pick * MH + rgd * ROWS + 4 * (lane // 16), [bf16_round(acc[i]) for i in range(4)])


@traced
def run_gate_up(c, expert, member, team_size, f0, nf):
    """This member's gate/up tasks p = member + k team_size (latency_trim): the xq rows are polled (here, not
    with the route: the units and the fan only need the route) and the xq tile's loads issued, then the first
    task's weights, so they stream while the tile is staged (its LDS stores wait only for its own loads; at
    the memory those still queue behind the other CTAs' first weights, so the first MFMA starts no earlier,
    but the weight stream starts one staging round trip earlier). The loop keeps run_experts' shape (``while``
    on p): a counted loop changed the expert stage's code generation and cost 1-6 us a layer (M8..40)."""
    tid = c["tid"]
    if tid < c["S"]:
        c["poll"]([(c["xqrdy"], tid, 1)])
    gpu.barrier()
    xq_tile = fan_tile_loads(c, f0, nf, xq_tile_parts(c))
    rocdl.sched_barrier(0)  # keep the xq loads ahead of the weight loads
    p = member
    first = ug_loads(c, expert, p, p < UG_PAIRS)
    store_fan_tile(xq_tile)
    c["stamp"](23)
    if p < UG_PAIRS:
        ug_task(c, p, first, f0, nf)
    p = p + team_size
    while p < UG_PAIRS:
        ug_task(c, p, ug_loads(c, expert, p, p < UG_PAIRS), f0, nf)
        p = p + team_size


@traced
def run_experts(c):
    """Work units (expert, fan tile), adaptive teams of BLOCKS / units CTAs a unit; a CTA stages
    its unit's xq tile once, runs gate/up tasks (8 waves, 64 columns a task), then down tiles
    (a wave's 16 W2 rows), the first down tile's weights in flight during the gate/up wait."""
    tid, bid = c["tid"], c["bid"]
    rt = c["rt"]
    units = build_units(c)
    if units > 0:
        team = uniform(bid % units)
        member = uniform(bid // units)
        team_size = uniform((BLOCKS - 1 - team) // units + 1)
        expert = uniform(fx.ptr_load(rt["unit_e"] + team))
        f0 = uniform(fx.ptr_load(rt["unit_f"] + team)) * FT
        nf = build_fan(c, expert)
        p = member
        if const_expr(c["latency_trim"]):
            run_gate_up(c, expert, member, team_size, f0, nf)
        else:
            load_xq_tile(c, f0, nf)
            c["stamp"](23)
            while p < UG_PAIRS:
                ug_task(c, p, ug_loads(c, expert, p, p < UG_PAIRS), f0, nf)
                p = p + team_size
        c["stamp"](24)
        publish(c["put"], c["ugrdy"], team * BLOCKS + member, fx.Int32(1), tid == 0)  # one flag a CTA
        if const_expr(c["wave_major_down"]):
            # wave-major: a team's last, partial round of down tiles spreads over every member (3-4 waves each)
            # instead of filling the first members' 8 waves; a round that few CTAs run is bounded by a CTA's
            # load rate (~30 GB/s), not by the HBM. Same tiles, other waves (bitwise).
            rgd = c["wave"] * team_size + member
        else:
            rgd = member * 8 + c["wave"]
        dops = down_loads(c, expert, fx.min(rgd, DOWN_TILES - 1), rgd < DOWN_TILES)
        if tid < team_size:
            c["poll"]([(c["ugrdy"], team * BLOCKS + tid, 1)])
        gpu.barrier()
        c["stamp"](25)
        if const_expr(c["latency_trim"]):
            store_fan_tile(fan_tile_loads(c, f0, nf, inter_tile_parts(c)))
        else:
            load_inter_tile(c, f0, nf)
        if rgd < DOWN_TILES:
            down_store(c, rgd, down_mfma(c, rgd, dops, f0, nf), f0, nf)
        rgd = rgd + team_size * 8
        while rgd < DOWN_TILES:
            dacc = down_mfma(c, rgd, down_loads(c, expert, rgd, rgd < DOWN_TILES), f0, nf)
            rocdl.sched_barrier(0)
            down_store(c, rgd, dacc, f0, nf)
            rgd = rgd + team_size * 8


# ---------------------------------------------------------------- finalize
def _opaque(v):
    """An f32 through a v_mov the compiler cannot see through (keeps a product rounded
    before its add: no fma contraction)."""
    return fresh(fx.Float32(v).bitcast(fx.Int32)).bitcast(fx.Float32)


@traced
def finalize_token(c, t):
    """Token t on one CTA: muse_moe_finalize over its local routes (slot order), then the
    bf16 partial to every rank's EPFIN (and the local debug copy).

    Thread (half h, tq): the finalize kernel's thread tq (columns tq 4 + 1024 chunk + 0..3)
    for slots 4 h .. 4 h + 3, every load issued up front: v = bf16(y bf16(w_post + 1)) and
    the sum of squares in the kernel's order (chunk, element; wave butterfly; 4 waves in
    order) -> sc = score / sqrt(ss / 4096 + eps). The slot-order sum acc += sc v (each
    product rounded before its add) runs slots 0..3 on half 0, hands acc to half 1 through
    LDS, which adds slots 4..7 and sends the bf16 partial."""
    m, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rank = c["rank"]
    rt = c["rt"]
    half = wave // 4
    tq = tid % 256
    r_y = rsrc(c["expert_y"], m * TOPK * MH * 2)
    r_w = rsrc(c["args"]["w_post"], MH * 2)
    slots, local = [], []
    for j in range_constexpr(4):
        slot = 4 * half + j
        e = fx.ptr_load(rt["route"] + t * 16 + slot)
        slots.append(slot)
        local.append((e >= rank * E) & (e < rank * E + E))
    wraw = [fx.Vector(bo.buffer_load(r_w, (tq * 4 + chunk * 1024) // 2, vec_width=2, dtype=T.i32)) for chunk in range(4)]
    yraw = [[fx.Vector(bo.buffer_load(r_y, ((t * TOPK + slots[j]) * MH + tq * 4 + chunk * 1024) // 2, vec_width=2,
                                      dtype=T.i32, cache_modifier=CM_DEV, mask=local[j])) for chunk in range(4)]
            for j in range(4)]
    scales = []
    for chunk in range_constexpr(4):
        for k in range_constexpr(4):
            word = wraw[chunk][k // 2]
            wf = bf_lo(word) if k % 2 == 0 else bf_hi(word)
            scales.append(bf16_round(wf + fx.Float32(1.0)))
    values = []
    for j in range_constexpr(4):
        vj = []
        ss = fx.Float32(0.0)
        for chunk in range_constexpr(4):
            for k in range_constexpr(4):
                word = yraw[j][chunk][k // 2]
                yf = bf_lo(word) if k % 2 == 0 else bf_hi(word)
                v = bf16_round(yf * scales[chunk * 4 + k])
                vj.append(v)
                ss = ss + _opaque(v * v)
        values.append(vj)
        ss = butterfly(ss, (32, 16, 8, 4, 2, 1))
        if lane == 0:
            fx.ptr_store(ss, red + ((4 * half + j) * 4 + wave % 4))
    gpu.barrier()
    c["stamp"](26)
    sc = []
    for j in range_constexpr(4):
        total = fx.Float32(0.0)
        for w in range_constexpr(4):
            total = total + fx.ptr_load(red + ((4 * half + j) * 4 + w))
        mean = total * fx.Float32(1.0 / MH) + fx.Float32(c["eps_post"])
        score = fx.ptr_load(rt["route"] + t * 16 + 8 + slots[j]).bitcast(fx.Float32)
        sc.append(score * (fx.Float32(1.0) / fx.Float32(fx.rocdl.sqrt(T.f32, _to_raw(mean)))))
    hand = c["xl"]  # half 0's acc for half 1: 16 words a thread

    def chain(acc):
        for j in range_constexpr(4):
            acc = [local[j].select(acc[i] + _opaque(sc[j] * values[j][i]), acc[i]) for i in range(16)]
        return acc

    acc = chain([fx.Float32(0.0) for _ in range(16)])  # half 0: slots 0..3 from 0
    if half == 0:
        for i in range_constexpr(16):
            fx.ptr_store(acc[i].bitcast(fx.Int32), hand + (i * 256 + tq))
    gpu.barrier()
    acc = chain([fx.ptr_load(hand + (i * 256 + tq)).bitcast(fx.Float32) for i in range(16)])  # half 1: slots 4..7
    if half == 1:
        owner = t // (m // TP)
        for chunk in range_constexpr(4):
            col = tq * 4 + chunk * 1024
            offset = (rank * m + t) * MH + col
            vs = acc[4 * chunk:4 * chunk + 4]
            for p in range_constexpr(TP):
                if const_expr(moe_rs(m, c["moe_rs_from"])):
                    if owner == p:  # reduce-scatter: the row block's owner only
                        c["put_bf"](c["peer"]["epfin"][p], offset, vs)
                else:
                    c["put_bf"](c["peer"]["epfin"][p], offset, vs)
            put_bf_local(c["part_dbg"], t * MH + col, vs)
    gpu.barrier()


def token_uses_cta(c, t, cta):
    """Whether CTA ``cta`` works for one of token t's local experts: at M <= FT every expert with a pick is
    one work unit, and unit u's team is the CTAs b with b % units == u (run_experts)."""
    rank, rt = c["rank"], c["rt"]
    units = fx.max(uniform(fx.ptr_load(rt["count"])), 1)
    team = cta % units
    used = fx.Boolean(False)
    for j in range_constexpr(TOPK):
        e = fx.ptr_load(rt["route"] + t * 16 + j) - rank * E
        local = (e >= 0) & (e < E)
        used = used | (local & (fx.ptr_load(rt["ustart"] + fx.min(fx.max(e, 0), E - 1)) == team))
    return used


@traced
def stage_finalize(c, bid):
    """Every CTA waits for every CTA's experts; CTA t < M finalizes token t. Trimmed at M <= FT, CTA t waits
    only for the teams of token t's own local experts, so a token's finalize and its partial's link transfer
    run while the other experts' teams still stream weights (the expert stage ends over a ~15 us spread)."""
    m, tid = c["S"], c["tid"]
    if const_expr(c["latency_trim"] and m <= FT):
        if tid < BLOCKS:
            if (bid >= m) | token_uses_cta(c, fx.min(bid, m - 1), tid):
                c["poll"]([(c["expertdone"], tid, 1)])
    else:
        if tid < BLOCKS:
            c["poll"]([(c["expertdone"], tid, 1)])
    gpu.barrier()
    c["stamp"](7)
    if bid < m:
        finalize_token(c, bid)
    c["stamp"](12)


@traced
def reduce_unit(c, u):
    """Unit u = slice j + 32 group: columns 128 j .. of tokens 8 group + wave, the four
    ranks' partials in the custom AR's order -> bf16 FIN, then FRDY[u]."""
    m = c["S"]
    j = u % FIN_SLICES
    t = (u // FIN_SLICES) * 8 + c["wave"]
    col = j * FIN_COLS + c["lane"] * 2
    if t < m:
        got = c["poll"]([(c["peer"]["epfin_own"], ((src * m + t) * MH + col) // 2, 1) for src in range(TP)])
        c["stamp"](27)
        start = ar_start(t, m, MH * 2)
        lo = sum_from([bf_lo(g[0]) for g in got], start)
        hi = sum_from([bf_hi(g[0]) for g in got], start)
        out = bf16_pair(lo, hi).bitcast(fx.Int32)
        bo.buffer_store(out, rsrc(c["fin"], m * MH * 2), (t * MH + col) // 2, cache_modifier=CM_DEV)
    gpu.barrier()
    publish(c["put"], c["frdy"], u, fx.Int32(1), c["tid"] == 0)
    gpu.barrier()


@traced
def owner_reduce_unit(c, u):
    """RS mode, unit u = slice j + 32 group of this rank's owned tokens (row block rank):
    the four ranks' partials in the 2-stage order (owner first) -> bf16 -> every rank's FING."""
    m, rank = c["S"], c["rank"]
    per = m // TP
    j = u % FIN_SLICES
    i = (u // FIN_SLICES) * 8 + c["wave"]
    col = j * FIN_COLS + c["lane"] * 2
    if i < per:
        t = rank * per + i
        got = c["poll"]([(c["peer"]["epfin_own"], ((src * m + t) * MH + col) // 2, 1) for src in range(TP)])
        start = owner_sum_start(m, MH * 2, rank)
        lo = sum_from([bf_lo(g[0]) for g in got], start)
        hi = sum_from([bf_hi(g[0]) for g in got], start)
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["fing"][p], t * MH + col, [lo, hi])


@traced
def fin_gather_unit(c, u):
    """RS mode, unit u (the reduce units' indexing): FING pairs -> FIN, then FRDY[u]."""
    m = c["S"]
    j = u % FIN_SLICES
    t = (u // FIN_SLICES) * 8 + c["wave"]
    col = j * FIN_COLS + c["lane"] * 2
    if t < m:
        w = c["poll"]([(c["peer"]["fing_own"], (t * MH + col) // 2, 1)])[0][0]
        bo.buffer_store(w, rsrc(c["fin"], m * MH * 2), (t * MH + col) // 2, cache_modifier=CM_DEV)
    gpu.barrier()
    publish(c["put"], c["frdy"], u, fx.Int32(1), c["tid"] == 0)
    gpu.barrier()


@traced
def stage_reduce(c, bid):
    m = c["S"]
    if const_expr(moe_rs(m, c["moe_rs_from"])):
        n = FIN_SLICES * groups8(m // TP)
        for r in range_constexpr((n + BLOCKS - 1) // BLOCKS):
            u = bid + BLOCKS * r
            if u < n:
                owner_reduce_unit(c, u)
        for r in range_constexpr((fin_units(m) + BLOCKS - 1) // BLOCKS):
            u = bid + BLOCKS * r
            if u < fin_units(m):
                fin_gather_unit(c, u)
    else:
        for r in range_constexpr((fin_units(m) + BLOCKS - 1) // BLOCKS):
            u = bid + BLOCKS * r
            if u < fin_units(m):
                reduce_unit(c, u)


# ---------------------------------------------------------------- unpool / gather
def up_loads(c, j, live):
    """Unpool task j: branch columns rank 2048 + 16 j .., K 4096 (chunks wave + 8 i); the loads
    of a CTA that runs no unpool task (``live`` false) are masked off."""
    lane, wave = c["lane"], c["wave"]
    per, _n = gemv.chunks(MH)
    r_w = rsrc(c["args"]["w_unpool"], H * MH * 2)  # sized: a masked load moves its offset past the end
    row = c["rank"] * UP_N + j * ROWS + lane % ROWS
    g = lane // ROWS
    out = []
    for i in range(per):
        base = (row * MH + (wave + 8 * i) * gemv.KCH + g * 8) // 2
        out.append([fx.Vector(bo.buffer_load(r_w, base + 16 * q, vec_width=4, dtype=T.i32, cache_modifier=CM_NT,
                                             mask=live)) for q in range(4)])
    return out


@traced
def stage_up(c, j, ops):
    """Unpool task j for every 16-token tile: a wave's K chunks wave + 8 i (i = 0..3) of FIN
    read from global one tile ahead (no LDS staging), one MFMA chain a wave, the 8 waves
    summed in wave order -> bf16 -> every rank's BR."""
    m, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    nflags = fin_units(m)
    for r in range_constexpr((nflags + 511) // 512):
        if tid + 512 * r < nflags:
            c["poll"]([(c["frdy"], tid + 512 * r, 1)])
    gpu.barrier()
    c["stamp"](13)
    g = lane // ROWS
    r_f = rsrc(c["fin"], m * MH * 2)
    ntiles = (m + 15) // 16

    def fin_loads(tt):
        rows = min(16, m - 16 * tt)
        t = fx.min(lane % ROWS, rows - 1)
        return [[fx.Vector(bo.buffer_load(r_f, ((16 * tt + t) * MH + (wave + 8 * i) * gemv.KCH + g * 8 + 32 * qq) // 2,
                                          vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)) for qq in range(4)]
                for i in range(4)]

    xb = fin_loads(0)
    for tt in range_constexpr(ntiles):
        rows = min(16, m - 16 * tt)
        nxt = fin_loads(tt + 1) if tt + 1 < ntiles else None
        rocdl.sched_barrier(0)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for i in range_constexpr(4):
            for qq in range_constexpr(4):
                acc = mfma_bf16(ops[i][qq].bitcast(fx.BFloat16), xb[i][qq].bitcast(fx.BFloat16), acc)
        rocdl.sched_barrier(0)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        c["stamp"](28)
        if tid < (ROWS // 4) * rows:
            rq = tid % (ROWS // 4)
            tk = tid // (ROWS // 4)
            vs = [bf16_round(row_sum(red, 4 * rq + d, tk)) for d in range(4)]
            col = c["rank"] * UP_N + j * ROWS + 4 * rq
            for p in range_constexpr(TP):
                c["put_bf"](c["peer"]["br"][p], br_index(16 * tt + tk, col, m), vs)
        gpu.barrier()
        xb = nxt


@traced
def stage_gather(c, bid):
    """Units slice j + 64 group: branch columns 128 j .. of tokens 8 group + wave."""
    m, lane, wave = c["S"], c["lane"], c["wave"]
    n = GATHER_SLICES * groups8(m)
    for r in range_constexpr((n + BLOCKS - 1) // BLOCKS):
        u = bid + BLOCKS * r
        t = (u // GATHER_SLICES) * 8 + wave
        col = (u % GATHER_SLICES) * 128 + 2 * lane
        if (u < n) & (t < m):
            w = c["poll"]([(c["peer"]["br_own"], br_index(t, col, m) // 2, 1)])[0][0]
            bo.buffer_store(w, rsrc(c["args"]["branch_out"]), (t * H + col) // 2)
    gpu.barrier()
