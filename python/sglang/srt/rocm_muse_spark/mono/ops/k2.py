# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2: one Muse Spark sparse decoder layer from the attention's core output
(o_proj's input) to the layer's outputs, in one persistent launch a layer and
TP rank, every TP collective inside the kernel. Adapted from Kimi-K3's K2
``layer_post`` (``ops/mono/layer.py`` of PR #60683).

    CTAs 0 .. 255     o_proj (512 x 16 rows, two a CTA) -> bf16 partials to
                      every rank's AR region (AR#1); the opt-in 512-CTA build
                      keeps this original dense producer geometry
    CTAs 0 .. 63      seam B (a 128-column slice each): AR#1's rank-ordered sum
                      -> post_attn_norm -> gated residual -> ``stream_out`` ->
                      pre_ffn_norm -> the bf16 x rows + XRDY
    CTAs 0 .. T-1     route (token t)
    CTAs 64 .. 79     router (16 tasks, weights in flight before x lands)
    CTAs 80 .. 207    pooler (128 tasks, likewise)
    every CTA         xe quant and the route table; the opt-in q8 build uses
                      32 teams of 16 CTAs for joint expert-local UG and down
    CTAs 0 .. 31      finalize (AR#3's rank-ordered sum, post_expert_norm,
                      the weighted top-k sum) -> FIN rows + FRDY; at q8 these
                      CTAs each finalize one owned (128-column slice, token),
                      then the same CTAs 0 .. 31
                      receive the tagged FIN all-gather and publish FRDY
    CTAs 192 .. 255   unpooler (64 x 16 of this rank's 1024 columns) -> AG#4
    CTAs 128 .. 191   gather: AG#4 -> ``branch_out``

The opt-in q8 schedule uses 512 resident CTAs. CTAs 0 .. 31 perform
quarter-token normalization; 32 teams of 16 CTAs process expert-local UG/down
dependencies; CTAs 480 .. 511 finalize and distribute the expert results.
The original o_proj, router, pooler, unpooler and gather owners remain unchanged.

No CTA waits on a stage that comes later in its own program, so the grid
(all CTAs co-resident) cannot deadlock; every wait gives up after
``WAIT_SECONDS`` (fail-stop). Mailbox tags carry the step epoch and the slot
128 + layer; the peer regions alternate by the epoch's parity.
"""

import os
import re
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import Int32, Int64

from sglang.srt.rocm_muse_spark.mono.ops.common.debug import (
    region_ids,
    stamp,
    stamp_begin,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.epoch import (
    EPOCH_TAG_PERIOD,
    TAG_SLOTS,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    ld_i32,
    load_ptr64,
    rsrc,
    traced,
    uniform,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    enter_stage,
    key_tuple,
    pair_layout,
    recording,
    source_digest,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.ranks import peer_bases
from sglang.srt.rocm_muse_spark.mono.ops.common.sync import (
    Mailbox,
    import_alarms,
    preg,
    publish,
    shift,
    sreg,
    stall,
)
from sglang.srt.rocm_muse_spark.mono.ops.k2_experiment import (
    finish_counts,
    grid_blocks,
    health_words_for_key,
)
from sglang.srt.rocm_muse_spark.mono.ops.stages import gemv, moe, oproj, quarter_seam
from sglang.srt.rocm_muse_spark.mono.ops.stages import seam_b as seam
from sglang.srt.rocm_muse_spark.mono.runtime.abi import K2_DONE_BYTES, K2_DONE_OFFSET

TP = 8
MAX_TOKENS = 8
K2_SLOT0 = 128  # K2 of layer i is slot 128 + i (K1: slot i)

NSL = seam.NSL  # 64 seam slices: CTAs 0 ..
R0 = NSL  # router CTAs
P0 = R0 + moe.R_TASKS  # pooler CTAs
FIN0 = 0  # finalize CTAs
G0 = 128  # gather CTAs
U0 = BLOCKS - moe.UP_TASKS  # 192: unpooler CTAs
assert P0 + moe.P_TASKS <= BLOCKS
assert moe.DOWN_TASKS == BLOCKS and oproj.OTASKS == 2 * BLOCKS
assert G0 + moe.GATHER_TASKS <= U0 and FIN0 + moe.FIN_TASKS <= G0
# timeline points: 0 start, 1 o_proj, 2 early (router / pooler), 3 seam,
# 4 route, 5 xq, 6 route table, 7 ug, 8 down, 9 fin, 10 up, 11 gather.
# Candidate points 7/8 bracket the joint expert pipeline and are not isolated
# UG/down phase timings; default builds retain the original phase meaning.
# inside stages: 12 seam AR#1 summed, 13 seam MSQ1, 14 fin YR summed,
# 15 fin FMSQ, 16 down INTER loaded
TL_POINTS = 17

REGIONS = (
    (
        "ar",
        "msq1",
        "msq2",
        "xrdy",
        "finish",
        "done",
    )
    + moe.REGIONS
    + ("rstd1", "rstd2")
)
FINISH_FANIN = 8
FINISH_GROUPS = (BLOCKS + FINISH_FANIN - 1) // FINISH_FANIN
FINISH_ROOTS = (FINISH_GROUPS + FINISH_FANIN - 1) // FINISH_FANIN
_PEER_REGIONS = ("ar", "rlog", "pool", "yr", "br", "owned_fmsq", "owned_fin")
_SCRATCH_REGIONS = ("msq1", "msq2", "xrdy", "finish") + tuple(moe.scratch_pairs(1))
_SCRATCH_PLAIN = tuple(moe.scratch_plain(1))

# the K2 sources (and the shared helpers): in every build's JIT cache key
_SOURCES = source_digest(
    "common",
    "stages/gemv.py",
    "stages/mxq.py",
    "stages/oproj.py",
    "stages/seam_b.py",
    "stages/quarter_seam.py",
    "stages/moe.py",
    "k2_experiment.py",
    "k2.py",
)

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K2Build:
    tokens: int  # T, rows of the step (1 .. 8)
    eps_post_attn: float = 1e-8  # post_attn_norm (no weight)
    eps_ffn: float = 1e-5  # pre_ffn_norm
    eps_pre_expert: float = 1e-5  # feed_forward.pre_expert_norm
    eps_post_expert: float = 1e-8  # feed_forward.post_expert_norm (weight first)
    # Applied to the raw router logits before sigmoid, matching
    # FusedMuseSparkRouting's router_output_multiplier.  Original-vLLM
    # fixtures have this folded into their weights and retain the default.
    router_output_multiplier: float = 1.0
    # the fault hook (sync.stall): rank fault_rank stalls fault_ticks once, in
    # slot fault_layer's launch of the step at epoch fault_epoch
    fault_rank: int = -1
    fault_epoch: int = 0
    fault_layer: int = 0
    fault_ticks: int = 0
    timeline: bool = False
    experimental_two_cta: bool = False

    def __post_init__(self):
        if self.experimental_two_cta and self.tokens != MAX_TOKENS:
            raise ValueError("experimental_two_cta is restricted to tokens == 8")


def make_key(
    *,
    tokens: int,
    eps_post_attn: float,
    eps_ffn: float,
    eps_pre_expert: float,
    eps_post_expert: float,
    router_output_multiplier: float = 1.0,
    experimental_two_cta: bool = False,
    **fault,
) -> K2Build:
    return K2Build(
        tokens=tokens,
        eps_post_attn=float(eps_post_attn),
        eps_ffn=float(eps_ffn),
        eps_pre_expert=float(eps_pre_expert),
        eps_post_expert=float(eps_post_expert),
        router_output_multiplier=float(router_output_multiplier),
        experimental_two_cta=experimental_two_cta,
        **fault,
    )


ABI = KernelAbi(
    (
        "core",
        "stream_a",
        "w_o",
        "alpha_a",
        "beta_a",
        "w_fn",
        "w_router",
        "expert_bias",
        "w_pool",
        "w_pre",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "w_post",
        "w_unpool",
        "stream_out",
        "branch_out",
        "scratch",
        "peers",
        "rank",
        "epoch",
        "layer",
        "health",
        "alarm",
        "tl",
    )
)


@traced
def stamp_flush(on, tls, tl, tid, bid, points):
    """The CTA's timeline record -> ``tl`` [CTA][points] int64 (a buffer store:
    the pinned FlyDSL has no ``generic_store`` for ``debug.stamp_flush``)."""
    if const_expr(on):  # noqa: SIM102
        if tid < points:
            bo.buffer_store(fx.ptr_load(tls + tid), rsrc(tl), bid * points + tid)


def step_tag(epoch, slot):
    """This launch's mailbox tag (Kimi-K3's): the step's epoch (a device counter
    the runner bumps once a step, in the graph) and the launch's slot, so no
    step's or other launch's pair ever passes for this one's."""
    e = uniform(ld_i32(epoch, 0))
    return (e % EPOCH_TAG_PERIOD) * TAG_SLOTS + slot + 1


@traced
def finish_consensus(c, done, health, alarm):
    """Keep CTA 0 live until every local CTA and TP rank reached K2's tail.

    The device cannot report kernel completion while CTA 0 remains live, so
    other CTAs may retire after publishing their terminal pair.  CTA 0 imports
    the shared alarms before it retires and therefore before output can return.
    """
    tid, bid = c["tid"], c["bid"]
    publish(c["put"], c["finish"], bid, 1, tid == 0)
    blocks = c["blocks"]
    finish_groups = c["finish_groups"]
    finish_roots = c["finish_roots"]
    if (bid < finish_groups) & (tid == 0):
        lo = bid * FINISH_FANIN
        c["poll_terminal"]([(c["finish"], lo + i, 1) for i in range(FINISH_FANIN)])
        c["put"](c["finish"], blocks + bid, 1)
    if (bid < finish_roots) & (tid == 0):
        lo = blocks + bid * FINISH_FANIN
        c["poll_terminal"]([(c["finish"], lo + i, 1) for i in range(FINISH_FANIN)])
        c["put"](c["finish"], blocks + finish_groups + bid, 1)
    if (bid == 0) & (tid == 0):
        c["poll_terminal"](
            [(c["finish"], blocks + finish_groups + i, 1) for i in range(finish_roots)]
        )
        c["put"](done, c["rank"], 1)
        rocdl.s_waitcnt(vmcnt=0)
        c["poll_terminal"]([(done, r, 1) for r in range(TP)])
        import_alarms(health, alarm, c["rank"], TP)


# ---------------------------------------------------------------- layouts
def peer_layout() -> dict:
    """One parity's peer regions (name -> (byte offset, bytes)), laid out for
    MAX_TOKENS rows whatever the build's width: a parity's half never moves
    when the width changes between steps."""
    pairs = {"ar": oproj.ar_pairs(MAX_TOKENS), **moe.peer_pairs(MAX_TOKENS)}
    return pair_layout(pairs.items(), align=256)


def _half() -> int:
    end = max(o + n for o, n in peer_layout().values())
    return (end + 255) // 256 * 256


def peer_bytes(tokens: int = MAX_TOKENS) -> int:
    """Both parities' peer bytes (past the runner's PEER_HEAD); the layout is
    the same for every width <= MAX_TOKENS."""
    assert 1 <= tokens <= MAX_TOKENS
    return 2 * _half()


def scratch_layout(s: int, *, experimental_two_cta: bool = False) -> dict:
    if experimental_two_cta and s != MAX_TOKENS:
        raise ValueError("experimental_two_cta is restricted to tokens == 8")
    pairs = {"msq1": NSL * s, "msq2": NSL * s, "xrdy": NSL}
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    lay.update(
        moe.scratch_layout(
            s,
            start=(end + 15) // 16 * 16,
            wave_splitk=experimental_two_cta,
        )
    )
    end = max(o + n for o, n in lay.values())
    lay["xrow"] = ((end + 15) // 16 * 16, s * seam.H * 2)  # plain bf16 x rows
    end = lay["xrow"][0] + lay["xrow"][1]
    blocks = 512 if experimental_two_cta else BLOCKS
    groups, roots = finish_counts(blocks)
    lay["finish"] = (
        (end + 15) // 16 * 16,
        (blocks + groups + roots) * 8,
    )
    if experimental_two_cta:
        end = lay["finish"][0] + lay["finish"][1]
        lay.update(
            pair_layout(
                (("rstd1", 8), ("rstd2", 8)),
                start=(end + 15) // 16 * 16,
            )
        )
    return lay


def scratch_bytes(key: K2Build) -> int:
    return max(
        o + n
        for o, n in scratch_layout(
            key.tokens, experimental_two_cta=key.experimental_two_cta
        ).values()
    )


def max_scratch_bytes(
    tokens: int = MAX_TOKENS, *, experimental_two_cta: bool = False
) -> int:
    """The largest scratch any K2 build of <= ``tokens`` rows needs."""
    if experimental_two_cta:
        if tokens != MAX_TOKENS:
            raise ValueError("experimental_two_cta is restricted to tokens == 8")
        return max(
            o + n
            for o, n in scratch_layout(MAX_TOKENS, experimental_two_cta=True).values()
        )
    return max(
        max(o + n for o, n in scratch_layout(s).values()) for s in range(1, tokens + 1)
    )


# ---------------------------------------------------------------- build
_BUILDS: dict = {}


def _make(key: K2Build):
    s = key.tokens
    assert 1 <= s <= MAX_TOKENS
    blocks = grid_blocks(key)
    finish_groups, finish_roots = finish_counts(blocks)
    # A step has at most TOPK * s unique experts.  The eight waves consume
    # strided slots, so no wave can have more than s live slots.  Avoid issuing
    # W2 loads for statically impossible slots at the narrow widths; rows >= 4
    # retain the latency-hiding batch used by the original schedule.
    down_batch = min(moe.DOWN_BATCH, s)
    # At rows=2, at most 16 experts produce 128 default two-group UG tasks,
    # leaving half of gfx950's 256-CU persistent grid idle.  One group per
    # task exposes 256 independent tasks while preserving the exact bytes and
    # arithmetic of each expert group.  Wider builds retain two groups/CTA.
    ug_units = moe.ug_units(s)
    fin_tasks = moe.OWNER_FIN_CTAS if s == 8 else moe.FIN_TASKS
    lay = scratch_layout(s, experimental_two_cta=key.experimental_two_cta)
    play = peer_layout()
    half = _half()
    xrow_max = max(oproj.CORE_ROW, moe.EARLY_K + gemv.LDS_PAD, moe.MH + gemv.LDS_PAD)

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * moe.TOPK, 16]
        flag: fx.Array[fx.Int32, moe.E, 16]
        expert: fx.Array[fx.Int32, moe.MAX_U, 16]
        kof: fx.Array[fx.Int32, moe.MAX_U * 8, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class XqLds:
        xq: fx.Array[fx.Int32, s * moe.XQ_W, 16]
        xsc: fx.Array[fx.Int32, s * moe.XQ_G, 16]

    @fx.struct
    class RowsLds:
        xl: fx.Array[fx.Int32, s * xrow_max // 2, 16]

    if key.experimental_two_cta:

        @fx.struct
        class ExpertDownLds:
            interq: fx.Array[fx.Int32, s * moe.INTER_W, 16]
            intsc: fx.Array[fx.Int32, s * moe.UG_GROUPS, 16]

        @fx.struct
        class ExpertWorkLds:
            xq: XqLds
            down: ExpertDownLds

        @fx.union
        class StageLds:
            rows: RowsLds
            expert: ExpertWorkLds

        def allocate_work(alloc):
            stage = alloc.allocate(StageLds)
            work = stage.expert.peek()
            return work.xq, stage.rows.peek(), work.down

    else:

        @fx.struct
        class DownLds:
            interq: fx.Array[fx.Int32, s * moe.TOPK * moe.INTER_W, 16]
            intsc: fx.Array[fx.Int32, s * moe.TOPK * moe.UG_GROUPS, 16]
            contrib: fx.Array[fx.Float32, s * moe.TOPK * moe.ROWS, 16]

        @fx.union
        class StageLds:
            rows: RowsLds
            down: DownLds

        def allocate_work(alloc):
            xq = alloc.allocate(XqLds).peek()
            stage = alloc.allocate(StageLds)
            return xq, stage.rows.peek(), stage.down.peek()

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        av: fx.Array[fx.Float32, THREADS, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = f"muse_mono_k2_s{s}_t{int(key.timeline)}"
    if key.experimental_two_cta:
        name += "_x2_quarter_expert_stack"
    if key.fault_rank >= 0:
        name += (
            f"_f{key.fault_rank}_{key.fault_epoch}_{key.fault_layer}_{key.fault_ticks}"
        )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
        stream_a: Int64,
        w_o: Int64,
        alpha_a: Int64,
        beta_a: Int64,
        w_fn: Int64,
        w_router: Int64,
        expert_bias: Int64,
        w_pool: Int64,
        w_pre: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        w_post: Int64,
        w_unpool: Int64,
        stream_out: Int64,
        branch_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        xql, rows, dl = allocate_work(alloc)
        if const_expr(key.fault_rank >= 0):
            stall(
                key.fault_rank,
                key.fault_epoch,
                key.fault_layer,
                key.fault_ticks,
                rank,
                epoch,
                layer,
            )
        mb = Mailbox(
            step_tag(epoch, K2_SLOT0 + layer) - 1, health, alarm, region_ids(REGIONS)
        )
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, TP)
        own = load_ptr64(peers, rank)
        # The peer table points past PEER_HEAD.  ``alarm`` points at this
        # rank's word inside the output rank's header, so subtracting rank*4
        # recovers the common header base without adding an ABI argument.
        output_head = alarm - fx.Int64(rank) * 4
        done = preg(output_head, K2_DONE_OFFSET, "done", K2_DONE_BYTES)
        peer = {}
        for region in _PEER_REGIONS:
            off, extent = play[region]
            peer[region] = [
                shift(preg(b, off, region, extent), fx.Int64(par) * half) for b in bases
            ]
            peer[region + "_own"] = shift(
                preg(own, off, region, extent), fx.Int64(par) * half
            )
        c = {
            "S": s,
            "blocks": blocks,
            "finish_groups": finish_groups,
            "finish_roots": finish_roots,
            "experimental_two_cta": key.experimental_two_cta,
            "quarter_seam": key.experimental_two_cta,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "nsl": NSL,
            "eps_attn": key.eps_post_attn,
            "eps_ffn": key.eps_ffn,
            "eps_pre": key.eps_pre_expert,
            "eps_post": key.eps_post_expert,
            "router_output_multiplier": key.router_output_multiplier,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "poll_terminal": mb.poll_terminal,
            "red": lds.red.ptr,
            "av": lds.av.ptr,
            "peer": peer,
            "ar_peer": peer["ar"],
            "ar_own": peer["ar_own"],
            "xl": rows.xl.ptr,
            "xq": xql.xq.ptr,
            "xsc": xql.xsc.ptr,
            "interq": dl.interq.ptr,
            "intsc": dl.intsc.ptr,
            "scan": rl.scan.ptr,
            "rt": {
                "route": rl.route.ptr,
                "flag": rl.flag.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
            },
            "xbuf": scratch + fx.Int64(lay["xrow"][0]),
            "args": {
                "core": core,
                "stream_a": stream_a,
                "w_o": w_o,
                "alpha_a": alpha_a,
                "beta_a": beta_a,
                "w_fn": w_fn,
                "w_router": w_router,
                "expert_bias": expert_bias,
                "w_pool": w_pool,
                "w_pre": w_pre,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "w_post": w_post,
                "w_unpool": w_unpool,
                "stream_out": stream_out,
                "branch_out": branch_out,
            },
        }
        if not key.experimental_two_cta:
            c["contrib"] = dl.contrib.ptr
        if const_expr(s == 8):
            # Candidate members own 16 contiguous W2 tiles; two adjacent
            # members share one output rank. Default CTAs retain one tile.
            owner_rank = (
                (bid % 16) // 2
                if key.experimental_two_cta
                else bid // (moe.DOWN_TASKS // TP)
            )
            owner_base = load_ptr64(peers, owner_rank)
            off, extent = play["yr"]
            c["yr_owner"] = shift(
                preg(owner_base, off, "yr", extent), fx.Int64(par) * half
            )
        for region in _SCRATCH_REGIONS:
            c[region] = sreg(scratch, lay[region][0], region, lay[region][1])
        if const_expr(key.experimental_two_cta):
            for region in ("rstd1", "rstd2"):
                c[region] = sreg(scratch, lay[region][0], region, lay[region][1])
        for name in _SCRATCH_PLAIN:
            c[name] = scratch + fx.Int64(lay[name][0])
        on, tls = key.timeline, lds.tls.ptr
        c["stamp"] = lambda k: stamp(on, tls, tid, k)
        stamp_begin(on, tls, tid, TL_POINTS)
        # ---- original 256-producer o_proj, then early tasks' weights, then x
        eb = bid - R0
        pb = bid - P0
        is_r = (bid >= R0) & (eb < moe.R_TASKS)
        is_p = (pb >= 0) & (pb < moe.P_TASKS)
        enter_stage("oproj")
        if const_expr(key.experimental_two_cta):
            if bid < BLOCKS:
                oproj.stage_oproj(c, bid)
        else:
            oproj.stage_oproj(c, bid)
        stamp(on, tls, tid, 1)
        if is_r:
            enter_stage("router")
            ri = eb // moe.R_PARTS
            rq = eb % moe.R_PARTS
            moe.router_compute(c, ri, rq, moe.router_loads(c, ri, rq))
        if is_p:
            enter_stage("pool")
            pi = pb // moe.P_PARTS
            pq = pb % moe.P_PARTS
            moe.pool_compute(c, pi, pq, moe.pool_loads(c, pi, pq))
        stamp(on, tls, tid, 2)
        # ---- seam B, then the route
        if const_expr(key.experimental_two_cta):
            if bid < 32:
                enter_stage("seam")
                quarter_seam.stage_seam(c, bid)
                stamp(on, tls, tid, 3)
                if bid < s:
                    enter_stage("route")
                    moe.stage_route(c, bid)
                if (bid >= moe.XQ0) & (bid < moe.XQ0 + s):
                    enter_stage("xq")
                    moe.stage_xq(c, bid - moe.XQ0)
                stamp(on, tls, tid, 4)
        else:
            if bid < NSL:
                enter_stage("seam")
                seam.stage_seam(c, bid)
                stamp(on, tls, tid, 3)
                if bid < s:
                    enter_stage("route")
                    moe.stage_route(c, bid)
                if (bid >= moe.XQ0) & (bid < moe.XQ0 + s):
                    enter_stage("xq")
                    moe.stage_xq(c, bid - moe.XQ0)
                stamp(on, tls, tid, 4)
        # ---- the routed experts
        enter_stage("xq_wait")
        moe.load_xq(c)
        stamp(on, tls, tid, 5)
        enter_stage("route_wait")
        nu = moe.load_route(c)
        stamp(on, tls, tid, 6)
        if const_expr(key.experimental_two_cta):
            # Points 7/8 are joint-stage boundaries for the candidate.  The
            # helper publishes this team's UG groups before waiting on them,
            # then completes expert-local down before advancing its slot.
            enter_stage("ug")
            stamp(on, tls, tid, 7)
            moe.run_expert_wave_pipeline(c, nu)
            stamp(on, tls, tid, 8)
        else:
            enter_stage("ug")
            if const_expr(s == 8):
                moe.run_ug_q8(c, nu)
            else:
                moe.run_ug(c, nu, ug_units)
            stamp(on, tls, tid, 7)
            dops = moe.down_loads(c, bid, nu, fx.min(c["wave"], nu - 1), down_batch)
            enter_stage("inter_wait")
            if const_expr(s == 8):
                moe.load_inter_q8(c, nu)
            else:
                moe.load_inter(c, nu, ug_units)
            stamp(on, tls, tid, 16)
            enter_stage("down")
            moe.stage_down(c, bid, nu, dops, down_batch, prefetch_next=s != 2)
            stamp(on, tls, tid, 8)
        if const_expr(key.experimental_two_cta):
            fin_base = blocks - moe.OWNER_FIN_CTAS
            if bid >= fin_base:
                enter_stage("finalize")
                owner = bid - fin_base
                j = rank * moe.OWNED_FIN_TASKS + owner // s
                moe.stage_fin_routes(c, j, owner % s)
        else:
            if bid < FIN0 + fin_tasks:
                enter_stage("finalize")
                if const_expr(s == 8):
                    owner = bid - FIN0
                    j = rank * moe.OWNED_FIN_TASKS + owner // s
                    moe.stage_fin_token(c, j, owner % s)
                else:
                    moe.stage_fin(c, bid - FIN0)
        # Keep the compile-time gate outside the traced device branch.
        if const_expr(s == 8):  # noqa: SIM102
            if bid < FIN0 + moe.FIN_TASKS:
                enter_stage("fin_gather")
                moe.stage_fin_gather(c, bid - FIN0)
        stamp(on, tls, tid, 9)
        if (bid >= U0) & (bid < U0 + moe.UP_TASKS):
            enter_stage("up")
            moe.stage_up(c, bid - U0, moe.up_loads(c, bid - U0))
        stamp(on, tls, tid, 10)
        if (bid >= G0) & (bid < G0 + moe.GATHER_TASKS):
            enter_stage("gather")
            moe.stage_gather(c, bid - G0)
        stamp(on, tls, tid, 11)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)
        # Every CTA drains its output stores before publishing local completion.
        # CTA 0 then proves all resident CTAs reached the tail, publishes this
        # rank's completion in the output-rank header, waits for every TP rank,
        # and imports any shared alarm before CTA 0 (and therefore the kernel)
        # can return.
        enter_stage("finish")
        finish_consensus(c, done, health, alarm)

    @flyc.jit
    def launch(
        core: Int64,
        stream_a: Int64,
        w_o: Int64,
        alpha_a: Int64,
        beta_a: Int64,
        w_fn: Int64,
        w_router: Int64,
        expert_bias: Int64,
        w_pool: Int64,
        w_pre: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        w_post: Int64,
        w_unpool: Int64,
        stream_out: Int64,
        branch_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
            stream_a,
            w_o,
            alpha_a,
            beta_a,
            w_fn,
            w_router,
            expert_bias,
            w_pool,
            w_pre,
            w13,
            w13s,
            w2,
            w2s,
            w_post,
            w_unpool,
            stream_out,
            branch_out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
            health,
            alarm,
            tl,
        ).launch(grid=(blocks,), block=(THREADS,), stream=stream)

    ABI.check(k2, launch)
    return launch


def build(key: K2Build):
    if key not in _BUILDS:
        _BUILDS[key] = _make(key)
    return _BUILDS[key]


def _compile(key: K2Build, launch) -> bool:
    # FlyDSL's compile-only switch has no FLYDSL_ prefix
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        with recording() as trace:
            launch(*ABI.zeros(), stream=torch.cuda.current_stream())
        from sglang.srt.rocm_muse_spark.mono.ops.common.validation import (
            validate_k2_trace,
        )

        return validate_k2_trace(key, trace)
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old


def compile_only(key: K2Build) -> bool:
    """Compile ``key``'s kernel without running it (a launch on null arguments
    would fault)."""
    return _compile(key, build(key))


_META_INTS = (
    ".vgpr_count",
    ".agpr_count",
    ".sgpr_count",
    ".vgpr_spill_count",
    ".sgpr_spill_count",
    ".group_segment_fixed_size",
    ".private_segment_fixed_size",
)


def _msgpack_uint(buf: bytes, i: int) -> int | None:
    b = buf[i]
    if b < 0x80:
        return b
    width = {0xCC: 1, 0xCD: 2, 0xCE: 4}.get(b)
    if width is None:
        return None
    return int.from_bytes(buf[i + 1 : i + 1 + width], "big")


def _note_ints(elf: bytes) -> dict:
    """The kernel descriptor metadata's integers (the code object's msgpack
    note): VGPR / AGPR / SGPR / spill counts, LDS and scratch bytes."""
    out = {}
    for k in _META_INTS:
        kb = k.encode()
        m = re.search(re.escape(bytes([0xA0 | len(kb)]) + kb), elf)
        if m is not None:
            v = _msgpack_uint(elf, m.end())
            if v is not None:
                out[k.lstrip(".")] = v
    return out


def resources(key: K2Build) -> dict:
    """Register / LDS / spill use of ``key``'s compiled kernel: the code
    object's metadata and the runtime's occupancy at THREADS a CTA (needs a
    current HIP device; compiles if needed)."""
    from sglang.srt.rocm_muse_spark.mono.ops.common.residency import (
        code_objects,
        residency,
    )

    launch = build(key)
    _compile(key, launch)
    out: dict = {}
    for elf in code_objects(launch):
        out.update(_note_ints(elf))
    for r in residency(launch, THREADS):
        out.update(
            name=r.name,
            blocks_per_cu=r.blocks_per_cu,
            regs=r.regs,
            lds=r.lds,
            scratch=r.scratch,
            problems=r.problems(
                grid_blocks(key),
                torch.cuda.get_device_properties(0).multi_processor_count,
            ),
        )
    return out


def ptr(t):
    return 0 if t is None else t.data_ptr()


def _rows(t, s, width, dtype):
    assert t.dtype == dtype and t.dim() == 2 and t.shape[1] == width, (
        t.dtype,
        tuple(t.shape),
    )
    assert t.shape[0] >= s and t.stride(1) == 1 and t.stride(0) == width, (
        "rows must be contiguous",
        tuple(t.stride()),
    )


def k2_launch(
    key: K2Build,
    *,
    core,
    stream_a,
    w_o,
    alpha_a,
    beta_a,
    w_fn,
    w_router,
    expert_bias,
    w_pool,
    w_pre,
    w13,
    w13s,
    w2,
    w2s,
    w_post,
    w_unpool,
    stream_out,
    branch_out,
    scratch,
    peers,
    rank,
    epoch,
    layer,
    health,
    alarm,
    tl=None,
):
    """Host launcher. ``core`` bf16 [>= T, 1024] and ``stream_a`` fp32
    [>= T, 8192] (leading rows used); ``stream_out`` fp32 / ``branch_out``
    bf16 [>= T, 8192] written; the router / pooler / unpooler are the full
    replicated tensors (the kernel takes the rank's rows); ``w13`` / ``w2`` /
    their scales are the rank's AITER-shuffled expert tensors, read in place.
    ``scratch`` (zeroed once), ``peers`` (the int64 peer table past
    PEER_HEAD), ``epoch``, ``health`` and ``alarm`` as the runner's."""
    s = key.tokens
    bf = torch.bfloat16
    _rows(core, s, oproj.CORE, bf)
    _rows(stream_a, s, seam.H, torch.float32)
    _rows(stream_out, s, seam.H, torch.float32)
    _rows(branch_out, s, seam.H, bf)
    assert w_o.shape == (seam.H, oproj.CORE) and w_o.is_contiguous()
    assert w_router.shape == (moe.E, seam.H) and w_router.dtype == torch.float32
    assert w_pool.shape == (moe.MH, seam.H) and w_pool.is_contiguous()
    assert w_unpool.shape == (seam.H, moe.MH) and w_unpool.is_contiguous()
    assert w13.is_contiguous() and w2.is_contiguous()
    assert scratch.numel() * scratch.element_size() >= scratch_bytes(key)
    assert health.numel() >= health_words_for_key(key), (
        health.numel(),
        health_words_for_key(key),
    )
    f = build(key)
    f(
        core.data_ptr(),
        stream_a.data_ptr(),
        w_o.data_ptr(),
        alpha_a.data_ptr(),
        beta_a.data_ptr(),
        w_fn.data_ptr(),
        w_router.data_ptr(),
        expert_bias.data_ptr(),
        w_pool.data_ptr(),
        w_pre.data_ptr(),
        w13.data_ptr(),
        w13s.data_ptr(),
        w2.data_ptr(),
        w2s.data_ptr(),
        w_post.data_ptr(),
        w_unpool.data_ptr(),
        stream_out.data_ptr(),
        branch_out.data_ptr(),
        scratch.data_ptr(),
        peers.data_ptr(),
        rank,
        epoch.data_ptr(),
        layer,
        health.data_ptr(),
        alarm,
        ptr(tl),
        stream=torch.cuda.current_stream(),
    )
