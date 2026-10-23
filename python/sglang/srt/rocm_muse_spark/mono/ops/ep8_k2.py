# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""q8 TP8/EP8 full K2 in one256-CTA persistent launch per layer/rank.

Original o_proj, residual/pre-FFN normalization, router and pooler stages
feed local complete E32 experts. Adaptive teams execute FP4 gate/up and down;
per-route normalization precedes the local weighted sum. A tagged token-sized
peer reduction supplies FIN/FRDY to original unpool and branch gather stages.
All ranks and CTAs join bounded completion consensus, including U0 ranks.

This is an isolated qualification prototype, not enabled serving code.
"""

import os
import hashlib
from pathlib import Path
import re
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, rocdl, gpu
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
from sglang.srt.rocm_muse_spark.mono.ops.stages import gemv, moe, oproj
from sglang.srt.rocm_muse_spark.mono.ops import ep8_stages as ep8
from sglang.srt.rocm_muse_spark.mono.ops import ep8_frontend as frontend
from sglang.srt.rocm_muse_spark.mono.ops import ep8_owner_ar as owner_ar
from sglang.srt.rocm_muse_spark.mono.ops import ep8_quarter_seam as quarter_seam
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

REGIONS = ("ar", "msq1", "msq2", "rstd1", "rstd2", "xrdy", "finish", "done", "rpart", "ppart", "route", "xqrdy", "ugrdy", "frdy", "rlog", "pool", "br", "epfin", "expertdone", "normrdy", "ar_reduced", "owned_fin")
FINISH_FANIN = 8
FINISH_GROUPS = 32
FINISH_ROOTS = 4
_PEER_REGIONS = ("ar", "rlog", "pool", "br", "epfin", "ar_reduced", "owned_fin")
_SCRATCH_REGIONS = ("msq1", "msq2", "rstd1", "rstd2", "xrdy", "finish", "rpart", "ppart", "route", "xqrdy", "ugrdy", "frdy", "expertdone", "normrdy")
_SCRATCH_PLAIN = ("xqb", "xscb", "interb", "iscb", "fin", "expert_y", "normy")

# the K2 sources (and the shared helpers): in every build's JIT cache key
_SOURCES = source_digest(
    "common",
    "stages/gemv.py",
    "stages/mxq.py",
    "stages/oproj.py",
    "stages/seam_b.py",
    "stages/moe.py",
    "k2_experiment.py",
    "k2.py",
)

_SOURCES += hashlib.sha256(Path(__file__).read_bytes() + Path(ep8.__file__).read_bytes() + Path(frontend.__file__).read_bytes() + Path(owner_ar.__file__).read_bytes() + Path(quarter_seam.__file__).read_bytes()).hexdigest()

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
        if self.experimental_two_cta or self.tokens != 8:
            raise ValueError("EP8 full K2 requires q8 and exactly256 resident CTAs")


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
    pairs = {"ar": oproj.ar_pairs(MAX_TOKENS), **{n:v for n,v in moe.peer_pairs(MAX_TOKENS).items() if n in ("rlog","pool","br")}, "epfin": TP * MAX_TOKENS * moe.MH // 2, "ar_reduced": MAX_TOKENS * oproj.H // 2, "owned_fin": MAX_TOKENS * moe.MH // 2}
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
    assert s == 8 and not experimental_two_cta
    pairs = {"msq1": NSL*s,"msq2": NSL*s,"rstd1":s,"rstd2":s,"xrdy":32,"rpart":moe.R_TASKS*s*WAVES,"ppart":moe.P_TASKS*s*moe.ROWS,"route":s*16,"xqrdy":s,"ugrdy":32*128,"frdy":32,"expertdone":256,"normrdy":64,"finish":292}
    lay=pair_layout(pairs.items())
    plain={"xqb":s*512*4,"xscb":s*128*4,"interb":64*512*4,"iscb":64*128*4,"fin":s*4096*2,"expert_y":64*4096*2,"normy":64*4096*4,"xrow":s*8192*2}
    for name,size in plain.items():
        end=max(o+n for o,n in lay.values())
        lay[name]=((end+15)//16*16,size)
    return lay


def scratch_bytes(key: K2Build) -> int:
    return max(
        o + n
        for o, n in scratch_layout(
            key.tokens, experimental_two_cta=key.experimental_two_cta
        ).values()
    )


def max_scratch_bytes(tokens: int = MAX_TOKENS, *, experimental_two_cta: bool = False) -> int:
    assert tokens == 8 and not experimental_two_cta
    return max(o+n for o,n in scratch_layout(8).values())


# ---------------------------------------------------------------- build
_BUILDS: dict = {}


def _make(key: K2Build):
    s = key.tokens
    assert 1 <= s <= MAX_TOKENS
    blocks = 256
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
        route: fx.Array[fx.Int32,128,16]
        expert: fx.Array[fx.Int32,32,16]
        kof: fx.Array[fx.Int32,256,16]
        active_ids: fx.Array[fx.Int32,32,16]
        count: fx.Array[fx.Int32,1,16]

    @fx.struct
    class XqLds:
        xq: fx.Array[fx.Int32,4096,16]
        xsc: fx.Array[fx.Int32,1024,16]

    @fx.struct
    class RowsLds:
        xl: fx.Array[fx.Int32,s*xrow_max//2,16]

    @fx.struct
    class ExpertDownLds:
        interq: fx.Array[fx.Int32,4096,16]
        intsc: fx.Array[fx.Int32,1024,16]

    @fx.struct
    class ExpertWorkLds:
        xq: XqLds
        down: ExpertDownLds

    @fx.union
    class StageLds:
        rows: RowsLds
        expert: ExpertWorkLds

    def allocate_work(alloc):
        stage=alloc.allocate(StageLds)
        work=stage.expert.peek()
        return work.xq,stage.rows.peek(),work.down

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        av: fx.Array[fx.Float32, THREADS, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = f"p255_muse_ep8_full_k2_s{s}_t{int(key.timeline)}"
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
            "quarter_seam": True,
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
            "rt": {
                "route": rl.route.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
                "active_ids": rl.active_ids.ptr,
                "count": rl.count.ptr,
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
        for region in _SCRATCH_REGIONS:
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
                owner_ar.stage_oproj(c, bid)
        else:
            owner_ar.stage_oproj(c, bid)
        stamp(on, tls, tid, 1)
        if bid < 8:
            enter_stage("oproj_owner_reduce")
            owner_ar.stage_reduce(c, bid)
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
                    frontend.stage_xq(c, bid - moe.XQ0)
                stamp(on, tls, tid, 4)
        else:
            if bid < 32:
                enter_stage("seam")
                quarter_seam.stage_seam(c, bid)
                stamp(on, tls, tid, 3)
                if bid < s:
                    enter_stage("route")
                    moe.stage_route(c, bid)
                if (bid >= moe.XQ0) & (bid < moe.XQ0 + s):
                    enter_stage("xq")
                    frontend.stage_xq(c, bid - moe.XQ0)
                stamp(on, tls, tid, 4)
        # ---- full local EP8 experts; frontend has produced quantized pooled input.
        enter_stage("xq_wait")
        moe.load_xq(c)
        if tid < 128:
            value = c["poll"]([(c["route"],tid,1)])[0][0]
            fx.ptr_store(value,c["rt"]["route"]+tid)
        gpu.barrier()
        enter_stage("ep8_experts")
        ep8.run_experts(c)
        # Every CTA, including U0, must join publication and all collectives.
        publish(c["put"],c["expertdone"],bid,fx.Int32(1),tid==0)
        if bid < 64:
            enter_stage("ep8_norm")
            ep8.normalize_route(c,bid)
        if bid < 8:
            enter_stage("ep8_local_sum")
            ep8.local_sum(c,bid)
        if bid < 32:
            if rank == bid // 4:
                enter_stage("ep8_owner_reduce")
                ep8.token_reduce(c,bid)
            enter_stage("ep8_fin_gather")
            moe.stage_fin_gather(c,bid)
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
        return True

    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old


def compile_only(key: K2Build) -> bool:
    """Compile ``key``'s kernel without running it (a launch on null arguments
    would fault)."""
    return _compile(key, build(key))

