# SPDX-License-Identifier: Apache-2.0
"""TP4/EP4 Muse Spark target-verify K2 for M rows (one build per M, M a multiple of
8 up to 128): one 256-CTA persistent launch per layer and rank from o_proj to the
branch output.

    all CTAs       o_proj row tiles b, b + 256 (K = 2048) x 16-token tiles -> push-all AR
    0..71          joint router/pooler (group, K quarter): MFMA split chains a token tile
    128..255       seam units (token, quarter), production sum order
    72..           route: 8 tokens a CTA (a wave each)
    88..88+M       pre-expert norm + MXFP4 xe (token)
    all            local experts: adaptive teams over each expert's fan (16-row tiles)
    all            per-route norm (route r on CTA r % 256), local sum (token t on CTA t)
    all            MoE reduce units (128-column slice, 8 tokens) -> FIN, FRDY
    128..255       unpool slice (2048 branch columns a rank), 16-token tiles
    all            gather units -> branch_out; finish consensus over every CTA and rank

With ``ffn_boundary`` the gather is replaced by the FFN boundary (``tpm_front.ffn_boundary_token``):
    0..M           token t on CTA t: stream'' -> stream_out, the next attention input -> next_in
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, rocdl, range_constexpr
from flydsl.expr.typing import Int32, Int64

from sglang.srt.rocm_muse_spark.tp4mono.ops import tpm_experts as ex
from sglang.srt.rocm_muse_spark.tp4mono.ops import tpm_front as fr
from sglang.srt.rocm_muse_spark.tp4mono.ops.common.debug import region_ids, stamp, stamp_begin
from sglang.srt.rocm_muse_spark.tp4mono.ops.common.epoch import EPOCH_TAG_PERIOD, TAG_SLOTS
from sglang.srt.rocm_muse_spark.tp4mono.ops.common.ops import CM_DEV, ld_i32, load_ptr64, rsrc, traced, uniform
from sglang.srt.rocm_muse_spark.tp4mono.ops.common.plan import THREADS, WAVES, KernelAbi, enter_stage, key_tuple, pair_layout, recording
from sglang.srt.rocm_muse_spark.tp4mono.ops.common.ranks import peer_bases
from sglang.srt.rocm_muse_spark.tp4mono.ops.common.sync import Mailbox, import_alarms, preg, publish, shift, sreg
from sglang.srt.rocm_muse_spark.tp4mono.runtime.abi import K2_DONE_BYTES, K2_DONE_OFFSET

TP = 4
BLOCKS = 256
K2_SLOT0 = 128
J0 = 0  # joint CTAs
ROUTE0 = 72
XQ0 = 88
SEAM0 = BLOCKS - fr.SEAM_CTAS  # 128
U0 = BLOCKS - ex.UP_TASKS  # 128
# Timeline builds: the stamp points (a CTA's thread 0 passes point k; the name says after what).
TL_NAMES = (
    "start", "oproj", "joint", "seam", "route_xq", "xq_wait", "experts", "expertdone", "reduce", "up", "gather",
    "end", "finalize", "up_frdy_wait", "finish", "joint_tiles", "joint_sums", "seam_ar", "seam_norm1",
    "seam_norm2", "joint_x", "route", "xq", "experts_staged", "experts_ug", "experts_ugrdy", "finalize_loads",
    "reduce_poll", "up_mfma", "finish_local", "finish_ranks",
)
TL_POINTS = len(TL_NAMES)

REGIONS = ("ar", "arg", "msq1", "msq2", "rstd1", "rstd2", "xrdy", "finish", "done", "ppart", "route", "xqrdy", "ugrdy",
           "frdy", "rlog", "pool", "br", "epfin", "fing", "expertdone")
FINISH_FANIN = 8
FINISH_GROUPS = 32
FINISH_ROOTS = 4
_PEER_REGIONS = ("ar", "arg", "rlog", "pool", "br", "epfin", "fing")
_SCRATCH_REGIONS = ("msq1", "msq2", "rstd1", "rstd2", "xrdy", "finish", "ppart", "route", "xqrdy", "ugrdy", "frdy",
                    "expertdone")
_SCRATCH_PLAIN = ("xqb", "xscb", "interb", "iscb", "fin", "expert_y", "xe_dbg", "part_dbg", "logit_dbg")

_HERE = Path(__file__).resolve().parent
_SOURCES = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(_HERE.parent.rglob("*.py")))).hexdigest()
_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K2Build:
    tokens: int = 8
    eps_post_attn: float = 1e-8
    eps_ffn: float = 1e-5
    eps_pre_expert: float = 1e-5
    eps_post_expert: float = 1e-8
    router_output_multiplier: float = 1.0
    route_scale: float = 1.0
    timeline: bool = False
    x_override: bool = False  # router/pooler read x from ``x_in`` (stage check); the seam still runs
    # The FFN boundary (post-FFN norm, gated residual, the next layer's attention norm) in the tail instead of
    # the gather: stream_out is then the layer's output stream and next_in the next attention input.
    ffn_boundary: bool = False
    eps_post_ffn: float = 1e-8
    eps_next_attn: float = 1e-5
    # The tail's BR wait (its CTAs spin while the unpool streams its weights): s_sleep units between rounds, and
    # whether a thread first waits on one of its four BR words before it loads all of them.
    ffn_tail_poll_sleep: int = 1
    ffn_tail_two_phase: bool = False
    # Shorter latency chains, the same arithmetic in the same order (SGLANG_MUSE_TP4_K2_LATENCY_TRIM sets this and
    # the fields below, tp4_k2_runner.latency_trim_fields): o_proj row tiles one after the other, so half of the
    # AR transfer overlaps the weight loads (tpm_front.stage_oproj_by_row_tile); the seam norms with one exchange
    # (tpm_front._seam_norm_all); the pre-expert norm weight in flight during the pool poll; the expert tiles
    # staged with one round trip and the first weights issued behind them (tpm_experts.run_gate_up,
    # fan_tile_loads); at M <= 16 a token's finalize waits only for its own experts' teams.
    latency_trim: bool = False
    # The widths from which the o_proj / MoE all-reduce run as reduce-scatter + all-gather (tpm_front.ar_rs,
    # tpm_experts.moe_rs).
    oproj_rs_from: int = fr.RS_MIN_M
    moe_rs_from: int = fr.RS_MIN_M
    # -1: every launch ends in the cross-rank finish consensus (all CTAs and ranks done, alarms imported); a
    # layer index: only the launch of that layer (the step's last K2 layer) does. The dataflow alone keeps the
    # peer regions safe between layers: a rank starts layer L + 1 only after it gathered every rank's branch of
    # layer L, so every rank has consumed its layer-L inputs, and each region of L + 1 is written only after
    # its owner has reached L + 1. A failed wait still reaches every rank's fail-stop at that last launch.
    rendezvous_layer: int = -1
    # Mailbox polls read the FAILED word only after a long wait (common/sync._spin).
    poll_health_grace: bool = False
    # Down tiles assigned wave-major inside an expert's team (tpm_experts.run_experts).
    wave_major_down: bool = False

    def __post_init__(self):
        if not (self.tokens % 8 == 0 and 8 <= self.tokens <= 128):
            raise ValueError("TP4 K2: M must be a multiple of 8 in 8..128")


ABI = KernelAbi((
    "core", "stream_a", "w_o", "alpha_a", "beta_a", "w_fn", "w_joint", "expert_bias", "w_pre",
    "w13", "w13s", "w2", "w2s", "w_post", "w_unpool", "stream_out", "branch_out", "x_in",
    "w_post_ffn", "alpha_f", "beta_f", "w_next_attn", "next_in",
    "scratch", "peers", "rank", "epoch", "layer", "health", "alarm", "tl",
))


def step_tag(epoch, slot):
    e = uniform(ld_i32(epoch, 0))
    return (e % EPOCH_TAG_PERIOD) * TAG_SLOTS + slot + 1


@traced
def stamp_flush(on, tls, tl, tid, bid, points):
    if const_expr(on):  # noqa: SIM102
        if tid < points:
            bo.buffer_store(fx.ptr_load(tls + tid), rsrc(tl), bid * points + tid)


@traced
def finish_consensus(c, done, health, alarm):
    """Every CTA's finish flag; CTA 0 polls all of them in one round (a thread a CTA), then
    this rank's done word, every rank's done word, and every rank's alarm."""
    tid, bid = c["tid"], c["bid"]
    publish(c["put"], c["finish"], bid, 1, tid == 0)
    if bid == 0:
        if tid < BLOCKS:
            c["poll_terminal"]([(c["finish"], tid, 1)])
        gpu.barrier()
        c["stamp"](29)
        if tid == 0:
            c["put"](done, c["rank"], 1)
            rocdl.s_waitcnt(vmcnt=0)
            c["poll_terminal"]([(done, r, 1) for r in range(TP)])
            c["stamp"](30)
            import_alarms(health, alarm, c["rank"], TP)


# ---------------------------------------------------------------- layouts
def peer_layout(m: int) -> dict:
    pairs = {"ar": TP * m * fr.H // 2, "arg": m * fr.H // 2, "rlog": m * fr.E, "pool": m * fr.MH // 2, "br": m * fr.H // 2,
             "epfin": TP * m * fr.MH // 2, "fing": m * fr.MH // 2}
    return pair_layout(pairs.items(), align=256)


def _half(m: int) -> int:
    end = max(o + n for o, n in peer_layout(m).values())
    return (end + 255) // 256 * 256


def peer_bytes(m: int) -> int:
    return 2 * _half(m)


def scratch_layout(m: int) -> dict:
    pairs = {
        "msq1": m * 4 * fr.SEAM_T * 2, "msq2": m * 4 * fr.SEAM_T * 2, "rstd1": m, "rstd2": m, "xrdy": 4 * m,
        "ppart": fr.J_GROUPS * fr.J_SPLITS * 4 * 16 * m, "route": m * 16, "xqrdy": m,
        "ugrdy": ex.max_units(m) * BLOCKS, "frdy": ex.fin_units(m), "expertdone": BLOCKS,
        "finish": BLOCKS,
    }
    lay = pair_layout(pairs.items())
    plain = {
        "xqb": m * 512 * 4, "xscb": m * 128 * 4, "interb": m * 8 * 512 * 4, "iscb": m * 8 * 128 * 4,
        "fin": m * 4096 * 2, "expert_y": m * 8 * 4096 * 2, "xrow": m * 8192 * 2,
        "xe_dbg": m * 4096 * 2, "part_dbg": m * 4096 * 2, "logit_dbg": m * 256 * 4,
    }
    for name, size in plain.items():
        end = max(o + n for o, n in lay.values())
        lay[name] = ((end + 255) // 256 * 256, size)
    return lay


def scratch_bytes(m: int) -> int:
    return max(o + n for o, n in scratch_layout(m).values())


# ---------------------------------------------------------------- build
_BUILDS: dict = {}


def _make(key: K2Build):
    m = key.tokens
    lay = scratch_layout(m)
    play = peer_layout(m)
    half = _half(m)

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, m * 16, 16]
        ecnt: fx.Array[fx.Int32, 64, 16]
        ustart: fx.Array[fx.Int32, 64, 16]
        unit_e: fx.Array[fx.Int32, ex.max_units(m), 16]
        unit_f: fx.Array[fx.Int32, ex.max_units(m), 16]
        count: fx.Array[fx.Int32, 4, 16]
        fan_t: fx.Array[fx.Int32, m, 16]
        fan_k: fx.Array[fx.Int32, m, 16]
        fan_n: fx.Array[fx.Int32, 4, 16]

    @fx.struct
    class RowsLds:
        xl: fx.Array[fx.Int32, 16 * fr.XROW // 2, 16]

    @fx.struct
    class JointLds:
        fl: fx.Array[fx.Float32, 192 * m, 16]

    @fx.struct
    class XqLds:
        xq: fx.Array[fx.Int32, 16 * 512, 16]
        xsc: fx.Array[fx.Int32, 16 * 128, 16]

    @fx.struct
    class ExpertDownLds:
        interq: fx.Array[fx.Int32, 16 * 512, 16]
        intsc: fx.Array[fx.Int32, 16 * 128, 16]

    @fx.struct
    class ExpertWorkLds:
        xq: XqLds
        down: ExpertDownLds

    @fx.union
    class StageLds:
        rows: RowsLds
        expert: ExpertWorkLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        av: fx.Array[fx.Float32, 2 * THREADS, 16]
        fl: fx.Array[fx.Float32, 192 * m, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = (f"tp4_muse_k2_m{m}_t{int(key.timeline)}_x{int(key.x_override)}" + ("_fb" if key.ffn_boundary else "")
            + ("_lt" if key.latency_trim else "")
            + (f"_rs{key.oproj_rs_from}_{key.moe_rs_from}" if (key.oproj_rs_from, key.moe_rs_from) != (fr.RS_MIN_M,) * 2
               else "")
            + (f"_rv{key.rendezvous_layer}" if key.rendezvous_layer >= 0 else "")
            + ("_hg" if key.poll_health_grace else "")
            + ("_wmd" if key.wave_major_down else ""))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(core: Int64, stream_a: Int64, w_o: Int64, alpha_a: Int64, beta_a: Int64, w_fn: Int64, w_joint: Int64,
           expert_bias: Int64, w_pre: Int64, w13: Int64, w13s: Int64, w2: Int64, w2s: Int64, w_post: Int64,
           w_unpool: Int64, stream_out: Int64, branch_out: Int64, x_in: Int64, w_post_ffn: Int64, alpha_f: Int64,
           beta_f: Int64, w_next_attn: Int64, next_in: Int64, scratch: Int64, peers: Int64,
           rank: Int32, epoch: Int64, layer: Int32, health: Int64, alarm: Int64, tl: Int64):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        stage = alloc.allocate(StageLds)
        work = stage.expert.peek()
        rows = stage.rows.peek()
        xql, dl = work.xq, work.down
        mb = Mailbox(step_tag(epoch, K2_SLOT0 + layer) - 1, health, alarm, region_ids(REGIONS),
                     health_grace=key.poll_health_grace)
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, TP)
        own = load_ptr64(peers, rank)
        output_head = alarm - fx.Int64(rank) * 4
        done = preg(output_head, K2_DONE_OFFSET, "done", K2_DONE_BYTES)
        peer = {}
        for region in _PEER_REGIONS:
            off, extent = play[region]
            peer[region] = [shift(preg(b, off, region, extent), fx.Int64(par) * half) for b in bases]
            peer[region + "_own"] = shift(preg(own, off, region, extent), fx.Int64(par) * half)
        xbuf = scratch + fx.Int64(lay["xrow"][0])
        c = {
            "S": m, "tid": tid, "bid": bid, "lane": tid % 64, "wave": tid // 64, "rank": rank,
            "eps_attn": key.eps_post_attn, "eps_ffn": key.eps_ffn, "eps_pre": key.eps_pre_expert,
            "eps_post": key.eps_post_expert, "router_output_multiplier": key.router_output_multiplier,
            "route_scale": key.route_scale, "eps_post_ffn": key.eps_post_ffn, "eps_next_attn": key.eps_next_attn,
            "stream_store_cm": CM_DEV if key.ffn_boundary else 0,
            "tail_poll_sleep": key.ffn_tail_poll_sleep, "tail_two_phase": key.ffn_tail_two_phase,
            "latency_trim": key.latency_trim, "oproj_rs_from": key.oproj_rs_from, "moe_rs_from": key.moe_rs_from,
            "wave_major_down": key.wave_major_down,
            "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words, "poll": mb.poll,
            "poll_terminal": mb.poll_terminal,
            "red": lds.red.ptr, "av": lds.av.ptr, "fl": lds.fl.ptr,
            "peer": peer, "ar_peer": peer["ar"], "ar_own": peer["ar_own"],
            "xl": rows.xl.ptr, "xq": xql.xq.ptr, "xsc": xql.xsc.ptr, "interq": dl.interq.ptr, "intsc": dl.intsc.ptr,
            "rt": {"route": rl.route.ptr, "ecnt": rl.ecnt.ptr, "ustart": rl.ustart.ptr, "unit_e": rl.unit_e.ptr,
                   "unit_f": rl.unit_f.ptr, "count": rl.count.ptr, "fan_t": rl.fan_t.ptr, "fan_k": rl.fan_k.ptr, "fan_n": rl.fan_n.ptr},
            "xbuf": xbuf,
            "x_src": x_in if key.x_override else xbuf,
            "args": {"core": core, "stream_a": stream_a, "w_o": w_o, "alpha_a": alpha_a, "beta_a": beta_a,
                     "w_fn": w_fn, "w_joint": w_joint, "expert_bias": expert_bias, "w_pre": w_pre, "w13": w13,
                     "w13s": w13s, "w2": w2, "w2s": w2s, "w_post": w_post, "w_unpool": w_unpool,
                     "stream_out": stream_out, "branch_out": branch_out, "w_post_ffn": w_post_ffn,
                     "alpha_f": alpha_f, "beta_f": beta_f, "w_next_attn": w_next_attn, "next_in": next_in},
        }
        for region in _SCRATCH_REGIONS:
            c[region] = sreg(scratch, lay[region][0], region, lay[region][1])
        for nm in _SCRATCH_PLAIN:
            c[nm] = scratch + fx.Int64(lay[nm][0])
        on, tls = key.timeline, lds.tls.ptr
        c["stamp"] = lambda k: stamp(on, tls, tid, k)
        stamp_begin(on, tls, tid, TL_POINTS)
        enter_stage("oproj")
        if const_expr(key.latency_trim):
            fr.stage_oproj_by_row_tile(c, bid)
        else:
            fr.stage_oproj(c, bid)
        if const_expr(fr.ar_rs(m, key.oproj_rs_from)):
            if bid < fr.AR_RED_CTAS:
                enter_stage("ar_reduce")
                fr.stage_ar_reduce(c, bid)
        stamp(on, tls, tid, 1)
        if bid < J0 + fr.J_TASKS:
            enter_stage("joint")
            grp = (bid - J0) // fr.J_QUARTERS
            q = (bid - J0) % fr.J_QUARTERS
            fr.joint_compute(c, grp, q, fr.joint_loads(c, grp, q))
        stamp(on, tls, tid, 2)
        if bid >= SEAM0:
            enter_stage("seam")
            fr.stage_seam(c, bid - SEAM0)
        stamp(on, tls, tid, 3)
        if (bid >= ROUTE0) & (bid < ROUTE0 + ex.groups8(m)):
            enter_stage("route")
            fr.stage_route(c, bid - ROUTE0)
            stamp(on, tls, tid, 21)
        if (bid >= XQ0) & (bid < XQ0 + m):
            enter_stage("xq")
            fr.stage_xq(c, bid - XQ0)
            stamp(on, tls, tid, 22)
        stamp(on, tls, tid, 4)
        enter_stage("xq_wait")
        if const_expr(not key.latency_trim):  # trimmed: tpm_experts.run_gate_up polls the xq rows itself
            if tid < m:
                c["poll"]([(c["xqrdy"], tid, 1)])
        for r in range_constexpr((m * 16 + THREADS - 1) // THREADS):
            i = tid + THREADS * r
            if i < m * 16:
                value = c["poll"]([(c["route"], i, 1)])[0][0]
                fx.ptr_store(value, c["rt"]["route"] + i)
        gpu.barrier()
        stamp(on, tls, tid, 5)
        enter_stage("experts")
        ex.run_experts(c)
        publish(c["put"], c["expertdone"], bid, fx.Int32(1), tid == 0)
        gpu.barrier()
        stamp(on, tls, tid, 6)
        enter_stage("finalize")
        ex.stage_finalize(c, bid)
        up_ops = ex.up_loads(c, fx.max(bid - U0, 0), bid >= U0)
        enter_stage("reduce")
        ex.stage_reduce(c, bid)
        stamp(on, tls, tid, 8)
        if bid >= U0:
            enter_stage("up")
            ex.stage_up(c, bid - U0, up_ops)
        stamp(on, tls, tid, 9)
        if const_expr(key.ffn_boundary):
            enter_stage("ffn_boundary")
            gpu.barrier()  # LDS ``red`` is free from here
            if bid < m:
                fr.ffn_boundary_token(c, bid, fr.ffn_boundary_loads(c, bid))
        else:
            enter_stage("gather")
            ex.stage_gather(c, bid)
        stamp(on, tls, tid, 10)
        stamp(on, tls, tid, 11)
        enter_stage("finish")
        if const_expr(key.rendezvous_layer < 0):
            finish_consensus(c, done, health, alarm)
        else:
            if layer == key.rendezvous_layer:
                finish_consensus(c, done, health, alarm)
        stamp(on, tls, tid, 14)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(core: Int64, stream_a: Int64, w_o: Int64, alpha_a: Int64, beta_a: Int64, w_fn: Int64, w_joint: Int64,
               expert_bias: Int64, w_pre: Int64, w13: Int64, w13s: Int64, w2: Int64, w2s: Int64, w_post: Int64,
               w_unpool: Int64, stream_out: Int64, branch_out: Int64, x_in: Int64, w_post_ffn: Int64,
               alpha_f: Int64, beta_f: Int64, w_next_attn: Int64, next_in: Int64, scratch: Int64, peers: Int64,
               rank: Int32, epoch: Int64, layer: Int32, health: Int64, alarm: Int64, tl: Int64,
               stream: fx.Stream = _STREAM):
        _ = keyed
        k2(core, stream_a, w_o, alpha_a, beta_a, w_fn, w_joint, expert_bias, w_pre, w13, w13s, w2, w2s, w_post,
           w_unpool, stream_out, branch_out, x_in, w_post_ffn, alpha_f, beta_f, w_next_attn, next_in, scratch, peers,
           rank, epoch, layer, health, alarm, tl).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k2, launch)
    return launch


def build(key: K2Build):
    if key not in _BUILDS:
        _BUILDS[key] = _make(key)
    return _BUILDS[key]


def compile_only(key: K2Build) -> bool:
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        with recording():
            build(key)(*ABI.zeros(), stream=torch.cuda.current_stream())
        return True
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old
