"""K2 on MI455X (gfx1250) at TP4 / EP1: one Muse Spark sparse decoder layer from the attention's core output
(o_proj's input) to the MoE branch, one persistent launch a layer and TP rank, every TP collective inside the
kernel. Port of the MI355X ``k2bundle/ops/k2.py`` (TP8 / EP1, q8 schedule); stages from the helpers' modules
(``port/gemv`` o_proj / seam / router / pooler / route / unpooler / gather, ``port/moe`` xq / UG / down /
finalize), sync from ``port/sync``. Contract: ``port/INTERFACE.md``.

Width: T = 8 rows only (C1 verify step). Other widths take the served path (``step_disable_reason``).

CTA roles (256 CTAs x 512 threads, one a CU, all co-resident). No CTA waits on a stage later in its own program:

    CTAs 0 .. 255     o_proj: tiles bid, bid + 256 (16 rows x K 2048 each) -> AR#1 reduce-scatter to the owners
    CTAs 0 .. 31      AR#1 owner sum (quarter seam) -> XG / XSQ; CTAs 0 .. 7 then seam token t -> stream_out, x, XRDY
    CTAs 0 .. 7       route token t (after the router's RLOG_PARTS) -> ROUTE
    CTAs 32 .. 95     router: 64 q8 tasks (16 experts x K quarter x token pair), weights in flight before XRDY
    CTAs 0 .. 255     pooler task bid (64 groups x K quarter) -> POOL all-gather (CTAs 0 .. 31 after the seam)
    CTAs 8 .. 15      xq token t: POOL row -> pre_expert_norm -> MXFP8 -> XQ / XSC (+ XQRDY)
    every CTA         route table (LDS); UG tasks u = bid + 256 i < |U| 32 (slot, 32-column group)
    every wave        down tasks v = 16 bid + wave + 4096 i < |U| 256 (slot, 16-row MH tile) -> YR (row owner)
    CTAs 192 .. 255   finalize (owned row tl, 128-column slice j) -> FIN rows (plain) + FRDY on every rank
    CTAs 0 .. 127     unpooler task bid (16 branch columns) -> BR all-gather; W_unpool issued before FRDY
    CTAs 128 .. 191   gather: BR -> ``branch_out``
    every CTA         finish consensus (fan-in tree, then the 4 ranks' done pairs)
"""

import os
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import Int32, Int64, T

from port import consts as K
from port.common.ops import ld_i32, load_ptr64, rsrc, traced, uniform, vm_wait
from port.gemv import moe_dense as MD
from port.gemv import oproj, quarter_seam
from port.moe import moe as X
from port.sync.abi import K2_DONE_BYTES, K2_DONE_OFFSET
from port.sync.debug import region_ids
from port.sync.epoch import EPOCH_TAG_PERIOD, TAG_SLOTS
from port.sync.memmodel import CM_DEV, CM_SYS, memrealtime
from port.sync.plan import KernelAbi, enter_stage, key_tuple, pair_layout, source_digest
from port.sync.ranks import peer_bases
from port.sync.abi import FAILED
from port.sync.memmodel import wait_stores
from port.sync.sync import Mailbox, import_alarms, ld_health, preg, publish, publish_wt, shift, sreg, stamp

TP = K.TP
BLOCKS = K.BLOCKS
THREADS = K.THREADS
WAVES = K.WAVES
MAX_TOKENS = K.MAX_TOKENS
K2_SLOT0 = 128  # K2 of layer i is mailbox slot 128 + i

# ---- CTA roles (T = 8)
OWNER0, OWNER_CTAS = 0, 4 * MAX_TOKENS  # AR#1 owner sum: 32 CTAs (P4: four a token)
R0 = OWNER0 + OWNER_CTAS  # 32: router CTAs
P0 = 0  # pooler task p on CTA p
XQ0 = MAX_TOKENS  # 8: xq CTAs (seam CTAs after their pooler task)
FIN_CTAS = (MAX_TOKENS // TP) * X.FIN_SLICES  # 64: (owned row, slice)
FIN0 = BLOCKS - FIN_CTAS  # 192
U0 = 0  # unpooler CTAs
_FIN_PAIRS = os.environ.get("K2_FIN_PAIRS", "1") == "1"  # FIN as tagged pairs: no drain / write-back / FRDY
_FINISH_LITE = os.environ.get("K2_FINISH_LITE", "1") == "1"  # finish flag: store drain + flag, no L2 write-back
_FINISH2 = os.environ.get("K2_FINISH2", "1") == "1"  # one-level finish fan-in, done pushed to every rank
# k2v2ho (hand-offs): every rank unpools all 8192 branch columns from the FIN all-gather (w_unpool is replicated),
# two tasks a CTA, the words stored straight to branch_out: no BR all-gather, no gather stage
_UP_ALL = os.environ.get("K2_UP_ALL", "1") == "1"
_UP_FINW = os.environ.get("K2_UP_FINW", "pre")  # K2_UP_ALL: the finalize CTAs' unpool weights before (pre) / after (post) finalize
# no end-of-launch consensus: the data dependencies of the next launch's mailboxes order the ranks; peers' give-ups
# are imported from the output rank's alarm words before the unpooler (best effort; the runner reads them after a step)
_FINISH_ASYNC = os.environ.get("K2_FINISH_ASYNC", "1") == "1"
RT0 = 96  # route CTAs (K2_ROUTE_CTAS): pooler-only CTAs before the experts
G0 = MD.UP_TASKS  # 128: gather CTAs
UG_GROUPS = K.RI // 32  # 32 UG tasks a slot
DOWN_TILES = K.MH // K.ROWS  # 256 down wave-tasks a slot
assert MD.P_TASKS == BLOCKS
assert R0 + MD.R_TASKS_Q8 <= BLOCKS
assert G0 + MD.GATHER_TASKS <= FIN0 and U0 + MD.UP_TASKS <= G0
assert oproj.OTASKS == 2 * BLOCKS
assert R0 + MD.R_TASKS_Q8 <= RT0 and RT0 + MAX_TOKENS <= U0 + MD.UP_TASKS and RT0 + MAX_TOKENS <= FIN0
assert (FIN0 - R0) % MD.P_PARTS == 0 and BLOCKS - FIN0 == MD.R_TASKS_Q8  # K2_POOL_MOVE pairing
assert MD.UP_ALL_TASKS == 2 * BLOCKS
assert R0 % MD.R_PARTS == 0 and MD.R_PARTS == MD.P_PARTS  # a router CTA's K quarter is its pooler task's

FINISH_FANIN = 8
FINISH_GROUPS = BLOCKS // FINISH_FANIN  # 32
FINISH_ROOTS = FINISH_GROUPS // FINISH_FANIN  # 4

# ---- regions
_PEER_PAIRS = {
    "ar": oproj.ar_pairs(MAX_TOKENS),
    **MD.peer_pairs(MAX_TOKENS),  # rlog, rlog_parts, pool, br
    "xg": MAX_TOKENS * K.H // 2,
    "xsq": MAX_TOKENS * 64,
    "yr": TP * MAX_TOKENS * K.TOPK * K.MH // 2,  # [src][t][k][MH] bf16 pairs (owned rows written)
    "frdy": MAX_TOKENS * X.FIN_SLICES,  # row t, slice j
    "finp": MAX_TOKENS * K.MH // 2,  # [t][MH] bf16 pairs: FIN as tagged pairs (K2_FIN_PAIRS; no FRDY)
    "done4": TP,  # finish consensus (K2_FINISH2): rank r's sticky FAILED word, pushed to every rank
}
_PEER_PLAIN = {"fin": MAX_TOKENS * K.MH * 2}  # [t][MH] bf16, written by the row owners
_SCRATCH_PAIRS = {
    "msq1": 64 * MAX_TOKENS,
    "msq2": 64 * MAX_TOKENS,
    "xrdy": max(64, 4 * MAX_TOKENS),
    "p1": 64 * MAX_TOKENS,
    "p2": 64 * MAX_TOKENS,
    "rpart": MD.R_TASKS * MAX_TOKENS * WAVES,
    "ppart": MD.P_TASKS * MAX_TOKENS * K.ROWS,
    "route": MAX_TOKENS * 2 * K.TOPK,
    "xqrdy": MAX_TOKENS,
    "ugrdy": X.MAX_U * UG_GROUPS,
    "fmsq": MAX_TOKENS * K.TOPK * X.FIN_SLICES,
    "finish": BLOCKS + FINISH_GROUPS + FINISH_ROOTS,
}
# Expert activations (XQ, INTER) are read through this XCD's L2 (plain loads, after a per-CTA device-scope
# invalidate), not past it: every wave of the grid broadcast-reads the same few KB, and device-scope reads of them
# queue on one or two memory channels (a CTA's first down load took ~45 us to return).
_ACT_L2 = os.environ.get("K2_ACT_L2", "1") == "1"
_UG_WT = os.environ.get("K2_UG_WT", "1") == "1"  # INTER stored write-through, UGRDY without the L2 write-back
_UG_PIPE = os.environ.get("K2_UG_PIPE", "1") == "1"  # UG loop software-pipelined (next task's weights early)
_DOWN_WFIRST = os.environ.get("K2_DOWN_WFIRST", "1") == "1"  # a down task's w2 loads issued before its UGRDY poll
_DOWN_REV = os.environ.get("K2_DOWN_REV", "1") == "1"  # down CTA order reversed (see _down_cta)
# lane F (pre-expert / tail)
_ROUTER_X8 = os.environ.get("K2_ROUTER_X8", "0") == "1"  # router CTAs stage all 8 x rows for their pooler task
_POOL_EARLY = os.environ.get("K2_POOL_EARLY", "0") == "1"  # pooler weights issued right after o_proj on every CTA
_POOL_FIRST = os.environ.get("K2_POOL_FIRST", "0") == "1"  # router CTAs: pooler task first, then the router on its LDS x
_POOL_MOVE = os.environ.get("K2_POOL_MOVE", "0") == "1"  # router CTAs' pooler tasks run on CTAs 192.. as a 2nd task
_SEAM_DIRECT = os.environ.get("K2_SEAM_DIRECT", "1") == "1"  # AR#1 pushed to every rank, seam sums it (no XG hop)
_SEAM_CTAS = MAX_TOKENS if _SEAM_DIRECT else OWNER_CTAS  # CTAs with a seam stage
_ROUTE_CTAS = os.environ.get("K2_ROUTE_CTAS", "1") == "1"  # route token t on CTA RT0 + t, after its pooler task
# k2v2 pre-expert chain
_RPOOL = os.environ.get("K2_RPOOL", "0") == "1"  # router CTAs: pooler weights issued with the router's, x8 staging shared
# every rank computes the router logits of all 256 experts (256 tasks = every CTA: 16 experts x K quarter x token
# pair, with the CTA's pooler task on the same x quarter) into local RLOGL: the route's wait is a device hop
_RLOCAL = os.environ.get("K2_RLOCAL", "1") == "1"
assert MD.R_PARTS * (MD.E // WAVES) * (MAX_TOKENS // MD.ROUTER_PAIR) == BLOCKS
_RT_FAST = os.environ.get("K2_RT_FAST", "1") == "1"  # route table by LDS bitmap / first-entry atomics (moe.load_route_fast)
# each pool task's fp32 part pushed to every rank's POOLP; xq adds the 4 parts in part order (no PPART hop)
_POOL_DIRECT = os.environ.get("K2_POOL_DIRECT", "1") == "1"
assert not _POOL_DIRECT or _RLOCAL, "K2_POOL_DIRECT runs on the K2_RLOCAL router / pooler program"
if _POOL_DIRECT:
    _PEER_PAIRS["poolp"] = MAX_TOKENS * MD.P_PARTS * MD.MH  # [t][q][4096] fp32 pool parts
if _RLOCAL:
    _SCRATCH_PAIRS["rlogl"] = MD.R_PARTS * MAX_TOKENS * MD.E  # [q][t][256] fp32 router parts (this rank's)
# Expert phase on the vres2 pattern (moe.py ``vr_*``): K split over the waves, activations resident, partial
# sums ds_add'ed into LDS, the last wave of a task runs its epilogue (no CTA barrier in the weight stream);
# down = a contiguous run of tiles a CTA, balanced against its UG tasks.
_VRES2 = os.environ.get("K2_VRES2", "1") == "1"
_VR_YR_TILE = os.environ.get("K2_VR_YR", "end") == "tile"  # vres2 YR puts: a tile's last K half (tile) or after the run (end)
_VR_DNACQ = os.environ.get("K2_VR_DNACQ", "1") == "1"  # vres2: acquire_cta again between UG and down
_VR_REV = os.environ.get("K2_VR_REV", "1") == "1"  # vres2: UG CTA order reversed (_vr_ub)
_VR_LATE = os.environ.get("K2_VR_LATE", "1") == "1"  # vres2: the xq CTAs poll XQ before their first weights
_VR_LAZY = os.environ.get("K2_VR_LAZY", "1") == "1"  # vres2: a UG epilogue's UGRDY published at the wave's next task
_VR_UGW = int(os.environ.get("K2_VR_UGW", "16"))  # vres2: down tiles an extra UG task counts for (bytes: 16)
_VR_EPI_ROT = os.environ.get("K2_VR_EPI", "rot") == "rot"  # vres2: task i's epilogue on wave i % 16 (rot) or the last to add (last)
assert not _VRES2 or (_ACT_L2 and _UG_WT), "K2_VRES2 needs K2_ACT_L2=1 (trash pick row) and K2_UG_WT=1"
_SCRATCH_PLAIN = dict(X.scratch_plain(MAX_TOKENS, "a8", l2=_ACT_L2))  # xqb, xscb, interb, iscb (fmsq: pairs above)
_SCRATCH_PLAIN.pop("fmsq", None)
_SCRATCH_PLAIN["xrow"] = MAX_TOKENS * K.H * 2

REGIONS = tuple(_PEER_PAIRS) + tuple(_SCRATCH_PAIRS) + ("done",)

_SOURCES = source_digest("common", "sync", "gemv", "moe", "consts.py", "k2.py")
_STREAM = fx.Stream(None)


def _layout(pairs, plain, align=256):
    lay = pair_layout(pairs.items(), align=align)
    end = max(o + n for o, n in lay.values())
    for name, nbytes in plain.items():
        off = (end + align - 1) // align * align
        lay[name] = (off, nbytes)
        end = off + nbytes
    return lay, (end + align - 1) // align * align


def peer_layout() -> dict:
    """One parity's peer regions (name -> (byte offset, bytes))."""
    return _layout(_PEER_PAIRS, _PEER_PLAIN)[0]


def _half() -> int:
    return _layout(_PEER_PAIRS, _PEER_PLAIN)[1]


def peer_bytes() -> int:
    """Both parities' peer bytes (past the runner's PEER_HEAD)."""
    return 2 * _half()


_SCRATCH_ALIGN = 256 if _ACT_L2 else 16  # an L2-cached region shares no cache line with another region


def scratch_layout() -> dict:
    return _layout(_SCRATCH_PAIRS, _SCRATCH_PLAIN, align=_SCRATCH_ALIGN)[0]


def scratch_bytes() -> int:
    return _layout(_SCRATCH_PAIRS, _SCRATCH_PLAIN, align=_SCRATCH_ALIGN)[1]


# ---- build key / ABI
@dataclass(frozen=True)
class K2Build:
    tokens: int = MAX_TOKENS
    eps_post_attn: float = K.EPS_POST_ATTN
    eps_ffn: float = K.EPS_FFN
    eps_pre_expert: float = K.EPS_PRE_EXPERT
    eps_post_expert: float = K.EPS_POST_EXPERT
    router_output_multiplier: float = K.ROUTER_MULT
    act: str = "a8"
    ug_ksplit: int = 8  # in-CTA split-K of a UG task (16 waves: 2 tiles x 8 K parts)
    ug_prefetch: int = 4
    down_prefetch: int = 8
    fault_rank: int = -1
    fault_epoch: int = 0
    fault_layer: int = 0
    fault_ticks: int = 0

    def __post_init__(self):
        if self.tokens != MAX_TOKENS:
            raise ValueError(f"gfx1250 K2 supports T = {MAX_TOKENS} only (got {self.tokens})")
        if self.act not in ("a8", "a4"):
            raise ValueError(self.act)


def make_key(**kw) -> K2Build:
    return K2Build(**kw)


ABI = KernelAbi(
    (
        "core", "stream_a", "w_o", "alpha_a", "beta_a", "w_fn", "w_router", "expert_bias", "w_pool", "w_pre",
        "w13", "w13s", "w2", "w2s", "w_post", "w_unpool", "stream_out", "branch_out", "scratch", "peers", "rank",
        "epoch", "layer", "health", "alarm", "tl",
    )
)  # fmt: skip


def step_tag(epoch, slot):
    """The launch's mailbox tag (epoch and slot packed; never 0)."""
    return (epoch % EPOCH_TAG_PERIOD) * TAG_SLOTS + slot + 1


def step_disable_reason(rows: int) -> str | None:
    if rows != MAX_TOKENS:
        return f"K2 gfx1250 runs T={MAX_TOKENS} only (step has {rows} rows)"
    return None


@traced
def finish_consensus(c, done, health, alarm):
    """Keep CTA 0 live until every local CTA and every TP rank reached the tail (MI355X default fan-in tree)."""
    tid, bid = c["tid"], c["bid"]
    publish(c["put"], c["finish"], bid, 1, tid == 0)
    if (bid < FINISH_GROUPS) & (tid == 0):
        lo = bid * FINISH_FANIN
        c["poll_terminal"]([(c["finish"], lo + i, 1) for i in range(FINISH_FANIN)])
        c["put"](c["finish"], BLOCKS + bid, 1)
    if (bid < FINISH_ROOTS) & (tid == 0):
        lo = BLOCKS + bid * FINISH_FANIN
        c["poll_terminal"]([(c["finish"], lo + i, 1) for i in range(FINISH_FANIN)])
        c["put"](c["finish"], BLOCKS + FINISH_GROUPS + bid, 1)
    if (bid == 0) & (tid == 0):
        c["poll_terminal"]([(c["finish"], BLOCKS + FINISH_GROUPS + i, 1) for i in range(FINISH_ROOTS)])
        c["put"](done, c["rank"], 1)
        vm_wait()
        c["poll_terminal"]([(done, r, 1) for r in range(TP)])
        import_alarms(health, alarm, c["rank"], TP)


@traced
def finish_consensus2(c, health):
    """``finish_consensus`` without remote pulls: CTA 0's wave 0 polls the 256 local finish flags (one level),
    then thread 0 pushes this rank's sticky FAILED word as its done pair to every rank and polls its own four;
    any rank's FAILED becomes this rank's (what ``import_alarms`` did from the output rank's header)."""
    tid, bid, lane, wave = c["tid"], c["bid"], c["lane"], c["wave"]
    if const_expr(_FINISH_LITE):
        # nothing reads plain data on the strength of a finish flag (the host reads after the kernel): the
        # drain keeps a give-up's FAILED store ahead of the flag, no write-back of the XCD's L2
        publish_wt(c["put"], c["finish"], bid, 1, tid == 0)
    else:
        publish(c["put"], c["finish"], bid, 1, tid == 0)
    if bid == 0:
        if wave == 0:
            c["poll_terminal"]([(c["finish"], lane + K.LANES * m, 1) for m in range(BLOCKS // K.LANES)],
                               BLOCKS // K.LANES)
        gpu.barrier()
        if tid == 0:
            f = ld_health(health, FAILED)
            for p in range_constexpr(TP):
                c["put"](c["peer"]["done4"][p], c["rank"], f)
            got = c["poll_terminal"]([(c["peer"]["done4_own"], r, 1) for r in range(TP)], TP)
            anyf = got[0][0] != 0
            for r in range_constexpr(1, TP):
                anyf = anyf | (got[r][0] != 0)
            if anyf:
                bo.buffer_store(fx.Int32(1), rsrc(health), FAILED, cache_modifier=CM_SYS)
                wait_stores()


_TL = os.environ.get("K2_DEBUG_TL") == "1"  # per-CTA stage clock stamps into ``tl`` (TL_WORDS int32 a CTA)
TL_EXTRA = {14: "down1_polled", 15: "down1_chain", 24: "down_acquired", 25: "xq_ready", 26: "down1_yr_drained",
            22: "seam_xsq_seen", 23: "owner_ar_seen", 27: "pool_x_seen", 28: "xq_pool_seen", 29: "up_fin_seen",
            30: "gather_br_seen", 31: "route_rlog_seen", 32: "pool_x_lds", 33: "pool_wmma", 34: "pool_ppart_seen",
            35: "pool_done", 36: "fin_yr_seen", 37: "fin_fmsq_seen", 38: "fin_done", 39: "up_lds", 40: "up_done",
            41: "router_x_seen", 42: "router_done", 43: "xq_done", 44: "seam_done", 45: "route_done", 46: "xq_all_pool", 47: "xq_stored",
            48: "route_table", 49: "ug_w_issued", 50: "route_seen"}
TL_STAGES = ['oproj', 'router', 'seam', 'route', 'pool', 'xq', 'route_wait', 'ug', 'down', 'finalize', 'up', 'gather', 'finish'] + ["end"]
TL_WORDS = 52


def _tlx(tl, tid, bid):
    """The ``sync.stamp`` hook (None: no timeline build)."""
    if not _TL:
        return None
    return lambda k: _tl_stamp(tl, tid, bid, k)


def _first_ctx(c, i):
    """The stage context with the first-down-task timeline flag (plain Python: outside the kernel rewriter)."""
    if _TL and i == 0:
        return dict(c, tl_first=True)
    return c


@traced
def _tl_stamp(tl, tid, bid, k):
    if (tid == 0) & (tl != 0):
        bo.buffer_store(fx.Int32(memrealtime() & 0x7FFFFFFF), rsrc(tl), bid * TL_WORDS + k, cache_modifier=CM_DEV)


@traced
def acquire_cta():
    """Drop this CU's and this XCD's clean lines (one wave's device-scope invalidate), then the CTA meets: plain
    loads after it see every store published before it."""
    from flydsl._mlir.dialects import llvm as _llvm

    if fx.thread_idx.x < K.LANES:
        _llvm.inline_asm(T.i32, [], "global_inv scope:SCOPE_DEV\n s_wait_loadcnt 0x0\n s_mov_b32 $0, 0", "=s",
                         has_side_effects=True)
    gpu.barrier()


def _ug_first_weights(c, bid, n_ug):
    """Pipelined UG: round 0's weights, issued before the XQ wait (they need only the route table)."""
    if _UG_PIPE:
        return X.ug_weights(c, bid // UG_GROUPS, bid % UG_GROUPS, bid < n_ug)
    return []


@traced
def _wait_xq_if(c, cond, tl, tid, bid):
    """The XQ wait, then the CTA's acquire (its plain XQ / INTER reads see this launch's data)."""
    if cond:
        X.wait_xq(c)
        if const_expr(_TL):
            _tl_stamp(tl, tid, bid, 25)
        acquire_cta()


def _vr_first(c, slot, g, live, late, tl, tid, bid):
    """vres2: the first UG task's weights (they need only the route table) and the XQ wait + acquire. A CTA
    that reaches the wait late (the xq CTAs, which just published XQ) waits first: a poll behind its weight
    loads returns only after them, and the acquire's load drain would wait for them too."""
    _wait_xq_if(c, late, tl, tid, bid)
    w = X.vr_ug_weights(c, slot, g, live)
    stamp(c, 49)
    _wait_xq_if(c, ~late, tl, tid, bid)
    return w


def _vr_ub(bid):
    """vres2: the CTA's UG index (UG task u runs on index u % 256; the lowest indices get one task more).
    Reversed, the extra tasks avoid the CTAs that start the expert phase late (seam / route / xq CTAs)."""
    if _VR_REV:
        return BLOCKS - 1 - bid
    return bid


@traced
def _vr_ug_loop(c, bid, ub, n_ug, w, xs, tl, tid):
    """vres2 UG: this CTA's tasks u = ub + 256 i, each issuing the next one's weights. Returns the UGRDY this
    wave still has to publish (``X.vr_flush``)."""
    pend = fx.Int32(-1)
    for i in range(fx.max((n_ug - ub + BLOCKS - 1) // BLOCKS, 0)):
        u = ub + BLOCKS * i
        un = u + BLOCKS
        w, pend = X.vr_ug_task(c, u // UG_GROUPS, u % UG_GROUPS, i, w, xs, un // UG_GROUPS, un % UG_GROUPS,
                               un < n_ug, pend)
        if const_expr(_TL):
            _tl_stamp(tl, tid, bid, 16 + i)
    return pend


def _expert_vres2(c, bid, nu, tl, tid):
    """The routed experts on the vres2 pattern (``_VRES2``): UG tasks u = ub + 256 i, then this CTA's down run."""
    n_ug = nu * UG_GROUPS
    ub = _vr_ub(bid)
    X.vr_init(c)
    late = ((bid >= XQ0) & (bid < XQ0 + c["S"])) if _VR_LATE else (bid < 0)
    w = _vr_first(c, ub // UG_GROUPS, ub % UG_GROUPS, ub < n_ug, late, tl, tid, bid)
    enter_stage("ug")
    if _TL:
        _tl_stamp(tl, tid, bid, 7)
    xs = X.vr_ug_acts(c)
    pend = _vr_ug_loop(c, bid, ub, n_ug, w, xs, tl, tid)
    enter_stage("down")
    if _TL:
        _tl_stamp(tl, tid, bid, 8)
    d0, d1 = X.vr_down_range(nu, ub, _VR_UGW)
    ring = X.vr_down_prime(c, d0, d1)
    X.vr_flush(c, pend)
    if _VR_DNACQ:
        acquire_cta()
    if _TL:
        _tl_stamp(tl, tid, bid, 24)
    X.vr_down(c, d0, d1, ring)
    gpu.barrier()
    if not _VR_YR_TILE:
        X.vr_yr_out(c, d0, d1)
    if _TL:
        _tl_stamp(tl, tid, bid, 28)


@traced
def _tail_split(c, bid, tid, tl, health):
    """The k2v2pre tail: finalize (CTAs 192 ..; unpooler weights in flight first), unpooler task bid (CTAs
    0 .. 127) -> BR all-gather, gather (128 .. 191)."""
    up_live = (bid >= U0) & (bid < U0 + MD.UP_TASKS)
    up_ops = MD.up_loads_masked(c, bid - U0, up_live)
    if bid >= FIN0:
        _fin_task(c, bid, tid, tl)
    if up_live:
        enter_stage("up")
        if const_expr(_TL):
            _tl_stamp(tl, tid, bid, 10)
        if const_expr(os.environ.get("K2_DEBUG_EPOCH") == "1"):
            if tid == 0:
                bo.buffer_store(fx.Int32(memrealtime() & 0x7FFFFFFF), rsrc(health), (bid + 1) * 8 + 6, cache_modifier=CM_SYS)
        MD.stage_up(c, bid - U0, up_ops)
        if const_expr(os.environ.get("K2_DEBUG_EPOCH") == "1"):
            if tid == 0:
                bo.buffer_store(fx.Int32(memrealtime() & 0x7FFFFFFF), rsrc(health), (bid + 1) * 8 + 4, cache_modifier=CM_SYS)
    if (bid >= G0) & (bid < G0 + MD.GATHER_TASKS):
        enter_stage("gather")
        if const_expr(_TL):
            _tl_stamp(tl, tid, bid, 11)
        MD.stage_gather(c, bid - G0)


@traced
def _import_alarms_async(c, health, alarm):
    """K2_FINISH_ASYNC without K2_UP_ALL: a finalize CTA (done well before the gather) imports the output rank's
    alarm words into this rank's health; nothing waits for the other ranks."""
    if (c["bid"] == FIN0) & (c["tid"] == 0):
        import_alarms(health, alarm, c["rank"], TP)


def _fin_task(c, bid, tid, tl):
    enter_stage("finalize")
    if const_expr(_TL):
        _tl_stamp(tl, tid, bid, 9)
    f = bid - FIN0
    X.stage_fin(c, f // X.FIN_SLICES, f % X.FIN_SLICES, 2)


@traced
def _up_all(c, bid, tid, tl, gs, ops, health, alarm):
    enter_stage("up")
    if const_expr(_TL):
        _tl_stamp(tl, tid, bid, 10)
    if const_expr(_FINISH_ASYNC):
        # wave 1 (the FIN gate polls on wave 0): the import's round trip hides under the FIN all-gather hop
        if (bid == 0) & (tid == K.LANES):
            import_alarms(health, alarm, c["rank"], TP)
    MD.stage_up_all(c, gs, ops)


@traced
def _tail_up_all(c, bid, tid, tl, health, alarm):
    """K2_UP_ALL tail: finalize (CTAs 192 ..), then every CTA unpools branch columns 32 bid .. + 32 (tasks
    2 bid, 2 bid + 1) of every token; its weights issued before the finalize (``K2_UP_FINW=pre``) or, on the
    finalize CTAs, after it (``post``: the YR polls do not return behind 256 KB of weights)."""
    gs = [2 * bid + i for i in range(MD.UP_ALL_TASKS // BLOCKS)]
    if const_expr(_UP_FINW == "post"):
        if bid >= FIN0:
            _fin_task(c, bid, tid, tl)
            _up_all(c, bid, tid, tl, gs, [MD.up_loads_all(c, g) for g in gs], health, alarm)
        else:
            _up_all(c, bid, tid, tl, gs, [MD.up_loads_all(c, g) for g in gs], health, alarm)
    else:
        ops = [MD.up_loads_all(c, g) for g in gs]
        if bid >= FIN0:
            _fin_task(c, bid, tid, tl)
        _up_all(c, bid, tid, tl, gs, ops, health, alarm)


def _pool_ops(c, pi, pq, ops, early):
    """The pooler task's weight loads: issued right after o_proj (``K2_POOL_EARLY``) or at the pool stage."""
    if ops is not None:
        return ops
    if early != _POOL_EARLY:
        return None
    return MD.pool_loads(c, pi, pq)


def _load_route(c):
    if _RT_FAST:
        return X.load_route_fast(c)
    return X.load_route(c)


def _router_pool(c, ri, rq, tg, pi, pq):
    """K2_RPOOL: a router CTA's router task, then its pooler task on the router's x8 staging; both tasks'
    weights issued first (router's first: the router consumes them first)."""
    r_ops = MD.router_loads(c, ri, rq)
    p_ops = MD.pool_loads(c, pi, pq)
    MD.router_compute_pair(c, ri, rq, tg, r_ops, x8=True)
    enter_stage("pool")
    stamp(c, 4)
    MD.pool_compute(c, pi, pq, p_ops, True)


def _router_pool_local(c, bid, pi, pq):
    """K2_RLOCAL: every CTA's router task (experts 16 gi .. of all 256, K quarter pq, token pair tg) and its
    pooler task, both tasks' weights in flight before the x wait, the 8 x rows of quarter pq staged once; the
    pool's PPART round trip overlaps the router dots."""
    gi = (bid // MD.R_PARTS) % (MD.E // WAVES)
    tg = bid // (MD.R_PARTS * (MD.E // WAVES))
    enter_stage("pool")
    stamp(c, 4)
    r_ops = MD.router_loads(c, gi, pq, local=True)
    p_ops = MD.pool_loads(c, pi, pq)
    MD.early_x(c, pq)
    v = MD.pool_partial(c, pi, pq, p_ops)
    enter_stage("router")
    stamp(c, 1)
    MD.router_local_pair(c, gi, pq, tg, r_ops)
    if const_expr(not _POOL_DIRECT):
        enter_stage("pool")
        MD.pool_finish(c, pi, pq, v)


def _down_cta(bid):
    """The CTA order of the down wave-tasks: reversed, the last (partial) round lands on the CTAs with one UG
    round fewer (UG task u = bid + 256 i gives the low CTAs the extra round)."""
    if _DOWN_REV:
        return BLOCKS - 1 - bid
    return bid


def _lane_f_key():
    return f"rx{int(_ROUTER_X8)}pe{int(_POOL_EARLY)}lb{int(_S_LDS_BARRIER())}rc{int(_ROUTE_CTAS)}f2{int(_FINISH2)}pf{int(_POOL_FIRST)}fp{int(_FIN_PAIRS)}pm{int(_POOL_MOVE)}ra{int(MD.ROUTE_ARGMAX)}xw{int(X._XQ_WT)}xd{X._DEBUG_XQ_DELAY}sd{int(_SEAM_DIRECT)}fl{int(_FINISH_LITE)}"


def _v2_key():
    return f"rp{int(_RPOOL)}rl{int(_RLOCAL)}rt{int(_RT_FAST)}sp{int(quarter_seam._SEAM_POLL1)}pd{int(_POOL_DIRECT)}"


def _ho_key():
    return f"ua{int(_UP_ALL)}{_UP_FINW}fa{int(_FINISH_ASYNC)}"


def _S_LDS_BARRIER():
    from port.sync import sync as _S

    return _S._LDS_BARRIER


# ---- build
_BUILDS: dict = {}


def _make(key: K2Build):
    s = key.tokens
    lay = scratch_layout()
    play = peer_layout()
    half = _half()
    xrow_max = max(oproj.CORE_ROW, MD.EARLY_K + 8, K.MH + 8)
    nk = (X.MAX_U + 1) * 8
    _vr_words = X.vr_lds_words() if _VRES2 else (1, 1, 1)
    assert X.VR_CTAS == BLOCKS and THREADS // K.LANES == 16 and s == X.MAX_TOKENS

    @fx.struct
    class RouteLds:
        ids: fx.Array[fx.Int32, s * K.TOPK, 16]
        fl: fx.Array[fx.Int32, s * K.TOPK, 16]
        expert: fx.Array[fx.Int32, X.MAX_U + 1, 16]
        kof: fx.Array[fx.Int32, nk, 16]
        own: fx.Array[fx.Int32, X.E if _RT_FAST else 1, 16]
        bm: fx.Array[fx.Int32, X.E // 32 if _RT_FAST else 1, 16]

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 256, 16]
        av: fx.Array[fx.Float32, THREADS, 16]
        xl: fx.Array[fx.Int32, s * xrow_max // 2, 16]
        kred: fx.Array[fx.Float32, 2 * key.ug_ksplit * 2 * 8 * 32, 16]
        amx: fx.Array[fx.Float32, 32, 16]
        vr_uy: fx.Array[fx.Float32, _vr_words[0], 16]
        vr_dy: fx.Array[fx.Float32, _vr_words[1], 16]
        vr_cnt: fx.Array[fx.Int32, _vr_words[2], 16]

    from port.sync import sync as _S

    keyed = key_tuple(key, f"{_SOURCES}|inv{int(_S._INV_POLL)}e{_S._INV_EVERY}f|{_S._INV_SCOPE}|th{_S._POLL_TH}|dbg{os.environ.get('K2_DEBUG_EPOCH', '0')}|tl{int(_TL)}|pf{int(_S._PUB_FENCE)}|pg{int(_S._POLL_GLOBAL)}a{int(_S._POLL_ATOMIC)}|pw{int(_S._PUT_WB)}w2|sl{_S._POLL_SLEEP}b{int(_S._POLL_BACKOFF)}|al2{int(_ACT_L2)}wt{int(_UG_WT)}up{int(_UG_PIPE)}wf{int(_DOWN_WFIRST)}dr{int(_DOWN_REV)}|mon{int(_S._POLL_MONITOR)}|vr{int(_VRES2)}y{int(_VR_YR_TILE)}q{int(_VR_DNACQ)}r{int(_VR_REV)}l{int(_VR_LATE)}z{int(_VR_LAZY)}w{_VR_UGW}e{int(_VR_EPI_ROT)}|F{_lane_f_key()}|V{_v2_key()}|H{_ho_key()}|wt{_S.WAIT_TICKS}")
    name = f"muse_mono_k2g_s{s}_{key.act}"
    if key.fault_rank >= 0:
        name += f"_f{key.fault_rank}_{key.fault_epoch}_{key.fault_layer}_{key.fault_ticks}"
    rids = region_ids(REGIONS)
    act_cm, act_isc = (0, X.ISC_PS_L2) if _ACT_L2 else (CM_DEV, K.RI // 32)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64, stream_a: Int64, w_o: Int64, alpha_a: Int64, beta_a: Int64, w_fn: Int64, w_router: Int64,
        expert_bias: Int64, w_pool: Int64, w_pre: Int64, w13: Int64, w13s: Int64, w2: Int64, w2s: Int64,
        w_post: Int64, w_unpool: Int64, stream_out: Int64, branch_out: Int64, scratch: Int64, peers: Int64,
        rank: Int32, epoch: Int64, layer: Int32, health: Int64, alarm: Int64, tl: Int64,
    ):  # fmt: skip
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        # The epoch word is written by the previous launch (epoch.add_); per-XCD L2s are not coherent, so a
        # plain load can return a stale epoch and pass every wait on that epoch's leftover tags.
        ep = uniform(fx.Int32(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_SYS)))
        if const_expr(os.environ.get("K2_DEBUG_EPOCH") == "1"):
            if tid == 0:
                bo.buffer_store(ep, rsrc(health), (bid + 1) * 8 + 7, cache_modifier=CM_SYS)
                bo.buffer_store(fx.Int32(memrealtime() & 0x7FFFFFFF), rsrc(health), (bid + 1) * 8 + 5, cache_modifier=CM_SYS)
        mb = Mailbox(step_tag(ep, K2_SLOT0 + layer) - 1, health, alarm, rids)
        par = ep % 2
        bases = peer_bases(peers, TP)
        own = load_ptr64(peers, rank)
        output_head = alarm - fx.Int64(rank) * 4
        done = preg(output_head, K2_DONE_OFFSET, "done", K2_DONE_BYTES)
        poff = fx.Int64(par) * half
        peer = {}
        for region in _PEER_PAIRS:
            off, extent = play[region]
            peer[region] = [shift(preg(b, off, region, extent), poff) for b in bases]
            peer[region + "_own"] = shift(preg(own, off, region, extent), poff)
        args = dict(
            core=core, stream_a=stream_a, w_o=w_o, alpha_a=alpha_a, beta_a=beta_a, w_fn=w_fn, w_router=w_router,
            expert_bias=expert_bias, w_pool=w_pool, w_pre=w_pre, w13=w13, w13s=w13s, w2=w2, w2s=w2s,
            w_post=w_post, w_unpool=w_unpool, stream_out=stream_out, branch_out=branch_out,
        )  # fmt: skip
        fin_off = play["fin"][0]
        c = {
            "S": s, "T": s, "tid": tid, "bid": bid, "lane": tid % K.LANES, "wave": tid // K.LANES, "rank": rank,
            "threads": THREADS, "act": key.act, "nt": 1, "ksplit": key.ug_ksplit, "prefetch": key.ug_prefetch,
            "dprefetch": key.down_prefetch,
            "quarter_seam": True, "rlog_parts": True, "rlog_local": _RLOCAL, "pool_direct": _POOL_DIRECT, "experimental_two_cta": True,
            "eps_attn": key.eps_post_attn, "eps_ffn": key.eps_ffn, "eps_pre": key.eps_pre_expert,
            "eps_post": key.eps_post_expert, "router_output_multiplier": key.router_output_multiplier,
            "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words, "poll": mb.poll,
            "poll_terminal": mb.poll_terminal,
            "red": lds.red.ptr, "av": lds.av.ptr, "xl": lds.xl.ptr, "kred": lds.kred.ptr, "amx": lds.amx.ptr, "vr_uy": lds.vr_uy.ptr, "vr_dy": lds.vr_dy.ptr, "vr_cnt": lds.vr_cnt.ptr,
            "rt": {"ids": rl.ids.ptr, "fl": rl.fl.ptr, "expert": rl.expert.ptr, "kof": rl.kof.ptr, "own": rl.own.ptr,
                   "bm": rl.bm.ptr},
            "peer": peer, "ar_peer": peer["ar"], "ar_own": peer["ar_own"],
            "yr": peer["yr_own"], "yr_owner": peer["yr"], "frdy_peer": peer["frdy"], "frdy": peer["frdy_own"],
            "fin": own + fx.Int64(fin_off) + poff, "fin_pairs": _FIN_PAIRS, "finp": peer["finp_own"],
            "finp_peer": peer["finp"],
            "fin_peer": [b + fx.Int64(fin_off) + poff for b in bases],
            "xbuf": scratch + fx.Int64(lay["xrow"][0]),
            "args": args, "stamp": lambda k: None, "tl_hook": lambda k: _tl_stamp(tl, tid, bid, k), "tlx": _tlx(tl, tid, bid), "xcm": act_cm, "isc_ps": act_isc, "trash_pick": _ACT_L2, "ug_wt": _UG_WT, "down_wfirst": _DOWN_WFIRST, "frdy_flags": s * X.FIN_SLICES, "tl_on": _TL, "vr_yr_tile": _VR_YR_TILE, "tag": mb.tag, "vr_lazy": _VR_LAZY, "vr_epi_rot": _VR_EPI_ROT,
        }  # fmt: skip
        for region in _SCRATCH_PAIRS:
            c[region] = sreg(scratch, lay[region][0], region, lay[region][1])
        for region in _SCRATCH_PLAIN:
            c[region] = scratch + fx.Int64(lay[region][0])
        off_ar, ext_ar = play["ar"]
        c["ar_owner"] = [
            shift(preg(load_ptr64(peers, oproj.owner_of_tile(bid + BLOCKS * j)), off_ar, "ar", ext_ar), poff)
            for j in range(oproj.O_ROUNDS)
        ]

        # ---- o_proj (every CTA) -> AR#1 to the column owners
        enter_stage("oproj")
        if const_expr(_TL):
            _tl_stamp(tl, tid, bid, 0)
        if const_expr(_SEAM_DIRECT):
            oproj.stage_oproj(c, bid)
        else:
            oproj.stage_oproj_deep(c, bid)
        pi = bid // MD.P_PARTS
        pq = bid % MD.P_PARTS
        pool_ops = _pool_ops(c, pi, pq, None, True)
        # ---- router CTAs: weights in flight, then x
        is_r = (bid >= R0) & (bid < R0 + MD.R_TASKS_Q8)
        r_task = fx.max(fx.min(bid - R0, MD.R_TASKS_Q8 - 1), 0)
        ri = (r_task % MD.R_TASKS) // MD.R_PARTS
        rq = r_task % MD.R_PARTS
        if const_expr(_POOL_FIRST):
            r_ops = MD.router_loads(c, ri, rq, mask=is_r)
        elif const_expr(_RLOCAL):
            pass  # every CTA's router task runs with its pooler task (_router_pool_local)
        elif const_expr(_RPOOL):
            # router CTA: its pooler task's weights in flight with the router's before the x wait; the router's
            # staging of all 8 rows of K quarter rq (= pq) is the pooler task's x
            if is_r:
                enter_stage("router")
                if const_expr(_TL):
                    _tl_stamp(tl, tid, bid, 1)
                _router_pool(c, r_task // MD.R_PARTS if _RLOCAL else ri, rq, r_task // MD.R_TASKS, pi, pq)
        else:
            if is_r:
                enter_stage("router")
                if const_expr(_TL):
                    _tl_stamp(tl, tid, bid, 1)
                MD.router_compute_pair(c, ri, rq, r_task // MD.R_TASKS, MD.router_loads(c, ri, rq), x8=_ROUTER_X8)
        # ---- seam B (CTAs 0 .. 31), route (0 .. 7)
        if bid < _SEAM_CTAS:
            enter_stage("seam")
            if const_expr(_TL):
                _tl_stamp(tl, tid, bid, 2)
            if const_expr(not _SEAM_DIRECT):
                quarter_seam.stage_ar_owner_sum(c, bid)
            if bid < s:
                quarter_seam.stage_seam_token(c, bid, _SEAM_DIRECT)
                if const_expr(not _ROUTE_CTAS):
                    enter_stage("route")
                    if const_expr(_TL):
                        _tl_stamp(tl, tid, bid, 3)
                    MD.stage_route(c, bid)
        # ---- pooler task bid (every CTA)
        if const_expr(_RLOCAL):
            _router_pool_local(c, bid, pi, pq)
        elif const_expr(_RPOOL):
            if ~is_r:
                enter_stage("pool")
                if const_expr(_TL):
                    _tl_stamp(tl, tid, bid, 4)
                MD.pool_compute(c, pi, pq, MD.pool_loads(c, pi, pq))
        else:
            enter_stage("pool")
            if const_expr(_TL):
                _tl_stamp(tl, tid, bid, 4)
            pool_ops = _pool_ops(c, pi, pq, pool_ops, False)
        if const_expr(_RPOOL or _RLOCAL):
            pass
        elif const_expr(_POOL_MOVE):
            # CTA FIN0 + m also runs router CTA R0 + m's pooler task (same K quarter: its x part is staged)
            # the 2nd task's weights are issued inside the branch that uses them, before its x wait (issued
            # outside under a mask, the compiler sinks them to their use: no prefetch)
            live2 = bid >= FIN0
            if live2:
                ops2 = MD.pool_loads(c, (bid - (FIN0 - R0)) // MD.P_PARTS, pq)
                MD.pool_compute(c, pi, pq, pool_ops)
                MD.pool_compute(c, (bid - (FIN0 - R0)) // MD.P_PARTS, pq, ops2, True)
            if (bid < R0) | ((bid >= R0 + MD.R_TASKS_Q8) & (bid < FIN0)):
                MD.pool_compute(c, pi, pq, pool_ops)
        elif const_expr(_ROUTER_X8 and not _POOL_FIRST):
            MD.pool_compute(c, pi, pq, pool_ops, is_r.select(fx.Int32(1), fx.Int32(0)))
        else:
            MD.pool_compute(c, pi, pq, pool_ops)
        if const_expr(_POOL_FIRST):
            if is_r:
                enter_stage("router")
                if const_expr(_TL):
                    _tl_stamp(tl, tid, bid, 1)
                MD.router_compute_pair(c, ri, rq, r_task // MD.R_TASKS, r_ops, x8=2)
        if const_expr(_ROUTE_CTAS):
            # ---- route token t on CTA RT0 + t (pooler only: the RLOG wait delays no seam / pool task)
            if (bid >= RT0) & (bid < RT0 + s):
                enter_stage("route")
                if const_expr(_TL):
                    _tl_stamp(tl, tid, bid, 3)
                MD.stage_route(c, bid - RT0)
        # ---- xq (CTAs 8 .. 15)
        if (bid >= XQ0) & (bid < XQ0 + s):
            enter_stage("xq")
            if const_expr(_TL):
                _tl_stamp(tl, tid, bid, 5)
            X.stage_xq(c, bid - XQ0)
        # ---- routed experts: UG (CTA tasks), down (wave tasks)
        enter_stage("route_wait")
        if const_expr(_TL):
            _tl_stamp(tl, tid, bid, 6)
        nu = _load_route(c)
        stamp(c, 48)
        n_ug = nu * UG_GROUPS
        if const_expr(_VRES2):
            _expert_vres2(c, bid, nu, tl, tid)
        else:
            wcur = _ug_first_weights(c, bid, n_ug)
            X.wait_xq(c)
            if const_expr(_TL):
                _tl_stamp(tl, tid, bid, 25)
            if const_expr(_ACT_L2):
                acquire_cta()
            enter_stage("ug")
            if const_expr(_TL):
                _tl_stamp(tl, tid, bid, 7)
            if const_expr(_UG_PIPE):
                # task u's epilogue overlaps task u + BLOCKS's weight stream (one CTA a CU: nothing else would)
                for i in range(fx.max((n_ug - bid + BLOCKS - 1) // BLOCKS, 0)):
                    u = bid + BLOCKS * i
                    un = u + BLOCKS
                    wcur = X.stage_ug_pipe(c, u // UG_GROUPS, u % UG_GROUPS, wcur, un // UG_GROUPS, un < n_ug)
                    if const_expr(_TL):
                        _tl_stamp(tl, tid, bid, 16 + i)
            else:
                for i in range((X.MAX_U * UG_GROUPS + BLOCKS - 1) // BLOCKS):
                    u = bid + BLOCKS * i
                    if u < n_ug:
                        X.stage_ug(c, u // UG_GROUPS, u % UG_GROUPS)
                    if const_expr(_TL):
                        _tl_stamp(tl, tid, bid, 16 + i)
            enter_stage("down")
            if const_expr(_TL):
                _tl_stamp(tl, tid, bid, 8)
            if const_expr(_ACT_L2):
                acquire_cta()
            if const_expr(_TL):
                _tl_stamp(tl, tid, bid, 24)
            n_dn = nu * DOWN_TILES
            v0 = _down_cta(bid) * WAVES + c["wave"]
            for i in range((X.MAX_U * DOWN_TILES + BLOCKS * WAVES - 1) // (BLOCKS * WAVES)):
                v = v0 + BLOCKS * WAVES * i
                cd = _first_ctx(c, i)
                if v < n_dn:
                    X.stage_down(cd, v // DOWN_TILES, v % DOWN_TILES)
            gpu.barrier()
        # ---- finalize on the row owners (CTAs 192 .. 255), unpooler, gather
        if const_expr(_UP_ALL):
            _tail_up_all(c, bid, tid, tl, health, alarm)
        else:
            _tail_split(c, bid, tid, tl, health)
        enter_stage("finish")
        if const_expr(_TL):
            _tl_stamp(tl, tid, bid, 12)
        if const_expr(_FINISH_ASYNC):
            if const_expr(not _UP_ALL):
                _import_alarms_async(c, health, alarm)
        elif const_expr(_FINISH2):
            finish_consensus2(c, health)
        else:
            finish_consensus(c, done, health, alarm)
        if const_expr(_TL):
            _tl_stamp(tl, tid, bid, 13)

    @flyc.jit
    def launch(
        core: Int64, stream_a: Int64, w_o: Int64, alpha_a: Int64, beta_a: Int64, w_fn: Int64, w_router: Int64,
        expert_bias: Int64, w_pool: Int64, w_pre: Int64, w13: Int64, w13s: Int64, w2: Int64, w2s: Int64,
        w_post: Int64, w_unpool: Int64, stream_out: Int64, branch_out: Int64, scratch: Int64, peers: Int64,
        rank: Int32, epoch: Int64, layer: Int32, health: Int64, alarm: Int64, tl: Int64,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        k2(
            core, stream_a, w_o, alpha_a, beta_a, w_fn, w_router, expert_bias, w_pool, w_pre, w13, w13s, w2, w2s,
            w_post, w_unpool, stream_out, branch_out, scratch, peers, rank, epoch, layer, health, alarm, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(k2, launch)
    return launch


def build(key: K2Build):
    if key not in _BUILDS:
        _BUILDS[key] = _make(key)
    return _BUILDS[key]


def k2_launch(key: K2Build, t: dict, rank: int, layer: int, epoch, peers, scratch, health, alarm, stream=None, tl=0):
    """Launch one layer's K2 on this rank. ``t``: the ABI tensors by name (``core`` .. ``branch_out``)."""
    launch = build(key)
    ptr = lambda x: x if isinstance(x, int) else x.data_ptr()  # noqa: E731
    vals = [ptr(t[n]) for n in ABI.names[:18]]
    st = fx.Stream(stream if stream is not None else 0)
    launch(*vals, ptr(scratch), ptr(peers), rank, ptr(epoch), layer, ptr(health), ptr(alarm), ptr(tl), stream=st)


if os.environ.get("K2G_PRINT_LAYOUT"):
    print("peer", peer_layout(), "half", _half())
    print("scratch", scratch_layout(), "bytes", scratch_bytes())
