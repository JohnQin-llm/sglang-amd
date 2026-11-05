"""Single-GPU harness for the gfx1250 K2 small-GEMM stages: one generic kernel a stage (the stage function called
with the same context dict ``c`` ``ops/k2.py`` builds), and a rig that simulates the TP4 ranks on one GPU
(four peer buffers behind a peer table, a scratch buffer a rank), launching rank by rank.

(No ``from __future__ import annotations`` here: ``fx.struct`` needs real annotations.)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.typing import Int32, Int64

from . import common as C
from . import gemv, moe_dense, oproj
from .common import load_ptr64
from .mailbox import Addr, Mailbox

S = 8
ARGS = (
    "core", "stream_a", "w_o", "alpha_a", "beta_a", "w_fn", "w_router", "expert_bias", "w_pool", "w_unpool",
    "stream_out", "branch_out", "scratch", "peers", "err", "branch_in", "stream_in", "w_pf", "alpha_f", "beta_f",
    "w_qn",
)  # fmt: skip
XROW_MAX = max(oproj.CORE_ROW, moe_dense.EARLY_K + gemv.LDS_PAD, moe_dense.MH + gemv.LDS_PAD)


def _pair_layout(items, align=256):
    lay, off = {}, 0
    for name, pairs in items:
        lay[name] = (off, pairs * 8)
        off = (off + pairs * 8 + align - 1) // align * align
    return lay, off


def peer_layout(s=S):
    pairs = {
        "ar": oproj.ar_pairs(s),
        **moe_dense.peer_pairs(s),
        "xg": s * C.H // 2,
        "xsq": s * 64,
    }
    return _pair_layout(pairs.items())


def scratch_layout(s=S):
    pairs = {
        "msq1": 64 * s, "msq2": 64 * s, "xrdy": max(64, 4 * s), "p1": 64 * s, "p2": 64 * s,
        **moe_dense.scratch_pairs(s),
    }  # fmt: skip
    lay, off = _pair_layout(pairs.items())
    lay["xrow"] = (off, s * C.H * 2)
    off += s * C.H * 2
    lay["fin"] = (off, s * moe_dense.MH * 2)
    off += s * moe_dense.MH * 2
    return lay, off


SCRATCH_REGIONS = ("msq1", "msq2", "xrdy", "p1", "p2", "rpart", "ppart", "route", "frdy")


def variant_key():
    """Build-time module settings a binary depends on (they must reach FlyDSL's JIT key)."""
    return (C.THREADS, C.BLOCKS, gemv.INTERLEAVE, gemv.SHUFFLED, C.CM_NT, C.CM_XROWS)


def build(name, body, grid, s=S, quarter=True, eps=None, mult=1.0):
    """The launcher of one stage kernel: ``body(c, bid)`` on ``grid`` CTAs of THREADS threads."""
    eps = eps or {}
    play, _ = peer_layout(s)
    slay, _ = scratch_layout(s)
    eps_t = tuple(sorted(eps.items()))
    key = (name, s, grid, quarter, eps_t, float(mult)) + variant_key()

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, C.RED_FLOATS, 16]
        av: fx.Array[fx.Float32, C.THREADS, 16]
        xl: fx.Array[fx.Int32, s * XROW_MAX // 2, 16]

    @flyc.kernel(name=f"k2g_{name}_s{s}", known_block_size=[C.THREADS, 1, 1])
    def kern(
        core: Int64, stream_a: Int64, w_o: Int64, alpha_a: Int64, beta_a: Int64, w_fn: Int64, w_router: Int64,
        expert_bias: Int64, w_pool: Int64, w_unpool: Int64, stream_out: Int64, branch_out: Int64, scratch: Int64,
        peers: Int64, err: Int64, branch_in: Int64, stream_in: Int64, w_pf: Int64, alpha_f: Int64, beta_f: Int64,
        w_qn: Int64, rank: Int32, tag: Int32,
    ):  # fmt: skip
        _ = key
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        mb = Mailbox(tag, err)
        bases = [load_ptr64(peers, p) for p in range(C.TP)]
        own = load_ptr64(peers, rank)
        peer = {}
        for region, (off, _n) in play.items():
            peer[region] = [Addr(b + fx.Int64(off), True) for b in bases]
            peer[region + "_own"] = Addr(own + fx.Int64(off), True)
        args = dict(
            core=core, stream_a=stream_a, w_o=w_o, alpha_a=alpha_a, beta_a=beta_a, w_fn=w_fn, w_router=w_router,
            expert_bias=expert_bias, w_pool=w_pool, w_unpool=w_unpool, stream_out=stream_out,
            branch_out=branch_out, branch_in=branch_in, stream_in=stream_in, w_pf=w_pf, alpha_f=alpha_f,
            beta_f=beta_f, w_qn=w_qn,
        )  # fmt: skip
        c = {
            "S": s, "T": s, "tid": tid, "bid": bid, "lane": tid % C.WAVE, "wave": tid // C.WAVE, "rank": rank,
            "quarter_seam": quarter, "rlog_parts": quarter, "experimental_two_cta": quarter,
            "eps_attn": eps.get("eps_attn", 1e-8), "eps_ffn": eps.get("eps_ffn", 1e-5),
            "post_eps": eps.get("post_eps", 1e-8), "eps": eps.get("eps", 1e-5),
            "router_output_multiplier": float(mult),
            "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words, "poll": mb.poll,
            "red": lds.red.ptr, "av": lds.av.ptr, "xl": lds.xl.ptr,
            "peer": peer, "ar_peer": peer["ar"], "ar_own": peer["ar_own"],
            "xbuf": scratch + fx.Int64(slay["xrow"][0]), "fin": scratch + fx.Int64(slay["fin"][0]),
            "args": args, "stamp": lambda k: None,
        }  # fmt: skip
        for region in SCRATCH_REGIONS:
            c[region] = Addr(scratch + fx.Int64(slay[region][0]), False)
        off_ar = play["ar"][0]
        c["ar_owner"] = [
            Addr(load_ptr64(peers, oproj.owner_of_tile(bid + C.BLOCKS * j)) + fx.Int64(off_ar), True)
            for j in range(oproj.O_ROUNDS)
        ]
        body(c, bid)

    @flyc.jit
    def launch(
        core: Int64, stream_a: Int64, w_o: Int64, alpha_a: Int64, beta_a: Int64, w_fn: Int64, w_router: Int64,
        expert_bias: Int64, w_pool: Int64, w_unpool: Int64, stream_out: Int64, branch_out: Int64, scratch: Int64,
        peers: Int64, err: Int64, branch_in: Int64, stream_in: Int64, w_pf: Int64, alpha_f: Int64, beta_f: Int64,
        w_qn: Int64, rank: Int32, tag: Int32, stream: fx.Stream = fx.Stream(None),
    ):  # fmt: skip
        _ = key
        kern(
            core, stream_a, w_o, alpha_a, beta_a, w_fn, w_router, expert_bias, w_pool, w_unpool, stream_out,
            branch_out, scratch, peers, err, branch_in, stream_in, w_pf, alpha_f, beta_f, w_qn, rank, tag,
        ).launch(grid=(grid,), block=(C.THREADS,), stream=stream)  # fmt: skip

    return launch


class Rig:
    """Four simulated ranks on one GPU: peer buffers (+ table), scratch buffers, an error word."""

    def __init__(self, s=S, dev="cuda"):
        self.s = s
        self.play, pb = peer_layout(s)
        self.slay, sb = scratch_layout(s)
        self.peer = [torch.zeros(pb // 4, dtype=torch.int32, device=dev) for _ in range(C.TP)]
        self.scratch = [torch.zeros(sb // 4, dtype=torch.int32, device=dev) for _ in range(C.TP)]
        self.table = torch.tensor([p.data_ptr() for p in self.peer], dtype=torch.int64, device=dev)
        self.err = torch.zeros(4, dtype=torch.int32, device=dev)
        self.ptrs = {}

    def zero(self):
        for t in self.peer + self.scratch:
            t.zero_()
        self.err.zero_()

    def set(self, **tensors):
        for k, v in tensors.items():
            assert k in ARGS, k
            self.ptrs[k] = v

    def argv(self, rank, tag, **over):
        vals = []
        for name in ARGS:
            if name == "scratch":
                t = self.scratch[rank]
            elif name == "peers":
                t = self.table
            elif name == "err":
                t = self.err
            else:
                t = over.get(name, self.ptrs.get(name))
            vals.append(0 if t is None else (t if isinstance(t, int) else t.data_ptr()))
        return vals + [rank, tag]

    def run(self, launch, rank, tag=1, **over):
        launch(*self.argv(rank, tag, **over), stream=fx.Stream(torch.cuda.current_stream().cuda_stream))

    # ---- region views (int32 words of a rank's buffer)
    def peer_words(self, rank, region):
        off, n = self.play[region]
        return self.peer[rank][off // 4: (off + n) // 4]

    def scratch_words(self, rank, region):
        off, n = self.slay[region]
        return self.scratch[rank][off // 4: (off + n) // 4]

    def pairs_value(self, words):
        """(value words, tag words) of a pair region."""
        w = words.view(-1, 2)
        return w[:, 0], w[:, 1]


def bf16_of_pairs(values):
    """A pair region's value words, each two packed bf16 -> bf16 tensor."""
    return values.contiguous().view(torch.bfloat16)


def f32_of_pairs(values):
    return values.contiguous().view(torch.float32)
