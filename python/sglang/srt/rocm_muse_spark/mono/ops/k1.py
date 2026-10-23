# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1: one Muse Spark sparse decoder layer from the previous layer's FFN
boundary to o_proj's input, one persistent launch a layer and rank (no
collective): seam A, the QKV GEMV, QK-norm + RoPE + KV insert, split-KV
attention over the paged cache, and the chunk combine with the gated o-norm.

``BLOCKS`` CTAs x ``THREADS`` threads, T = R x L rows (R requests of q_len L):

    CTAs 0 .. 143     QKV GEMV (16 rows x K 8192 a task): weights issued first,
                      then the x rows (XRDY) -> bf16 qkv -> QKV pairs
    CTAs 144 .. 207   seam A (``stages.seam``): stream_a out, the bf16 attn_in
                      columns -> x rows + XRDY
    CTAs 144 ..       attention units (``stages.attn.stage_unit``): (request,
                      kv head, chunk), NCH chunks; the last chunk inserts the
                      step's K / V rows -> partials (APART pairs)
    CTAs 0 .. 16 T-1  combine + rms_norm_gated (``stages.attn.stage_combine``)
                      -> ``core`` [T, 1024] bf16

No stage waits on a later one; every wait is a ``Mailbox`` poll (10 s give-up,
health record, alarm). Tags: ``step_tag(epoch, layer)`` (the epoch the host
bumps once a step; this launch's slot is the layer index).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu
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
    bf16_round,
    ld_i32,
    row_sum,
    rsrc,
    traced,
    uniform,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.plan import (
    BLOCKS,
    LDS_BYTES,
    THREADS,
    WAVES,
    KernelAbi,
    enter_stage,
    first_task,
    key_tuple,
    pair_layout,
    recording,
    source_digest,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.sync import Mailbox, sreg, stall
from sglang.srt.rocm_muse_spark.mono.ops.stages import attn, gemv, seam

HIDDEN = seam.HIDDEN  # 8192
CORE = attn.CORE  # 1024: o_proj's input width a rank
QKV_ROWS = attn.QKV_ROWS  # 2304
ROWS = gemv.ROWS  # 16
GEMV_TASKS = QKV_ROWS // ROWS  # 144
GEMV0 = 0
SEAM0 = GEMV_TASKS  # 144
ATT0 = GEMV_TASKS  # attention units from CTA 144 (the CTAs without weights)
COMB0 = 0
XROW = HIDDEN + gemv.LDS_PAD  # bf16 a padded x row in LDS
MAX_TOKENS = 8
MAX_LEN = 32768  # max_model_len the builds are sized for
WINDOW = 2048
# timeline points: 0 start, 1 seam done, 2 x rows in LDS, 3 qkv out, 4 unit
# prologue (block ids, first tiles), 5 q ready, 6 tiles done, 7 partials out,
# 8 combine done, 9 XRDY seen (the GEMV CTA's wave 0 weights landed)
TL_POINTS = 10

assert SEAM0 + seam.NS <= BLOCKS

REGIONS = ("p1", "p2", "xrdy", "qkv", "apart")
_STREAM = fx.Stream(None)
# this kernel's sources: every K1 build's JIT cache key
_SOURCES = source_digest(
    "common", "stages/gemv.py", "stages/seam.py", "stages/attn.py", "k1.py"
)


def default_nch(tokens: int, q_len: int) -> int:
    """Chunks a (request, kv head): up to a unit a CTA (a long context's KV
    stream wants every CU), the partials bounded (units x q_len <= 1024: 4.3
    MB). A short context fills only the first chunks (``attn.MIN_PER``)."""
    if tokens == 8 and q_len == 8:
        # Measured rows=8 verify optimum: one request * two KV heads *
        # 16 chunks = 32 attention work units.  Keep rows=8 decode unchanged.
        return 16
    r = tokens // q_len
    return max(1, min(BLOCKS // (attn.NKV * r), 512 // tokens))


@dataclass(frozen=True)
class K1Build:
    tokens: int  # T rows
    q_len: int  # L rows a request (T = R L)
    window: int  # 0: full attention; 2047/2048: visible causal keys
    rope: bool  # GPT-J RoPE (sliding layers); False: NoPE
    scale: float  # softmax scale: 64 ** -0.5 * qk_scale_factor
    eps: float = 1e-5  # pre_qkv_proj_norm
    post_eps: float = 1e-8  # the previous layer's post_ffn_norm
    qk_eps: float = 1e-5  # QK-norm and the gated o-norm
    nch: int = 0  # chunks a (request, kv head); 0: ``default_nch``
    max_len: int = MAX_LEN
    # the fault hook (sync.stall): rank fault_rank stalls fault_ticks once, in
    # slot fault_layer's launch of the step at epoch fault_epoch
    fault_rank: int = -1
    fault_epoch: int = 0
    fault_layer: int = 0
    fault_ticks: int = 0
    timeline: bool = False

    @property
    def requests(self) -> int:
        return self.tokens // self.q_len

    @property
    def chunks(self) -> int:
        return self.nch or default_nch(self.tokens, self.q_len)

    @property
    def units(self) -> int:
        return self.requests * attn.NKV * self.chunks


def make_key(
    *,
    tokens: int,
    q_len: int,
    window: int,
    rope: bool,
    scale: float,
    eps: float,
    post_eps: float,
    qk_eps: float,
    **fault,
) -> K1Build:
    return K1Build(
        tokens=tokens,
        q_len=q_len,
        window=window,
        rope=rope,
        scale=float(scale),
        eps=float(eps),
        post_eps=float(post_eps),
        qk_eps=float(qk_eps),
        **fault,
    )


def _check_key(key: K1Build) -> None:
    if not (1 <= key.tokens <= MAX_TOKENS and key.q_len >= 1):
        raise ValueError(f"K1: tokens {key.tokens} outside 1..{MAX_TOKENS}")
    if key.tokens % key.q_len:
        raise ValueError(f"K1: q_len {key.q_len} does not divide {key.tokens} rows")
    # Current SGLang passes an effective 2047-key window; the original
    # adapter uses 2048. The per-query mask handles either left boundary.
    if key.window not in (0, WINDOW - 1, WINDOW):
        raise ValueError(f"K1: window {key.window} (0, {WINDOW - 1}, or {WINDOW})")
    if not (1 <= key.max_len <= MAX_LEN):
        raise ValueError(f"K1: max_len {key.max_len} outside 1..{MAX_LEN}")
    if key.units > BLOCKS:
        raise ValueError(f"K1: {key.units} attention units > {BLOCKS} CTAs")


ABI = KernelAbi(
    (
        "stream_in",
        "branch_in",
        "w_pf",
        "alpha_f",
        "beta_f",
        "w_qn",
        "w_qkv",
        "positions",
        "cos_sin",
        "slot_mapping",
        "kv_cache",
        "kv_sb",
        "kv_sh",
        "kv_st",
        "block_table",
        "bt_stride",
        "seq_lens",
        "stream_a",
        "core",
        "scratch",
        "rank",
        "epoch",
        "layer",
        "health",
        "alarm",
        "tl",
    )
)


def scratch_layout(key: K1Build) -> dict:
    s = key.tokens
    lay = pair_layout(
        (
            ("p1", seam.NS * s),
            ("p2", seam.NS * s),
            ("xrdy", seam.NS),
            ("qkv", s * QKV_ROWS),
            ("apart", key.units * attn.GQA * key.q_len * attn.PART),
        ),
        align=16,
    )
    end = max(o + n for o, n in lay.values())
    # the plain bf16 x rows (not pairs), 16 B aligned
    lay["x"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    return lay


def scratch_bytes(key: K1Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def max_scratch_bytes(tokens: int) -> int:
    """The largest scratch a K1 build of <= ``tokens`` rows needs (any q_len
    dividing its rows, either layer kind, any context)."""
    best = 0
    for s in range(1, tokens + 1):
        for q in range(1, s + 1):
            if s % q == 0:
                k = K1Build(tokens=s, q_len=q, window=0, rope=False, scale=1.0)
                best = max(best, scratch_bytes(k))
    return best


def step_tag(epoch, slot):
    """This launch's mailbox tag: the step's epoch (a device counter the host
    bumps once a step, in the graph) and the launch's slot (K1: the layer)."""
    e = uniform(ld_i32(epoch, 0))
    return (e % EPOCH_TAG_PERIOD) * TAG_SLOTS + slot + 1


@traced
def stamp_flush(on, tls, tl, tid, bid, points):
    """A timeline build's LDS stamps -> ``tl`` [CTA][points] int64 (the
    common ``debug.stamp_flush`` needs ``fx.generic_store``, which this FlyDSL
    lacks: two dwords a stamp through a buffer store instead)."""
    if const_expr(on):  # noqa: SIM102
        if tid < points:
            v = fx.ptr_load(tls + tid)
            words = fx.Vector.from_elements(
                [fx.Int32(v & 0xFFFFFFFF), fx.Int32(v >> 32)], fx.Int32
            )
            bo.buffer_store(words, rsrc(tl), (bid * points + tid) * 2)


# ---------------------------------------------------------------- QKV GEMV

XCDS = 8  # MI355X: CTA b runs on XCD b % 8


def gemv_task(b):
    """The row group CTA ``b`` (of the GEMV's 144) takes: each XCD a contiguous
    eighth of the weight. CTA b -> rows b, b + 8, .. put every XCD's CTAs on
    addresses 2 MiB apart, and the stream ran at 3.4 TB/s instead of 4.9
    (measured: 37.7 MB, 144 CTAs, 11.0 vs 7.7 us)."""
    return (b % XCDS) * (GEMV_TASKS // XCDS) + b // XCDS


assert sorted(gemv_task(b) for b in range(GEMV_TASKS)) == list(range(GEMV_TASKS))


@traced
def stage_qkv(c, rg, ops, tensile_split=False):
    """Rows 16 rg .. of the QKV projection against every x row -> bf16 ->
    QKV pairs (t * 2304 + row)."""
    s, tid, lane, wave, red = c["T"], c["tid"], c["lane"], c["wave"], c["red"]
    if const_expr(tensile_split):
        first = gemv.load_tensile_chunk(lane, wave, ops, rg * ROWS, HIDDEN, 0)
    if tid < seam.NS:
        c["poll"]([(c["xrdy"], tid, 1)])
    gpu.barrier()
    stamp(c["tl_on"], c["tls"], tid, 9)
    gemv.dev_rows_to_lds(tid, c["xbuf"], HIDDEN, s, c["xl"], XROW)
    stamp(c["tl_on"], c["tls"], tid, 2)
    if const_expr(tensile_split):
        acc = gemv.mfmas_tensile_split(
            lane,
            wave,
            ops,
            rg * ROWS,
            c["xl"],
            XROW,
            HIDDEN,
            s,
            first,
        )
    else:
        acc = gemv.mfmas(lane, wave, c["xl"], XROW, HIDDEN, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * s:
        r = tid % ROWS
        tt = tid // ROWS
        if const_expr(tensile_split):
            value = gemv.reduce_tensile_split(red, r, tt)
        else:
            value = row_sum(red, r, tt)
        c["put"](
            c["qkv"],
            tt * QKV_ROWS + rg * ROWS + r,
            bf16_round(value),
        )
    gpu.barrier()


# ---------------------------------------------------------------- kernel

_BUILDS: dict = {}


def build(key: K1Build):
    if key in _BUILDS:
        return _BUILDS[key]
    _check_key(key)
    s, L = key.tokens, key.q_len
    nch = key.chunks
    nu = key.units
    nqt = attn.q_tiles(L)
    pm = attn.per_max(key.max_len, nch, key.q_len)
    lay = scratch_layout(key)

    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, s * XROW // 2, 16]

    @fx.struct
    class UnitLds:
        ql: fx.Array[fx.Int32, attn.GQA * L * attn.QROW, 16]
        btl: fx.Array[fx.Int32, 2 * pm + 4, 16]
        mb: fx.Array[fx.Float32, WAVES * 64, 16]
        lb: fx.Array[fx.Float32, WAVES * 64, 16]
        ob: fx.Array[fx.Float32, WAVES * 64 * 16, 16]

    @fx.struct
    class CombLds:
        cb: fx.Array[fx.Float32, 2 * WAVES, 16]
        cbo: fx.Array[fx.Float32, WAVES * 64, 16]

    @fx.union
    class RoleLds:
        gemv: GemvLds
        unit: UnitLds
        comb: CombLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = (
        f"muse_mono_k1_t{s}_l{L}_w{key.window}_r{int(key.rope)}_c{nch}"
        f"_t{int(key.timeline)}"
    )
    if key.fault_rank >= 0:
        name += (
            f"_f{key.fault_rank}_{key.fault_epoch}_{key.fault_layer}_{key.fault_ticks}"
        )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        stream_in: Int64,
        branch_in: Int64,
        w_pf: Int64,
        alpha_f: Int64,
        beta_f: Int64,
        w_qn: Int64,
        w_qkv: Int64,
        positions: Int64,
        cos_sin: Int64,
        slot_mapping: Int64,
        kv_cache: Int64,
        kv_sb: Int32,
        kv_sh: Int32,
        kv_st: Int32,
        block_table: Int64,
        bt_stride: Int32,
        seq_lens: Int64,
        stream_a: Int64,
        core: Int64,
        scratch: Int64,
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
        role = alloc.allocate(RoleLds)
        gl, ul, cl = role.gemv.peek(), role.unit.peek(), role.comb.peek()
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
        mb = Mailbox(step_tag(epoch, layer) - 1, health, alarm, region_ids(REGIONS))
        on, tls = key.timeline, lds.tls.ptr
        c = {
            "T": s,
            "L": L,
            "R": key.requests,
            "NCH": nch,
            "NQT": nqt,
            "per_max": pm,
            "window": key.window,
            "rope": key.rope,
            "scale": key.scale,
            "eps": key.eps,
            "post_eps": key.post_eps,
            "qk_eps": key.qk_eps,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "mbox": mb,
            "put": mb.put,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "tl_on": on,
            "tls": tls,
            "xl": gl.xl.ptr,
            "ql": ul.ql.ptr,
            "btl": ul.btl.ptr,
            "mb": ul.mb.ptr,
            "lb": ul.lb.ptr,
            "ob": ul.ob.ptr,
            "cb": cl.cb.ptr,
            "cbo": cl.cbo.ptr,
            "xbuf": scratch + fx.Int64(lay["x"][0]),
            "kv_blk_bytes": fx.Int64(kv_sb) * 2,
            "kv_span": (15 * kv_st + 2 * attn.HD) * 2,
            "args": {
                "stream_in": stream_in,
                "branch_in": branch_in,
                "w_pf": w_pf,
                "alpha_f": alpha_f,
                "beta_f": beta_f,
                "w_qn": w_qn,
                "positions": positions,
                "cos_sin": cos_sin,
                "slot_mapping": slot_mapping,
                "kv_cache": kv_cache,
                "kv_sb": kv_sb,
                "kv_sh": kv_sh,
                "kv_st": kv_st,
                "block_table": block_table,
                "bt_stride": bt_stride,
                "seq_lens": seq_lens,
                "stream_a": stream_a,
                "core": core,
            },
        }
        for region in REGIONS:
            c[region] = sreg(scratch, lay[region][0], region, lay[region][1])
        stamp_begin(on, tls, tid, TL_POINTS)
        # ---- 1b QKV GEMV: the weights in flight first, then the x rows
        if bid < GEMV0 + GEMV_TASKS:
            enter_stage("qkv")
            rg = gemv_task(bid - GEMV0)
            if const_expr(key.tokens > 4):
                stage_qkv(c, rg, w_qkv, tensile_split=True)
            else:
                ops = gemv.loads(c["lane"], c["wave"], w_qkv, rg * ROWS, HIDDEN)
                stage_qkv(c, rg, ops)
            stamp(on, tls, tid, 3)
        # ---- 1a seam A
        if (bid >= SEAM0) & (bid < SEAM0 + seam.NS):
            enter_stage("seam")
            seam.stage_seam(c, bid - SEAM0)
            stamp(on, tls, tid, 1)
        # ---- 1c / 1d attention units (at most one a CTA: a branch, not a
        # loop, so LLVM hoists no lane math into registers held across it)
        u = first_task(bid, ATT0)
        if u < nu:
            enter_stage("attention")
            attn.stage_unit(c, u)
            stamp(on, tls, tid, 7)
        # ---- 1e combine + gated o-norm (at most one a CTA)
        task = first_task(bid, COMB0)
        if task < s * attn.NQ:
            enter_stage("combine")
            attn.stage_combine(c, task)
            stamp(on, tls, tid, 8)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        stream_in: Int64,
        branch_in: Int64,
        w_pf: Int64,
        alpha_f: Int64,
        beta_f: Int64,
        w_qn: Int64,
        w_qkv: Int64,
        positions: Int64,
        cos_sin: Int64,
        slot_mapping: Int64,
        kv_cache: Int64,
        kv_sb: Int32,
        kv_sh: Int32,
        kv_st: Int32,
        block_table: Int64,
        bt_stride: Int32,
        seq_lens: Int64,
        stream_a: Int64,
        core: Int64,
        scratch: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            stream_in,
            branch_in,
            w_pf,
            alpha_f,
            beta_f,
            w_qn,
            w_qkv,
            positions,
            cos_sin,
            slot_mapping,
            kv_cache,
            kv_sb,
            kv_sh,
            kv_st,
            block_table,
            bt_stride,
            seq_lens,
            stream_a,
            core,
            scratch,
            rank,
            epoch,
            layer,
            health,
            alarm,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k1, launch)
    _BUILDS[key] = launch
    return launch


def lds_bytes(key: K1Build) -> int:
    """The LDS a build's CTA takes (static), from the layout above."""
    s, L = key.tokens, key.q_len
    pm = attn.per_max(key.max_len, key.chunks, key.q_len)
    gemv_b = s * XROW * 2
    unit_b = (attn.GQA * L * attn.QROW + 2 * pm + 4) * 4 + WAVES * 64 * 18 * 4
    comb_b = (2 * WAVES + WAVES * 64) * 4
    return WAVES * 64 * 4 * 4 + TL_POINTS * 8 + max(gemv_b, unit_b, comb_b)


assert (
    lds_bytes(K1Build(tokens=8, q_len=8, window=0, rope=False, scale=1.0)) < LDS_BYTES
)


def compile_only(key: K1Build) -> bool:
    """Compile ``key``'s kernel without launching it (FlyDSL's compile-only
    mode returns before the launch)."""
    import os

    launch = build(key)
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        with recording() as trace:
            launch(*ABI.zeros(), stream=torch.cuda.current_stream())
        from sglang.srt.rocm_muse_spark.mono.ops.common.validation import (
            validate_k1_trace,
        )

        return validate_k1_trace(key, trace)
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old


def resources(key: K1Build) -> dict:
    """VGPR / AGPR / SGPR counts, LDS and scratch (spill) bytes of the built
    kernel, from its code object's AMDGPU metadata (compiles it if needed)."""
    from sglang.srt.rocm_muse_spark.mono.ops.common.residency import code_objects

    launch = build(key)
    if not code_objects(launch):
        compile_only(key)
    out = {}
    for elf in code_objects(launch):
        out.update(_elf_resources(elf))
    return out


def _elf_resources(elf: bytes) -> dict:
    """The kernels' resource fields from the NT_AMDGPU_METADATA note (msgpack)."""
    import struct

    import msgpack

    (shoff,) = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", elf, 0x3A)
    out = {}
    for i in range(shnum):
        sh = shoff + i * shentsize
        (sh_type,) = struct.unpack_from("<I", elf, sh + 4)
        if sh_type != 7:  # SHT_NOTE
            continue
        off, size = struct.unpack_from("<QQ", elf, sh + 0x18)
        p = off
        while p < off + size:
            namesz, descsz, ntype = struct.unpack_from("<III", elf, p)
            p += 12
            name = elf[p : p + namesz].rstrip(b"\0")
            p += (namesz + 3) // 4 * 4
            desc = elf[p : p + descsz]
            p += (descsz + 3) // 4 * 4
            if name == b"AMDGPU" and ntype == 32:
                meta = msgpack.unpackb(desc, raw=False, strict_map_key=False)
                for k in meta.get("amdhsa.kernels", []):
                    out[k[".name"]] = {
                        "vgpr": k.get(".vgpr_count"),
                        "agpr": k.get(".agpr_count"),
                        "sgpr": k.get(".sgpr_count"),
                        "lds": k.get(".group_segment_fixed_size"),
                        "scratch": k.get(".private_segment_fixed_size"),
                        "vgpr_spill": k.get(".vgpr_spill_count"),
                        "sgpr_spill": k.get(".sgpr_spill_count"),
                        "wavefront_size": k.get(".wavefront_size"),
                    }
    return out


# ---------------------------------------------------------------- host


def ptr(t):
    return 0 if t is None else t.data_ptr()


def k1_launch(
    key: K1Build,
    *,
    stream_in,
    branch_in,
    w_pf,
    alpha_f,
    beta_f,
    w_qn,
    w_qkv,
    positions,
    cos_sin,
    slot_mapping,
    kv_cache,
    block_table,
    seq_lens,
    stream_a,
    core,
    scratch,
    epoch,
    layer,
    health,
    alarm,
    tl=None,
    rank: int = 0,
):
    """Host launcher (graph-safe: pointers and build constants only).

    ``stream_in`` fp32 / ``branch_in`` bf16 [T, 8192] contiguous rows;
    ``w_pf`` / ``w_qn`` bf16 [8192]; ``alpha_f`` / ``beta_f`` fp32 [8192];
    ``w_qkv`` bf16 [2304, 8192] row-major (q, k, v, o_gate rows); ``positions``
    / ``slot_mapping`` int64 [T]; ``cos_sin`` fp32 [max_pos, 64] (None on a
    NoPE layer); ``kv_cache`` bf16 [blocks, 2, 16, 128] (strides read, the
    content dim contiguous); ``block_table`` int32 [R, *] (row stride read);
    ``seq_lens`` int32 [R]; outputs ``stream_a`` fp32 [T, 8192] and ``core``
    bf16 [T, 1024] (contiguous rows); ``scratch`` (``scratch_bytes``), zeroed
    once by the runner; ``epoch`` the device step counter; ``layer`` this
    launch's tag slot (the layer index); ``health`` / ``alarm`` the fail-stop
    words; ``tl`` int64 [256, TL_POINTS] for a timeline build."""
    t = key.tokens
    assert stream_in.dtype == torch.float32 and stream_in.is_contiguous()
    assert branch_in.dtype == torch.bfloat16 and branch_in.is_contiguous()
    assert tuple(stream_in.shape) == (t, HIDDEN) == tuple(branch_in.shape)
    assert stream_a.dtype == torch.float32 and stream_a.is_contiguous()
    assert core.dtype == torch.bfloat16 and core.is_contiguous()
    assert tuple(core.shape) == (t, CORE)
    assert tuple(w_qkv.shape) == (QKV_ROWS, HIDDEN) and w_qkv.is_contiguous()
    assert (cos_sin is not None) == key.rope
    assert cos_sin is None or (
        cos_sin.dtype == torch.float32 and cos_sin.is_contiguous()
    )
    assert kv_cache.dtype == torch.bfloat16 and kv_cache.stride(3) == 1
    assert tuple(kv_cache.shape[1:]) == (attn.NKV, attn.BLK, 2 * attn.HD)
    assert kv_cache.stride(2) % 2 == 0
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert seq_lens.dtype == torch.int32
    assert positions.dtype == torch.int64 and slot_mapping.dtype == torch.int64
    assert scratch.numel() * scratch.element_size() >= scratch_bytes(key)
    f = build(key)
    f(
        stream_in.data_ptr(),
        branch_in.data_ptr(),
        w_pf.data_ptr(),
        alpha_f.data_ptr(),
        beta_f.data_ptr(),
        w_qn.data_ptr(),
        w_qkv.data_ptr(),
        positions.data_ptr(),
        ptr(cos_sin),
        slot_mapping.data_ptr(),
        kv_cache.data_ptr(),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.data_ptr(),
        block_table.stride(0),
        seq_lens.data_ptr(),
        stream_a.data_ptr(),
        core.data_ptr(),
        scratch.data_ptr(),
        rank,
        epoch.data_ptr(),
        layer,
        health.data_ptr(),
        alarm,
        ptr(tl),
        stream=torch.cuda.current_stream(),
    )
