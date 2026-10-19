# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 stages 1c-1e: QK-norm + RoPE + KV insert, split-KV attention over the
paged bf16 cache, and the chunk combine with the gated o-norm.

Per TP rank: 16 q heads, 2 kv heads (GQA 8), head_dim 64, the fused QKV row
``[q 0..1023 | k 1024..1151 | v 1152..1279 | o_gate 1280..2303]`` (bf16 values
handed over as tagged pairs by the QKV GEMV). R requests of q_len L rows each.

Units (``stage_unit``): (request j, kv head g, chunk ch), NCH chunks a (request,
head), numbered chunk-major (unit = ch * 2R + 2 j + g). The request's context
keys are cut into 32-key tiles.  Single-row AITER attention starts at the first
whole 16-key cache block in the sliding window and uses at least 256 keys per
partition.  Multi-row Triton attention starts at the first whole 64-key tile
selected by its query-aware window bound and divides that range into
whole 64-key segments (including short contexts).  Runs are derived on device from seq_lens (a short context fills only
the first chunks); the step's L new keys [seq - L, seq) belong to their
production partition. A unit:

  1. its tiles' block ids -> LDS, each wave's first tile of K / V issued (no
     hand-off needed: the context is already in the cache);
  2. the raw q of its 8 heads x L rows polled from the QKV pairs, normed and
     rotated exactly as ``fused_muse_spark_qknorm_rope_kv_insert`` does (a head
     on 8 lanes of 8 dims, fma sum of squares, xor 4 / 2 / 1, rsq, GPT-J pairs
     with the fp32 cos / sin, one bf16 rounding) -> LDS;
  3. the chunk owning the step's new-key partition also norms / rotates k,
     takes v raw, stores both rows at
     ``slot_mapping`` (skipping slot < 0), drains, and only then reads the new
     keys back through the block table;
  4. each wave walks its tiles (wave w: tiles w, w + 8, .., the next tile's
     loads in flight): QK^T and PV on bf16 16x16x32 MFMAs with an online fp32
     softmax (exp2 domain);
  5. the 8 waves merged in LDS in wave order -> the unit's partial (max, sum,
     o[64]) for each of its 8 L query rows, fp32 tagged pairs.  The PV MFMA
     accumulator stays fp32 across this boundary; rounding it to bf16 here
     needlessly compounds the probability rounding and diverges from the
     production paged-attention output at multi-row widths.

MFMA layout (16x16x32 bf16: lane (g4, r16) holds row / column r16, K values
8 g4 .. 8 g4 + 7; the accumulator's lane holds rows 4 g4 + i, column r16):
QK^T has A = K (lane: key r16 of a 16-key block, dims 8 g4 ..) loaded straight
from the cache, B = q from LDS, so the scores of keys 4 g4 + i sit in column
(query) r16. PV contracts over 32 keys numbered so that lane (g4, n) of P
holds exactly its own scores (key 16 s + 4 g4 + i is k-index 8 g4 + 4 s + i):
B = P needs no data movement, and A = V^T is built from 8-byte loads of V (four
keys x four dims a lane), A row r standing for dim 4 r + c in MFMA c, so the
accumulators of a lane hold dims 16 g4 .. 16 g4 + 15 of its query.

Combine (``stage_combine``): one (row, q head) a CTA, chunks merged in a fixed
order (wave w: chunks w, w + 8, .., then waves in order), o rounded to bf16 as
the attention output is, then ``rms_norm_gated`` (sigmoid, no weight).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value
from sglang.srt.rocm_muse_spark.mono.ops.common.debug import stamp
from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    CM_DEV,
    bf16_round,
    butterfly,
    hw_exp,
    hw_exp2,
    hw_rsq,
    ld_i32,
    mfma_bf16,
    rsrc,
    traced,
    uniform,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.plan import THREADS, WAVES, note
from sglang.srt.rocm_muse_spark.mono.ops.common.sync import _spin

HD = 64
GQA = 8  # q heads a kv head
NQ = 16  # q heads a rank
NKV = 2  # kv heads a rank
QKV_ROWS = 2304
K_ROW0 = 1024
V_ROW0 = 1152
OG_ROW0 = 1280
CORE = NQ * HD  # 1024
TILE = 32  # keys a tile (two cache blocks)
BLK = 16  # the cache's block size
PART = 2 + HD  # pairs a partial: max, sum, o[64]
QROW = HD // 2 + 4  # i32 words an LDS q row: 64 bf16 + 16 B
NEG = -3.0e38  # a masked score (finite: NEG - NEG = 0)
LOG2E = 1.4426950408889634


def q_tiles(qlen: int) -> int:
    """16-query MFMA columns tiles of a unit (8 heads x qlen rows)."""
    return (GQA * qlen + 15) // 16


MIN_PER = 8  # tiles a unit takes at least (one a wave)


def per_max(max_len: int, nch: int, q_len: int = 1) -> int:
    """The most tiles a unit can own at a context of ``max_len``."""
    if q_len > 1:
        blocks = (max_len + 63) // 64
        return max(2 * ((blocks + nch - 1) // nch), MIN_PER)
    tiles = (max_len + TILE - 1) // TILE
    return max((tiles + nch - 1) // nch, MIN_PER)


def _gptr(addr):
    return llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr))
    ).result


def gstore(addr, value):
    """Four i32 (a Vector) at a 64-bit global address (a cache past 4 GiB)."""
    llvm.StoreOp(as_ir_value(value), _gptr(addr), alignment=16)


def _fma(a, b, c):
    return fx.Float32(fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


def _f32(w):
    return w.bitcast(fx.Float32)


def pack8(vals):
    """Eight f32 -> four words of bf16 pairs (RNE)."""
    return fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)


# ---------------------------------------------------------------- q / k prep


def rope_vals(c, t, li):
    """Cos / sin of lane li's four pairs at row t's position (fp32 cache)."""
    if const_expr(not c["rope"]):
        return None, None
    a = c["args"]
    pos = ld_i32(a["positions"], 2 * t)  # int64, low word
    r = rsrc(a["cos_sin"])
    cs = fx.Vector(bo.buffer_load(r, pos * HD + 4 * li, vec_width=4, dtype=T.f32))
    sn = fx.Vector(
        bo.buffer_load(r, pos * HD + HD // 2 + 4 * li, vec_width=4, dtype=T.f32)
    )
    return [cs[p] for p in range(4)], [sn[p] for p in range(4)]


def qk_norm_rope(c, x, cs, sn):
    """A lane's 8 dims of a head (8 lanes a head, lane-aligned groups): the
    weightless RMSNorm and GPT-J pair rotation of the fused CUDA kernel, fp32
    (its fma chain, xor 4 / 2 / 1 reduction, rsq, contracted rotation)."""
    ss = x[0] * x[0]
    for k in range_constexpr(1, 8):
        ss = _fma(x[k], x[k], ss)
    ss = butterfly(ss, (4, 2, 1))
    r = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(c["qk_eps"]))
    y = [x[k] * r for k in range(8)]
    if const_expr(c["rope"]):
        out = []
        for p in range_constexpr(4):
            v0, v1 = y[2 * p], y[2 * p + 1]
            out.append(_fma(v0, cs[p], -(v1 * sn[p])))
            out.append(_fma(v0, sn[p], v1 * cs[p]))
        y = out
    return y


@traced
def unit_q(c, j, g):
    """The 8 q heads of kv head g, request j's L rows: polled raw, normed,
    rotated, bf16 -> LDS ``ql`` row qvec = row * 8 + head (8 lanes a row)."""
    L, tid = c["L"], c["tid"]
    if tid < 64 * L:
        hr = tid // 8
        li = tid % 8
        i = hr // GQA
        hq = hr % GQA
        t = j * L + i
        cs, sn = rope_vals(c, t, li)
        base = t * QKV_ROWS + (g * GQA + hq) * HD + 8 * li
        got = c["poll"]([(c["qkv"], base + 2 * e, 2) for e in range(4)])
        x = [_f32(got[e // 2][e % 2]) for e in range(8)]
        y = qk_norm_rope(c, x, cs, sn)
        fx.ptr_store(pack8(y), c["ql"] + (hr * QROW + 4 * li))
    gpu.barrier()


@traced
def unit_insert(c, j, g, kvh, token_start=None, token_end=None, n_old=0):
    """KV head g's k (normed, rotated) and v (raw) of request j's L rows ->
    the cache at their slots (slot < 0: a pad row, skipped); drained before
    the CTA reads them back."""
    L, tid = c["L"], c["tid"]
    a = c["args"]
    if tid < 16 * L:
        kvs = tid // (8 * L)  # 0: k, 1: v (a whole 8-lane group each)
        i = (tid // 8) % L
        li = tid % 8
        t = j * L + i
        slot = ld_i32(a["slot_mapping"], 2 * t)  # int64, low word (-1: pad)
        cs, sn = rope_vals(c, t, li)
        base = t * QKV_ROWS + K_ROW0 + kvs * (V_ROW0 - K_ROW0) + g * HD + 8 * li
        got = c["poll"]([(c["qkv"], base + 2 * e, 2) for e in range(4)])
        x = [_f32(got[e // 2][e % 2]) for e in range(8)]
        y = qk_norm_rope(c, x, cs, sn)
        is_v = kvs == 1
        words = pack8([is_v.select(x[e], y[e]) for e in range(8)])
        owns = slot >= 0
        if const_expr(token_start is not None):
            owns = owns & (n_old + i >= token_start) & (n_old + i < token_end)
        if owns:
            off = fx.Int64(slot // BLK) * fx.Int64(a["kv_sb"]) + fx.Int64(
                (slot % BLK) * a["kv_st"] + kvs * HD + 8 * li
            )
            gstore(kvh + off * 2, words)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()


# ---------------------------------------------------------------- K / V tiles


def _kv_rsrc(c, kvh, blk):
    """The (block, head) rows of the cache as a buffer: 16 tokens x [K | V]."""
    return rsrc(kvh + fx.Int64(blk) * c["kv_blk_bytes"], c["kv_span"])


def tile_loads(c, kvh, b0, b1, pred):
    """K and V of a 32-key tile in cache blocks b0 (keys 0..15) and b1 (16..31),
    masked off (no access, zeros) unless ``pred``: [K s=0 lo, hi, s=1 lo, hi]
    (dwordx4: key r16, dims 8 g4 .. / 32 + 8 g4 ..), then [V s, i] (dwordx2:
    key 16 s + 4 g4 + i, dims 4 r16 ..)."""
    lane = c["lane"]
    st = c["args"]["kv_st"]
    g4 = lane // 16
    r16 = lane % 16
    ks, vs = [], []
    for b in (b0, b1):
        r = _kv_rsrc(c, kvh, b)
        ko = (r16 * st + 8 * g4) // 2
        for h in range(2):
            ks.append(
                fx.Vector(
                    bo.buffer_load(r, ko + 16 * h, vec_width=4, dtype=T.i32, mask=pred)
                )
            )
        for i in range(4):
            vo = ((4 * g4 + i) * st + HD + 4 * r16) // 2
            vs.append(
                fx.Vector(bo.buffer_load(r, vo, vec_width=2, dtype=T.i32, mask=pred))
            )
    return ks + vs


def old_tile_loads(c, kvh, ti, ta, tb):
    """Tile ``ti`` of the unit's run [ta, tb) (its block ids from LDS); past the
    run: masked loads of nothing."""
    pred = ti < tb
    idx = fx.max(fx.min(ti, tb - 1) - ta, 0)
    btl = c["btl"]
    b0 = uniform(pred.select(fx.ptr_load(btl + 2 * idx), fx.Int32(0)))
    b1 = uniform(pred.select(fx.ptr_load(btl + 2 * idx + 1), fx.Int32(0)))
    return tile_loads(c, kvh, b0, b1, pred)


def new_loads(c, kvh, n_old, seq):
    """The step's new keys n_old .. seq - 1 (< 16 of them, in at most two cache
    blocks: block ids in LDS ``btl`` past the run's) as a tile based at n_old;
    each lane's key from whichever block holds it (two masked loads)."""
    lane = c["lane"]
    st = c["args"]["kv_st"]
    g4 = lane // 16
    r16 = lane % 16
    btl = c["btl"]
    pm = c["per_max"]
    ra = _kv_rsrc(c, kvh, uniform(fx.ptr_load(btl + 2 * pm)))
    rb = _kv_rsrc(c, kvh, uniform(fx.ptr_load(btl + 2 * pm + 1)))
    blk_a = fx.max(n_old, 0) // BLK

    def two(key, words, inner):
        ok = (key >= 0) & (key < seq)
        in_a = (key // BLK) == blk_a
        off = ((key % BLK) * st + inner) // 2
        la = fx.Vector(
            bo.buffer_load(ra, off, vec_width=words, dtype=T.i32, mask=ok & in_a)
        )
        lb = fx.Vector(
            bo.buffer_load(rb, off, vec_width=words, dtype=T.i32, mask=ok & ~in_a)
        )
        return la | lb

    zero4 = fx.Vector.filled(4, 0, fx.Int32)
    zero2 = fx.Vector.filled(2, 0, fx.Int32)
    ks = [two(n_old + r16, 4, 8 * g4 + 32 * h) for h in range(2)] + [zero4, zero4]
    vs = [two(n_old + 4 * g4 + i, 2, HD + 4 * r16) for i in range(4)]
    return ks + vs + [zero2] * 4


# ---------------------------------------------------------------- tile math


def init_stats(nqt):
    """Per-wave online-softmax ``[max] + [sum]`` state."""
    return [fx.Float32(NEG)] * nqt + [fx.Float32(0.0)] * nqt


def init_output(nqt):
    """Per-wave partition output, four MFMA accumulator groups per query."""
    return [fx.Vector.filled(4, 0.0, fx.Float32)] * (4 * nqt)


def _vt_operands(vs):
    """V^T MFMA A operands c = 0..3 (A row r16 <-> dim 4 r16 + c; k-index
    8 g4 + j <-> V load (s, i) = (j // 4, j % 4))."""
    out = []
    for cc in range(4):
        words = []
        for d in range(4):
            w0 = vs[(2 * d) // 4 * 4 + (2 * d) % 4][cc // 2]
            w1 = vs[(2 * d + 1) // 4 * 4 + (2 * d + 1) % 4][cc // 2]
            if cc % 2 == 0:
                words.append((w0 & 0xFFFF) | (w1 << 16))
            else:
                words.append(((w0 >> 16) & 0xFFFF) | (w1 & fx.Int32(-65536)))
        out.append(fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16))
    return out


def _tile_scores(c, ks, kb, khi, n_old, n):
    """The eight lane-local scaled QK scores and their validity masks."""
    L, window = c["L"], c["window"]
    lane = c["lane"]
    g4 = lane // 16
    n16 = lane % 16
    keys = [kb + 16 * s + 4 * g4 + i for s in range(2) for i in range(4)]
    kok = [(k >= 0) & (k < khi) for k in keys]
    qvec = 16 * (n + c.get("qtile", 0)) + n16
    qv = fx.min(qvec, GQA * L - 1)
    qb = c["ql"] + qv * QROW + 4 * g4
    q0 = fx.Vector(
        fx.ptr_load(qb, result_type=fx.Vector.make_type(4, fx.Int32))
    ).bitcast(fx.BFloat16)
    q1 = fx.Vector(
        fx.ptr_load(qb + 16, result_type=fx.Vector.make_type(4, fx.Int32))
    ).bitcast(fx.BFloat16)
    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    accs = []
    for s in range(2):
        acc = mfma_bf16(ks[2 * s].bitcast(fx.BFloat16), q0, zero)
        accs.append(mfma_bf16(ks[2 * s + 1].bitcast(fx.BFloat16), q1, acc))
    pos = n_old + qvec // GQA
    qok = qvec < GQA * L
    scale = fx.Float32(c["scale"])
    sv, ok = [], []
    for s in range(2):
        for i in range(4):
            k = keys[4 * s + i]
            v = kok[4 * s + i] & (k <= pos) & qok
            if window > 0:
                v = v & (k > pos - window)
            ok.append(v)
            sv.append(v.select(accs[s][i] * scale, fx.Float32(NEG)))
    return sv, ok


def tile_stats(c, st, kv, kb, khi, n_old):
    """Accumulate one tile into a wave's online max/sum state."""
    nqt = c["NQT"]
    ms, ls = st[:nqt], st[nqt:]
    ms2, ls2 = [], []
    for n in range(nqt):
        sv, ok = _tile_scores(c, kv[:4], kb, khi, n_old, n)
        tmax = fx.max(
            fx.max(fx.max(sv[0], sv[1]), fx.max(sv[2], sv[3])),
            fx.max(fx.max(sv[4], sv[5]), fx.max(sv[6], sv[7])),
        )
        tmax = butterfly(tmax, (16, 32), fx.max)
        m_new = fx.max(ms[n], tmax)
        alpha = hw_exp(ms[n] - m_new)
        p = [ok[e].select(hw_exp(sv[e] - m_new), fx.Float32(0.0)) for e in range(8)]
        psum = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]))
        ls2.append(ls[n] * alpha + psum)
        ms2.append(m_new)
    return ms2 + ls2


def tile_output(c, out, kv, kb, khi, n_old):
    """Accumulate one tile with partition-global, BF16-rounded probabilities."""
    nqt = c["NQT"]
    n16 = c["lane"] % 16
    av = _vt_operands(kv[4:])
    out2 = []
    for n in range(nqt):
        sv, ok = _tile_scores(c, kv[:4], kb, khi, n_old, n)
        qvec = 16 * n + n16
        mm = fx.ptr_load(c["mb"] + qvec)
        ll = fx.ptr_load(c["lb"] + qvec)
        den = ll + fx.Float32(1.0e-6)
        p = [ok[e].select(hw_exp(sv[e] - mm) / den, fx.Float32(0.0)) for e in range(8)]
        pb = fx.Vector.from_elements(p, fx.Float32).to(fx.BFloat16)
        for cc in range(4):
            out2.append(mfma_bf16(av[cc], pb, out[4 * n + cc]))
    return out2


def tile_maxima(c, kv, kb, khi, n_old):
    """Per-query maxima for one 32-key half of a Triton 64-key block."""
    maxima = []
    for n in range(c["NQT"]):
        sv, _ = _tile_scores(c, kv[:4], kb, khi, n_old, n)
        maxima.append(
            fx.max(
                fx.max(fx.max(sv[0], sv[1]), fx.max(sv[2], sv[3])),
                fx.max(fx.max(sv[4], sv[5]), fx.max(sv[6], sv[7])),
            )
        )
    return maxima


def _natural_pv_order(c, values):
    """Map [4g:4g+4,16+4g:20+4g] keys to native [8g:8g+8].

    Both MFMA operands receive the identical permutation. This preserves
    the mathematical dot while matching native's FP32 accumulation order.
    """
    words = values.bitcast(fx.Int32)
    lane = c["lane"]
    group = lane // 16
    out = []
    for e in range(4):
        src = (2 * (group % 2) + e // 2) * 16 + lane % 16
        lo = fx.Int32(rocdl.ds_bpermute(T.i32, src * 4, words[e % 2].ir_value()))
        hi = fx.Int32(rocdl.ds_bpermute(T.i32, src * 4, words[2 + e % 2].ir_value()))
        out.append((group < 2).select(lo, hi))
    return fx.Vector.from_elements(out, fx.Int32).bitcast(fx.BFloat16)


def tile_online_output(c, out, kv, kb, khi, n_old, maxima):
    """One 32-key half of Triton's BF16-P, fp32-accumulator PV update."""
    av = [_natural_pv_order(c, v) for v in _vt_operands(kv[4:])]
    out2, sums = [], []
    for n in range(c["NQT"]):
        sv, ok = _tile_scores(c, kv[:4], kb, khi, n_old, n)
        p = [ok[e].select(hw_exp(sv[e] - maxima[n]), fx.Float32(0.0)) for e in range(8)]
        # Original Triton MFMA-transposed reduction: register XOR2,1,
        # lane XOR32,16 (keys8,4), then warp keys32,16.
        sums.append(
            [
                butterfly((p[0] + p[2]) + (p[1] + p[3]), (32, 16)),
                butterfly((p[4] + p[6]) + (p[5] + p[7]), (32, 16)),
            ]
        )
        pb = _natural_pv_order(
            c, fx.Vector.from_elements(p, fx.Float32).to(fx.BFloat16)
        )
        for cc in range(4):
            out2.append(mfma_bf16(av[cc], pb, out[4 * n + cc]))
    return out2, sums


# ---------------------------------------------------------------- unit


@traced
def unit_stats(c, st):
    """Merge per-wave max/sum states and publish partition-global statistics."""
    nqt = c["NQT"]
    lane, wave = c["lane"], c["wave"]
    g4 = lane // 16
    n16 = lane % 16
    mb, lb = c["mb"], c["lb"]
    ms, ls = st[:nqt], st[nqt:]
    ls = [butterfly(ls[n], (16, 32)) for n in range(nqt)]
    for n in range_constexpr(nqt):
        qvec = 16 * n + n16
        if g4 == 0:
            fx.ptr_store(ms[n], mb + wave * 64 + qvec)
            fx.ptr_store(ls[n], lb + wave * 64 + qvec)
    gpu.barrier()
    for n in range_constexpr(nqt):
        qvec = 16 * n + n16
        mm = fx.ptr_load(mb + qvec)
        for w in range_constexpr(1, WAVES):
            mm = fx.max(mm, fx.ptr_load(mb + w * 64 + qvec))
        if g4 == 0:
            fx.ptr_store(ls[n] * hw_exp(ms[n] - mm), lb + wave * 64 + qvec)
    gpu.barrier()
    if (wave == 0) & (g4 == 0):
        for n in range_constexpr(nqt):
            qvec = 16 * n + n16
            mm = fx.ptr_load(mb + qvec)
            ltot = fx.ptr_load(lb + qvec)
            for w in range_constexpr(1, WAVES):
                mm = fx.max(mm, fx.ptr_load(mb + w * 64 + qvec))
                ltot = ltot + fx.ptr_load(lb + w * 64 + qvec)
            fx.ptr_store(mm, mb + qvec)
            fx.ptr_store(ltot, lb + qvec)
    gpu.barrier()


@traced
def unit_finish(c, out, u):
    """Sum wave outputs and publish the normalized fp32 partial."""
    L, nqt = c["L"], c["NQT"]
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    mb, lb, ob = c["mb"], c["lb"], c["ob"]
    qv = tid // 32
    dp = tid % 32
    for n in range_constexpr(nqt):
        for cc in range_constexpr(4):
            fx.ptr_store(out[4 * n + cc], ob + ((wave * 64 + lane) * 16 + 4 * cc))
        gpu.barrier()
        qvec = 16 * n + qv
        vals = []
        for e in range_constexpr(2):
            d = 2 * dp + e
            src = 16 * (d // 16) + qv
            idx = 4 * (d % 4) + (d % 16) // 4
            tot = fx.ptr_load(ob + (src * 16 + idx))
            for w in range_constexpr(1, WAVES):
                tot = tot + fx.ptr_load(ob + ((w * 64 + src) * 16 + idx))
            if const_expr(L == 1):
                # AITER decode writes each normalized partition result through
                # BF16 before its fp32 partition combine.
                tot = tot.to(fx.BFloat16).to(fx.Float32)
            vals.append(tot)
        ltot = fx.ptr_load(lb + qvec)
        mm = fx.ptr_load(mb + qvec)
        if qvec < GQA * L:
            base = (u * (GQA * L) + qvec) * PART
            c["put_words"](
                c["apart"], base + 2 + 2 * dp, [v.bitcast(fx.Int32) for v in vals]
            )
            if dp == 0:
                c["put_words"](
                    c["apart"],
                    base,
                    [mm.bitcast(fx.Int32), ltot.bitcast(fx.Int32)],
                )
        gpu.barrier()


def context_tiles(c, j):
    """Request j's production-aligned tile run.

    Returns ``(seq, n_old, origin, t_hi, per, new_ch)``. Tile indices are
    relative to ``origin``. Single-row AITER decode uses a shifted 16-key
    cache-block origin and a 256-key minimum run. Multi-row Triton verify uses
    a query-aware origin and segments aligned to 64 keys. ``t_hi``
    includes the newly inserted rows so every online-softmax block sees the
    same key extent as the production attention kernel.
    """
    L, nch, window = c["L"], c["NCH"], c["window"]
    seq = uniform(ld_i32(c["args"]["seq_lens"], j))
    n_old = seq - L
    nonneg = fx.max(n_old, 0)
    if const_expr((L > 1) and window > 0):
        # Equivalent to Triton's first_tile_numerator =
        # seq_len - query_len - sliding_window + 1, rounded down to its
        # 64-key BLOCK_N.  Our internal attention tile remains 32 keys.
        first_tile = fx.max(seq - L - window + 1, 0) // 64
        origin = first_tile * 64
        min_per = MIN_PER  # unused by the multi-query partition rule
    elif const_expr(window > 0):
        first_block = fx.max(seq - window, 0) // BLK
        origin = first_block * BLK
        min_per = MIN_PER
    else:
        origin = fx.Int32(0)
        min_per = MIN_PER
    old_extent = fx.max(nonneg - origin, 0)
    all_extent = fx.max(seq - origin, 1)
    t_hi = (old_extent + TILE - 1) // TILE
    all_tiles = (all_extent + TILE - 1) // TILE
    t_hi = all_tiles
    if const_expr(L > 1):
        # Original Triton partitions in whole 64-key blocks, including
        # short contexts. Express that count in our 32-key tile units.
        blocks = (all_extent + 63) // 64
        per = 2 * ((blocks + nch - 1) // nch)
    else:
        per = fx.min(fx.max((all_tiles + nch - 1) // nch, min_per), c["per_max"])
    new_tile = fx.max(nonneg - origin, 0) // TILE
    new_ch = fx.min(new_tile // per, nch - 1)
    return seq, n_old, origin, t_hi, per, new_ch


@traced
def stage_unit(c, u):
    """Attention unit ``u`` = (request j, kv head g, chunk ch) -> partials; a
    unit past its request's run does nothing (the combine skips it)."""
    ngrp = NKV * c["R"]
    grp = u % ngrp  # unit = chunk-major: every group's chunk 0 first
    ch = u // ngrp
    j = grp // NKV
    seq, n_old, origin, t_hi, per, new_ch = context_tiles(c, j)
    ta = ch * per
    tb = fx.min(ta + per, t_hi)
    if tb > ta:
        if const_expr(c["L"] == 1):
            unit_body(c, u, j, grp % NKV, ch, seq, n_old, origin, ta, tb, new_ch)
        else:
            unit_body_triton(c, u, j, grp % NKV, ch, seq, n_old, origin, ta, tb, new_ch)


@traced
def unit_body_triton(c, u, j, g, ch, seq, n_old, origin, ta, tb, new_ch):
    """Unit matching production's sequential 64-key online softmax.

    Each active wave owns one query tile and walks its segment in the same
    fixed 64-key block order as production. Query tiles are independent, so
    distributing them across waves preserves all arithmetic and rounding.
    """
    tiles = c["NQT"]
    nqt = 1
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    a = c["args"]
    pm = c["per_max"]
    btl = c["btl"]
    row0 = j * a["bt_stride"]
    nbt = 2 * (tb - ta)
    last_block = fx.max(seq - 1, 0) // BLK
    for r in range_constexpr((2 * pm + THREADS - 1) // THREADS):
        e = tid + THREADS * r
        if e < nbt:
            block = fx.min(origin // BLK + 2 * ta + e, last_block)
            fx.ptr_store(ld_i32(a["block_table"], row0 + block), btl + e)
    gpu.barrier()
    kvh = a["kv_cache"] + fx.Int64(g * a["kv_sh"]) * 2
    stamp(c["tl_on"], c["tls"], tid, 4)
    unit_q(c, j, g)
    # A verification batch can straddle partition boundaries. Each partition
    # inserts only its own new positions before reading its cache tiles.
    token_start = origin + ta * TILE
    token_end = origin + tb * TILE
    if (token_start < seq) & (token_end > n_old):
        unit_insert(c, j, g, kvh, token_start, token_end, n_old)
    stamp(c["tl_on"], c["tls"], tid, 5)

    tc = dict(c)
    tc["NQT"] = 1
    tc["qtile"] = wave
    ms = [fx.Float32(float("-inf"))] * nqt
    ls = [fx.Float32(1.0)] * nqt
    out = init_output(nqt)
    # Each query wave covers two tokens, as native BLOCK_M=16/GQA=8.
    # Native pruning can skip an entirely masked leading or trailing block.
    ti = ta
    if const_expr(c["window"] > 0):
        first = fx.max(n_old + 2 * wave - c["window"] + 1, 0) // 64
        ti = fx.max(ta, 2 * first - origin // TILE)
    last = (fx.min(seq, n_old + 2 * wave + 2) + 63) // 64
    limit = (wave < tiles).select(fx.min(tb, 2 * last - origin // TILE), ta)
    while ti < limit:
        # Production Triton updates online-softmax state once per 64 keys.
        # First find the block maximum, then reload the two halves for the
        # BF16 probability conversion and fp32 PV accumulation.
        kv0 = old_tile_loads(c, kvh, ti, ta, tb)
        mx0 = tile_maxima(tc, kv0, origin + ti * TILE, seq, n_old)
        kv1 = old_tile_loads(c, kvh, ti + 1, ta, tb)
        mx1 = tile_maxima(tc, kv1, origin + (ti + 1) * TILE, seq, n_old)
        mnew, alpha = [], []
        scaled = []
        for n in range_constexpr(nqt):
            block_max = butterfly(fx.max(mx0[n], mx1[n]), (16, 32), fx.max)
            mn = fx.max(ms[n], block_max)
            mn = (mn > fx.Float32(NEG)).select(mn, fx.Float32(0.0))
            an = hw_exp(ms[n] - mn)
            mnew.append(mn)
            alpha.append(an)
            for cc in range_constexpr(4):
                scaled.append(out[4 * n + cc] * an)
        kv0 = old_tile_loads(c, kvh, ti, ta, tb)
        out, sum0 = tile_online_output(
            tc, scaled, kv0, origin + ti * TILE, seq, n_old, mnew
        )
        kv1 = old_tile_loads(c, kvh, ti + 1, ta, tb)
        out, sum1 = tile_online_output(
            tc, out, kv1, origin + (ti + 1) * TILE, seq, n_old, mnew
        )
        for n in range_constexpr(nqt):
            block_sum = (sum0[n][0] + sum1[n][0]) + (sum0[n][1] + sum1[n][1])
            ls[n] = _fma(ls[n], alpha[n], block_sum)
            ms[n] = mnew[n]
        ti = ti + 2

    # Publish directly from each query owner's MFMA accumulators. In this
    # layout lane%16 owns the query and lane//16 owns a 16-feature group;
    # register r, accumulator cc correspond to feature 16*g4 + 4*r + cc.
    # No cross-wave sum is needed: query tiles have disjoint outputs.
    if wave < tiles:
        qvec = 16 * wave + lane % 16
        # Native multi-query attention publishes unnormalized PV accumulators.
        # Dividing here and multiplying by the sum in combine loses FP32 bits.
        partial = out
        if qvec < GQA * c["L"]:
            base = (u * (GQA * c["L"]) + qvec) * PART
            for r in range_constexpr(4):
                for pair in range_constexpr(2):
                    d = 16 * (lane // 16) + 4 * r + 2 * pair
                    c["put_words"](
                        c["apart"],
                        base + 2 + d,
                        [
                            partial[2 * pair][r].bitcast(fx.Int32),
                            partial[2 * pair + 1][r].bitcast(fx.Int32),
                        ],
                    )
            if lane // 16 == 0:
                c["put_words"](
                    c["apart"],
                    base,
                    [ms[0].bitcast(fx.Int32), ls[0].bitcast(fx.Int32)],
                )
    gpu.barrier()
    stamp(c["tl_on"], c["tls"], tid, 6)


@traced
def unit_body(c, u, j, g, ch, seq, n_old, origin, ta, tb, new_ch):
    nqt = c["NQT"]
    tid, wave = c["tid"], c["wave"]
    a = c["args"]
    pm = c["per_max"]
    nonneg = fx.max(n_old, 0)
    nbt = 2 * fx.max(tb - ta, 0)
    owns_new = ch == new_ch
    btl = c["btl"]
    row0 = j * a["bt_stride"]
    for r in range_constexpr((2 * pm + THREADS - 1) // THREADS):
        e = tid + THREADS * r
        if e < nbt:
            block = origin // BLK + 2 * ta + e
            fx.ptr_store(ld_i32(a["block_table"], row0 + block), btl + e)
    if tid < 2:
        bi = fx.min(nonneg // BLK + tid, fx.max(seq - 1, 0) // BLK)
        fx.ptr_store(ld_i32(a["block_table"], row0 + bi), btl + (2 * pm + tid))
    gpu.barrier()
    kvh = a["kv_cache"] + fx.Int64(g * a["kv_sh"]) * 2
    ti = ta + wave
    cur = old_tile_loads(c, kvh, ti, ta, tb)
    stamp(c["tl_on"], c["tls"], tid, 4)
    unit_q(c, j, g)
    if owns_new:
        unit_insert(c, j, g, kvh)
    stamp(c["tl_on"], c["tls"], tid, 5)
    st = init_stats(nqt)
    while ti < tb:
        nxt = old_tile_loads(c, kvh, ti + WAVES, ta, tb)
        st = tile_stats(c, st, cur, origin + ti * TILE, n_old, n_old)
        ti = ti + WAVES
        cur = nxt
    kn = (owns_new & (wave == WAVES - 1)).select(fx.Int32(1), fx.Int32(0))
    k = fx.Int32(0)
    while k < kn:
        nv = new_loads(c, kvh, n_old, seq)
        st = tile_stats(c, st, nv, n_old, seq, n_old)
        k = k + 1
    unit_stats(c, st)
    ti = ta + wave
    cur = old_tile_loads(c, kvh, ti, ta, tb)
    out = init_output(nqt)
    while ti < tb:
        nxt = old_tile_loads(c, kvh, ti + WAVES, ta, tb)
        out = tile_output(c, out, cur, origin + ti * TILE, n_old, n_old)
        ti = ti + WAVES
        cur = nxt
    k = fx.Int32(0)
    while k < kn:
        nv = new_loads(c, kvh, n_old, seq)
        out = tile_output(c, out, nv, n_old, seq, n_old)
        k = k + 1
    stamp(c["tl_on"], c["tls"], tid, 6)
    unit_finish(c, out, u)


def poll_live(c, specs, lives):
    """``Mailbox.poll`` of ``specs`` (one batch, scratch pairs) that waits only
    on the specs whose ``lives`` entry (a traced bool, or None: always) holds;
    the others are loaded and ignored."""
    mb = c["mbox"]
    tag = mb.tag
    for a, _, _ in specs:
        note(a.region, a.space, "poll", a.offset, a.extent)
    bases = [a.value for a, _, _ in specs]

    def load_all():
        words = []
        for b, (_, i, n) in zip(bases, specs):
            w = fx.Vector(
                bo.buffer_load(
                    rsrc(b),
                    fx.Int32(i) * 2,
                    vec_width=2 * n,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            words += [w[e] for e in range(2 * n)]
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        bad = None
        e = 0
        for (_, _, n), live in zip(specs, lives):
            for q in range(n):
                stale = v[e + 2 * q + 1] != tag
                if live is not None:
                    stale = stale & live
                bad = stale if bad is None else bad | stale
            e += 2 * n
        return bad

    rid = mb.region_ids.get(specs[-1][0].region, -1)
    v = _spin(load_all, pending, mb.health, mb.alarm, rid, specs[-1][1], tag)
    outs, e = [], 0
    for _, _, n in specs:
        outs.append([v[e + 2 * q] for q in range(n)])
        e += 2 * n
    return outs


# ---------------------------------------------------------------- combine


@traced
def stage_combine(c, task):
    """(row t, q head h) = ``task``: the NCH chunk partials merged, o -> bf16,
    rms_norm_gated (o * rsqrt(mean(o^2) + eps) * sigmoid(og)) -> bf16 core."""
    L, nch = c["L"], c["NCH"]
    lane, wave = c["lane"], c["wave"]
    a = c["args"]
    cb, cbo = c["cb"], c["cbo"]
    t = task // NQ
    h = task % NQ
    j = t // L
    qvec = (t % L) * GQA + h % GQA
    grp = j * NKV + h // GQA
    ngrp = NKV * c["R"]
    kpw = (nch + WAVES - 1) // WAVES
    # The new-key owner is the last live production-aligned partition.
    _, _, _, t_hi, per, new_ch = context_tiles(c, j)
    if const_expr(L > 1):
        # Count every partition through the LAST new token, not only the
        # partition containing the first token of the verification batch.
        nact = (t_hi + per - 1) // per
    else:
        nact = new_ch + 1
    specs = [(c["qkv"], t * QKV_ROWS + OG_ROW0 + h * HD + lane, 1)]
    lives = [None]
    for k in range_constexpr(kpw):
        chk = fx.min(wave + WAVES * k, nch - 1)
        base = ((chk * ngrp + grp) * (GQA * L) + qvec) * PART
        specs += [(c["apart"], base, 2), (c["apart"], base + 2 + lane, 1)]
        live = (wave + WAVES * k) < nact
        lives += [live, live]
    # one batch: the partials are all out by the time the last one is
    got = poll_live(c, specs, lives)
    og = _f32(got[0][0])
    mm, ll, oo = [], [], []
    mw = fx.Float32(NEG)
    for k in range_constexpr(kpw):
        live = (wave + WAVES * k) < nact
        mk = live.select(_f32(got[1 + 2 * k][0]), fx.Float32(NEG))
        mm.append(mk)
        ll.append(live.select(_f32(got[1 + 2 * k][1]), fx.Float32(0.0)))
        oo.append(live.select(_f32(got[2 + 2 * k][0]), fx.Float32(0.0)))
        mw = fx.max(mw, mk)
    if const_expr(L > 1 and nch == 16):
        combine_triton16(c, mm, ll, oo, mw, t, h, og)
    else:
        lw = fx.Float32(0.0)
        ow = fx.Float32(0.0)
        for k in range_constexpr(kpw):
            wk = hw_exp(mm[k] - mw)
            lw = lw + ll[k] * wk
            if const_expr(L > 1):
                ow = ow + oo[k] * wk
            else:
                ow = ow + oo[k] * ll[k] * wk
        if lane == 0:
            fx.ptr_store(mw, cb + 2 * wave)
            fx.ptr_store(lw, cb + (2 * wave + 1))
        fx.ptr_store(ow, cbo + (wave * 64 + lane))
        gpu.barrier()
        if wave == 0:
            mt = fx.ptr_load(cb + 0)
            for w in range_constexpr(1, WAVES):
                mt = fx.max(mt, fx.ptr_load(cb + 2 * w))
            lt = fx.Float32(0.0)
            ot = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                sc = hw_exp(fx.ptr_load(cb + 2 * w) - mt)
                lt = lt + fx.ptr_load(cb + (2 * w + 1)) * sc
                ot = ot + fx.ptr_load(cbo + (w * 64 + lane)) * sc
            live = lt > fx.Float32(0.0)
            if const_expr(L > 1):
                den = live.select(lt, fx.Float32(1.0))
            else:
                den = live.select(lt + fx.Float32(1.0e-6), fx.Float32(1.0))
            o = bf16_round(live.select(ot / den, fx.Float32(0.0)))
            # Triton's D=64 gated-RMS kernel packs adjacent bf16 features into
            # one lane, adds their squares first, then reduces the 32 pair sums
            # with logical XOR offsets 16,8,4,2,1.  This wave holds one feature
            # per lane, so the equivalent physical-lane tree is 1 first and then
            # 32,16,8,4,2.  The older generic wave_sum paired i with i+32 first;
            # it has the same real sum but can move a bf16 output by one ULP.
            var = butterfly(o * o, (1, 32, 16, 8, 4, 2)) * fx.Float32(1.0 / HD)
            rstd = fx.Float32(1.0) / fx.Float32(
                fmath.sqrt(var + fx.Float32(c["qk_eps"]))
            )
            sig = fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(og * fx.Float32(-LOG2E)))
            y = (o * rstd) * sig
            bo.buffer_store(
                y.to(fx.BFloat16), rsrc(a["core"]), t * CORE + h * HD + lane
            )
        gpu.barrier()


def _segment_sum(values, strides):
    # Native reduce_segments: scalar sums use XOR 8,4,2,1. The [16,64]
    # output layout first reduces four segments per wave (2,1), then waves
    # (8,4). Keep these distinct FP32 addition trees.
    remaining = dict(enumerate(values))
    for stride in strides:
        remaining = {
            i: remaining[i] + remaining[i + stride] for i in remaining if not i & stride
        }
    return remaining[0]


@traced
def combine_triton16(c, mm, ll, oo, mw, t, h, og):
    """Native 16-segment reduction: one global maximum, raw PV, one divide."""
    lane, wave = c["lane"], c["wave"]
    red, cb = c["red"], c["cb"]
    if lane == 0:
        fx.ptr_store(mw, cb + wave)
    gpu.barrier()
    mt = fx.ptr_load(cb)
    for w in range_constexpr(1, WAVES):
        mt = fx.max(mt, fx.ptr_load(cb + w))
    for k in range_constexpr(2):
        ch = wave + WAVES * k
        scale = hw_exp(mm[k] - mt)
        fx.ptr_store(oo[k] * scale, red + ch * HD + lane)
        if lane == 0:
            fx.ptr_store(ll[k] * scale, red + 16 * HD + ch)
    gpu.barrier()
    if wave == 0:
        sums = [fx.ptr_load(red + 16 * HD + ch) for ch in range_constexpr(16)]
        outputs = [fx.ptr_load(red + ch * HD + lane) for ch in range_constexpr(16)]
        lt = _segment_sum(sums, (8, 4, 2, 1))
        ot = _segment_sum(outputs, (2, 1, 8, 4))
        live = lt > fx.Float32(0.0)
        den = live.select(lt, fx.Float32(1.0))
        o = bf16_round(live.select(ot / den, fx.Float32(0.0)))
        var = butterfly(o * o, (1, 32, 16, 8, 4, 2)) * fx.Float32(1.0 / HD)
        rstd = fx.Float32(1.0) / fx.Float32(fmath.sqrt(var + fx.Float32(c["qk_eps"])))
        sig = fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(og * fx.Float32(-LOG2E)))
        y = (o * rstd) * sig
        bo.buffer_store(
            y.to(fx.BFloat16), rsrc(c["args"]["core"]), t * CORE + h * HD + lane
        )
    gpu.barrier()
