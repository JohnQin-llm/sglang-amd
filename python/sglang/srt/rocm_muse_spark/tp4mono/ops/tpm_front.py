# SPDX-License-Identifier: Apache-2.0
"""TP4/EP4 K2 front half for M verify rows (one build per M): o_proj K=2048, the
attention seam, the joint router/pooler slice of this rank, the top-8 route and
the MXFP4 expert input.

Numerics follow the production TP4 target-verify path:
- o_proj partials bf16; the all-reduce sums in the custom AR's order (1-stage:
  ranks 0..3; 2-stage, at >= 160 KiB: the owner of row block p sums p, p+1, ..).
- seam: _fused_norm_residual's gfx950 code: per thread T (columns T + 512 j) a
  fixed tree, DPP row reduction, waves ((0+4)+(2+6))+((1+5)+(3+7)); bf16 post
  norm; alpha s + beta nb as two rounded products and an add; bf16 x.
- router + pooler: rows of the packed [pooler; router_hi; router_lo] weight,
  8 K splits of 1024, one 16x16x32 MFMA chain a split (bf16_splitk_gemm_partials,
  block_k 128); _joint_epilogue's split trees: pooler ((0+1)+(2+3))+((4+5)+(6+7)),
  router ((0+4)+(2+6))+((1+5)+(3+7)), logit = hi + lo.
- pre-expert norm in the epilogue's 4096 order; MXFP4 xe by aiter's rule.
Token rows are processed in MFMA tiles of 16.
"""

import math

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from sglang.srt.rocm_muse_spark.tp4mono.ops.common.ops import (
    CM_DEV,
    CM_NT,
    ballot,
    bf2_f32,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fresh,
    hw_exp2,
    hw_rsq,
    lanes_below,
    mfma_bf16,
    popcount,
    row_sum,
    rsrc,
    traced,
    xshfl,
)
from sglang.srt.rocm_muse_spark.tp4mono.ops.common.sync import publish
from sglang.srt.rocm_muse_spark.tp4mono.ops.stages import gemv, mxq

TP = 4
H = 8192
MH = 4096
E = 256
EPR = E // TP  # 64 router experts a rank
TOPK = 8
ROWS = 16
TT = 16  # token rows an MFMA tile
LOG2E = 1.0 / math.log(2.0)
ROUTE_EPS = 1e-15
AR_ONE_STAGE_BYTES = 160 * 1024  # custom AR: 1-stage below this at TP <= 4

CORE = 2048  # o_proj K a rank (32 q heads x 64)
XROW = 2048 + gemv.LDS_PAD  # bf16 a staged 2048-wide row
J_POOL_GROUPS = 16
J_GROUPS = J_POOL_GROUPS + 2
J_QUARTERS = 4
J_TASKS = J_GROUPS * J_QUARTERS  # 72
J_SPLITS = 8
J_SPLIT_K = H // J_SPLITS  # 1024
J_QK = H // J_QUARTERS  # 2048
POOL_N = MH // TP  # 1024 pooled columns a rank
XQ_W = MH // 8
XQ_G = MH // 32
SEAM_T = 512  # the boundary kernel's threads: thread T holds columns T + 512 j
SEAM_CTAS = 128


def tiles(m):
    return (m + TT - 1) // TT


def _fma(a, b, c):
    return fx.Float32(fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


def _opaque(v):
    """An f32 through a v_mov the compiler cannot see through."""
    return fresh(fx.Float32(v).bitcast(fx.Int32)).bitcast(fx.Float32)


def f32_of(w):
    return w.bitcast(fx.Float32)


def ar_start(t, m, row_bytes):
    """First rank of the custom AR's sum for row ``t``: 0 (1-stage), else its row block."""
    if m * row_bytes < AR_ONE_STAGE_BYTES:
        return 0
    return t // (m // TP)


RS_MIN_M = 32  # o_proj all-reduce as reduce-scatter + all-gather from this M (2-stage widths only)
AR_RED_CTAS = 128


def ar_rs(m, first=RS_MIN_M):
    """Whether the o_proj all-reduce runs as reduce-scatter to the row-block owner + all-gather of
    the owner's bf16 sum (half the link bytes of push-all, one more hop): from ``first`` rows
    (K2Build.oproj_rs_from; at the default, 32, every such width is a 2-stage one)."""
    return m >= first and (m * H * 2 >= AR_ONE_STAGE_BYTES or first < RS_MIN_M)


def owner_sum_start(m, row_bytes, rank):
    """First rank of a reduce-scatter owner's sum: the owner itself at 2-stage widths (the custom AR's
    order), rank 0 at 1-stage widths (its rank order 0..3, every row)."""
    return rank if m * row_bytes >= AR_ONE_STAGE_BYTES else 0


def ar_index(src, token, col, m):
    """Element of (src, token, col) in the o_proj AR region, blocked by 32-column group: a CTA's
    tile (16 tokens x its 32 columns) is one contiguous run."""
    return ((src * (H // 32) + col // 32) * m + token) * 32 + col % 32


def br_index(token, col, m):
    """Element of (token, col) in the branch region, blocked by 16-column group (an unpool task)."""
    return ((col // 16) * m + token) * 16 + col % 16


def sum_from(vals, start):
    """fp32 sum of the ranks' values in the order start, start + 1, .. (mod TP)."""
    if isinstance(start, int):
        acc = vals[start]
        for k in range(1, TP):
            acc = acc + vals[(start + k) % TP]
        return acc

    def pick(i):
        v = vals[TP - 1]
        for r in range(TP - 2, -1, -1):
            v = (i == r).select(vals[r], v)
        return v

    acc = pick(start % TP)
    for k in range(1, TP):
        acc = acc + pick((start + k) % TP)
    return acc


@traced
def stage_rows(tid, src, ldk, k0, k, rows, row0, m, xl, xrow, cm):
    """Token rows row0 .. row0 + rows of the bf16 [m, ldk] at ``src`` (columns k0 ..
    k0 + k) -> LDS (``xrow`` a row); rows past m are not loaded."""
    words = rows * k // 8
    for i in range_constexpr((words + 511) // 512):
        e = fx.min(tid + 512 * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        live = (tid + 512 * i < words) & (row0 + t < m)
        v = fx.Vector(bo.buffer_load(rsrc(src), ((row0 + t) * ldk + k0 + kk) // 2, vec_width=4, dtype=T.i32,
                                     cache_modifier=cm, mask=live))
        if live:
            fx.ptr_store(v, xl + (t * xrow + kk) // 2)
    gpu.barrier()


# ---------------------------------------------------------------- o_proj
def o_loads(c, rg):
    return gemv.loads(c["lane"], c["wave"], c["args"]["w_o"], rg * ROWS, CORE)


OPROJ_PASS = 4  # token tiles a pipelined pass (B operands of a pass held in registers)


@traced
def stage_oproj(c, bid):
    """CTA b: rows 32 b .. 32 b + 31 (row tiles 2 b, 2 b + 1) of this rank's o_proj shard against every
    16-token tile -> bf16 partials -> every rank's AR region. Production's a16w16 GEMM
    (NUM_KSPLIT 1, 16x16 MFMA) is one chain over K ascending: wave w holds K chunks w and
    w + 8 of both row tiles, and a chain's accumulators pass wave to wave through LDS.
    Token tiles are pipelined through the chain: tile tt takes chunk k at step tt + k on
    wave k % 8 (no two tiles of a pass of <= 8 share a wave at a step), each wave's token
    rows read from global memory ahead of its steps."""
    m, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    accs = c["fl"]  # [tile][j][lane][4] accumulators
    ops = [o_loads(c, 2 * bid + j) for j in range(2)]  # adjacent row tiles: 32 contiguous columns
    g = lane // ROWS
    nch = CORE // gemv.KCH  # 16
    ntiles = tiles(m)
    r_x = rsrc(c["args"]["core"])
    for p0 in range_constexpr(0, ntiles, OPROJ_PASS):
        tts = list(range(p0, min(ntiles, p0 + OPROJ_PASS)))
        tp = len(tts)
        xb = []  # step s of this wave: chunk wave + 8 (s // tp), tile tts[s % tp]
        for s_ in range_constexpr(2 * tp):
            tt = tts[s_ % tp]
            rows = min(TT, m - TT * tt)
            t = fx.min(lane % ROWS, rows - 1)
            kk = (wave + 8 * (s_ // tp)) * gemv.KCH + g * 8
            xb.append([fx.Vector(bo.buffer_load(r_x, ((TT * tt + t) * CORE + kk + 32 * qq) // 2, vec_width=4, dtype=T.i32))
                       for qq in range(4)])
        for tau in range_constexpr(tp + nch - 1):
            for s_ in range_constexpr(2 * tp):
                i = s_ // tp
                tl_ = s_ % tp
                wv = tau - 8 * i - tl_
                if const_expr(0 <= wv < 8):
                    if wave == wv:
                        k = wv + 8 * i
                        slot = tl_ * 2
                        for j in range_constexpr(2):
                            if const_expr(k == 0):
                                acc = fx.Vector.filled(4, 0.0, fx.Float32)
                            else:
                                acc = fx.Vector(fx.ptr_load(accs + ((slot + j) * 64 + lane) * 4, result_type=fx.Vector.make_type(4, fx.Float32)))
                            for qq in range_constexpr(4):
                                acc = mfma_bf16(ops[j][i][qq].bitcast(fx.BFloat16), xb[s_][qq].bitcast(fx.BFloat16), acc)
                            fx.ptr_store(acc, accs + ((slot + j) * 64 + lane) * 4)
            gpu.barrier()
        for tl_ in range_constexpr(tp):
            tt = tts[tl_]
            rows = min(TT, m - TT * tt)
            if tid < 2 * rows * (ROWS // 4):  # a wave writes contiguous (token, 32 columns) runs
                cq = tid % 8
                j = cq // 4
                rq = cq % 4
                t2 = tid // 8
                vs = [bf16_round(fx.ptr_load(accs + (((tl_ * 2 + j) * 64 + 16 * rq + t2) * 4 + d))) for d in range(4)]
                at = ar_index(c["rank"], TT * tt + t2, 32 * bid + 16 * j + 4 * rq, m)
                if const_expr(ar_rs(m, c["oproj_rs_from"])):
                    owner = (TT * tt + t2) // (m // TP)
                    for p in range_constexpr(TP):
                        if owner == p:
                            c["put_bf"](c["ar_peer"][p], at, vs)
                else:
                    for p in range_constexpr(TP):
                        c["put_bf"](c["ar_peer"][p], at, vs)
        gpu.barrier()


@traced
def _push_oproj_tile(c, bid, j, tts):
    """Row tile 2 b + j's bf16 partials of every token tile -> the AR region of every rank (or of the row
    block's owner in RS mode): a thread's 4 columns, the values and addresses stage_oproj pushes."""
    m, tid = c["S"], c["tid"]
    accs = c["fl"]
    for tl_ in range_constexpr(len(tts)):
        tt = tts[tl_]
        rows = min(TT, m - TT * tt)
        if tid < rows * (ROWS // 4):
            rq = tid % 4
            t2 = tid // 4
            vs = [bf16_round(fx.ptr_load(accs + (((tl_ * 2 + j) * 64 + 16 * rq + t2) * 4 + d))) for d in range(4)]
            at = ar_index(c["rank"], TT * tt + t2, 32 * bid + 16 * j + 4 * rq, m)
            if const_expr(ar_rs(m, c["oproj_rs_from"])):
                owner = (TT * tt + t2) // (m // TP)
                for p in range_constexpr(TP):
                    if owner == p:
                        c["put_bf"](c["ar_peer"][p], at, vs)
            else:
                for p in range_constexpr(TP):
                    c["put_bf"](c["ar_peer"][p], at, vs)


@traced
def stage_oproj_by_row_tile(c, bid):
    """stage_oproj with the CTA's two row tiles one after the other (latency_trim): row tile 2 b's weights
    and the core rows are issued first, row tile 2 b + 1's after them, so tile 2 b's chain and AR push run
    while tile 2 b + 1's weights stream and half of the AR bytes cross the links during the o_proj loads
    (at M8..24 the push-all transfer is link-bound, about 50 GB/s a link). Each chain has stage_oproj's K
    order and every pushed value and address is the same (bitwise)."""
    m, lane, wave = c["S"], c["lane"], c["wave"]
    accs = c["fl"]  # [tile][j][lane][4] accumulators
    g = lane // ROWS
    nch = CORE // gemv.KCH  # 16
    tts = list(range(tiles(m)))
    assert len(tts) <= OPROJ_PASS, m
    tp = len(tts)
    r_x = rsrc(c["args"]["core"])
    first = o_loads(c, 2 * bid)
    xb = []  # step s of this wave: chunk wave + 8 (s // tp), tile tts[s % tp]
    for s_ in range_constexpr(2 * tp):
        tt = tts[s_ % tp]
        rows = min(TT, m - TT * tt)
        t = fx.min(lane % ROWS, rows - 1)
        kk = (wave + 8 * (s_ // tp)) * gemv.KCH + g * 8
        xb.append([fx.Vector(bo.buffer_load(r_x, ((TT * tt + t) * CORE + kk + 32 * qq) // 2, vec_width=4, dtype=T.i32))
                   for qq in range(4)])
    rocdl.sched_barrier(0)  # issue order = return order: tile 2 b's chain waits only for its own loads
    second = o_loads(c, 2 * bid + 1)
    for j, ops in ((0, first), (1, second)):
        for tau in range_constexpr(tp + nch - 1):
            for s_ in range_constexpr(2 * tp):
                i = s_ // tp
                tl_ = s_ % tp
                wv = tau - 8 * i - tl_
                if const_expr(0 <= wv < 8):
                    if wave == wv:
                        k = wv + 8 * i
                        at = accs + ((tl_ * 2 + j) * 64 + lane) * 4
                        if const_expr(k == 0):
                            acc = fx.Vector.filled(4, 0.0, fx.Float32)
                        else:
                            acc = fx.Vector(fx.ptr_load(at, result_type=fx.Vector.make_type(4, fx.Float32)))
                        for qq in range_constexpr(4):
                            acc = mfma_bf16(ops[i][qq].bitcast(fx.BFloat16), xb[s_][qq].bitcast(fx.BFloat16), acc)
                        fx.ptr_store(acc, at)
            gpu.barrier()
        _push_oproj_tile(c, bid, j, tts)
    gpu.barrier()


# ---------------------------------------------------------------- o_proj AR owner reduce (RS mode)
@traced
def stage_ar_reduce(c, cta):
    """RS mode, CTA ``cta`` < AR_RED_CTAS: units (owned token, 1024-column slice); a thread's
    column pair: the four ranks' partials summed in the custom AR's 2-stage order (owner first)
    -> bf16 -> every rank's ARG."""
    m, tid, rank = c["S"], c["tid"], c["rank"]
    per = m // TP
    n = per * (H // 1024)
    for r in range_constexpr((n + AR_RED_CTAS - 1) // AR_RED_CTAS):
        u = cta + AR_RED_CTAS * r
        if u < n:
            t = rank * per + u // (H // 1024)
            col = (u % (H // 1024)) * 1024 + 2 * tid
            got = c["poll"]([(c["ar_own"], ar_index(src, t, col, m) // 2, 1) for src in range(TP)])
            start = owner_sum_start(m, H * 2, rank)
            lo = sum_from([bf_lo(g[0]) for g in got], start)
            hi = sum_from([bf_hi(g[0]) for g in got], start)
            for p in range_constexpr(TP):
                c["put_bf"](c["peer"]["arg"][p], t * H + col, [lo, hi])


# ---------------------------------------------------------------- seam (production order)
def _seam_part_index(token, q, T_):
    return ((token * 4 + q) * SEAM_T + T_) * 2


@traced
def _load_scalar(c, region, i):
    if c["tid"] == 0:
        word = c["poll"]([(region, i, 1)])[0][0]
        fx.ptr_store(f32_of(word), c["red"])
    gpu.barrier()
    v = fx.ptr_load(c["red"])
    gpu.barrier()
    return v


@traced
def _seam_norm(c, region, rstd_region, token, q, lo, hi, eps):
    """One 8192-wide sum of squares in _fused_norm_residual's order (this quarter's
    j = 4q .. 4q+3 give a thread's lo, hi; quarter 0 combines) -> rsq(sum/8192 + eps)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    if q > 0:
        c["put_words"](region, _seam_part_index(token, q, tid), [lo.bitcast(fx.Int32), hi.bitcast(fx.Int32)])
    if q == 0:
        got = c["poll"]([(region, _seam_part_index(token, qq, tid), 2) for qq in range(1, 4)])
        los = [lo] + [f32_of(got[k][0]) for k in range(3)]
        his = [hi] + [f32_of(got[k][1]) for k in range(3)]
        tot = ((los[0] + los[1]) + (los[2] + los[3])) + ((his[0] + his[1]) + (his[2] + his[3]))
        tot = butterfly(tot, (8, 4, 2, 1, 16, 32))
        if lane == 0:
            fx.ptr_store(tot, red + 8 + wave)
        gpu.barrier()
        if tid == 0:
            w = [fx.ptr_load(red + 8 + i) for i in range(8)]
            total = ((w[0] + w[4]) + (w[2] + w[6])) + ((w[1] + w[5]) + (w[3] + w[7]))
            c["put"](rstd_region, token, hw_rsq(total * fx.Float32(1.0 / H) + fx.Float32(eps)))
        gpu.barrier()
    return _load_scalar(c, rstd_region, token)


@traced
def _seam_norm_all(c, region, token, q, lo, hi, eps):
    """``_seam_norm`` with one exchange: every quarter publishes its thread partials and reads the other
    three, then forms the same total in the same order (quarters 0..3, the DPP butterfly, the wave tree)
    and the same rsq; no owner broadcast, so one round trip where ``_seam_norm`` takes two."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    c["put_words"](region, _seam_part_index(token, q, tid), [lo.bitcast(fx.Int32), hi.bitcast(fx.Int32)])
    got = c["poll"]([(region, _seam_part_index(token, (q + d) % 4, tid), 2) for d in range(1, 4)])

    def by_quarter(own, k):
        """The four quarters' value k in quarter order: this quarter's own, the others from ``got``
        (quarter (q + d) % 4 is got[d - 1])."""
        out = []
        for qq in range_constexpr(4):
            v = own
            for d in range_constexpr(1, 4):
                v = (q == (qq - d) % 4).select(f32_of(got[d - 1][k]), v)
            out.append(v)
        return out

    los, his = by_quarter(lo, 0), by_quarter(hi, 1)
    tot = ((los[0] + los[1]) + (los[2] + los[3])) + ((his[0] + his[1]) + (his[2] + his[3]))
    tot = butterfly(tot, (8, 4, 2, 1, 16, 32))
    if lane == 0:
        fx.ptr_store(tot, red + 8 + wave)
    gpu.barrier()
    w = [fx.ptr_load(red + 8 + i) for i in range(8)]
    total = ((w[0] + w[4]) + (w[2] + w[6])) + ((w[1] + w[5]) + (w[3] + w[7]))
    r = hw_rsq(total * fx.Float32(1.0 / H) + fx.Float32(eps))
    gpu.barrier()  # ``red`` is read before the next norm writes it
    return r


@traced
def seam_unit(c, u):
    """Unit u = 4 token + q; thread T: columns T + 512 j, j = 4q .. 4q + 3."""
    m, tid = c["S"], c["tid"]
    token, q = u // 4, u % 4
    a = c["args"]
    cols = [tid + SEAM_T * (4 * q + j) for j in range(4)]
    es = [token * H + col for col in cols]
    sv = [fx.Float32(bo.buffer_load(rsrc(a["stream_a"]), e, vec_width=1, dtype=T.f32)) for e in es]
    al = [fx.Float32(bo.buffer_load(rsrc(a["alpha_a"]), col, vec_width=1, dtype=T.f32)) for col in cols]
    be = [fx.Float32(bo.buffer_load(rsrc(a["beta_a"]), col, vec_width=1, dtype=T.f32)) for col in cols]
    odd = (tid % 2) == 1
    wf = []
    for col in cols:
        word = fx.Int32(bo.buffer_load(rsrc(a["w_fn"]), col // 2, vec_width=1, dtype=T.i32))
        wf.append(odd.select(bf_hi(word), bf_lo(word)))
    xs = []
    if const_expr(ar_rs(m, c["oproj_rs_from"])):
        ag = c["poll"]([(c["peer"]["arg_own"], (token * H + col) // 2, 1) for col in cols])
        for j in range_constexpr(4):
            lo_, hi_ = bf2_f32(ag[j][0])
            xs.append(odd.select(hi_, lo_))
    else:
        ar = c["poll"]([(c["ar_own"], ar_index(src, token, col, m) // 2, 1) for col in cols for src in range(TP)], batch=16)
        c["stamp"](17)
        start = ar_start(token, m, H * 2)
        for j in range_constexpr(4):
            vals = []
            for src in range_constexpr(TP):
                lo_, hi_ = bf2_f32(ar[j * TP + src][0])
                vals.append(odd.select(hi_, lo_))
            xs.append(bf16_round(sum_from(vals, start)))
    lo1 = _fma(xs[0], xs[0], xs[2] * xs[2])
    hi1 = _fma(xs[1], xs[1], xs[3] * xs[3])
    if const_expr(c["latency_trim"]):
        r1 = _seam_norm_all(c, c["msq1"], token, q, lo1, hi1, c["eps_attn"])
    else:
        r1 = _seam_norm(c, c["msq1"], c["rstd1"], token, q, lo1, hi1, c["eps_attn"])
    c["stamp"](18)
    # two rounded products and an add, as the boundary kernel's code (opaque moves stop fma contraction)
    ns = [_opaque(al[j] * sv[j]) + _opaque(be[j] * bf16_round(xs[j] * r1)) for j in range(4)]
    for j in range_constexpr(4):
        # device scope when the K2 tail (ffn_boundary_token) reads stream' back on another CTA
        bo.buffer_store(ns[j], rsrc(a["stream_out"]), es[j], cache_modifier=c["stream_store_cm"])
    lo2 = _fma(ns[0], ns[0], ns[2] * ns[2])
    hi2 = _fma(ns[1], ns[1], ns[3] * ns[3])
    if const_expr(c["latency_trim"]):
        r2 = _seam_norm_all(c, c["msq2"], token, q, lo2, hi2, c["eps_ffn"])
    else:
        r2 = _seam_norm(c, c["msq2"], c["rstd2"], token, q, lo2, hi2, c["eps_ffn"])
    c["stamp"](19)
    for j in range_constexpr(4):
        y = bf16_round((ns[j] * r2) * wf[j])
        y_next = xshfl(y, 1)  # column + 1 lives on thread T + 1
        if (tid % 2) == 0:
            pk = fx.Vector.from_elements([y, y_next], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
            bo.buffer_store(pk[0], rsrc(c["xbuf"]), es[j] // 2, cache_modifier=CM_DEV)
    publish(c["put"], c["xrdy"], token * 4 + q, fx.Int32(1), c["tid"] == 0)
    gpu.barrier()


@traced
def stage_seam(c, cta):
    """Seam CTA ``cta`` (0 .. SEAM_CTAS): units cta, cta + SEAM_CTAS, .. < 4 M."""
    m = c["S"]
    for r in range_constexpr((4 * m + SEAM_CTAS - 1) // SEAM_CTAS):
        u = cta + SEAM_CTAS * r
        if u < 4 * m:
            seam_unit(c, u)


# ---------------------------------------------------------------- FFN boundary (K2 tail)
# The FFN boundary of _fused_norm_residual_triton_kernel (input-scaled post-FFN norm, gated residual add, the
# next layer's attention pre-norm) for one token row on one CTA, in that kernel's gfx950 order: its thread T holds
# columns 8 T .. 8 T + 7 and 4096 + 8 T .. 4096 + 8 T + 7 (layout sizePerThread 8, reduced through a reorderable
# reshape that moves no data), register j = 0..15 in that order, and both sums of squares combine a thread's
# registers as the seam's quarters j = 4q .. 4q + 3 do, then the same lane butterfly and wave tree (ISA:
# small-kernels/RESULTS.md, item 1).
TAIL_COLS = 16  # a thread's columns


def _tail_columns(c):
    """This thread's first column of each run of 8 (registers 0..7, 8..15). From an opaque copy of the thread
    index (``fresh``): otherwise LLVM CSEs the tail's lane math with earlier stages' and holds it in registers
    through them (the expert stage was 2-5 us slower, results/timeline-v6)."""
    tid = fresh(c["tid"])
    return [8 * tid, H // 2 + 8 * tid]


def _quarter_squares(v):
    """A thread's per-quarter partial sums of squares as the seam forms them (j = 4q .. 4q + 3)."""
    lo = [_fma(v[4 * q], v[4 * q], v[4 * q + 2] * v[4 * q + 2]) for q in range(4)]
    hi = [_fma(v[4 * q + 1], v[4 * q + 1], v[4 * q + 3] * v[4 * q + 3]) for q in range(4)]
    return lo, hi


@traced
def _row_rsq(c, lo, hi, slot, eps):
    """rsq(sum / 8192 + eps) of the CTA's token row, summed as _seam_norm sums it: the thread's quarter
    partials ((0+1)+(2+3)) for lo and for hi, the DPP butterfly, then the waves ((0+4)+(2+6))+((1+5)+(3+7)).
    ``slot``: 8 free words of LDS ``red``; every thread computes the same value."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    tot = ((lo[0] + lo[1]) + (lo[2] + lo[3])) + ((hi[0] + hi[1]) + (hi[2] + hi[3]))
    tot = butterfly(tot, (8, 4, 2, 1, 16, 32))
    if lane == 0:
        fx.ptr_store(tot, red + slot + wave)
    gpu.barrier()
    w = [fx.ptr_load(red + slot + i) for i in range(8)]
    total = ((w[0] + w[4]) + (w[2] + w[6])) + ((w[1] + w[5]) + (w[3] + w[7]))
    return hw_rsq(total * fx.Float32(1.0 / H) + fx.Float32(eps))


def _f32_run(ptr, first, cm=0):
    """Eight fp32 elements first .. first + 7 (two 16-byte loads)."""
    out = []
    for h in range(2):
        v = fx.Vector(bo.buffer_load(rsrc(ptr), first + 4 * h, vec_width=4, dtype=T.f32, cache_modifier=cm))
        out += [fx.Float32(v[e]) for e in range(4)]
    return out


def _bf16_run(ptr, first):
    """Eight bf16 elements first .. first + 7 (first a multiple of 8) as f32 (one 16-byte load)."""
    v = fx.Vector(bo.buffer_load(rsrc(ptr), first // 2, vec_width=4, dtype=T.i32))
    out = []
    for e in range(4):
        lo, hi = bf2_f32(fx.Int32(v[e]))
        out += [lo, hi]
    return out


def ffn_boundary_loads(c, t):
    """Token t's boundary inputs that do not wait for the branch, issued before it arrives: alpha * stream'
    (stream' as the seam stored it; a rounded product, kept out of fma contraction), beta, the post-FFN norm
    weight and the next attention norm weight."""
    a = c["args"]
    out = dict(scaled=[], beta=[], w_post=[], w_next=[])
    for col in _tail_columns(c):
        stream = _f32_run(a["stream_out"], t * H + col, CM_DEV)
        alpha = _f32_run(a["alpha_f"], col)
        out["scaled"] += [_opaque(alpha[e] * stream[e]) for e in range(8)]
        out["beta"] += _f32_run(a["beta_f"], col)
        out["w_post"] += _bf16_run(a["w_post_ffn"], col)
        out["w_next"] += _bf16_run(a["w_next_attn"], col)
    return out


@traced
def ffn_boundary_token(c, t, pre):
    """Token t's FFN boundary on this CTA: the unpooled branch from this rank's BR (every rank's slices),
    post = bf16(branch * w_post), post * rsq -> bf16, s'' = fma(beta, post, alpha s') (the boundary kernel's
    contraction, ``v_pk_fma``), then the next attention norm bf16((s'' * rsq) * w_next) -> stream_out (fp32),
    next_in (bf16)."""
    m, a = c["S"], c["args"]
    cols = _tail_columns(c)
    br = c["peer"]["br_own"]  # a run of 8 columns is 4 consecutive (bf16 pair, tag) words
    specs = [(br, br_index(t, col + 4 * h, m) // 2, 2) for col in cols for h in range(2)]
    if const_expr(c["tail_two_phase"]):
        c["poll"](specs[:1], sleep=c["tail_poll_sleep"])
    got = c["poll"](specs, sleep=c["tail_poll_sleep"])
    post = []
    for words in got:
        for w in words:
            lo, hi = bf2_f32(w)
            post += [lo, hi]
    post = [bf16_round(post[j] * pre["w_post"][j]) for j in range(TAIL_COLS)]
    lo1, hi1 = _quarter_squares(post)
    r1 = _row_rsq(c, lo1, hi1, 8, c["eps_post_ffn"])
    ns = [_fma(pre["beta"][j], bf16_round(post[j] * r1), pre["scaled"][j]) for j in range(TAIL_COLS)]
    for r in range_constexpr(2):
        for h in range_constexpr(2):
            j = 8 * r + 4 * h
            bo.buffer_store(fx.Vector.from_elements(ns[j:j + 4], fx.Float32), rsrc(a["stream_out"]),
                            t * H + cols[r] + 4 * h)
    lo2, hi2 = _quarter_squares(ns)
    r2 = _row_rsq(c, lo2, hi2, 16, c["eps_next_attn"])
    y = [(ns[j] * r2) * pre["w_next"][j] for j in range(TAIL_COLS)]
    for r in range_constexpr(2):
        words = [bf16_pair(y[8 * r + 2 * e], y[8 * r + 2 * e + 1]).bitcast(fx.Int32) for e in range(4)]
        bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(a["next_in"]), (t * H + cols[r]) // 2)


# ---------------------------------------------------------------- router / pooler
def joint_row0(c, grp, tile):
    rank = c["rank"]
    pool = rank * POOL_N + grp * 64 + tile * ROWS
    g = grp - J_POOL_GROUPS
    router = 4096 + (tile // 2) * E + rank * EPR + g * 32 + (tile % 2) * ROWS
    return (grp >= J_POOL_GROUPS).select(router, pool)


def joint_loads(c, grp, q):
    """Wave w: tile w // 2, split 2 q + w % 2: its 8 K chunks of 128 (all in flight)."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_joint"])
    row = joint_row0(c, grp, wave // 2) + lane % ROWS
    g = lane // ROWS
    sp = 2 * q + wave % 2
    out = []
    for j in range(J_SPLIT_K // gemv.KCH):
        base = (row * H + sp * J_SPLIT_K + j * gemv.KCH + g * 8) // 2
        out.append([fx.Vector(bo.buffer_load(r_w, base + 16 * qq, vec_width=4, dtype=T.i32, cache_modifier=CM_NT))
                    for qq in range(4)])
    return out


def _ppart(c, grp, split, tile, row, t):
    return (((grp * J_SPLITS + split) * 4 + tile) * ROWS + row) * c["S"] + t


@traced
def joint_compute(c, grp, q, ops):
    m, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    g = lane // ROWS
    tile = wave // 2
    split = 2 * q + wave % 2
    part = c["fl"]  # [tile][split 0/1][row][m] then the owner's sums [tile][row][m]
    r_x = rsrc(c["x_src"])

    def x_ready(tt):
        rows = min(TT, m - TT * tt)
        if tid < rows:
            c["poll"]([(c["xrdy"], (TT * tt + tid) * 4 + q, 1)])
        gpu.barrier()

    def x_loads(tt):
        """Token tile tt's rows, this quarter's 2048 columns: a thread's dwordx4 words (all live)."""
        rows = min(TT, m - TT * tt)
        out = []
        for i in range_constexpr(rows * J_QK // 8 // 512):
            e = tid + 512 * i
            out.append(fx.Vector(bo.buffer_load(r_x, ((TT * tt + e // (J_QK // 8)) * H + q * J_QK + e % (J_QK // 8) * 8) // 2,
                                                vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)))
        return out

    x_ready(0)
    c["stamp"](20)
    xv = x_loads(0)
    for tt in range_constexpr(tiles(m)):
        rows = min(TT, m - TT * tt)
        for i in range_constexpr(len(xv)):
            e = tid + 512 * i
            fx.ptr_store(xv[i], c["xl"] + ((e // (J_QK // 8)) * XROW + e % (J_QK // 8) * 8) // 2)
        gpu.barrier()
        if const_expr(tt + 1 < tiles(m)):  # the next tile's rows load during this tile's MFMAs
            x_ready(tt + 1)
            xv = x_loads(tt + 1)
        t = fx.min(lane % ROWS, rows - 1)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for j in range_constexpr(J_SPLIT_K // gemv.KCH):
            kk = (wave % 2) * J_SPLIT_K + j * gemv.KCH + g * 8
            for qq in range_constexpr(4):
                xb = fx.Vector(fx.ptr_load(c["xl"] + (t * XROW + kk + 32 * qq) // 2, result_type=fx.Vector.make_type(4, fx.Int32)))
                acc = mfma_bf16(ops[j][qq].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc)
        tk = TT * tt + lane % ROWS
        live = lane % ROWS < rows
        for i in range_constexpr(4):
            row = 4 * (lane // ROWS) + i
            if live:
                if q == 0:
                    fx.ptr_store(acc[i], part + ((tile * 2 + wave % 2) * ROWS + row) * m + tk)
                if q > 0:
                    c["put"](c["ppart"], _ppart(c, grp, split, tile, row, tk), acc[i])
        gpu.barrier()
    gpu.barrier()
    c["stamp"](15)
    if q == 0:
        n = 4 * ROWS * m  # (tile, row, token) owned values
        sums = part + 8 * ROWS * m  # [tile][row][token], past the split partials
        rounds = (n + 511) // 512
        idxs = [fx.min(tid + 512 * r, n - 1) for r in range(rounds)]
        loc = [(idx // (ROWS * m), (idx // m) % ROWS, idx % m) for idx in idxs]
        # every round's remote splits in one poll (one round trip, not one a round)
        got = c["poll"]([(c["ppart"], _ppart(c, grp, sp, ot, orow, otk), 1) for ot, orow, otk in loc
                         for sp in range(2, J_SPLITS)], batch=48)
        for r in range_constexpr(rounds):
            ot, orow, otk = loc[r]
            p = [fx.ptr_load(part + ((ot * 2 + sp) * ROWS + orow) * m + otk) for sp in range(2)]
            p += [f32_of(got[r * (J_SPLITS - 2) + k][0]) for k in range(J_SPLITS - 2)]
            pool_v = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]))
            rout_v = ((p[0] + p[4]) + (p[2] + p[6])) + ((p[1] + p[5]) + (p[3] + p[7]))
            v = (grp >= J_POOL_GROUPS).select(rout_v, pool_v)
            if tid + 512 * r < n:
                fx.ptr_store(v, sums + idxs[r])
        gpu.barrier()
        c["stamp"](16)
        if grp < J_POOL_GROUPS:
            n4 = 4 * (ROWS // 4) * m
            for r in range_constexpr((n4 + 511) // 512):
                i = tid + 512 * r
                if i < n4:  # token-major: a wave writes runs of 64 contiguous columns
                    tk = i // 16
                    cq = i % 16  # column quad of the group's 64
                    vs = [bf16_round(fx.ptr_load(sums + ((4 * cq + d) * m + tk))) for d in range(4)]
                    col = c["rank"] * POOL_N + grp * 64 + 4 * cq
                    for pr in range_constexpr(TP):
                        c["put_bf"](c["peer"]["pool"][pr], tk * MH + col, vs)
        if grp >= J_POOL_GROUPS:
            n2 = 2 * ROWS * m
            for r in range_constexpr((n2 + 511) // 512):
                i = tid + 512 * r
                if i < n2:  # token-major: a wave writes runs of 32 contiguous experts
                    tk = i // 32
                    eo = i % 32
                    logit = fx.ptr_load(sums + (eo * m + tk)) + fx.ptr_load(sums + (eo * m + tk + n2))
                    e = c["rank"] * EPR + (grp - J_POOL_GROUPS) * 32 + eo
                    for pr in range_constexpr(TP):
                        c["put"](c["peer"]["rlog"][pr], tk * E + e, logit)
    gpu.barrier()


# ---------------------------------------------------------------- route
def _okey(v):
    b = v.bitcast(fx.Int32)
    neg = b < fx.Int32(0)
    return fx.Uint32(neg.select(~b, b ^ fx.Int32(-2147483648)))


def bo_ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


@traced
def _route_token(c, t, red):
    """Token t on one wave: 256 logits -> sigmoid(l mult) + bias -> top-8 (ties: lower
    id) ranked by biased score -> weights s / (sum + 1e-15) -> ROUTE."""
    lane = c["lane"]
    a = c["args"]
    per = E // 64
    bias = [bo_ld_f32(a["expert_bias"], lane + 64 * i) for i in range(per)]
    got = c["poll"]([(c["peer"]["rlog_own"], t * E + lane + 64 * i, 1) for i in range(per)])
    keys, sigs = [], []
    for i in range_constexpr(per):
        raw = f32_of(got[i][0])
        bo.buffer_store(raw, rsrc(c["logit_dbg"]), t * E + lane + 64 * i, cache_modifier=CM_DEV)
        lg = raw * fx.Float32(c["router_output_multiplier"])
        sg = fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(lg * fx.Float32(-LOG2E)))
        sc = sg + bias[i]
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
            fx.ptr_store(fx.Int32(lane + 64 * i).bitcast(fx.Float32), red + (pos * 3 + 1))
            fx.ptr_store(sigs[i], red + (pos * 3 + 2))
        eq_before = eq_before + popcount(eqb)
        sel_before = sel_before + popcount(selb)
    qd = fx.min(lane, TOPK - 1)
    mk = fx.Uint32(fx.ptr_load(red + qd * 3).bitcast(fx.Int32))
    me = fx.ptr_load(red + (qd * 3 + 1)).bitcast(fx.Int32)
    ms = fx.ptr_load(red + (qd * 3 + 2))
    rank = fx.Int32(0)
    for j in range_constexpr(TOPK):
        ok = fx.Uint32(fx.ptr_load(red + j * 3).bitcast(fx.Int32))
        oe = fx.ptr_load(red + (j * 3 + 1)).bitcast(fx.Int32)
        before = (ok > mk) | ((ok == mk) & (oe < me))
        rank = rank + before.select(fx.Int32(1), fx.Int32(0))
    if lane < TOPK:
        fx.ptr_store(ms, red + (64 + rank))
    total = fx.Float32(0.0)
    for j in range_constexpr(TOPK):
        total = total + fx.ptr_load(red + (64 + j))
    inv = fx.Float32(c["route_scale"]) / (total + fx.Float32(ROUTE_EPS))
    if lane < TOPK:
        c["put"](c["route"], t * 2 * TOPK + rank, fx.min(fx.max(me, 0), fx.Int32(E - 1)))
        c["put"](c["route"], t * 2 * TOPK + TOPK + rank, ms * inv)


@traced
def stage_route(c, cta):
    """Route CTA ``cta``: tokens 8 cta + wave (each wave its own LDS scratch)."""
    t = cta * 8 + c["wave"]
    if t < c["S"]:
        _route_token(c, t, c["red"] + c["wave"] * 128)
    gpu.barrier()


# ---------------------------------------------------------------- expert input
@traced
def stage_xq(c, t):
    """Token t: pooled row -> pre-expert norm in the joint epilogue's order -> bf16 xe
    -> aiter-rule MXFP4 words / codes (device scope), then XQRDY[t]."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    tt = fx.min(tid, 255)  # threads 0..255 play the epilogue's 256 threads
    w_pre = None
    if const_expr(c["latency_trim"]):  # in flight during the poll, not one round trip a rep behind the xe stores
        w_pre = [fx.Vector(bo.buffer_load(rsrc(a["w_pre"]), (rep * 1024 + tt * 4) // 2, vec_width=2, dtype=T.i32))
                 for rep in range(4)]
    got = c["poll"]([(c["peer"]["pool_own"], (t * MH + rep * 1024 + tt * 4) // 2, 2) for rep in range(4)])
    xs = []
    for rep in range_constexpr(4):
        xs.append([bf_lo(got[rep][0]), bf_hi(got[rep][0]), bf_lo(got[rep][1]), bf_hi(got[rep][1])])
    lo = [_fma(xs[r][0], xs[r][0], xs[r][2] * xs[r][2]) for r in range(4)]
    hi = [_fma(xs[r][1], xs[r][1], xs[r][3] * xs[r][3]) for r in range(4)]
    th = ((lo[0] + lo[1]) + (lo[2] + lo[3])) + ((hi[0] + hi[1]) + (hi[2] + hi[3]))
    th = butterfly(th, (8, 4, 2, 1, 16, 32))
    if (lane == 0) & (wave < 4):
        fx.ptr_store(th, red + wave)
    gpu.barrier()
    tot = (fx.ptr_load(red + 0) + fx.ptr_load(red + 2)) + (fx.ptr_load(red + 1) + fx.ptr_load(red + 3))
    r = hw_rsq(tot * fx.Float32(1.0 / MH) + fx.Float32(c["eps_pre"]))
    if tid < 256:
        for rep in range_constexpr(4):
            col = rep * 1024 + tid * 4
            if const_expr(w_pre is not None):
                wv = w_pre[rep]
            else:
                wv = fx.Vector(bo.buffer_load(rsrc(a["w_pre"]), col // 2, vec_width=2, dtype=T.i32))
            ws = [bf_lo(wv[0]), bf_hi(wv[0]), bf_lo(wv[1]), bf_hi(wv[1])]
            ys = [(xs[rep][e] * r) * ws[e] for e in range(4)]
            pk = fx.Vector.from_elements(ys, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
            fx.ptr_store(pk, c["xl"] + col // 2)
            bo.buffer_store(pk, rsrc(c["xe_dbg"]), (t * MH + col) // 2, cache_modifier=CM_DEV)
    gpu.barrier()
    wd = fx.Vector(fx.ptr_load(c["xl"] + tid * 4, result_type=fx.Vector.make_type(4, fx.Int32)))
    xe = []
    for d in range_constexpr(4):
        xe.append(bf_lo(wd[d]))
        xe.append(bf_hi(wd[d]))
    am = fx.Float32(0.0)
    for v in xe:
        am = fx.max(am, abs(v))
    am = butterfly(am, (2, 1), fx.max)
    code = mxq.a4_code(am)
    pk = mxq.pack8_a4(xe, code)
    bo.buffer_store(pk, rsrc(c["xqb"]), t * XQ_W + tid, cache_modifier=CM_DEV)
    if lane % 4 == 0:
        bo.buffer_store(code, rsrc(c["xscb"]), t * XQ_G + tid // 4, cache_modifier=CM_DEV)
    publish(c["put"], c["xqrdy"], t, fx.Int32(1), tid == 0)
    gpu.barrier()
