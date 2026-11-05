# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ops.py
"""Device-side primitives of the mono kernels, gfx1250 (MI455X, wave32) port of
the gfx950 ``ops.py``: wave reductions and shuffles, the approximate hardware
math, bf16 / fp8 packing, buffer resources, the execution-model constants and
the kernel symbol naming.

Every public name and signature of the gfx950 module is kept. Where wave32
changes a semantic the contract is in ``README.md`` next to this file; in
short:

* A wave is ``WAVE`` = 32 lanes and ``WAVES`` = ``THREADS // 32``.
* A *logical wave64 value* is a ``(lo, hi)`` tuple: logical lane L lives in
  physical lane L % 32, element L // 32. ``xshfl`` / ``xred`` / ``butterfly`` /
  ``wave_sum`` / ``wave_max`` / ``wave_incl_scan`` / ``ballot`` /
  ``permlane_swap(32, ...)`` accept one and then reproduce the gfx950 64-lane
  result bit for bit (offset 32 is the in-lane combine of the two elements). On
  a plain value they work over the 32-lane wave and refuse offset 32.
* ``mfma_bf16`` / ``mfma_fp8`` are the gfx1250 WMMAs: 8 f32 accumulators a
  lane, lane l = column l % 16, rows 8 (l // 16) + v; ``row_sum`` reads that
  layout as ``acc_store`` writes it.
* Cache policies are gfx12 CPol (TH / SCOPE), not gfx9 SC0 / SC1 / NT.
"""

from __future__ import annotations

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32

try:
    from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
except ImportError:  # aiter builds before kernel_signature

    def kernel_signature(**params):
        return "_".join(f"{k}{v}" for k, v in params.items())


from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

WAVE = 32  # lanes a wave (gfx1250 runs wave32 only)

try:
    from . import plan as _plan

    THREADS = _plan.THREADS
except ImportError:  # imported outside the k2bundle package (the unit tests)
    _plan = None
    THREADS = 512

WAVES = THREADS // WAVE
if _plan is not None and getattr(_plan, "WAVES", WAVES) != WAVES:
    raise RuntimeError(
        f"plan.WAVES = {_plan.WAVES} but THREADS // {WAVE} = {WAVES}: port plan.py to"
        " wave32 (WAVES = THREADS // 32) before importing the gfx1250 ops"
    )

# A wave's 16x16 f32 accumulator: 8 a lane, ACC_WORDS a wave (as on gfx950:
# 4 a lane x 64), so a WAVES * ACC_WORDS reduction buffer holds every wave's.
ACC_PER_LANE = 8
ACC_WORDS = WAVE * ACC_PER_LANE

# gfx12 cache policy (LLVM CPol, GFX12): TH bits 0-2 (RT 0, NT 1, HT 2, ...),
# SCOPE bits 3-4 (CU 0, SE 8, DEV 16, SYS 24). SCOPE_DEV: coherent with every
# CU of the device; SCOPE_SYS: with peers over XGMI and the host.
CM_DEV = 16


CM_SYS = 24


# TH_NT: nontemporal (streaming). A weight a layer reads once streams past the
# caches it would only evict the step's reused data from. On gfx950 NT was 2;
# 2 is TH_HT (high temporal) on gfx12, the opposite hint.
CM_NT = 1


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``atom::<stem>_<params>``. The
    amdhsa assembler rejects ``::`` in a symbol, so it is Itanium-mangled, as
    aiter's ``_ZN5aiter...E`` kernels are."""
    name = f"{stem}_{kernel_signature(**params)}"
    return f"_ZN4atom{len(name)}{name}E"


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def _pair(v):
    return isinstance(v, tuple)


def _bits(v):
    return v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)


def _like(v, bits):
    return bits.bitcast(fx.Float32) if isinstance(v, fx.Float32) else type(v)(bits)


def _swap16(x, y):
    pr = rocdl.permlane16_swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        as_ir_value(fx.Int32(x)),
        as_ir_value(fx.Int32(y)),
        False,
        False,
    )
    return tuple(fx.Int32(llvm.extractvalue(T.i32, pr, [j])) for j in range(2))


def permlane_swap(off, x, y):
    """v_permlane16_swap on registers (x, y): off 16 trades x's lanes 16-31 with
    y's lanes 0-15 -> (x', y') = ([x 0-15, y 0-15], [x 16-31, y 16-31]).

    Off 32 needs logical wave64 pairs (``x = (x_lo, x_hi)``): it trades x's
    upper 32 logical lanes with y's lower ones, (x_lo, y_lo), (x_hi, y_hi), a
    renaming with no instruction."""
    if off == 32:
        if not (_pair(x) and _pair(y)):
            raise ValueError("permlane_swap(32, ...) on wave32 takes (lo, hi) pairs")
        return (x[0], y[0]), (x[1], y[1])
    assert off == 16, off
    return _swap16(x, y)


def lane_gather(v, src_lane):
    """Lane ``src_lane``'s value of v, a source lane a lane (ds_bpermute; the
    lane is taken mod 32)."""
    return fx.Int32(
        rocdl.ds_bpermute(T.i32, fx.Int32(src_lane) * 4, fx.Int32(v).ir_value())
    )


def readlane(v, src_lane):
    """Lane ``src_lane``'s value of v (wave-uniform)."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    r = fx.Int32(rocdl.readlane(T.i32, bits, fx.Int32(src_lane)))
    return r.bitcast(fx.Float32) if isinstance(v, fx.Float32) else r


def memrealtime():
    """The constant-rate device clock (gfx12: s_sendmsg_rtn_b64 GET_REALTIME;
    s_memrealtime is gone). 100 MHz, measured: ``test_misc.py``."""
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))


def unlikely(cond):
    """``cond`` marked cold (llvm.expect): LLVM lays the branch out of the hot
    path. A rare path's code inline among the hot stages costs them even when it
    never runs (the long-context selection: S = 16, 3k context +1.4 us)."""
    return fx.Boolean(
        llvm.call_intrinsic(
            ir.IntegerType.get_signless(1),
            "llvm.expect.i1",
            [as_ir_value(cond), as_ir_value(fx.Boolean(False))],
            [],
            [],
        )
    )


def fresh(v):
    """``v`` (a per-lane Int32) through an opaque move, here: the lane math built
    on it is computed at its use, where LLVM would otherwise hoist it out of a task
    loop or CSE it across stages into registers held for the whole kernel (spills,
    and a kernel with scratch is refused: ``runtime.widths``)."""
    return fx.Int32(
        llvm.inline_asm(
            T.i32, [fx.Int32(v).ir_value()], "v_mov_b32 $0, $1", "=v,v",
            has_side_effects=True,
        )
    )  # fmt: skip


def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def _hw_f32(name, x):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], [])
    )


def hw_rsq(x):
    """v_rsq_f32 (~1 ulp, no libm range fixups)."""
    return _hw_f32("llvm.amdgcn.rsq.f32", x)


def hw_rcp(x):
    return _hw_f32("llvm.amdgcn.rcp.f32", x)


def hw_exp2(x):
    return _hw_f32("llvm.amdgcn.exp2.f32", x)


def hw_exp(x):
    """Natural exponential through the native base-two instruction."""
    return hw_exp2(fx.Float32(x) * fx.Float32(1.4426950408889634))


def div_rn(x, d, r):
    """``x / d`` rounded to nearest, bit-identical to the IEEE division, given
    ``r = 1.0 / d`` from one true division per divisor (Markstein: a quotient
    within 1 ulp plus its exact fma residual times RN(1/d) rounds correctly)."""
    q = x * r
    return fx.Float32(fmath.fma(fx.Float32(fmath.fma(-q, d, x)), r, q))


def _dpp(x, ctrl, old=None, bound_ctrl=False):
    return fx.Int32(
        update_dpp_i32(x if old is None else old, x, ctrl, 0xF, 0xF, bound_ctrl)
    )


def xshfl(v, off):
    """Value of lane ``lane ^ off``: offsets 1-8 by one DPP row_xmask, 16 by
    v_permlanex16 (VALU latency, not LDS). A ``(lo, hi)`` pair also takes 32
    (the two elements swapped). Powers of two only."""
    assert off in (1, 2, 4, 8, 16, 32), off
    if _pair(v):
        if off == 32:
            return (v[1], v[0])
        return (xshfl(v[0], off), xshfl(v[1], off))
    if off == 32:
        raise ValueError("xshfl(v, 32) on wave32 needs a (lo, hi) pair")
    if off == 16:
        return v.shuffle_xor(16, WAVE)
    return _like(v, _dpp(_bits(v), 0x160 | off))


def xred(v, off, op):
    """op(v, lane ^ off) for a symmetric op; 16 takes both rows of one swap,
    op(row 0, row 1) as gfx950; 32 (pairs only) op(lo, hi), one value after."""
    if _pair(v):
        if off == 32:
            return op(v[0], v[1])
        return (xred(v[0], off, op), xred(v[1], off, op))
    if off == 32:
        raise ValueError("xred(v, 32) on wave32 needs a (lo, hi) pair")
    if off < 16:
        return op(v, xshfl(v, off))
    a, b = _swap16(_bits(v), _bits(v))
    return op(_like(v, a), _like(v, b))


def _add(a, b):
    return a + b


def butterfly(v, offsets, op=_add):
    """Butterfly reduction over the given xor offsets, in the given order.

    The order is the numerical contract: the aiter per-head norm reduces 32
    lanes with offsets 16, 8, 4, 2, 1, and matching it keeps rstd bit-exact.
    A (lo, hi) pair is a logical wave64 value (offset 32 allowed); the result is
    a pair until the offset-32 step, one value after it."""
    for off in offsets:
        v = xred(v, off, op)
    return v


def _full(v):
    return (32, 16, 8, 4, 2, 1) if _pair(v) else (16, 8, 4, 2, 1)


def wave_sum(v):
    """Sum over the 32-lane wave; over the 64 logical lanes of a (lo, hi) pair
    in the gfx950 order (32 first: bit-identical to its wave_sum)."""
    return butterfly(v, _full(v))


def traced(fn):
    """Apply flydsl's AST rewriting to a helper, so its ``while`` / ``if`` over
    traced values lower to scf ops. Kernels rewrite only their own body."""
    return ASTRewriter.transform(fn)


def wave_max(v):
    return butterfly(v, _full(v), fx.max)


@traced
def block_max(v, lane, wave, red):
    """Max of v over the CTA's WAVES waves; ``red`` needs WAVES words."""
    wv = wave_max(v)
    gpu.barrier()
    if lane == 0:
        fx.ptr_store(wv, red + wave)
    gpu.barrier()
    t = fx.ptr_load(red + 0)
    for w in range_constexpr(1, WAVES):
        t = fx.max(t, fx.ptr_load(red + w))
    gpu.barrier()
    return t


def row_shr(v, lane, k):
    """row_shr:k of an f32: lane - k of the same 16-lane row, else 0 (one DPP
    with bound_ctrl for a constant k)."""
    if isinstance(k, int):
        if k == 0:
            return v
        assert 0 < k < 16, k
        return _dpp(v.bitcast(fx.Int32), 0x110 | k, fx.Int32(0), True).bitcast(fx.Float32)
    moved = lane_gather(v.bitcast(fx.Int32), fx.max(lane - k, 0)).bitcast(fx.Float32)
    return fx.Float32((lane % 16 >= k).select(moved, fx.Float32(0.0)))


def _scan32(v, lane):
    """(inclusive prefix of an Int32 over the 32-lane wave, the wave total)."""
    blk = v
    for off in (1, 2, 4, 8, 16):
        o = xshfl(blk, off)
        v = v + ((lane & off) != 0).select(o, 0)
        blk = blk + o
    return v, blk


def wave_incl_scan(v, lane):
    """Inclusive prefix sum of an Int32 over the wave, in lane order (``lane``
    0-31). A (lo, hi) pair scans the 64 logical lanes -> (lo', hi')."""
    if _pair(v):
        lo, tot = _scan32(v[0], lane)
        hi, _ = _scan32(v[1], lane)
        return lo, hi + tot
    return _scan32(v, lane)[0]


@traced
def block_excl_scan(v, lane, wave, buf):
    """(exclusive prefix of v over the CTA in thread order, CTA total); ``buf``
    needs WAVES words."""
    incl = wave_incl_scan(v, lane)
    gpu.barrier()
    if lane == WAVE - 1:
        fx.ptr_store(incl, buf + wave)
    gpu.barrier()
    off = fx.Int32(0)
    tot = fx.Int32(0)
    for w in range_constexpr(WAVES):
        cnt = fx.ptr_load(buf + w)
        off = off + (w < wave).select(cnt, 0)
        tot = tot + cnt
    return incl - v + off, tot


def global_load(addr, words=1, nt=False):
    """``words`` i32 (1, 2 or 4) at the 64-bit global address ``addr``. A pool
    past 4 GiB is reached only this way: a buffer load's offset (index x stride
    included) is 32 bits. ``nt``: nontemporal (th:TH_LOAD_NT)."""
    i32 = ir.IntegerType.get_signless(32)
    ty = i32 if words == 1 else ir.VectorType.get([words], i32)
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr)))
    return llvm.LoadOp(
        ty, ptr.result, alignment=4 * words, nontemporal=nt or None
    ).result


def ld_i32(ptr, i):
    """Word ``i`` of a buffer, a plain Int32 load."""
    return fx.Int32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32))


def row_live(batch_ids, k):
    """Whether row k of the step is a request's: its ``batch_id_per_q_token``
    entry, -1 on a CUDA-graph pad row (wave-uniform for a uniform k). A pad row
    runs like any other but joins no decision another row's work depends on."""
    return uniform(ld_i32(batch_ids, k)) >= 0


def ld_dev(ptr, i, words=1):
    """``words`` i32 (1, 2 or 4) of a buffer from word ``i``, at device scope
    (scope:SCOPE_DEV: coherent across the device's caches)."""
    return bo.buffer_load(
        rsrc(ptr), i, vec_width=words, dtype=T.i32, cache_modifier=CM_DEV
    )


LOAD_BATCH = 4  # rounds of a strided loop whose loads go out together


@traced
def batched_rounds(tid, n, load, use, batch=LOAD_BATCH):
    """A CTA's strided loop over elements v = tid + THREADS i < n (n a Python
    int): ``load(v)`` of ``batch`` rounds issued before any ``use(v, loaded)``.
    A round whose ``use`` stores to global memory keeps the next round's loads
    behind it (they may alias), so the rounds' loads would go out in series. A
    round past n loads element n - 1 (in range) and uses nothing."""
    rounds = -(-n // THREADS)
    for b0 in range_constexpr(0, rounds, batch):
        idx = range(b0, min(b0 + batch, rounds))
        got = [load(fx.min(tid + THREADS * i, n - 1)) for i in idx]
        for i, g in zip(idx, got):
            v = tid + THREADS * i
            if v < n:
                use(v, g)


def rows_to_lds(tid, src, rows, row_words, dst, dst_row):
    """``rows`` rows of ``row_words`` words (a multiple of 4) read at device scope
    from address ``src`` -> LDS ``dst``, a row every ``dst_row`` words: 16 B a
    lane, the loads batched (``batched_rounds``)."""
    assert row_words % 4 == 0, row_words
    batched_rounds(
        tid, rows * row_words // 4,
        lambda v: fx.Vector(ld_dev(src, v * 4, 4)),
        lambda v, w: fx.ptr_store(
            w, dst + (v * 4 // row_words * dst_row + v * 4 % row_words)
        ),
    )  # fmt: skip


def mfma_bf16(a, b, c):
    """16x16x32 bf16 WMMA (v_wmma_f32_16x16x32_bf16): ``a`` / ``b`` 16 bf16 a
    lane (lane l: row / column l % 16, elements 8 j .. 8 j + 7 = K 16 j + 8 (l //
    16) ..), ``c`` 8 f32 (column l % 16, rows 8 (l // 16) + v)."""
    return _wmma(rocdl.wmma_f32_16x16x32_bf16, a, b, c)


def _wmma(op, a, b, c):
    return fx.Vector(
        op(T.vec(8, T.f32), as_ir_value(a), as_ir_value(b), as_ir_value(c)).result
    )


def acc_slot(wave, lane):
    """Word of wave ``wave``'s lane ``lane`` accumulator in a reduction buffer
    (``ACC_WORDS`` a wave), as ``row_sum`` reads it."""
    return (wave * WAVE + lane) * ACC_PER_LANE


def acc_store(red, acc, wave, lane):
    """A wave's WMMA accumulator -> its ``ACC_WORDS`` of ``red``."""
    fx.ptr_store(acc, red + acc_slot(wave, lane))


def row_sum(red, r, t, waves=None):
    """Row r, token t of the waves' 16x16 WMMA accumulators (``acc_store``'d:
    wave w's at ``red`` + (w * 32 + lane) * 8), summed in wave order from 0
    (``waves``: which, in order; all of them by default)."""
    tot = fx.Float32(0.0)
    for w in waves if waves is not None else range(WAVES):
        tot = tot + fx.ptr_load(red + (acc_slot(w, 16 * (r // 8) + t) + r % 8))
    return tot


def mfma_fp8(a, b, c):
    """16x16x64 fp8 x fp8 WMMA, unscaled: ``a`` / ``b`` 8 i32 a lane (32 E4M3:
    dwords 4 j .. 4 j + 3 = K 32 j + 16 (l // 16) ..), ``c`` 8 f32. A 4-dword
    operand is a 16x16x32 product (its upper K half zero)."""
    a, b = (_pad8(x) for x in (a, b))
    return _wmma(rocdl.wmma_f32_16x16x64_fp8_fp8, a, b, c)


def _pad8(x):
    x = fx.Vector(x)
    n = len(x)
    if n == 8:
        return x
    assert n == 4, n
    return x.shuffle(fx.Vector.filled(4, 0, fx.Int32), list(range(8)))


def ballot(pred):
    """The wave's lanes where ``pred`` holds, a bit a lane: an Int64 with bits
    32-63 zero. A (lo, hi) pair of predicates -> the 64 logical lanes' mask."""
    if _pair(pred):
        return (ballot(pred[1]) << 32) | ballot(pred[0])
    return fx.Int64(rocdl.ballot(T.i64, as_ir_value(pred)))


def popcount(x):
    return fx.Int32(fx.ctpop(x))


def lanes_below(bal, lane):
    """How many of the lanes in ballot ``bal`` are below ``lane``."""
    return popcount(bal & ((fx.Int64(1) << fx.Int64(lane)) - 1))


def load_ptr64(table, i):
    """Entry ``i`` of an i64 table in memory, wave-uniform (each half through
    ``uniform``): a peer's buffer base, a kernel argument block's pointer."""
    pw = fx.Vector(bo.buffer_load(rsrc(table), i * 2, vec_width=2, dtype=T.i32))
    return (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(fx.Uint32(uniform(pw[0])))


def fp8x8_bf16_pk(words):
    """Eight fp8 of two dwords -> bf16 WMMA operand (exact) by
    v_cvt_scale_pk8_bf16_fp8 (scale 1), eight values an instruction."""
    words = list(words)
    assert len(words) % 2 == 0, len(words)
    out = []
    for j in range(0, len(words), 2):
        src = fx.Vector.from_elements([fx.Int32(words[j]), fx.Int32(words[j + 1])], fx.Int32)
        p = rocdl.cvt_scale_pk8_bf16_fp8(
            T.vec(8, T.bf16), as_ir_value(src), as_ir_value(fx.Int32(0x7F7F7F7F)), 0
        )
        out.extend(fx.Vector(p).bitcast(fx.Int32)[d] for d in range(4))
    return fx.Vector.from_elements(out, fx.Int32).bitcast(fx.BFloat16)


def bf16_round(a):
    return fx.Float32(fx.Float32(a).to(fx.BFloat16))


def bf16_pair(a, b):
    """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Float32)[0]
    )


def lds_bytes(struct) -> int:
    """The LDS an ``fx.struct`` class takes, alignment included."""
    return struct.__dsl_size_of__()


def bf_lo(w):
    """The f32 of a packed bf16 pair word's low half."""
    return (w << 16).bitcast(fx.Float32)


def bf_hi(w):
    """The f32 of a packed bf16 pair word's high half."""
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def bf2_f32(w):
    """Packed bf16 pair word -> (f32 low, f32 high)."""
    return bf_lo(w), bf_hi(w)


def fp8_pack4(a, b, c, d):
    """Four f32 (already in range) -> one dword of four E4M3 bytes, RNE."""
    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))


def vm_wait():
    """Wait for this wave's outstanding global loads and stores: gfx12 splits
    gfx9's ``s_waitcnt vmcnt(0)`` into s_wait_loadcnt 0 + s_wait_storecnt 0."""
    rocdl.s_wait_loadcnt(0)
    rocdl.s_wait_storecnt(0)
