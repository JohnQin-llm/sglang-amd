"""gfx1250 (MI455X, wave32) device primitives for the K2 small-GEMM stages.

The names mirror ``k2bundle/ops/common/ops.py`` / ``plan.py`` (the gfx950 originals) and the lead's
``port/common/ops.py`` (same accumulator layout, ``acc_slot`` / ``row_sum``; same cache constants), so the stage
modules read the same; the bodies are rewritten for gfx1250:

* waves are 32 lanes: ``wave_sum`` / ``butterfly`` reduce over xor offsets (16, 8, 4, 2, 1); offsets below 16
  use the 16-lane DPP patterns, 16 a ``shuffle_xor`` over the 32-lane wave; ballots are 32 bits.
* cache policy is the gfx12 CPol: TH in bits [2:0], SCOPE in bits [4:3] (CU 0, SE 8, DEV 16, SYS 24).
  gfx950's NT = 2 would be TH_HT here, so it is not reused.  A poll without the DEV scope never sees another
  CTA's store (measured: the hand-off times out), so every hand-off load keeps it.
* the bf16 matrix op is ``v_wmma_f32_16x16x32_bf16``: A 16 rows x 32 K (v16 bf16 a lane), B 32 K x 16 columns
  (v16 bf16 a lane), D v8 f32 a lane, lane l holding column l % 16, rows 8 (l // 16) + v.
* ``s_waitcnt vmcnt`` is split on gfx12: ``drain_stores`` / ``drain_loads``.

The execution model (THREADS, WAVES, BLOCKS) matches ``port/consts.py`` (256 CTAs x 512 threads); the
``K2G_*`` environment overrides exist for the measurements in README.md only.
"""

from __future__ import annotations

import os

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

WAVE = 32
THREADS = 512
WAVES = THREADS // WAVE  # 16
BLOCKS = int(os.environ.get("K2G_BLOCKS", 256))  # 256: one CTA a CU; 512: two (MI355X q8's grid)
TP = 4
H = 8192

# gfx12 CPol bits (LLVM SIDefines CPol::TH / CPol::SCOPE)
TH_NT = 1
TH_WB = 3  # stores: write back past the scope's caches
SCOPE_DEV = 16
SCOPE_SYS = 24
CM_NT = int(os.environ.get("K2G_CMW", TH_NT))  # a weight read once a layer: nontemporal
CM_DEV = SCOPE_DEV  # loads of data another CTA of this GPU wrote
CM_SYS = SCOPE_SYS  # loads of data a peer GPU wrote
CM_DEV_ST = SCOPE_DEV | TH_WB  # aiter's gfx1250 device-scope store policy
CM_SYS_ST = SCOPE_SYS | TH_WB
# activation rows (x, FIN) another CTA wrote, staged to LDS; overridable for timing experiments only
CM_XROWS = int(os.environ.get("K2G_CMX", SCOPE_DEV))


def traced(fn):
    return ASTRewriter.transform(fn)


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def load_ptr64(table, i):
    """Entry ``i`` of an i64 table in memory, wave-uniform: a peer's buffer base."""
    pw = fx.Vector(bo.buffer_load(rsrc(table), i * 2, vec_width=2, dtype=T.i32))
    return (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(fx.Uint32(uniform(pw[0])))


def _hw_f32(name, x):
    return fx.Float32(llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], []))


def hw_rsq(x):
    """v_rsq_f32 (~1 ulp)."""
    return _hw_f32("llvm.amdgcn.rsq.f32", x)


def hw_exp2(x):
    """v_exp_f32 with no denormal range fixup."""
    return _hw_f32("llvm.amdgcn.exp2.f32", x)


def fma(a, b, c):
    from flydsl.expr import math as fmath

    return fx.Float32(fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


# ---------------------------------------------------------------- wave32 reductions
def xshfl(v, off):
    """Lane ``lane ^ off``'s v (off a power of two below 32): DPP inside a 16-lane row, a shuffle across."""
    assert off in (1, 2, 4, 8, 16), off
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    if off == 16:
        y = x.shuffle_xor(16, WAVE)
    elif off == 8:
        y = fx.Int32(update_dpp_i32(x, x, 0x118, 0xF, 0xC, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x108, 0xF, 0x3, False))
    elif off == 4:
        y = fx.Int32(update_dpp_i32(x, x, 0x114, 0xF, 0xA, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x104, 0xF, 0x5, False))
    elif off == 2:
        y = fx.Int32(update_dpp_i32(x, x, 0x4E, 0xF, 0xF, False))
    else:
        y = fx.Int32(update_dpp_i32(x, x, 0xB1, 0xF, 0xF, False))
    return y.bitcast(fx.Float32) if is_f else y


def _add(a, b):
    return a + b


def butterfly(v, offsets, op=_add):
    """Butterfly over the given xor offsets, in order (the order is the numerical contract)."""
    for off in offsets:
        v = op(v, xshfl(v, off))
    return v


WAVE_OFFSETS = (16, 8, 4, 2, 1)


def wave_sum(v):
    return butterfly(v, WAVE_OFFSETS)


def wave_max(v):
    return butterfly(v, WAVE_OFFSETS, fx.max)


def ballot(pred):
    """The wave's lanes where ``pred`` holds, a bit a lane (32 bits)."""
    return fx.Int32(rocdl.ballot(T.i32, as_ir_value(pred)))


def popcount(x):
    return fx.Int32(fx.ctpop(x))


def lanes_below(bal, lane):
    """How many lanes of ballot ``bal`` are below ``lane`` (lane < 32)."""
    return popcount(bal & ((fx.Int32(1) << fx.Int32(lane)) - fx.Int32(1)))


# ---------------------------------------------------------------- bf16 helpers
def bf16_round(a):
    return fx.Float32(fx.Float32(a).to(fx.BFloat16))


def bf16_pair(a, b):
    """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
    return fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Float32)[0]


def bf_lo(w):
    return (w << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def bf2_f32(w):
    return bf_lo(w), bf_hi(w)


# ---------------------------------------------------------------- WMMA
def wmma_bf16(a, b, c):
    """D = A B + C, ``v_wmma_f32_16x16x32_bf16``: a / b v16 bf16 (or v8 i32), c v8 f32."""
    a = a.bitcast(fx.BFloat16) if a.dtype is not fx.BFloat16 else a
    b = b.bitcast(fx.BFloat16) if b.dtype is not fx.BFloat16 else b
    return fx.Vector(
        rocdl.wmma_f32_16x16x32_bf16(
            fx.Vector.make_type(8, fx.Float32),
            as_ir_value(a),
            as_ir_value(b),
            as_ir_value(c),
            reuseA=False,
            reuseB=False,
        ).result
    )


ACC = 8  # f32 of a lane's WMMA accumulator
RED_FLOATS = WAVES * WAVE * ACC  # LDS a CTA's per-wave accumulators take


def acc_slot(w, r, t):
    """Float index in ``red`` of (row r, column t) of wave w's accumulator (stored at (w * 32 + lane) * 8)."""
    return (w * WAVE + 16 * (r // 8) + t) * ACC + r % 8


def row_sum(red, r, t, waves=None):
    """Row r, column t of the waves' 16x16 accumulators, summed in wave order from 0."""
    tot = fx.Float32(0.0)
    for w in waves if waves is not None else range(WAVES):
        tot = tot + fx.ptr_load(red + acc_slot(w, r, t))
    return tot


# ---------------------------------------------------------------- memory ordering
def drain_stores():
    rocdl.s_wait_storecnt(0)


def drain_loads():
    rocdl.s_wait_loadcnt(0)


@traced
def publish(put, a, i, v, who):
    """This CTA's plain stores drained, a CTA barrier, then (thread ``who``) the mailbox pair (a, i) := v."""
    drain_stores()
    gpu.barrier()
    if who:
        put(a, i, v)


def range_c(n):
    return range_constexpr(n)
