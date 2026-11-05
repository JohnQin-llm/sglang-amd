"""gfx1250 port of ``k2bundle/ops/stages/mxq.py``: the routed experts' activation quantization rules and the
gated activation, in one place (``moe`` calls only these), plus the wave32 helpers the stage needs.

Rules (a build parameter ``act``; ``ref.py`` has the same rules in torch, bit for bit):
  "a4"  aiter ``_mxfp4_quant_op`` (production compact-A4 and the TP4 K2's ``mxq.a4_code``; the rule the gfx1250
        FlyDSL expert GEMMs were proven exact against): code = max(expfield(bits(amax) + 0x200000) - 2, 0),
        E2M1 RNE saturating at 6, an all-zero group (code 0) packs to 0.
  "asm" the MI355X K2 rule (the TP8 k2bundle ``act_code`` / ``inter_code``, read from AITER's asm FMoE):
        code = e8m0_ceil(fp32(amax * fp32(1/6))), the intermediate's amax floored at 1e-6.
  "a8"  OCP MXFP8 E4M3 (the lead's EP1 decision, ``s2/``): code = max(expfield(amax) - 8, 0),
        x * 2^(127 - code) clamped to +-448, RNE to E4M3.

gfx1250 differences from gfx950:
  * ``v_cvt_scalef32_pk_fp4_f32`` (two f32 into a byte of the old dword) does not exist (the LLVM backend
    asserts "Invalid opcode"); gfx1250 has ``v_cvt_scalef32_pk8_fp4_f32``: eight f32 -> one E2M1 dword at the
    scale 2^(code - 127) (passed as the fp32 2^(code-127), i.e. it divides by the scale's power of two).
    ``probe_ops.py`` proves it equal to the torch E2M1 RNE rule (ties, saturation, signs).
  * E4M3 is ``v_cvt_pk_fp8_f32`` (two f32 per half-dword, OCP e4m3fn), after an explicit +-448 clamp.
  * wave32: butterflies are xor 16, 8, 4, 2, 1 (``shuffle_xor`` width 32).
"""

from __future__ import annotations

import struct

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import math as fmath
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


RCP6 = _f32(0x3E2AAAAB)  # fp32(1 / 6)
LOG2E = _f32(0x3FB8AA3B)  # fp32(log2 e)
INTER_FLOOR = _f32(0x358637BD)  # fp32(1e-6)
ACTS = ("a4", "asm", "a8")
WAVE = 32


def words_per8(act):
    """Packed dwords for 8 consecutive activations: one E2M1 dword, or two E4M3 dwords."""
    return 2 if act == "a8" else 1


# ---------------------------------------------------------------- wave32 / scalar helpers
def opaque(v):
    """``v`` through an opaque move: no fma contraction across it (the sums' order is the contract)."""
    return fx.Int32(llvm.inline_asm(T.i32, [fx.Float32(v).bitcast(fx.Int32).ir_value()], "v_mov_b32 $0, $1", "=v,v",
                                    has_side_effects=True)).bitcast(fx.Float32)


def xsum(v, offsets=(16, 8, 4, 2, 1)):
    """Butterfly sum (wave32): equal to halving the lanes [:off] + [off:2 off] in ``offsets`` order."""
    for off in offsets:
        v = v + v.shuffle_xor(off, WAVE)
    return v


def xmax(v, offsets):
    for off in offsets:
        v = fx.max(v, v.shuffle_xor(off, WAVE))
    return v


def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def bf16_round(a):
    return fx.Float32(fx.Float32(a).to(fx.BFloat16))


def bf_lo(w):
    return (fx.Int32(w) << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (fx.Int32(w) & fx.Int32(-65536)).bitcast(fx.Float32)


def exp2(x):
    return fx.Float32(llvm.call_intrinsic(T.f32, "llvm.amdgcn.exp2.f32", [fx.Float32(x).ir_value()], [], []))


def inv_sqrt_ieee(x):
    """1 / sqrt(x), both correctly rounded (llvm.sqrt.f32 and fdiv without fast-math flags)."""
    return fx.Float32(1.0) / fx.Float32(fmath.sqrt(fx.Float32(x)))


# ---------------------------------------------------------------- the rules
def _expfield(v):
    return (v.bitcast(fx.Int32) >> 23) & 0xFF


def e8m0_ceil(t):
    b = t.bitcast(fx.Int32)
    return ((b >> 23) & 0xFF) + ((b & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))


def code(amax, act, inter=False):
    """The E8M0 code of a 32-group of amax ``amax`` (fp32)."""
    if act == "a4":
        b = (amax.bitcast(fx.Int32) + fx.Int32(0x200000)) >> 23
        return fx.max((b & 0xFF) - 2, fx.Int32(0))
    if act == "asm":
        a = fx.max(amax, fx.Float32(INTER_FLOOR)) if inter else amax
        return e8m0_ceil(a * fx.Float32(RCP6))
    assert act == "a8", act
    return fx.max(_expfield(amax) - 8, fx.Int32(0))


def code_scale(c):
    """2^(c - 127) as fp32 (its exponent field)."""
    return (fx.Int32(c) << 23).bitcast(fx.Float32)


def pack8(vals, c, act):
    """Eight f32 (consecutive columns) -> E2M1 dword (``[w]``) or E4M3 dwords (``[w0, w1]``) at code ``c``;
    the first column in the low bits."""
    if act == "a8":
        inv = ((fx.Int32(254) - c) << 23).bitcast(fx.Float32)
        q = [fx.max(fx.min(v * inv, fx.Float32(448.0)), fx.Float32(-448.0)) for v in vals]
        out = []
        for h in range(2):
            w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, as_ir_value(q[4 * h]), as_ir_value(q[4 * h + 1]),
                                              as_ir_value(fx.Int32(0)), False))
            w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, as_ir_value(q[4 * h + 2]), as_ir_value(q[4 * h + 3]),
                                              as_ir_value(w), True))
            out.append(w)
        return out
    pk = fx.Int32(rocdl.cvt_scalef32_pk8_fp4_f32(T.i32, as_ir_value(fx.Vector.from_elements(vals, fx.Float32)),
                                                 as_ir_value(code_scale(c))))
    if act == "a4":
        pk = (c == 0).select(fx.Int32(0), pk)
    return [pk]


def silu_mul(g, u):
    """``g * sigmoid(g) * u`` as Triton's ``acc * tl.sigmoid(acc) * up`` (the served A16 path and the gfx1250
    FlyDSL GEMM): r = 1 / (exp2(-g log2e) + 1) with an IEEE division, then (g r) u."""
    r = fx.Float32(1.0) / (exp2(g * fx.Float32(-LOG2E)) + fx.Float32(1.0))
    return (g * r) * u
