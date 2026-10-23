# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The routed experts' MXFP4 arithmetic, in one place: the activation and
intermediate quantization rules and the gated activation of production's
expert kernel. ``moe`` calls only these, so a rule is swapped here.

Production (Quark OCP-MXFP4 W4A4, ``AITER_MXFP4_MXFP4``) runs every decode
width T = 1..8 -- 8 T routes, top-1 each -- through AITER's FLAT 1-stage asm
kernel (``fmoe_bf16_pertokenMXfp4_g1u1_flat_vs_silu_16x32`` for T <= 4,
``..._novs_silu_16x128`` for T >= 5; the ``muse_spark_mxfp4_tuned_fmoe.csv``
configs). Its arithmetic, read from the kernels' disassembly:

- a1, the bf16 expert input, per 32 consecutive columns: amax = max |x| (the
  bf16 bits), t = amax * fp32(1/6) (a rounded multiply, not a division), the
  E8M0 code = t's exponent, plus one if any mantissa bit is set (no clamp; an
  all-zero group gets 0), x -> E2M1 by ``v_cvt_scalef32_pk_fp4_*`` at the scale
  2^(code - 127) (round to nearest even, saturating at 6);
- gate / up: one fp32 scaled-MFMA chain over K ascending, then silu(g) u with
  the raw hardware exp2 and rcp (``silu_mul``);
- a2, the fp32 intermediate, per 32 columns: the same rule with amax floored
  at 1e-6;
- down: one fp32 chain, then one bf16 rounding a route element (``moe``).
"""

import struct

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

from sglang.srt.rocm_muse_spark.mono.ops.common.ops import hw_exp2, hw_rcp


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


RCP6 = _f32(0x3E2AAAAB)  # fp32(1 / 6)
LOG2E = _f32(0x3FB8AA3B)  # fp32(log2 e)
INTER_FLOOR = _f32(0x358637BD)  # fp32(1e-6): the intermediate's amax floor


def e8m0_ceil(t):
    """The E8M0 code of the smallest power of two >= t (t >= 0, fp32):
    t's exponent field, plus one if any mantissa bit is set."""
    b = t.bitcast(fx.Int32)
    e = (b >> 23) & 0xFF
    return e + ((b & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))


def act_code(amax):
    """a1's code for a 32-group of amax ``amax`` (an fp32 holding a bf16)."""
    return e8m0_ceil(amax * fx.Float32(RCP6))


def inter_code(amax):
    """a2's code for a 32-group of fp32 amax ``amax``."""
    return e8m0_ceil(fx.max(amax, fx.Float32(INTER_FLOOR)) * fx.Float32(RCP6))


def code_scale(code):
    """The fp32 a ``v_cvt_scalef32_*`` takes for code ``code``: 2^(code - 127)
    (its exponent field)."""
    return (code << 23).bitcast(fx.Float32)


def silu_mul(g, u):
    """silu(g) u as the expert kernel: t = (-g) log2e, r = rcp(exp2(t) + 1),
    (g r) u -- raw v_exp_f32 / v_rcp_f32, every step an fp32 rounding."""
    t = g * fx.Float32(-LOG2E)
    r = hw_rcp(hw_exp2(t) + fx.Float32(1.0))
    return (g * r) * u


def pk_fp4_f32(old, a, b, scale, j):
    """Two f32 / scale -> E2M1 nibbles (a low) into byte j of ``old``."""
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32,
            as_ir_value(fx.Int32(old)),
            as_ir_value(fx.Float32(a)),
            as_ir_value(fx.Float32(b)),
            as_ir_value(fx.Float32(scale)),
            j,
        )
    )


def pack8(vals, scale):
    """Eight f32 (consecutive columns) / scale -> one word of E2M1, the first
    column in the low nibble of byte 0."""
    pk = fx.Int32(0)
    for j in range(4):
        pk = pk_fp4_f32(pk, vals[2 * j], vals[2 * j + 1], scale, j)
    return pk
