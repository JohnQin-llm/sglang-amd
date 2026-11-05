# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/mx.py
"""FP8 / MX arithmetic the mono kernels share, gfx1250 (wave32) port: the E4M3
range, E8M0 scale codes, the byte of a scale in aiter's ``shuffle_scale``
layout and in the gfx1250 WMMA scale order, an FP4 weight tile's operand loads
and the scaled WMMA that takes them.

An E8M0 code is a biased exponent; ``pow2(code)`` is the scale it stands for.

gfx1250 operand layout (V_WMMA_SCALE_F32_16X16X128_F8F6F4, hardware-verified):
lane l holds row (A) / column (B) l % 16; its 16-byte chunk j holds K run
2 j + l // 16 (32 fp4, low nibble the even k; 16 fp8). FP4 is 8 dwords a lane,
FP8 16. A scale operand is one dword a lane, byte j = E8M0 of K block j (32 k)
of that row / column; opsel 0 reads lanes 0-15's dwords, opsel 1 lanes 16-31's.
D: 8 f32 a lane, column l % 16, rows 8 (l // 16) + v.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import rocdl
from flydsl.expr.typing import T

try:
    from .ops import CM_NT
except ImportError:  # imported outside the k2bundle package (the unit tests)
    from ops import CM_NT

FP8_MAX = 448.0
# the f8f6f4 WMMA's operand formats (fmtA / fmtB; gfx950's cbsz / blgp codes)
FP8, FP4 = 0, 4
UNIT_SCALE = 127  # the E8M0 code of 1.0


def clamp_fp8(v):
    """``v`` clamped to the E4M3 range."""
    return fx.max(fx.min(v, fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX))


def code_ceil(v):
    """Biased exponent of the smallest power of two >= ``v`` (v >= 0, finite),
    clamped to [1, 254]: the codes a scale may take."""
    bits = v.bitcast(fx.Int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(
        fx.Int32(1), fx.Int32(0)
    )
    return fx.max(fx.min(e, fx.Int32(254)), fx.Int32(1))


def pow2(code):
    """The fp32 power of two a biased exponent stands for."""
    return (code << 23).bitcast(fx.Float32)


def rcp_pow2(code):
    """1 / ``pow2(code)``, exact (code in [1, 253]): x times it is x / scale
    bit for bit, a multiply where the division is a dozen instructions."""
    return ((254 - code) << 23).bitcast(fx.Float32)


def scale_index(row, col, cols):
    """Byte of e8m0 scale (row, col) in aiter's ``shuffle_scale`` layout:
    ``view(rows/32, 2, 16, cols/8, 2, 4).permute(0, 3, 5, 2, 4, 1)``."""
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, e = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * (cols // 8) + c8) * 4 + e) * 16 + b) * 2 + d) * 2 + a


def scale_index_gfx1250(row, col, cols):
    """Byte of e8m0 scale (row, col) in the gfx1250 WMMA scale order
    (``layout_gfx1250.shuffle_scale_gfx1250``): ``view(rows/16, 16, cols/8, 2,
    4).permute(0, 2, 3, 1, 4)`` -- per 16-row group and pair of 128-k steps, 32
    dwords, dword (parity * 16 + row % 16) = that row's 4 blocks of the step."""
    rg, rr = row // 16, row % 16
    p8, par, blk = col // 8, (col // 4) % 2, col % 4
    return (((rg * (cols // 8) + p8) * 2 + par) * 16 + rr) * 4 + blk


def fp4_tile_load(r_w, base, rg, k, st, at):
    """A lane's 32 B of an FP4 weight's (16, 16)-shuffled 16-row group ``rg``
    (K = ``k``) at 128-k step ``st``: dword ``base`` + the group's 2 k + 256 a
    step + ``at`` (``lane * 4``, lane 0-31) and + 128 -- K chunks lane / 16 and
    lane / 16 + 2 of row lane % 16, the WMMA A operand (8 dwords). The bytes are
    the gfx950 layout's. Nontemporal: a weight a step reads once."""
    def ld(off):
        return fx.Vector(
            bo.buffer_load(
                r_w,
                base + rg * (2 * k) + st * 256 + at + off,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        )

    return ld(0).shuffle(ld(128), list(range(8)))


def mfma_scaled(a, b, c, sa, sb, a_fmt=FP8, b_fmt=FP8, sa_byte=0, sb_byte=0):
    """One 16x16x128 scaled WMMA (gfx1250 f8f6f4): A ``a`` (``a_fmt``; 8 dwords
    FP4, 16 FP8) with E8M0 dword ``sa``, B ``b`` (``b_fmt``) with E8M0 dword
    ``sb``, ``c`` 8 f32. ``sa_byte`` / ``sb_byte`` are the opsel: which lane
    half's dwords carry the scales (0: lanes 0-15, 1: lanes 16-31; byte j of a
    dword = K block j). gfx950's byte select (0-3) has no gfx1250 meaning."""
    assert sa_byte in (0, 1) and sb_byte in (0, 1), (sa_byte, sb_byte)
    return fx.Vector(
        rocdl.wmma_scale_f32_16x16x128_f8f6f4(
            T.vec(8, T.f32), a, b, c, fx.Int32(sa), fx.Int32(sb),
            fmtA=a_fmt, fmtB=b_fmt, scaleAType=sa_byte, scaleBType=sb_byte,
        )
    )  # fmt: skip
