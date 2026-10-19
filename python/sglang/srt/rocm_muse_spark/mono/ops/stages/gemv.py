# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 GEMV pieces the mono kernels share (adapted from Kimi-K3's, PR #60683): 16 weight rows a task against up
to 16 token rows in LDS, one ``mfma_f32_16x16x32_bf16`` per 32 K.

K is walked in 128-wide chunks, wave w taking chunks w, w + 8, ...; a lane reads
64 contiguous bytes of its row a chunk (4 dwordx4 at q 32 + g 8 elements: lane
group g = lane / 16), and the token rows are read at the same K. The K order
inside a chunk is a permutation both operands share, so the dot is unchanged.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    CM_DEV,
    CM_NT,
    mfma_bf16,
    rsrc,
    traced,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.plan import THREADS, WAVES

ROWS = 16
KCH = 128
LDS_PAD = 8  # bf16 a padded token row
TENSILE_SPLIT_WAVES = 4


def chunks(k):
    """A wave's K chunks for K = ``k``: (count a wave, total)."""
    n = k // KCH
    return (n + WAVES - 1) // WAVES, n


def loads(lane, wave, w, row0, k, nt=True, ldk=None, k0=0):
    """This wave's operands of weight rows ``row0`` .. + 16 (row-major, ``ldk``
    bf16 a row at ``w``, K columns ``k0`` .. ``k0 + k``): a list a chunk
    (clamped into K: ``mfmas`` zeroes the extra)."""
    per, n = chunks(k)
    ldk = k if ldk is None else ldk
    r_w = rsrc(w)
    row = row0 + lane % ROWS
    g = lane // ROWS
    out = []
    for i in range(per):
        ch = fx.min(wave + WAVES * i, n - 1)
        base = (row * ldk + k0 + ch * KCH + g * 8) // 2
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        base + 16 * q,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT if nt else 0,
                    )
                )
                for q in range(4)
            ]
        )
    return out


def mfmas(lane, wave, xl, xrow, k, s, ops):
    """``loads``' operands against token rows 0 .. s - 1 of the LDS rows ``xl``
    (``xrow`` bf16 a row) -> this wave's 16x16 accumulator."""
    per, n = chunks(k)
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(per):
        ch = wave + WAVES * i
        live = ch < n
        kk = fx.min(ch, n - 1) * KCH + g * 8
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    xl + (t * xrow + kk + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            if const_live(per, n):
                xb = fx.Vector.from_elements(
                    [live.select(xb[d], fx.Int32(0)) for d in range(4)], fx.Int32
                )
            acc = mfma_bf16(
                ops[i][q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    return acc


def load_tensile_chunk(lane, wave, w, row0, k, i):
    """One LocalSplitU K chunk's four vectors for rolling prefetch.

    Inactive waves use wave 3's address for the single load issued before
    XRDY; they never consume it.  All later loads are inside the active-wave
    branch.
    """
    owner = fx.min(wave, TENSILE_SPLIT_WAVES - 1)
    row = row0 + lane % ROWS
    g = lane // ROWS
    ch = owner + TENSILE_SPLIT_WAVES * i
    kk = ch * KCH + g * 8
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(w),
                (row * k + kk + 32 * q) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        )
        for q in range(4)
    ]


@traced
def mfmas_tensile_split(lane, wave, w, row0, xl, xrow, k, s, first):
    """Stream one of four interleaved K shards; waves 4..7 contribute zero.

    The production M=8/N=2304/K=8192 solution is a 256-thread
    ``MT16x16x512 ... WG16_4_4`` Tensile kernel.  Load each weight vector next
    to its MFMA instead of materializing all 16 chunks in registers.
    """
    n = k // KCH
    per = n // TENSILE_SPLIT_WAVES
    assert n % TENSILE_SPLIT_WAVES == 0
    r_w = rsrc(w)
    row = row0 + lane % ROWS
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    # Only the four LocalSplitU waves own K shards.  Keeping waves 4..7 on the
    # load path (and merely zeroing their x operand) doubled weight traffic on
    # the persistent CTA and serialized it behind XRDY for rows 5..8.
    if wave < TENSILE_SPLIT_WAVES:
        cur = first
        # Keep only one chunk resident.  Each next vector is issued before the
        # matching current vector's MFMA, then replaces it, limiting the peak
        # to five vectors instead of materializing two whole chunks.
        for i in range_constexpr(per - 1):
            ch = wave + TENSILE_SPLIT_WAVES * i
            kk = ch * KCH + g * 8
            for q in range_constexpr(4):
                next_ch = wave + TENSILE_SPLIT_WAVES * (i + 1)
                next_kk = next_ch * KCH + g * 8
                nxt = fx.Vector(
                    bo.buffer_load(
                        r_w,
                        (row * k + next_kk + 32 * q) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )
                )
                xb = fx.Vector(
                    fx.ptr_load(
                        xl + (t * xrow + kk + 32 * q) // 2,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                acc = mfma_bf16(
                    cur[q].bitcast(fx.BFloat16),
                    xb.bitcast(fx.BFloat16),
                    acc,
                )
                cur[q] = nxt
        kk = (wave + TENSILE_SPLIT_WAVES * (per - 1)) * KCH + g * 8
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    xl + (t * xrow + kk + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            acc = mfma_bf16(
                cur[q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    return acc


def reduce_tensile_split(red, r, t):
    """Forward linear reduction of the four Tensile LocalSplitU shards."""
    def partial(w):
        return fx.ptr_load(red + ((w * 64 + 16 * (r // 4) + t) * 4 + r % 4))

    return ((partial(0) + partial(1)) + partial(2)) + partial(3)


def mfmas_full(lane, xl, xrow, k, s, ops):
    """One wave's full contiguous-K MFMA chain.

    Production's wider-token GEMMs keep a complete output tile in one wave
    instead of reducing K-shards from several waves.  ``ops`` has one entry per
    128-wide K chunk, in increasing K order.
    """
    _, n = chunks(k)
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for ch in range_constexpr(n):
        kk = ch * KCH + g * 8
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    xl + (t * xrow + kk + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            acc = mfma_bf16(
                ops[ch][q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    return acc


def const_live(per, n):
    """Whether some wave's last chunk is past K (its MFMA must add zero)."""
    return per * WAVES != n


@traced
def rows_to_lds(tid, src, k, s, xl, xrow, cm=0, ldk=None, k0=0):
    """Columns k0 .. k0 + k of token rows 0 .. s - 1 of the bf16 [s, ldk] at
    ``src`` -> LDS (``xrow`` a row), 16 B a lane."""
    ldk = k if ldk is None else ldk
    words = s * k // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        v = fx.Vector(
            bo.buffer_load(
                rsrc(src),
                (t * ldk + k0 + kk) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cm,
            )
        )
        if tid + THREADS * i < words:
            fx.ptr_store(v, xl + (t * xrow + kk) // 2)
    gpu.barrier()


def dev_rows_to_lds(tid, src, k, s, xl, xrow):
    """``rows_to_lds`` at device scope: rows another CTA of this launch wrote."""
    rows_to_lds(tid, src, k, s, xl, xrow, CM_DEV)
