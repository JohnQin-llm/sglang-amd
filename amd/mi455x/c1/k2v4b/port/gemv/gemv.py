"""bf16 GEMV pieces of the gfx1250 K2 (port of ``k2bundle/ops/stages/gemv.py``): 16 weight rows a task against
up to 16 token rows in LDS, one ``v_wmma_f32_16x16x32_bf16`` per 32 K, split-K over the CTA's waves.

K is walked in 128-wide chunks, wave w taking chunks w, w + WAVES, ...  A lane holds row ``lane % 16`` and
eight b128 loads (64 bf16) of its row a chunk; WMMA q of the chunk takes loads 2 q and 2 q + 1.  The B operand
(token ``lane % 16``) is read from LDS at the same K, so each half-wave's 16 K of a WMMA is a permutation both
operands share and the dot is unchanged (checked on hardware by ``probe.py``).  D: lane l holds token
``l % 16``, rows ``8 (l // 16) + v``; a CTA's per-wave accumulators go to ``red`` and are summed in wave order
(``common.row_sum``).

Weight byte order (``SHUFFLED``, the default): the dense weights are preshuffled once at load time
(``shuffle_rows``, the same bytes in load order) so every b128 instruction of a wave reads 512 contiguous
bytes.  In the plain row-major order the lane = row WMMA mapping makes an instruction touch 16 rows x 32 B;
that costs 25-45% more time on this GPU (README.md, ``bench_stream.py``).
"""

import os

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from port.sync.sync import cta_barrier

from . import common as C
from .common import CM_NT, rsrc, traced, wmma_bf16

ROWS = 16
KCH = 128
LDS_PAD = 8  # bf16 a padded token row
LD_A_CHUNK = 8  # b128 loads a lane a chunk
# Lane (row r, half g)'s K inside a 128-K chunk.  1: load j at 16 j + 8 g (lanes r and r + 16 adjacent 16 B);
# 0: 64 contiguous at 64 g.
INTERLEAVE = int(os.environ.get("K2G_IL", "1"))
SHUFFLED = int(os.environ.get("K2G_SHUF", "1"))


def shuffle_rows(w):
    """bf16 [N, K] row-major -> the same bytes in the ``SHUFFLED`` load order (N % 16 == 0, K % 128 == 0):
    [N / 16][K / 128][load j][lane][8 bf16]."""
    n, k = w.shape
    if INTERLEAVE:
        x = w.reshape(n // ROWS, ROWS, k // KCH, LD_A_CHUNK, 2, 8)  # rt, r, ch, j, g, e: k = 16 j + 8 g + e
        x = x.permute(0, 2, 3, 4, 1, 5)
    else:
        x = w.reshape(n // ROWS, ROWS, k // KCH, 2, LD_A_CHUNK, 8)  # rt, r, ch, g, j, e: k = 64 g + 8 j + e
        x = x.permute(0, 2, 4, 3, 1, 5)
    return x.contiguous().reshape(n, k)


def chunks(k, waves=None):
    """A wave's K chunks for K = ``k`` over ``waves`` waves: (count a wave, total)."""
    waves = C.WAVES if waves is None else waves
    n = k // KCH
    return (n + waves - 1) // waves, n


def chunk_loads(lane, w, row0, ch, ldk, k0=0, nt=True, mask=None, nbytes=None):
    """Chunk ``ch`` of weight rows ``row0`` .. + 16 (bf16, ``ldk`` a row, columns from ``k0``): four v8-i32
    WMMA A operands."""
    r_w = rsrc(w, nbytes)
    row = row0 + lane % ROWS
    g = lane // ROWS
    if SHUFFLED:
        base = ((row0 // ROWS) * (ldk // KCH) + k0 // KCH + ch) * (LD_A_CHUNK * C.WAVE * 4) + lane * 4
        step = C.WAVE * 4
    elif INTERLEAVE:
        base = (row * ldk + k0 + ch * KCH + g * 8) // 2
        step = 8
    else:
        base = (row * ldk + k0 + ch * KCH + g * 64) // 2
        step = 4
    v = [
        fx.Vector(
            bo.buffer_load(
                r_w, base + step * j, vec_width=4, dtype=T.i32, cache_modifier=CM_NT if nt else 0, mask=mask
            )
        )
        for j in range(LD_A_CHUNK)
    ]
    return [v[2 * q].shuffle(v[2 * q + 1], list(range(8))) for q in range(4)]


def loads(lane, wave, w, row0, k, nt=True, ldk=None, k0=0, waves=None, mask=None, nbytes=None):
    """This wave's operands of weight rows ``row0`` .. + 16 over K columns ``k0`` .. ``k0 + k``: a list a chunk
    (clamped into K: ``wmmas`` zeroes the extra)."""
    waves = C.WAVES if waves is None else waves
    per, n = chunks(k, waves)
    ldk = k if ldk is None else ldk
    return [
        chunk_loads(lane, w, row0, fx.min(wave + waves * i, n - 1), ldk, k0, nt, mask, nbytes)
        for i in range(per)
    ]


def x_operand(xl, xrow, t, kk, gap=8):
    """bf16 K ``kk`` .. + 8 and ``kk + gap`` .. + 8 of LDS token row t: a v8-i32 B operand."""
    a = fx.Vector(fx.ptr_load(xl + (t * xrow + kk) // 2, result_type=fx.Vector.make_type(4, fx.Int32)))
    b = fx.Vector(fx.ptr_load(xl + (t * xrow + kk + gap) // 2, result_type=fx.Vector.make_type(4, fx.Int32)))
    return a.shuffle(b, list(range(8)))


def x_chunk_operand(xl, xrow, t, kbase, g, q):
    """The B operand matching ``chunk_loads``' WMMA q for lane half g (``kbase``: the chunk's LDS K)."""
    if INTERLEAVE:
        return x_operand(xl, xrow, t, kbase + 32 * q + 8 * g, 16)
    return x_operand(xl, xrow, t, kbase + 64 * g + 16 * q)


def wmmas(lane, wave, xl, xrow, k, s, ops, waves=None, k_lds0=0):
    """``loads``' operands against token rows 0 .. s - 1 of the LDS rows ``xl`` (``xrow`` bf16 a row; the
    task's K starts at LDS column ``k_lds0``) -> this wave's 16x16 accumulator (v8 f32)."""
    waves = C.WAVES if waves is None else waves
    per, n = chunks(k, waves)
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    acc = fx.Vector.filled(8, 0.0, fx.Float32)
    for i in range_constexpr(per):
        ch = wave + waves * i
        kb = k_lds0 + fx.min(ch, n - 1) * KCH
        for q in range_constexpr(4):
            xb = x_chunk_operand(xl, xrow, t, kb, g, q)
            if const_live(per, n, waves):
                live = ch < n
                xb = fx.Vector.from_elements([live.select(xb[d], fx.Int32(0)) for d in range(8)], fx.Int32)
            acc = wmma_bf16(ops[i][q], xb, acc)
    return acc


def wmmas_multi(lane, wave, xl, xrow, k, s, opss, waves=None, k_lds0=0):
    """``wmmas`` of several tasks over the same LDS rows: each B operand loaded once for all tasks; every task's
    accumulator sees the same WMMA sequence as its own ``wmmas``."""
    waves = C.WAVES if waves is None else waves
    per, n = chunks(k, waves)
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    accs = [fx.Vector.filled(8, 0.0, fx.Float32) for _ in opss]
    for i in range_constexpr(per):
        ch = wave + waves * i
        kb = k_lds0 + fx.min(ch, n - 1) * KCH
        for q in range_constexpr(4):
            xb = x_chunk_operand(xl, xrow, t, kb, g, q)
            if const_live(per, n, waves):
                live = ch < n
                xb = fx.Vector.from_elements([live.select(xb[d], fx.Int32(0)) for d in range(8)], fx.Int32)
            accs = [wmma_bf16(opss[j][i][q], xb, accs[j]) for j in range(len(opss))]
    return accs


mfmas = wmmas  # the gfx950 module's name, for callers not yet renamed


def const_live(per, n, waves=None):
    """Whether some wave's last chunk is past K (its WMMA must add zero)."""
    waves = C.WAVES if waves is None else waves
    return per * waves != n


def store_acc(red, wave, lane, acc):
    """This wave's accumulator -> ``red`` (``common.acc_slot`` order)."""
    at = red + (wave * C.WAVE + lane) * C.ACC
    fx.ptr_store(fx.Vector.from_elements([acc[v] for v in range(4)], fx.Float32), at)
    fx.ptr_store(fx.Vector.from_elements([acc[v] for v in range(4, 8)], fx.Float32), at + 4)


@traced
def rows_to_lds(tid, src, k, s, xl, xrow, cm=0, ldk=None, k0=0):
    """Columns k0 .. k0 + k of token rows 0 .. s - 1 of the bf16 [s, ldk] at ``src`` -> LDS (``xrow`` a row),
    16 B a lane, every round's load issued before the stores."""
    ldk = k if ldk is None else ldk
    words = s * k // 8
    rounds = (words + C.THREADS - 1) // C.THREADS
    got = []
    for i in range_constexpr(rounds):
        e = fx.min(tid + C.THREADS * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        got.append(
            fx.Vector(bo.buffer_load(rsrc(src), (t * ldk + k0 + kk) // 2, vec_width=4, dtype=T.i32, cache_modifier=cm))
        )
    for i in range_constexpr(rounds):
        e = fx.min(tid + C.THREADS * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        if tid + C.THREADS * i < words:
            fx.ptr_store(got[i], xl + (t * xrow + kk) // 2)
    cta_barrier()


def dev_rows_to_lds(tid, src, k, s, xl, xrow):
    """``rows_to_lds`` at device scope: rows another CTA of this launch wrote."""
    rows_to_lds(tid, src, k, s, xl, xrow, C.CM_XROWS)
