# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2's o_proj: the core rows (T x 1024 bf16, this rank's K slice) against the
rank's RowParallel shard of o_proj (bf16 [8192, 1024]) -> bf16 partials, pushed
to every rank's AR region (system scope, this rank's slot). Adapted from
Kimi-K3's K2a (``attention/back.py`` of PR #60683: ``o_loads`` / ``load_core``
/ ``stage_o``).

512 tasks of 16 rows, two a CTA (b, b + 256); a task's K = 1024 is one 128-wide
chunk a wave, so a row's fp32 sum is the 8 waves' chunk sums in wave order.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl._mlir import ir
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    bf16_round,
    row_sum,
    rsrc,
    traced,
)
from sglang.srt.rocm_muse_spark.mono.ops.common.plan import BLOCKS, THREADS
from sglang.srt.rocm_muse_spark.mono.ops.stages import gemv

TP = 8
H = 8192  # hidden: o_proj's rows
CORE = 1024  # o_proj's K a rank (16 q heads x 64)
ROWS = 16
OTASKS = H // ROWS  # 512
O_ROUNDS = OTASKS // BLOCKS  # 2
CORE_ROW = CORE + gemv.LDS_PAD  # bf16 a padded core row in LDS


def ar_pairs(s):
    """One parity's AR region: [src rank][token][column pair]."""
    return TP * s * H // 2


def o_loads(c, rg):
    """Rows 16 rg .. of o_proj: this wave's K chunk (a lane 64 B of its row)."""
    return gemv.loads(c["lane"], c["wave"], c["args"]["w_o"], rg * ROWS, CORE)


def o_loads_full(c, rg):
    """All contiguous K chunks for one wave-local output tile."""
    return [
        gemv.loads(c["lane"], chunk, c["args"]["w_o"], rg * ROWS, CORE)[0]
        for chunk in range_constexpr(CORE // gemv.KCH)
    ]


def _mfma4(a, b, acc):
    return fx.Vector(
        rocdl.mfma_f32_4x4x4bf16_1k_(
            T.vec(4, T.f32),
            a.bitcast(fx.Int16).ir_value(),
            b.bitcast(fx.Int16).ir_value(),
            acc.ir_value(),
            0,
            0,
            ir.Attribute.parse("#rocdl<mfma_perm_b none>"),
        )
    )


def _dpp(v, control):
    bits = v.bitcast(fx.Int32)
    return fx.Int32(update_dpp_i32(bits, bits, control, 0xF, 0xF, True)).bitcast(
        fx.Float32
    )


def _wv_reduce(acc):
    """WvSplitK's gfx9 MFMA/DPP reduction, instruction for instruction."""
    value = acc[0]
    value = value + _dpp(acc[1], 0x101)
    value = value + _dpp(acc[2], 0x102)
    value = value + _dpp(acc[3], 0x103)
    value = value + _dpp(value, 0x104)
    value = value + _dpp(value, 0x108)
    value = _dpp(value, 0x11F)
    value = value + _dpp(value, 0x142)
    return value + _dpp(value, 0x143)


@traced
def stage_o_wvsplit(c, rg):
    """Small-width production ``wvSplitK`` arithmetic for one 16-row tile."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    r_w = rsrc(c["args"]["w_o"])
    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)] for _ in range(s)]
    # A_CHUNK=8, UNRL=2: lane k and lane k+512, two K4 MFMAs each.
    for unroll in range_constexpr(2):
        k0 = lane * 8 + unroll * 512
        for token in range_constexpr(s):
            for y in range_constexpr(2):
                row = rg * ROWS + wave * 2 + y
                for part in range_constexpr(2):
                    kk = k0 + part * 4
                    activation = fx.Vector(
                        fx.ptr_load(
                            c["xl"] + (token * CORE_ROW + kk) // 2,
                            result_type=fx.Vector.make_type(2, fx.Int32),
                        )
                    ).bitcast(fx.BFloat16)
                    weight = fx.Vector(
                        bo.buffer_load(
                            r_w,
                            (row * CORE + kk) // 2,
                            vec_width=2,
                            dtype=T.i32,
                        )
                    ).bitcast(fx.BFloat16)
                    acc[token][y] = _mfma4(activation, weight, acc[token][y])
    for token in range_constexpr(s):
        for y in range_constexpr(2):
            reduced = _wv_reduce(acc[token][y])
            if lane == 63:
                fx.ptr_store(
                    reduced,
                    red + token * ROWS + wave * 2 + y,
                )
    gpu.barrier()
    if tid < s * (ROWS // 4):
        token = tid // (ROWS // 4)
        rq = tid % (ROWS // 4)
        values = [
            bf16_round(fx.ptr_load(red + token * ROWS + 4 * rq + d)) for d in range(4)
        ]
        at = (c["rank"] * s + token) * H + rg * ROWS + 4 * rq
        for peer in range_constexpr(TP):
            c["put_bf"](c["ar_peer"][peer], at, values)
    gpu.barrier()


@traced
def load_core(c):
    """The core rows (T x 1024 bf16, rows ``CORE`` apart) -> LDS, rows padded."""
    s, tid = c["S"], c["tid"]
    words = s * CORE // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        v = fx.Vector(
            bo.buffer_load(rsrc(c["args"]["core"]), e * 4, vec_width=4, dtype=T.i32)
        )
        t = e // (CORE // 8)
        k = e % (CORE // 8) * 8
        if tid + THREADS * i < words:
            fx.ptr_store(v, c["xl"] + (t * CORE_ROW + k) // 2)
    gpu.barrier()


@traced
def stage_o(c, rg, ops):
    """Rows 16 rg ..: GEMV -> bf16 partials -> every rank's AR region, this
    rank's slot (a thread a (token, row pair), a put a peer)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    acc = gemv.mfmas(lane, wave, c["xl"], CORE_ROW, CORE, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < s * (ROWS // 4):
        tt = tid // (ROWS // 4)
        rq = tid % (ROWS // 4)
        vs = [bf16_round(row_sum(red, 4 * rq + d, tt)) for d in range(4)]
        at = (c["rank"] * s + tt) * H + rg * ROWS + 4 * rq
        for p in range_constexpr(TP):
            c["put_bf"](c["ar_peer"][p], at, vs)
    gpu.barrier()


@traced
def stage_oproj(c, bid):
    """CTA bid's o_proj tasks bid, bid + 256: both tasks' weights in flight at
    once (one memory latency), the core rows, then the MFMAs."""
    if const_expr(c["S"] <= 4):
        load_core(c)
        for j in range_constexpr(O_ROUNDS):
            stage_o_wvsplit(c, bid + BLOCKS * j)
    elif const_expr(c["S"] < 5):
        ops = [o_loads(c, bid + BLOCKS * j) for j in range(O_ROUNDS)]
        load_core(c)
        for j in range_constexpr(O_ROUNDS):
            stage_o(c, bid + BLOCKS * j, ops[j])
    else:
        # The production rows 5..8 AITER GEMM keeps K=1024 in one accumulator
        # chain. Waves 0/1 own this CTA's two row tiles; the remaining waves
        # help stage the input and publish the two completed tiles.
        s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
        load_core(c)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        if wave < O_ROUNDS:
            rg = bid + BLOCKS * wave
            acc = gemv.mfmas_full(lane, c["xl"], CORE_ROW, CORE, s, o_loads_full(c, rg))
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        for j in range_constexpr(O_ROUNDS):
            if tid < s * (ROWS // 4):
                tt = tid // (ROWS // 4)
                rq = tid % (ROWS // 4)
                vs = [
                    bf16_round(row_sum(red, 4 * rq + d, tt, waves=(j,)))
                    for d in range(4)
                ]
                at = (c["rank"] * s + tt) * H + (bid + BLOCKS * j) * ROWS + 4 * rq
                for peer in range_constexpr(TP):
                    c["put_bf"](c["ar_peer"][peer], at, vs)
        gpu.barrier()
