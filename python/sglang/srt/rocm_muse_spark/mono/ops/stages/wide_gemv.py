# SPDX-License-Identifier: Apache-2.0
"""M16/M24 tiled GEMV stages used by the B2/B3 split kernels.

All MFMA calls have a physical row count of sixteen.  ``base`` is carried
through every readiness, partial, and peer address so the eight-row M24 tail
can never alias rows 0..7.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    bf_hi,
    bf_lo,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    wave_sum,
)
from sglang.srt.rocm_muse_spark.mono.ops.stages import gemv

TP = 8
H = 8192
MID = 4096
EPR = 32
ROWS = 16
R_PARTS = 4
R_K = H // R_PARTS
POOL_GROUPS = 32

# Source-only qualification switch, deliberately off until actual-AITER and
# complete FullK2 gates pass for BOTH widths. Change in a fresh process; this
# file is covered by FullK2's source digest. No runtime/serving flag is added.
OPROJ_FULL_K = True


def _f32(word):
    return word.bitcast(fx.Float32)


@traced
def _front_rows(c, q, base, live):
    """Wait on this global row tile and stage its K quarter in LDS."""
    if c["tid"] < live:
        c["poll"]([(c["xrdy"], (base + c["tid"]) * R_PARTS + q, 1)])
    gpu.barrier()
    gemv.rows_to_lds(
        c["tid"],
        c["xbuf"] + fx.Int64(base * H * 2),
        R_K,
        live,
        c["xl"],
        R_K + gemv.LDS_PAD,
        CM_DEV,
        ldk=H,
        k0=q * R_K,
    )


def oproj_loads(c, rg):
    if OPROJ_FULL_K:
        return ()
    return gemv.loads(
        c["lane"], c["wave"], c["args"]["w_o"], rg * ROWS, 1024
    )


@traced
def oproj_full_k(lane, wave, w, row0, xl, live, red):
    """Experimental wave-owned 16x16 tile; xl contains live padded BF16 rows.

    A = weight (output channel), B = activation (token). Thus lane%16 is
    the token coordinate in C, while lane//16 and register d select channel
    4*(lane//16)+d. Match gemv.mfmas_full's increasing K32 chain, loading each
    weight vector beside its MFMA instead of keeping K1024 operands live.
    This is NOT proof of AITER's lowered K256 tl.dot accumulation topology.

    Only wave 0 publishes its four FP32 accumulators. The caller must execute
    a CTA barrier before reading red and before reusing xl/red. No CTA barrier
    belongs inside the wave-uniform branch. Other waves' red slots are stale
    and must never be read by this variant.
    """
    if wave == 0:
        rw = rsrc(w)
        row = row0 + lane % ROWS
        token = fx.min(lane % ROWS, live - 1)
        group = lane // ROWS
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for step in range_constexpr(1024 // 32):
            kk = step * 32 + group * 8
            wb = fx.Vector(bo.buffer_load(
                rw, (row * 1024 + kk) // 2,
                vec_width=4, dtype=T.i32, cache_modifier=CM_NT,
            ))
            xb = fx.Vector(fx.ptr_load(
                xl + (token * (1024 + gemv.LDS_PAD) + kk) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            ))
            acc = mfma_bf16(wb.bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc)
        fx.ptr_store(acc, red + lane * 4)


def oproj_full_k_values(red, token, quarter):
    """Read only wave 0, with the same BF16 boundary as the shard variant."""
    return [
        bf16_round(fx.Float32(fx.ptr_load(red + (16 * quarter + token) * 4 + d)))
        for d in range(4)
    ]


@traced
def oproj_tile(c, rg, base, live, ops):
    """Local BF16 o-proj partial for global rows ``base:base+live``."""
    gemv.rows_to_lds(
        c["tid"],
        c["args"]["core"] + fx.Int64(base * 1024 * 2),
        1024,
        live,
        c["xl"],
        1024 + gemv.LDS_PAD,
    )
    if const_expr(OPROJ_FULL_K):
        oproj_full_k(
            c["lane"], c["wave"], c["args"]["w_o"], rg * ROWS,
            c["xl"], live, c["red"],
        )
    else:
        acc = gemv.mfmas(
            c["lane"], c["wave"], c["xl"], 1024 + gemv.LDS_PAD, 1024, live, ops
        )
        fx.ptr_store(acc, c["red"] + (c["wave"] * 64 + c["lane"]) * 4)
    gpu.barrier()
    if c["tid"] < live * 4:
        t = c["tid"] // 4
        rq = c["tid"] % 4
        if const_expr(OPROJ_FULL_K):
            vals = oproj_full_k_values(c["red"], t, rq)
        else:
            vals = [bf16_round(row_sum(c["red"], 4 * rq + d, t)) for d in range(4)]
        col = rg * ROWS + 4 * rq
        at = ((c["rank"] * c["S"] + base + t) * H + col)
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["front_ar"][p], at, vals)
    gpu.barrier()


def router_loads(c, group, q):
    """Eight FP32 expert rows; accepts raw [H,E] or cached [E,H]."""
    lane, wave = c["lane"], c["wave"]
    expert = c["rank"] * EPR + group * 8 + wave
    rw = rsrc(c["args"]["w_router"])
    out = []
    for j in range(R_K // 256):
        k0 = q * R_K + j * 256 + lane * 4
        if const_expr(c["router_transposed"]):
            out.append(
                fx.Vector(
                    bo.buffer_load(
                        rw, expert * H + k0, vec_width=4, dtype=T.f32,
                        cache_modifier=CM_NT,
                    )
                )
            )
        else:
            # The raw native tensor is [H, E].  Scalar gather preserves its
            # authoritative stride instead of inventing a packed format.
            out.append(
                fx.Vector.from_elements(
                    [
                        fx.Float32(
                            bo.buffer_load(
                                rw, (k0 + d) * 256 + expert, vec_width=1,
                                dtype=T.f32, cache_modifier=CM_NT,
                            )
                        )
                        for d in range(4)
                    ],
                    fx.Float32,
                )
            )
    return out


@traced
def router_tile(c, group, q, base, live, ops):
    _front_rows(c, q, base, live)
    xrow = R_K + gemv.LDS_PAD
    for lt in range_constexpr(live):
        acc = fx.Float32(0.0)
        for j in range_constexpr(R_K // 256):
            xv = fx.Vector(
                fx.ptr_load(
                    c["xl"] + (lt * xrow + 256 * j + c["lane"] * 4) // 2,
                    result_type=fx.Vector.make_type(2, fx.Int32),
                )
            )
            w = ops[j]
            acc = acc + bf_lo(xv[0]) * w[0] + bf_hi(xv[0]) * w[1]
            acc = acc + bf_lo(xv[1]) * w[2] + bf_hi(xv[1]) * w[3]
        value = wave_sum(acc)
        if c["lane"] == 0:
            c["put"](
                c["rpart"],
                (group * R_PARTS + q) * c["S"] * 8 + (base + lt) * 8 + c["wave"],
                value,
            )
    gpu.barrier()
    if (q == 0) & (c["tid"] < live * 8):
        lt, w = c["tid"] // 8, c["tid"] % 8
        at = (group * R_PARTS) * c["S"] * 8 + (base + lt) * 8 + w
        value = _f32(c["poll"]([(c["rpart"], at, 1)])[0][0])
        got = c["poll"](
            [
                (c["rpart"], (group * R_PARTS + qq) * c["S"] * 8 +
                 (base + lt) * 8 + w, 1)
                for qq in range(1, R_PARTS)
            ]
        )
        for qq in range_constexpr(R_PARTS - 1):
            value = value + _f32(got[qq][0])
        expert = group * 8 + w
        for p in range_constexpr(TP):
            c["put"](
                c["peer"]["front_rlog"][p],
                (c["rank"] * c["S"] + base + lt) * EPR + expert,
                value,
            )
    gpu.barrier()


def pool_loads(c, group, q):
    return gemv.loads(
        c["lane"], c["wave"], c["args"]["w_pool"],
        c["rank"] * 512 + group * ROWS, R_K, ldk=H, k0=q * R_K,
    )


@traced
def pool_tile(c, group, q, base, live, ops):
    _front_rows(c, q, base, live)
    acc = gemv.mfmas(
        c["lane"], c["wave"], c["xl"], R_K + gemv.LDS_PAD,
        R_K, live, ops,
    )
    fx.ptr_store(acc, c["red"] + (c["wave"] * 64 + c["lane"]) * 4)
    gpu.barrier()
    if c["tid"] < live * ROWS:
        lt, r = c["tid"] // ROWS, c["tid"] % ROWS
        value = row_sum(c["red"], r, lt)
        at = (group * R_PARTS + q) * c["S"] * ROWS + (base + lt) * ROWS + r
        c["put"](c["ppart"], at, value)
        if q == 0:
            got = c["poll"](
                [
                    (c["ppart"], (group * R_PARTS + qq) * c["S"] * ROWS +
                     (base + lt) * ROWS + r, 1)
                    for qq in range(1, R_PARTS)
                ]
            )
            for qq in range_constexpr(R_PARTS - 1):
                value = value + _f32(got[qq][0])
            fx.ptr_store(value, c["av"] + lt * ROWS + r)
    gpu.barrier()
    if (q == 0) & (c["tid"] < live * 4):
        lt, rq = c["tid"] // 4, c["tid"] % 4
        vals = [bf16_round(fx.ptr_load(c["av"] + lt * ROWS + 4 * rq + d))
                for d in range(4)]
        col = c["rank"] * 512 + group * ROWS + 4 * rq
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["front_pool"][p], (base + lt) * MID + col, vals)
    gpu.barrier()


def unpool_loads(c, group):
    return gemv.loads(
        c["lane"], c["wave"], c["args"]["w_unpool"],
        c["rank"] * 1024 + group * ROWS, MID,
    )


@traced
def unpool_tile(c, group, base, live, ops):
    """Unpool from the explicitly materialized BF16 all-reduce boundary."""
    if c["tid"] < live:
        c["poll"]([(c["tail_ready"], base + c["tid"], 1)])
    gpu.barrier()
    gemv.dev_rows_to_lds(
        c["tid"], c["tail_bf16"] + fx.Int64(base * MID * 2), MID, live,
        c["xl"], MID + gemv.LDS_PAD,
    )
    acc = gemv.mfmas(
        c["lane"], c["wave"], c["xl"], MID + gemv.LDS_PAD,
        MID, live, ops,
    )
    fx.ptr_store(acc, c["red"] + (c["wave"] * 64 + c["lane"]) * 4)
    gpu.barrier()
    if c["tid"] < live * 4:
        lt, rq = c["tid"] // 4, c["tid"] % 4
        vals = [bf16_round(row_sum(c["red"], 4 * rq + d, lt)) for d in range(4)]
        col = c["rank"] * 1024 + group * ROWS + 4 * rq
        for p in range_constexpr(TP):
            c["put_bf"](
                c["peer"]["tail_branch"][p], (base + lt) * H + col, vals
            )
    gpu.barrier()

