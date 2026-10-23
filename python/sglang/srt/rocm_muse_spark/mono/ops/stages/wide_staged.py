# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wide K2 stages whose consumers only wait on earlier kernel launches.

The arithmetic and tagged peer ABI are inherited from the original Muse
monokernel. Local reductions belong to one CTA or cross a stream-ordered
kernel boundary; no stage requires its entire grid to be resident.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from sglang.srt.rocm_muse_spark.mono.ops.common.ops import (
    CM_DEV,
    bf16_round,
    rsrc,
    traced,
    uniform,
)
from sglang.srt.rocm_muse_spark.mono.ops.stages import wide_moe as moe


def _f32(value):
    return value.bitcast(fx.Float32)


@traced
def router_reduce(c, group, base):
    if c["tid"] < 8 * 8:
        row, expert = base + c["tid"] // 8, c["tid"] % 8
        got = c["poll"](
            [
                (c["rpart"], (group * 4 + q) * c["S"] * 8 + row * 8 + expert, 1)
                for q in range(4)
            ]
        )
        value = _f32(got[0][0])
        for q in range_constexpr(1, 4):
            value = value + _f32(got[q][0])
        for p in range_constexpr(8):
            c["put"](
                c["peer"]["front_rlog"][p],
                (c["rank"] * c["S"] + row) * 32 + group * 8 + expert,
                value,
            )
    gpu.barrier()


@traced
def pool_reduce(c, group):
    for base in range_constexpr(0, c["S"], 8):
        if c["tid"] < 8 * 16:
            row, col = base + c["tid"] // 16, c["tid"] % 16
            got = c["poll"](
                [
                    (c["ppart"], (group * 4 + q) * c["S"] * 16 + row * 16 + col, 1)
                    for q in range(4)
                ]
            )
            value = _f32(got[0][0])
            for q in range_constexpr(1, 4):
                value = value + _f32(got[q][0])
            fx.ptr_store(value, c["av"] + c["tid"])
        gpu.barrier()
        if c["tid"] < 8 * 4:
            row, pack = base + c["tid"] // 4, c["tid"] % 4
            vals = [
                bf16_round(fx.ptr_load(c["av"] + (row - base) * 16 + pack * 4 + i))
                for i in range(4)
            ]
            col = c["rank"] * 512 + group * 16 + pack * 4
            for p in range_constexpr(8):
                c["put_bf"](c["peer"]["front_pool"][p], row * 4096 + col, vals)
        gpu.barrier()


@traced
def published_union(c):
    # The single-CTA union kernel completed on this stream before either expert stage.
    return uniform(
        fx.Int32(
            bo.buffer_load(
                rsrc(c["union_nu"]), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
            )
        )
    )


@traced
def expert_up(c):
    if const_expr(c["S"] == 16):
        ctx = moe.expert_stage_context(c)
        nu = published_union(ctx)
        slot, member = c["bid"] // 16, c["bid"] % 16
        # One independent task per expert/group. Inactive experts return before
        # loading XQ, and no persistent-loop state spans the GEMM pipeline.
        if slot < nu:
            moe.load_xq(c)
            fan = fx.Int32(16)
            moe.stage_ug_expert(ctx, slot * 16 + member, fan)
    else:
        moe.load_xq(c)
        ctx = moe.expert_stage_context(c)
        nu = published_union(ctx)
        slot, member = c["bid"] // 16, c["bid"] % 16
        while slot < nu:
            fan = fx.Int32(16)
            if const_expr(c["S"] == 24):
                fan = uniform(moe._rt_load(ctx["rt"], "fan", slot))
            moe.stage_ug_expert(ctx, slot * 16 + member, fan)
            slot = slot + 32


@traced
def expert_down(c):
    ctx = moe.expert_stage_context(c)
    nu = published_union(ctx)
    slot, member = c["bid"] // 16, c["bid"] % 16
    # All UG producers finished in the preceding kernel. Each down task owns
    # one expert/output group; the final experts need no persistent-loop tail.
    if slot < nu:
        fan = fx.Int32(16)
        if const_expr(c["S"] == 24):
            fan = uniform(moe._rt_load(ctx["rt"], "fan", slot))
        moe.load_expert_inter(ctx, slot)
        for it in range_constexpr(2):
            rgd = member * 16 + ctx["wave"] + 8 * it
            moe.stage_down_expert_all_rows(ctx, slot, rgd, fan)


@traced
def publish_unpool(c):
    # AITER writes the local column shard before this stage is launched.
    at = c["bid"] * 512 + c["tid"]
    value = fx.Int32(bo.buffer_load(rsrc(c["unpooled"]), at, vec_width=1, dtype=T.i32))
    target = c["bid"] * 4096 + c["rank"] * 512 + c["tid"]
    for peer in range_constexpr(8):
        c["put_words"](c["peer"]["br"][peer], target, [value])
