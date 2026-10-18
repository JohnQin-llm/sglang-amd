"""Opt-in AITER A16W4 schedule for small Muse expert batches.

Retains a shuffled copy of each MXFP4 weight and scale tensor. The native
weights remain available for other shapes; this costs 816 MiB per eligible
layer on each rank. Buffers are derived from checkpoint parameters and are
excluded from the state dict.
"""

import torch
import triton
import triton.language as tl
from torch import nn

from sglang.kernels.ops.muse_spark.moe_align import align_local


@triton.jit
def _prepare_routes(
    Sorted,
    Total,
    Tokens,
    Ones,
    NUM_ROUTES: tl.constexpr,
    SIZE: tl.constexpr,
    B: tl.constexpr,
):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    slot = tl.load(
        Sorted + offsets,
        (offsets < SIZE) & (offsets < tl.load(Total)),
        other=NUM_ROUTES,
    )
    # AITER stage1 consumes a token index. Stage2 consumes the original route
    # slot, so its atomic scatter has exactly one contribution per output row.
    tl.store(Tokens + offsets, slot // 8, offsets < SIZE)
    tl.store(Ones + offsets, 1.0, offsets < SIZE)


class PrepackedMuseMoE(nn.Module):
    def __init__(self, w13, w13_scale, w2, w2_scale):
        super().__init__()
        self.update_weights(w13, w13_scale, w2, w2_scale)

    def update_weights(self, w13, w13_scale, w2, w2_scale):
        """Refresh derived layouts without invalidating captured addresses."""
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility.fp4_utils import e8m0_shuffle

        layouts = {
            "w13": shuffle_weight(w13, (16, 16)),
            "w13_scale": e8m0_shuffle(w13_scale.flatten(0, 1)),
            "w2": shuffle_weight(w2, (16, 16)),
            "w2_scale": e8m0_shuffle(w2_scale.flatten(0, 1)),
        }
        for name, value in layouts.items():
            if name in self._buffers:
                self._buffers[name].copy_(value)
            else:
                self.register_buffer(name, value, persistent=False)

    def forward(self, hidden_states, scores, ids, local_expert_start, norm_weight):
        from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
            flydsl_a16w4_gemm1,
            flydsl_a16w4_gemm2,
        )

        from sglang.srt.rocm_muse_spark.muse_moe_flydsl import (
            _cached_finalize_launch,
            _require_flydsl,
        )

        num_tokens = hidden_states.shape[0]
        num_routes = num_tokens * 8
        block_m = 16 if num_tokens < 64 else 32
        ids = ids.to(torch.int32).contiguous()
        scores = scores.to(torch.float32).contiguous()
        sorted_ids, experts, total = align_local(ids, 32, local_expert_start, block_m)
        size = sorted_ids.numel()
        tokens = torch.empty_like(sorted_ids)
        ones = torch.empty_like(sorted_ids, dtype=torch.float32)
        _prepare_routes[(triton.cdiv(size, 256),)](
            sorted_ids, total, tokens, ones, num_routes, size, 256
        )
        inter = torch.empty(
            (size, 4096), device=hidden_states.device, dtype=torch.bfloat16
        )
        # Stage2 performs an atomic add to each unique route slot. It must start
        # from zero on every replay, including after an all-remote route batch.
        y = torch.zeros(
            (num_routes, 4096), device=hidden_states.device, dtype=torch.bfloat16
        )
        flydsl_a16w4_gemm1(
            a_bf16=hidden_states,
            w1_u8=self.w13,
            w1_scale_u8=self.w13_scale,
            sorted_expert_ids=experts,
            cumsum_tensor=total,
            m_indices=tokens,
            inter_sorted_bf16=inter,
            n_tokens=num_tokens,
            NE=32,
            D_HIDDEN=4096,
            D_INTER=4096,
            topk=8,
            tile_m=block_m,
            tile_n=128,
            tile_k=256,
            b_nt=2,
        )
        flydsl_a16w4_gemm2(
            inter_sorted_bf16=inter,
            w2_u8=self.w2,
            w2_scale_u8=self.w2_scale,
            sorted_expert_ids=experts,
            cumsum_tensor=total,
            sorted_token_ids=sorted_ids,
            sorted_weights=ones,
            flat_out=y,
            M_logical=num_routes,
            max_sorted=size,
            NE=32,
            D_HIDDEN=4096,
            D_INTER=4096,
            topk=1,
            tile_m=block_m,
            tile_n=256,
            tile_k=256,
            b_nt=0,
        )
        # Reuse Muse's existing per-expert norm, followed by the original route
        # scores and top-k sum. Vendor stage2 must not merge tokens first.
        out = torch.empty_like(hidden_states)
        fx, run_compiled = _require_flydsl()
        launch = _cached_finalize_launch(4096, 8, True, True, "bf16", 1)
        run_compiled(
            launch,
            y,
            norm_weight,
            scores,
            ids,
            out,
            int(local_expert_start),
            32,
            1e-8,
            1.0,
            num_tokens,
            fx.Stream(torch.cuda.current_stream(hidden_states.device).cuda_stream),
        )
        return out
