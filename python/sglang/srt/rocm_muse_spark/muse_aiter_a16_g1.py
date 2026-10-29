"""Default-OFF AITER A16W4 G1 reuse. Numerical/performance coverage is separate.

Original native alignment, G2, normalization and collectives remain caller-owned.
This module neither sorts routes nor retains per-forward GPU tensors globally.
"""
import os

import torch

CONFIG = dict(
    tile_m=16, tile_n=128, tile_k=256, waves_per_eu=None,
    k_batch=1, k_wave=1, b_nt=2, xcd_swizzle=0,
    gate_mode="separated", act="silu", w_dtype="fp4", w_layout="standard",
)


def eligible(
    a, b, b_scale, c, sorted_ids, expert_ids, num_post_padded,
    N, K, num_valid, topk, a_row_div_topk, fuse_silu, mxfp4, cfg,
    split_k=1, single_layout=False,
):
    """Single dispatch contract, based only on host metadata and actual topology.

    No .item(), device readback, shape cache, or caller-supplied topology bypass.
    Valid routing values/num_post are inherited from the original alignment API.
    Small alignment excludes remote slots; AOT may include a remote -1 bucket.
    Both are compacted identically by keeping only live local expert blocks.
    """
    if os.environ.get("SGLANG_MUSE_AITER_A16_G1", "0") != "1":
        return False
    if not (single_layout and mxfp4 and fuse_silu and a_row_div_topk):
        return False
    if split_k != 1 or topk != 8 or N != 4096 or K != 4096:
        return False
    if cfg.get("BLOCK_M") != 16 or cfg.get("GEMM1_BLOCK_M", 16) != 16:
        return False
    if a.ndim != 2 or not 1 <= a.shape[0] <= 512 or a.shape[1] != 4096:
        return False
    m = a.shape[0]
    slots = m * 8
    if num_valid != slots or c.shape != (slots, 4096):
        return False
    if b.shape != (64, 8192, 2048) or b_scale is None or b_scale.shape != (64, 8192, 128):
        return False
    tensors = (a, b, b_scale, c, sorted_ids, expert_ids, num_post_padded)
    if any(not t.is_cuda or t.device != a.device or not t.is_contiguous() for t in tensors):
        return False
    if a.dtype != torch.bfloat16 or c.dtype != torch.bfloat16:
        return False
    if b.dtype != torch.uint8 or b_scale.dtype != torch.uint8:
        return False
    if any(t.dtype != torch.int32 for t in (sorted_ids, expert_ids, num_post_padded)):
        return False
    # Exact original small/AOT capacity; unusual aligners stay on native fallback.
    cap = slots + (64 if slots <= 64 else 65) * 15
    blocks = (cap + 15) // 16
    if sorted_ids.shape != (cap,) or expert_ids.shape != (blocks,) or num_post_padded.shape != (1,):
        return False
    try:
        from sglang.srt.rocm_muse_spark.muse_moe_triton import ALIGN_BACKEND
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        if (parallel.tp_size, parallel.moe_ep_size, parallel.moe_tp_size) != (4, 4, 1):
            return False
        if ALIGN_BACKEND != "sgl" or getattr(torch.version, "hip", None) is None:
            return False
        arch = getattr(torch.cuda.get_device_properties(a.device), "gcnArchName", "")
        return arch.startswith("gfx950")
    except (AttributeError, ImportError, RuntimeError, ValueError):
        # Unpublished context or missing platform metadata cannot opt in.
        return False


def launch_eligible(a, w, ws, out, meta):
    """Internal admitted leaf; returns per-call workspace for lifetime/diagnostics.

    The caller must first run eligible with the same tensors/metadata. Production
    try_launch discards this return after enqueuing the scatter. No host sync.
    Both original W and E8M0 tensors pass directly to the unmodified vendor API.
    """
    from aiter.ops.flydsl.kernels import moe_2stage_a16wmix as vendor
    from sglang.srt.rocm_muse_spark import muse_aiter_a16_g1_kernels as adapters

    m = a.shape[0]
    buffers = adapters.allocate(torch, meta, m)
    adapters.compact(meta, buffers, m)
    vendor.flydsl_a16w4_gemm1(
        a_bf16=a, w1_u8=w, w1_scale_u8=ws,
        sorted_expert_ids=buffers["veid"], cumsum_tensor=buffers["vpost"],
        m_indices=buffers["vsid"], inter_sorted_bf16=buffers["sorted_inter"],
        n_tokens=m, NE=64, D_HIDDEN=4096, D_INTER=4096, topk=8, **CONFIG,
    )
    adapters.scatter(buffers, out, m)
    return buffers


def try_launch(
    a, b, b_scale, c, sorted_ids, expert_ids, num_post_padded,
    N, K, num_valid, topk, a_row_div_topk, fuse_silu, mxfp4, cfg,
    split_k=1, single_layout=False,
):
    """True means an admitted G1 was enqueued; False requests native fallback."""
    if not eligible(
        a, b, b_scale, c, sorted_ids, expert_ids, num_post_padded,
        N, K, num_valid, topk, a_row_div_topk, fuse_silu, mxfp4, cfg,
        split_k, single_layout,
    ):
        return False
    launch_eligible(a, b, b_scale, c, (sorted_ids, expert_ids, num_post_padded))
    return True
