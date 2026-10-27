"""Default-OFF experimental G1-sorted -> G2-slot pair; no all-M qualification.

Native K32 G1 passed saved M256 and M48 screens; this pair has no all-M
qualification. Supporting shape guards does not qualify those shapes. Original caller owns alignment, routing and Muse post-normalization.
No G1 scatter/G2 gather, tensor cache, readback, synchronization or weight copy.
"""
import os
import torch

G1_CONFIG = dict(
    tile_m=16, tile_n=128, tile_k=256, waves_per_eu=None,
    k_batch=1, k_wave=1, b_nt=2, xcd_swizzle=0,
    gate_mode="separated", act="silu", w_dtype="fp4", w_layout="standard",
)
# Small verify batches benefit from additional N workgroups. K32 math is unchanged.
G1_SMALL_CONFIG = dict(G1_CONFIG, tile_n=64)
G2_CONFIG = dict(
    BM=16, NE=64, N_OUT=4096, D_INTER=4096, TILE_N=128, TILE_K=256,
    b_cache_mod=2, xcd_swizzle=0, waves_per_eu=None, w_dtype="fp4",
    persist=False, use_k16=False,
)


def eligible(a, w1, s1, w2, s2, meta, cfg, *, topk, split_k, single_layout):
    """Pure host metadata dispatch; no allocation or device-value inspection.

    Flag is independent of SGLANG_MUSE_AITER_A16_G1. If false, the caller's
    existing native/standalone-G1 dispatch retains its original behavior.
    Metadata contents are guaranteed by the original SGL alignment producer.
    """
    if os.environ.get("SGLANG_MUSE_AITER_A16_G1G2_SLOT", "0") != "1":
        return False
    if not single_layout or topk != 8 or split_k != 1:
        return False
    if cfg.get("BLOCK_M") != 16 or cfg.get("GEMM1_BLOCK_M", 16) != 16:
        return False
    if a.ndim != 2 or not 1 <= a.shape[0] <= 512 or a.shape[1] != 4096:
        return False
    if w1.shape != (64, 8192, 2048) or w2.shape != (64, 4096, 2048):
        return False
    if s1 is None or s2 is None or s1.shape != (64, 8192, 128) or s2.shape != (64, 4096, 128):
        return False
    if len(meta) != 3:
        return False
    sid, eid, post = meta
    tensors = (a, w1, s1, w2, s2, sid, eid, post)
    if any(not t.is_cuda or t.device != a.device or not t.is_contiguous() for t in tensors):
        return False
    if a.dtype != torch.bfloat16 or any(t.dtype != torch.uint8 for t in (w1, s1, w2, s2)):
        return False
    if any(t.dtype != torch.int32 for t in meta):
        return False
    slots = a.shape[0] * 8
    cap = slots + (64 if slots <= 64 else 65) * 15
    blocks = (cap + 15) // 16
    if sid.shape != (cap,) or eid.shape != (blocks,) or post.shape != (1,):
        return False
    try:
        from sglang.srt.rocm_muse_spark.muse_moe_triton import ALIGN_BACKEND
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        if (parallel.tp_size, parallel.moe_ep_size, parallel.moe_tp_size) != (4, 4, 1):
            return False
        if ALIGN_BACKEND != "sgl" or getattr(torch.version, "hip", None) is None:
            return False
        return getattr(torch.cuda.get_device_properties(a.device), "gcnArchName", "").startswith("gfx950")
    except (AttributeError, ImportError, RuntimeError, ValueError):
        return False


def launch_g2(buffers, w2, s2, out, m):
    """Full measured launcher ABI; only M/capacity/grid are dynamic.

    Already-admitted internal leaf. The caller retains all buffers on its current
    device stream until both stages and the original finalizer have been queued.
    Reserved routing-weight pointer stays exactly zero and is never read.
    """
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from sglang.srt.rocm_muse_spark.muse_aiter_a16_g2_slot_kernel import compile_gemm2_a16w4_port, gemm2_a16w4_grid

    a = buffers["sorted_inter"]
    sid, eid, post = buffers["vsid"], buffers["veid"], buffers["vpost"]
    blocks = eid.numel()
    assert 1 <= m <= 512
    slots = m * 8
    cap = slots + (64 if slots <= 64 else 65) * 15
    assert blocks == (cap + 15) // 16
    assert a.shape == (blocks * 16, 4096) and out.shape == (slots, 4096)
    assert sid.shape == (blocks * 16,) and post.shape == (1,)
    assert w2.shape == (64, 4096, 2048) and s2.shape == (64, 4096, 128)
    assert a.dtype == out.dtype == torch.bfloat16 and w2.dtype == s2.dtype == torch.uint8
    assert all(t.dtype == torch.int32 for t in (sid, eid, post))
    assert all(t.is_cuda and t.device == a.device and t.is_contiguous() for t in (a, sid, eid, post, w2, s2, out))
    exe = compile_gemm2_a16w4_port(**G2_CONFIG)
    grid = gemm2_a16w4_grid(16, N_OUT=4096, TILE_N=128, max_m_blocks=blocks, persist=False)
    _run_compiled(
        exe, a.data_ptr(), w2.data_ptr(), s2.data_ptr(), eid.data_ptr(),
        post.data_ptr(), sid.data_ptr(), 0, m, blocks, grid, out.data_ptr(),
        torch.cuda.current_stream(a.device),
    )
    return out


def launch_eligible(a, w1, s1, w2, s2, meta):
    """Return (per-slot output, workspace); no source activation/weights copied."""
    from sglang.srt.rocm_muse_spark import muse_aiter_native_order_g1 as vendor
    from sglang.srt.rocm_muse_spark import muse_aiter_a16_g1_kernels as adapters

    m = a.shape[0]
    buffers = adapters.allocate(torch, meta, m)
    adapters.compact(meta, buffers, m)
    vendor.flydsl_a16w4_gemm1(
        a_bf16=a, w1_u8=w1, w1_scale_u8=s1,
        sorted_expert_ids=buffers["veid"], cumsum_tensor=buffers["vpost"],
        m_indices=buffers["vsid"], inter_sorted_bf16=buffers["sorted_inter"],
        n_tokens=m, NE=64, D_HIDDEN=4096, D_INTER=4096, topk=8,
        stream=torch.cuda.current_stream(a.device),
        **(G1_SMALL_CONFIG if m <= 48 else G1_CONFIG),
    )
    out = torch.empty((m * 8, 4096), dtype=a.dtype, device=a.device)
    launch_g2(buffers, w2, s2, out, m)
    return out, buffers


def try_launch_pair(a, w1, s1, w2, s2, meta, cfg, *, topk, split_k, single_layout):
    if not eligible(a, w1, s1, w2, s2, meta, cfg,
                    topk=topk, split_k=split_k, single_layout=single_layout):
        return None
    return launch_eligible(a, w1, s1, w2, s2, meta)
