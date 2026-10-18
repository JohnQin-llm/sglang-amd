"""Compact page metadata shared by Muse sliding-window verification layers."""

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    kernel_unified_attention_3d,
    reduce_segments,
)
from aiter.ops.triton.attention.unified_attention import select_3d_config


@triton.jit
def _window_pages(
    table,
    lengths,
    compact,
    compact_lengths,
    table_stride: tl.constexpr,
    columns: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(lengths + row)
    # Preserve every key used by any of the eight tail-aligned queries.
    first_page = tl.maximum(0, length - 8 - 2047 + 1) // 16
    offset = tl.arange(0, BLOCK)
    page = first_page + offset
    valid = (offset < columns) & (page * 16 < length) & (page < table_stride)
    value = tl.load(table + row * table_stride + page, valid, other=0)
    tl.store(compact + row * columns + offset, value, offset < columns)
    tl.store(compact_lengths + row, length - first_page * 16)


def prepare_muse_swa_verify(
    seqused_k: torch.Tensor, block_table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build layer-independent window metadata once per target verify step."""
    batch = seqused_k.numel()
    columns = 132  # Pad 130 live pages to the four-page KV tile boundary.
    compact = torch.empty(
        (batch, columns), device=block_table.device, dtype=block_table.dtype
    )
    lengths = torch.empty_like(seqused_k)
    _window_pages[(batch,)](
        block_table, seqused_k, compact, lengths, block_table.stride(0), columns, 256
    )
    return compact, lengths


def muse_swa_verify_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    softmax_scale,
    k_descale,
    v_descale,
    prepared: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    tokens, heads, dim = q.shape
    batch = seqused_k.numel()
    kv_heads = k.shape[2]
    compact, lengths = (
        prepare_muse_swa_verify(seqused_k, block_table)
        if prepared is None
        else prepared
    )
    # At batch 32, reuse each KV tile across all eight query tokens. Smaller
    # batches benefit from the extra query-tile parallelism of BLOCK_M=16.
    packed = batch == 32
    block_m = 8 * (heads // kv_heads) if packed else 16
    block_q = block_m // (heads // kv_heads)
    blocks = tokens // block_q + batch
    attn, reduce = select_3d_config(
        dim,
        16,
        2080,
        torch.cuda.get_device_properties(q.device).multi_processor_count * 4,
        blocks * kv_heads,
        q.dtype,
        k.dtype,
        SLIDING_WINDOW=2047,
    )
    if packed:
        attn.update(NUM_SEGMENTS_PER_SEQ=8, num_warps=2)
        reduce.update(NUM_SEGMENTS_PER_SEQ=8, num_warps=1)
    segments = attn["NUM_SEGMENTS_PER_SEQ"]
    partial = torch.empty(
        (tokens, heads, segments, dim), device=q.device, dtype=torch.float32
    )
    maxima = torch.empty(
        (tokens, heads, segments), device=q.device, dtype=torch.float32
    )
    sums = torch.empty_like(maxima)
    kernel_unified_attention_3d[(blocks, kv_heads, segments)](
        segm_output_ptr=partial,
        segm_max_ptr=maxima,
        segm_expsum_ptr=sums,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=None,
        block_tables_ptr=compact,
        seq_lens_ptr=lengths,
        alibi_slopes_ptr=None,
        qq_bias_ptr=None,
        scale=softmax_scale,
        q_descale_ptr=None,
        k_descale_ptr=k_descale,
        v_descale_ptr=v_descale,
        out_scale_ptr=None,
        softcap=0.0,
        num_query_heads=heads,
        num_queries_per_kv=heads // kv_heads,
        block_table_stride=compact.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        qq_bias_stride_0=0,
        BLOCK_SIZE=16,
        HEAD_SIZE=dim,
        HEAD_SIZE_PADDED=dim,
        USE_ALIBI_SLOPES=False,
        USE_QQ_BIAS=False,
        USE_SOFTCAP=False,
        USE_SINKS=False,
        SLIDING_WINDOW=2047,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        num_seqs=batch,
        BLOCK_M=block_m,
        ALL_DECODE=False,
        SHUFFLED_KV_CACHE=False,
        K_WIDTH=16,
        IS_Q_FP8=False,
        IS_KV_FP8=True,
        **attn,
    )
    reduce_segments[(tokens, heads)](
        output_ptr=out,
        segm_output_ptr=partial,
        segm_max_ptr=maxima,
        segm_expsum_ptr=sums,
        seq_lens_ptr=lengths,
        num_seqs=batch,
        num_query_heads=heads,
        out_scale_ptr=None,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_table_stride=compact.stride(0),
        HEAD_SIZE=dim,
        HEAD_SIZE_PADDED=dim,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        **reduce,
    )
    return out
