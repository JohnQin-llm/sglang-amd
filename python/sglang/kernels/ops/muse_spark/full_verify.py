"""Bounded gfx950 q8 query-reuse prototype using installed AITER kernels."""

from numbers import Integral, Real

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    kernel_unified_attention_3d,
    reduce_segments,
)
from aiter.ops.triton.attention.unified_attention import (
    DEVICE_ARCH as _DEVICE_ARCH,
    unified_attention as _vendor_unified_attention,
)
from aiter.ops.triton.utils.types import e4m3_dtype


_NUM_QUERY_HEADS = 16
_NUM_KV_HEADS = 2
_NUM_QUERIES_PER_KV = 8
_HEAD_SIZE = 64
_BLOCK_SIZE = 16
_BLOCK_M = 64
_BLOCK_Q = 8
_TILE_SIZE = 64
_NUM_SEGMENTS = 256


def allocate_workspace(device):
    """Allocate one caller-owned scratch tuple for the fixed q8 path."""
    return (
        torch.empty(
            (8, _NUM_QUERY_HEADS, _NUM_SEGMENTS, _HEAD_SIZE),
            dtype=torch.float32,
            device=device,
        ),
        torch.empty(
            (8, _NUM_QUERY_HEADS, _NUM_SEGMENTS),
            dtype=torch.float32,
            device=device,
        ),
        torch.empty(
            (8, _NUM_QUERY_HEADS, _NUM_SEGMENTS),
            dtype=torch.float32,
            device=device,
        ),
    )


def _validate_workspace(workspace, device):
    """Validate scratch ownership using tensor metadata only."""
    if type(workspace) is not tuple or len(workspace) != 3:
        raise TypeError("workspace must be an exact tuple of three tensors")

    expected_shapes = (
        (8, _NUM_QUERY_HEADS, _NUM_SEGMENTS, _HEAD_SIZE),
        (8, _NUM_QUERY_HEADS, _NUM_SEGMENTS),
        (8, _NUM_QUERY_HEADS, _NUM_SEGMENTS),
    )
    storage_bases = []
    byte_ranges = []
    for tensor, shape in zip(workspace, expected_shapes):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("workspace entries must be tensors")
        if tensor.device != device:
            raise ValueError("workspace tensors must be on the query device")
        if tensor.dtype != torch.float32:
            raise ValueError("workspace tensors must have dtype float32")
        if tuple(tensor.shape) != shape:
            raise ValueError("workspace tensor has an invalid shape")
        if not tensor.is_contiguous():
            raise ValueError("workspace tensors must be contiguous")
        storage_bases.append(tensor.untyped_storage().data_ptr())
        start = tensor.data_ptr()
        byte_ranges.append((start, start + tensor.numel() * tensor.element_size()))

    if len(set(storage_bases)) != 3:
        raise ValueError("workspace tensors must use distinct buffers")
    for index, (start, end) in enumerate(byte_ranges):
        for other_start, other_end in byte_ranges[index + 1 :]:
            if start < other_end and other_start < end:
                raise ValueError("workspace tensors must not overlap")
    return workspace


def _delegate(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    q_scales,
    alibi_slopes,
    output_scale,
    qq_bias,
    sinks,
    shuffled_kv_cache,
    skip_reduce,
):
    return _vendor_unified_attention(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        q_scales=q_scales,
        alibi_slopes=alibi_slopes,
        output_scale=output_scale,
        qq_bias=qq_bias,
        sinks=sinks,
        shuffled_kv_cache=shuffled_kv_cache,
        skip_reduce=skip_reduce,
    )


def _linear_optional_tensor_is_valid(tensor, q, *, shape=None, numel=None):
    if tensor is None:
        return True
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.device != q.device:
        return False
    if shape is not None and tuple(tensor.shape) != shape:
        return False
    if numel is not None and tensor.numel() != numel:
        return False
    return tensor.numel() == 1 or tensor.stride(-1) == 1


def _eligible(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    q_scales,
    alibi_slopes,
    output_scale,
    qq_bias,
    sinks,
    shuffled_kv_cache,
    skip_reduce,
):
    """Check metadata only; never inspect a device-resident scalar value."""
    if _DEVICE_ARCH != "gfx950":
        return False
    if not isinstance(max_seqlen_q, Integral) or max_seqlen_q != 8:
        return False
    if not isinstance(max_seqlen_k, Integral) or max_seqlen_k < 0:
        return False
    if not isinstance(softmax_scale, Real) or not isinstance(softcap, Real):
        return False
    if causal is not True or window_size != (-1, -1):
        return False
    if shuffled_kv_cache or skip_reduce or q_scales is not None:
        return False
    # This experiment qualifies the actual C1 full-attention path only.
    # Preserve every additional semantic through the existing vendor wrapper.
    if softcap != 0 or any(
        value is not None for value in (alibi_slopes, output_scale, qq_bias, sinks)
    ):
        return False

    if q.device.type != "cuda":
        return False
    if tuple(q.shape) != (8, _NUM_QUERY_HEADS, _HEAD_SIZE):
        return False
    if tuple(out.shape) != tuple(q.shape):
        return False
    if q.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        return False
    if q.stride(2) != 1 or out.stride(2) != 1:
        return False

    expected_kv_tail = (_BLOCK_SIZE, _NUM_KV_HEADS, _HEAD_SIZE)
    if k.ndim != 4 or v.ndim != 4:
        return False
    if tuple(k.shape[1:]) != expected_kv_tail or tuple(v.shape) != tuple(k.shape):
        return False
    if k.dtype != e4m3_dtype or v.dtype != e4m3_dtype:
        return False

    tensors = (k, v, out, cu_seqlens_q, seqused_k, block_table)
    if any(tensor.device != q.device for tensor in tensors):
        return False
    if tuple(cu_seqlens_q.shape) != (2,) or cu_seqlens_q.stride(0) != 1:
        return False
    if tuple(seqused_k.shape) != (1,) or seqused_k.stride(0) != 1:
        return False
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        return False
    if block_table.stride(1) != 1:
        return False
    if block_table.shape[1] * _BLOCK_SIZE < max_seqlen_k:
        return False

    scalar_ptrs = (q_descale, k_descale, v_descale, output_scale)
    if any(
        not _linear_optional_tensor_is_valid(ptr, q, numel=1)
        for ptr in scalar_ptrs
    ):
        return False
    if not _linear_optional_tensor_is_valid(
        sinks, q, shape=(_NUM_QUERY_HEADS,)
    ):
        return False
    if not _linear_optional_tensor_is_valid(
        alibi_slopes, q, shape=(_NUM_QUERY_HEADS,)
    ):
        return False
    if qq_bias is not None:
        if qq_bias.device != q.device or tuple(qq_bias.shape) != (8, 8):
            return False
        if qq_bias.stride(1) != 1:
            return False
    return True


def unified_attention_q8(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
    *,
    workspace=None,
):
    """Run the fixed q8 prototype or delegate an unsupported call unchanged."""
    args = (
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        q_scales,
        alibi_slopes,
        output_scale,
        qq_bias,
        sinks,
        shuffled_kv_cache,
        skip_reduce,
    )
    if not _eligible(*args):
        return _delegate(*args)

    head_size_padded = triton.next_power_of_2(_HEAD_SIZE)
    if workspace is None:
        segm_output, segm_max, segm_expsum = allocate_workspace(q.device)
    else:
        segm_output, segm_max, segm_expsum = _validate_workspace(
            workspace, q.device
        )

    # Vendor upper bound: floor(total_q / BLOCK_Q) + num_seqs. The second
    # block is deliberately launched; find_seq_idx maps it to sequence zero
    # and the kernel rejects q_block_local_idx * BLOCK_Q >= query_len.
    num_seqs = len(seqused_k)
    total_num_q_blocks = q.shape[0] // _BLOCK_Q + num_seqs
    kernel_unified_attention_3d[
        (total_num_q_blocks, _NUM_KV_HEADS, _NUM_SEGMENTS)
    ](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        scale=softmax_scale,
        q_descale_ptr=q_descale,
        k_descale_ptr=k_descale,
        v_descale_ptr=v_descale,
        out_scale_ptr=None,
        softcap=softcap,
        num_query_heads=_NUM_QUERY_HEADS,
        num_queries_per_kv=_NUM_QUERIES_PER_KV,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if qq_bias is not None else 0,
        BLOCK_SIZE=_BLOCK_SIZE,
        TILE_SIZE=_TILE_SIZE,
        HEAD_SIZE=_HEAD_SIZE,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=alibi_slopes is not None,
        USE_QQ_BIAS=qq_bias is not None,
        USE_SOFTCAP=softcap > 0,
        USE_SINKS=sinks is not None,
        SLIDING_WINDOW=0,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=_BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=_BLOCK_M,
        num_warps=2,
        waves_per_eu=2,
        num_stages=2,
        NUM_SEGMENTS_PER_SEQ=_NUM_SEGMENTS,
        ALL_DECODE=False,
        SHUFFLED_KV_CACHE=False,
        K_WIDTH=16,
        IS_Q_FP8=False,
        IS_KV_FP8=True,
    )

    reduce_segments[(q.shape[0], _NUM_QUERY_HEADS)](
        output_ptr=out,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=seqused_k,
        num_seqs=num_seqs,
        num_query_heads=_NUM_QUERY_HEADS,
        out_scale_ptr=output_scale,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_table_stride=block_table.stride(0),
        TILE_SIZE=_TILE_SIZE,
        HEAD_SIZE=_HEAD_SIZE,
        HEAD_SIZE_PADDED=head_size_padded,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=_BLOCK_Q,
        NUM_SEGMENTS_PER_SEQ=_NUM_SEGMENTS,
        num_warps=2,
        waves_per_eu=2,
        num_stages=1,
    )
    return out
