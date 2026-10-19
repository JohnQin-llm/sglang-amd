# SPDX-License-Identifier: Apache-2.0
"""Zero-copy bridge from SGLang decode metadata to original K1's cache ABI."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.model_executor.forward_context import get_attn_backend


@dataclass(frozen=True)
class MonoAttentionInputs:
    kv_cache: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor


def attention_inputs(
    layer, forward_batch, *, seq_lens: torch.Tensor
) -> MonoAttentionInputs:
    """Return graph-live cache pointers and metadata without copying KV data.

    The monokernel cache pool exposes one combined BF16 page tensor per layer;
    the ordinary attention backend reads K/V views of that same allocation.
    Tables and slots remain owned by the backend/ForwardBatch. The runner
    converts SGLang's int64 lengths once per model step into a stable int32
    buffer, shared by every K1 launch in that step.
    """

    backend = get_attn_backend()
    if getattr(backend, "page_size", None) != 16:
        raise ValueError("Muse MonoKernel requires page_size=16")
    pool = backend.token_to_kv_pool
    getter = getattr(pool, "get_muse_mono_buffer", None)
    if getter is None:
        raise ValueError(
            "Muse MonoKernel requires the combined BF16 Muse KV pool; "
            "separate K/V storage cannot be adapted without a full cache copy"
        )
    kv_cache = getter(layer.layer_idx)
    metadata = backend.forward_metadata
    is_swa = layer.self_attn.attn.sliding_window_size > 0
    block_table = metadata.swa_page_table if is_swa else metadata.kv_indices
    if block_table is None or block_table.dim() != 2:
        raise ValueError(
            "Muse MonoKernel requires AITER's graph-live unified 2D page table"
        )
    slot_mapping = (
        metadata.swa_out_cache_loc
        if is_swa and metadata.swa_out_cache_loc is not None
        else forward_batch.out_cache_loc
    )
    if slot_mapping is None:
        raise ValueError("Muse MonoKernel requires a live KV slot mapping")
    values = MonoAttentionInputs(
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        slot_mapping=slot_mapping,
    )
    validate_attention_inputs(values)
    return values


def validate_attention_inputs(values: MonoAttentionInputs) -> None:
    kv, table, lens, slots = (
        values.kv_cache,
        values.block_table,
        values.seq_lens,
        values.slot_mapping,
    )
    if kv.dtype != torch.bfloat16 or kv.dim() != 4 or tuple(kv.shape[1:]) != (
        2,
        16,
        128,
    ):
        raise ValueError(
            "Muse MonoKernel KV cache must be BF16 [blocks,2,16,128], got "
            f"{tuple(kv.shape)} {kv.dtype}"
        )
    if kv.stride(-1) != 1 or kv.stride(-2) != 128:
        raise ValueError("Muse MonoKernel requires contiguous K|V token rows")
    if table.dtype != torch.int32 or table.dim() != 2 or table.stride(1) != 1:
        raise ValueError("Muse MonoKernel block table must be contiguous int32 [R,B]")
    if lens.dtype != torch.int32 or lens.dim() != 1:
        raise ValueError("Muse MonoKernel seq_lens must be int32 [R]")
    if slots.dtype != torch.int64 or slots.dim() != 1:
        raise ValueError("Muse MonoKernel slot mapping must be int64 [T]")
