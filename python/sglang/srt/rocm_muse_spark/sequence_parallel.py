"""Sequence-parallel Muse Spark prefill on ROCm, with Ulysses all-to-all attention.

Plain EXTEND batches of at least ``SP_MIN_TOKENS`` tokens run the norms, residuals, router and the replicated
pooler / unpooler on 1/TP of the tokens (each rank keeps a contiguous shard, padded to a multiple of TP).
Attention and the experts still see every token:

* Ulysses (``SGLANG_MUSE_PREFILL_ULYSSES``): each rank projects its own token shard for every head with the
  gathered QKV/gate weight, an all-to-all hands every rank all tokens of its own heads, attention and the output
  norm/gate run as under TP, a second all-to-all returns all heads of the rank's tokens and the full O projection
  finishes them without a reduction. Per 16K tokens and layer that moves ~95 MB per rank instead of the
  all-gather + reduce-scatter's ~450 MB. Without it the shard is all-gathered before attention and the
  attention output reduce-scattered.
* Lean QKV all-to-all (``SGLANG_MUSE_ULYSSES_LEAN_A2A``): the sender runs the QK norm and RoPE on its own
  tokens for every head group and converts K/V to the FP8 cache format with the cache writer's kernel, so the
  first all-to-all carries BF16 Q + FP8 K/V (4608 instead of 9216 bytes per token and head group) and keeps
  the output gate home. The receiver copies the FP8 K/V into its cache, and the output norm and gate run on
  the token owner after the second all-to-all. Every kernel sees the same rows as before, so output and KV
  cache are bitwise equal, and a 16K-token sliding-window layer takes 0.34 ms less.
* Experts: the shard's router outputs and expert inputs are all-gathered (as MXFP4 bytes when the a4w4 experts
  run, since per-row quantization commutes with the gather) and the expert output is reduce-scattered.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

SP_MIN_TOKENS = 2048
_PREFILL_QKV_LAYOUT = os.environ.get("SGLANG_MUSE_TP4_PREFILL_QKV_LAYOUT", "0") == "1"
if _PREFILL_QKV_LAYOUT:
    from sglang.srt.rocm_muse_spark.prefill_qkv_layout import run as _qkv_layout_run

_PREFILL_SP = envs.SGLANG_MUSE_PREFILL_SP.get()
_PREFILL_ULYSSES = envs.SGLANG_MUSE_PREFILL_ULYSSES.get()
_LEAN_ALL_TO_ALL = envs.SGLANG_MUSE_ULYSSES_LEAN_A2A.get()


@dataclass(frozen=True)
class SPShape:
    """Token counts of one sequence-parallel prefill: real rows and rows padded to a multiple of TP."""

    num_tokens: int
    padded: int


def sp_shape(forward_batch: ForwardBatch, num_tokens: int) -> SPShape | None:
    if not (
        _PREFILL_SP
        and forward_batch.forward_mode == ForwardMode.EXTEND
        and num_tokens >= SP_MIN_TOKENS
    ):
        return None
    world = get_tp_group().world_size
    if world == 1:
        return None
    return SPShape(num_tokens, -(-num_tokens // world) * world)


def sp_shard(stream: torch.Tensor, sp: SPShape) -> torch.Tensor:
    """This rank's contiguous token shard of ``stream`` (padded to ``sp.padded`` rows)."""
    group = get_tp_group()
    stream = F.pad(stream, (0, 0, 0, sp.padded - sp.num_tokens))
    return stream.tensor_split(group.world_size)[group.rank_in_group].contiguous()


def sp_gather(shard: torch.Tensor, rows: int) -> torch.Tensor:
    """All-gather token shards and keep the first ``rows`` rows."""
    group = get_tp_group()
    shard = shard.contiguous()
    if shard.dtype == torch.uint8 and shard.dim() == 2 and shard.shape[1] % 2 == 0:
        # MXFP4 rows / E8M0 scales: gathered as bf16 words, so the aiter custom all-gather (f32 / f16 / bf16
        # only) carries them instead of RCCL; a gather copies bytes, the result is exact.
        return sp_gather(shard.view(torch.bfloat16), rows).view(torch.uint8)
    out = shard.new_empty((shard.shape[0] * group.world_size, *shard.shape[1:]))
    group.all_gather_into_tensor(out, shard)
    return out[:rows]


def sp_reduce_scatter(full: torch.Tensor, padded: int) -> torch.Tensor:
    """Sum ``full`` over ranks and return this rank's token shard."""
    group = get_tp_group()
    if full.shape[0] != padded:
        full = F.pad(full, (0, 0, 0, padded - full.shape[0]))
    out = full.new_empty((padded // group.world_size, *full.shape[1:]))
    group.reduce_scatter_tensor(out, full.contiguous())
    return out


def ulysses_enabled() -> bool:
    return _PREFILL_SP and _PREFILL_ULYSSES and get_tp_group().world_size > 1


@torch.no_grad()
def prepare_ulysses(attention) -> None:
    """Gather the TP shards of the QKV/gate and O projections once after loading (rank-major head order)."""
    group = get_tp_group()
    attention.ulysses_qkv_weight = group.all_gather(
        attention.qkv_proj.weight.data.contiguous(), dim=0
    )
    attention.ulysses_o_weight = group.all_gather(
        attention.o_proj.weight.data.contiguous(), dim=1
    ).contiguous()


def _all_to_all(x: torch.Tensor, group) -> torch.Tensor:
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group.device_group)
    return out


@lru_cache(maxsize=8)
def _qkv_layout_platform(device):
    return bool(torch.version.hip) and torch.cuda.get_device_properties(
        device
    ).gcnArchName.startswith("gfx950")


def _qkv_layout_supported(attention, x, world):
    if not _PREFILL_QKV_LAYOUT or world != 4:
        return False
    from sglang.srt.runtime_context import get_parallel

    weight = attention.ulysses_qkv_weight
    return (
        get_parallel().moe_ep_size == 4
        and x.ndim == 2
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[0] == 4096
        and x.shape[1] == 8192
        and weight.shape == (18432, 8192)
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and weight.device == x.device
        and _qkv_layout_platform(x.device)
    )


def _qkv_rank_major_layout(x, weight):
    return _qkv_layout_run(x, weight)


def _project_qkv_by_head_group(attention, x: torch.Tensor, world: int) -> torch.Tensor:
    """[rows, hidden] shard -> [world, rows, q|k|v|gate of head group d] for every head group d."""
    if _qkv_layout_supported(attention, x, world):
        return _qkv_rank_major_layout(x, attention.ulysses_qkv_weight)
    qkv = F.linear(x, attention.ulysses_qkv_weight).view(x.shape[0], world, -1)
    return qkv.transpose(0, 1).contiguous()


def ulysses_attention(
    attention,
    x: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    sp: SPShape,
) -> torch.Tensor:
    """Attention block for this rank's token shard ``x`` [padded / world, hidden]; returns its output rows."""
    backend = (
        _lean_cache_backend(attention, forward_batch) if _LEAN_ALL_TO_ALL else None
    )
    if backend is not None:
        return _lean_ulysses_attention(
            attention, x, positions, forward_batch, sp, backend
        )
    group = get_tp_group()
    world, rows = group.world_size, x.shape[0]
    # all-to-all -> [world * rows, q|k|v|gate of this rank's heads]
    qkv = _project_qkv_by_head_group(attention, x, world)
    qkv = _all_to_all(qkv, group).view(world * rows, -1)
    q, k, v, o_gate = qkv[: sp.num_tokens].split(
        [attention.q_size, attention.kv_size, attention.kv_size, attention.q_size],
        dim=-1,
    )
    q, k = attention.qk_norm_rope(positions, q, k)
    out = attention.attn(q, k, v, forward_batch)
    out = attention._fused_o_norm_gate.forward(out, o_gate)
    # [tokens, this rank's heads] -> all-to-all -> full O projection of this rank's token shard
    out = F.pad(out, (0, 0, 0, sp.padded - out.shape[0])).contiguous()
    shard_rows = sp.padded // world
    heads = _all_to_all(out, group).view(world, shard_rows, -1).transpose(0, 1)
    return F.linear(heads.reshape(shard_rows, -1), attention.ulysses_o_weight)


def _lean_cache_backend(attention, forward_batch: ForwardBatch):
    """The attention backend when its extend stores this layer's K/V with ``launch_reshape_and_cache_flash``
    at the backend's scales and reads them back from the cache, the branch the lean all-to-all reproduces."""
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    layer = attention.attn
    if (
        isinstance(backend, AiterAttnBackend)
        and not forward_batch.attn_attend_prefix_cache
        and not backend.use_mla
        and not backend.kv_cache_is_vectorized_5d
        and backend.use_triton_unified_attention
        and backend.use_sliding_window_kv_pool
        and backend.kv_cache_dtype == torch.float8_e4m3fn
        and layer.k_scale is None
        and layer.v_scale is None
    ):
        return backend
    return None


def _as_one_token_pages(rows: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """[tokens, heads * head_dim] rows (any row stride) as a paged cache of one token per page."""
    stride = rows.stride(0)
    return rows.as_strided(
        (rows.shape[0], 1, heads, head_dim), (stride, stride, head_dim, 1)
    )


def _lean_ulysses_attention(
    attention, x, positions, forward_batch, sp: SPShape, backend
):
    """``ulysses_attention`` with BF16 Q + FP8 K/V in the first all-to-all (see the module docstring)."""
    from sglang.kernels.ops.kvcache.cache_ops import launch_reshape_and_cache_flash

    group = get_tp_group()
    world, rows, rank = group.world_size, x.shape[0], group.rank_in_group
    q_size, kv_size, head_dim = attention.q_size, attention.kv_size, attention.head_dim
    kv_heads = kv_size // head_dim
    # Row d * rows + i: this rank's token i, head group d, the row order the receiver gets.
    projected = _project_qkv_by_head_group(attention, x, world).view(world * rows, -1)
    q, k, v, o_gate = projected.split([q_size, kv_size, kv_size, q_size], dim=-1)
    token_positions = F.pad(positions, (0, sp.padded - sp.num_tokens))[
        rank * rows : (rank + 1) * rows
    ]
    q, k = attention.qk_norm_rope(token_positions.repeat(world), q, k)
    # Payload rows: BF16 q | FP8 k | FP8 v, the K/V written by the cache writer at the backend's scales.
    q_bytes = q_size * q.element_size()
    payload = torch.empty(
        (world * rows, q_bytes + 2 * kv_size), dtype=torch.uint8, device=x.device
    )
    payload[:, :q_bytes].view(q.dtype).copy_(q)
    fp8 = backend.kv_cache_dtype
    launch_reshape_and_cache_flash(
        k.view(-1, kv_heads, head_dim),
        v.reshape(-1, kv_heads, head_dim),
        _as_one_token_pages(
            payload[:, q_bytes : q_bytes + kv_size].view(fp8), kv_heads, head_dim
        ),
        _as_one_token_pages(
            payload[:, q_bytes + kv_size :].view(fp8), kv_heads, head_dim
        ),
        torch.arange(world * rows, device=x.device),
        k_scale=backend.k_scale,
        v_scale=backend.v_scale,
    )
    received = _all_to_all(payload.view(world, rows, -1), group).view(world * rows, -1)[
        : sp.num_tokens
    ]
    q = received[:, :q_bytes].view(q.dtype)
    k = received[:, q_bytes : q_bytes + kv_size].view(fp8)
    v = received[:, q_bytes + kv_size :].view(fp8)
    # The backend's cache write without its conversion, which the sender already ran.
    layer = attention.attn
    k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    launch_reshape_and_cache_flash(
        k.view(-1, kv_heads, head_dim),
        v.view(-1, kv_heads, head_dim),
        k_cache.view(-1, backend.page_size, kv_heads, head_dim),
        v_cache.view(-1, backend.page_size, kv_heads, head_dim),
        forward_batch.out_cache_loc,
        (
            backend.swa_kv_pool.full_to_swa_index_mapping.long()
            if layer.sliding_window_size > 0
            else None
        ),
    )
    out = attention.attn(q.contiguous(), k, v, forward_batch, save_kv_cache=False)
    # [tokens, this rank's heads] -> all-to-all -> this rank's tokens by head group; norm and gate here
    out = F.pad(out, (0, 0, 0, sp.padded - out.shape[0])).contiguous()
    heads = attention._fused_o_norm_gate.forward(_all_to_all(out, group), o_gate)
    heads = heads.view(world, rows, -1).transpose(0, 1)
    return F.linear(heads.reshape(rows, -1), attention.ulysses_o_weight)
