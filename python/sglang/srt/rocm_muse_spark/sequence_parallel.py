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
* Lean QKV all-to-all (``SGLANG_MUSE_ULYSSES_LEAN_A2A``): the sender runs the per-head QK norm and RoPE on its
  own token shard for every head group, converts K/V to the FP8 cache format there (same kernel as the cache
  writer), and keeps the output gate; the first all-to-all then carries bf16 Q + FP8 K/V (4608 instead of 9216
  bytes per token and head group), the receiver copies K/V into its cache, and the output norm/gate runs after
  the second all-to-all on the token owner. Every kernel sees the same per-row inputs as before: bitwise equal.
* Experts: the shard's router outputs and expert inputs are all-gathered (as MXFP4 bytes when the a4w4 experts
  run, since per-row quantization commutes with the gather) and the expert output is reduce-scattered.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

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
_LEAN_A2A = os.environ.get("SGLANG_MUSE_ULYSSES_LEAN_A2A", "0") == "1"
_PREFILL_ULYSSES = envs.SGLANG_MUSE_PREFILL_ULYSSES.get()
_ALIAS_TP_QKV = os.environ.get("SGLANG_MUSE_ULYSSES_ALIAS_QKV", "0") == "1"


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
    """Gather the TP shards of the QKV/gate and O projections once after loading (rank-major head order).

    With ``SGLANG_MUSE_ULYSSES_ALIAS_QKV`` the rank's own QKV/gate shard is then read from its rows of the gathered
    weight (same values, same contiguous layout) and the separate copy is freed: 72 MiB per layer at TP4."""
    group = get_tp_group()
    weight = attention.qkv_proj.weight
    gathered = group.all_gather(weight.data.contiguous(), dim=0)
    if _ALIAS_TP_QKV:
        rows = weight.shape[0]
        own = gathered[group.rank_in_group * rows:(group.rank_in_group + 1) * rows]
        assert own.is_contiguous() and own.shape == weight.shape and torch.equal(own, weight.data)
        weight.data = own
    attention.ulysses_qkv_weight = gathered
    attention.ulysses_o_weight = group.all_gather(
        attention.o_proj.weight.data.contiguous(), dim=1
    ).contiguous()


def _all_to_all(x: torch.Tensor, group) -> torch.Tensor:
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group.device_group)
    return out


@lru_cache(maxsize=8)
def _qkv_layout_platform(device):
    return bool(torch.version.hip) and torch.cuda.get_device_properties(device).gcnArchName.startswith('gfx950')


def _qkv_layout_supported(attention, x, world):
    if not _PREFILL_QKV_LAYOUT or world!=4:
        return False
    from sglang.srt.runtime_context import get_parallel
    weight=attention.ulysses_qkv_weight
    return (get_parallel().moe_ep_size==4 and x.ndim==2 and x.is_cuda
            and x.dtype==torch.bfloat16 and x.is_contiguous()
            and x.shape[0]==4096 and x.shape[1]==8192
            and weight.shape==(18432,8192) and weight.dtype==torch.bfloat16
            and weight.is_contiguous() and weight.device==x.device
            and _qkv_layout_platform(x.device))


def _qkv_rank_major_layout(x, weight):
    return _qkv_layout_run(x,weight)


def ulysses_attention(
    attention,
    x: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    sp: SPShape,
) -> torch.Tensor:
    """Attention block for this rank's token shard ``x`` [padded / world, hidden]; returns its output rows."""
    group = get_tp_group()
    world, rows = group.world_size, x.shape[0]
    backend = _lean_backend(attention, forward_batch) if _LEAN_A2A else None
    if backend is not None:
        return _lean_ulysses_attention(attention, x, positions, forward_batch, sp, backend)
    # [rows, hidden] shard -> all-to-all -> [world * rows, q|k|v|gate of this rank's heads]
    if _qkv_layout_supported(attention, x, world):
        qkv = _qkv_rank_major_layout(x, attention.ulysses_qkv_weight)
    else:
        qkv = F.linear(x, attention.ulysses_qkv_weight).view(rows, world, -1)
        qkv = qkv.transpose(0, 1).contiguous()
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


def _lean_backend(attention, forward_batch: ForwardBatch):
    """The attention backend when it would store this layer's K/V with ``launch_reshape_and_cache_flash`` and run
    the plain unified-attention extend (the only branch the lean path reproduces), else None."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    layer = attention.attn
    return backend if (
        type(backend).__name__ == "AiterAttnBackend"
        and forward_batch.forward_mode == ForwardMode.EXTEND
        and not forward_batch.attn_attend_prefix_cache
        and not getattr(backend, "use_mla", False)
        and not backend.kv_cache_is_vectorized_5d
        and backend.use_triton_unified_attention
        and backend.use_sliding_window_kv_pool
        and backend.kv_cache_dtype == torch.float8_e4m3fn
        and not layer.is_cross_attention
        and layer.k_scale is None
        and layer.v_scale is None
        and layer.qk_head_dim == layer.v_head_dim == attention.head_dim
        and layer.tp_k_head_num == layer.tp_v_head_num == attention.num_kv_heads
        and getattr(attention, "use_o_gate", False)
        and not getattr(attention, "use_attn_temperature_tuning", False)
    ) else None


def _lean_ulysses_attention(attention, x, positions, forward_batch, sp, backend):
    """``ulysses_attention`` with a bf16 Q + FP8 K/V payload in the first all-to-all (see the module docstring)."""
    from sglang.kernels.ops.kvcache.cache_ops import launch_reshape_and_cache_flash

    group = get_tp_group()
    world, rows, rank = group.world_size, x.shape[0], group.rank_in_group
    q_size, kv_size, head_dim = attention.q_size, attention.kv_size, attention.head_dim
    kv_heads = kv_size // head_dim
    if _qkv_layout_supported(attention, x, world):
        qkv = _qkv_rank_major_layout(x, attention.ulysses_qkv_weight)
    else:
        qkv = F.linear(x, attention.ulysses_qkv_weight).view(rows, world, -1)
        qkv = qkv.transpose(0, 1).contiguous()
    # Row (d, i): this rank's token i, head group d. Same per-row layout as the receiver's rows below.
    flat = qkv.view(world * rows, -1)
    q, k, v, o_gate = flat.split([q_size, kv_size, kv_size, q_size], dim=-1)
    token_pos = F.pad(positions, (0, sp.padded - sp.num_tokens))[rank * rows:(rank + 1) * rows]
    q, k = attention.qk_norm_rope(token_pos.repeat(world), q, k)
    # Payload rows: bf16 q | fp8 k | fp8 v. K/V go through the cache writer's own conversion kernel, writing into
    # the payload as a one-token-per-block "cache".
    q_bytes = q_size * 2
    payload = torch.empty((world * rows, q_bytes + 2 * kv_size), dtype=torch.uint8, device=x.device)
    payload[:, :q_bytes].view(torch.bfloat16).copy_(q)
    fp8 = torch.float8_e4m3fn
    k8 = payload[:, q_bytes:q_bytes + kv_size].view(fp8)
    v8 = payload[:, q_bytes + kv_size:].view(fp8)
    layer = attention.attn
    k_descale = backend.k_scale
    v_descale = backend.v_scale
    as_cache = lambda t: t.as_strided((t.shape[0], 1, kv_heads, head_dim), (t.stride(0), t.stride(0), head_dim, 1))
    launch_reshape_and_cache_flash(
        k.view(-1, kv_heads, head_dim), v.reshape(-1, kv_heads, head_dim), as_cache(k8), as_cache(v8),
        torch.arange(world * rows, device=x.device), None, k_scale=k_descale, v_scale=v_descale,
    )
    received = _all_to_all(payload.view(world, rows, -1), group).view(world * rows, -1)[: sp.num_tokens]
    q = received[:, :q_bytes].view(torch.bfloat16)
    k8 = received[:, q_bytes:q_bytes + kv_size].view(fp8)
    v8 = received[:, q_bytes + kv_size:].view(fp8)
    # The cache write the backend would do, minus its bf16 -> fp8 conversion (already done, same kernel).
    k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    page = backend.page_size
    launch_reshape_and_cache_flash(
        k8.view(-1, kv_heads, head_dim), v8.view(-1, kv_heads, head_dim),
        k_cache.view(-1, page, kv_heads, head_dim), v_cache.view(-1, page, kv_heads, head_dim),
        forward_batch.out_cache_loc,
        backend.swa_kv_pool.full_to_swa_index_mapping.long() if layer.sliding_window_size > 0 else None,
    )
    out = attention.attn(q.contiguous(), k8, v8, forward_batch, save_kv_cache=False)
    # [tokens, this rank's heads] -> all-to-all -> this rank's tokens by head group; norm + gate on the token owner
    out = F.pad(out, (0, 0, 0, sp.padded - out.shape[0])).contiguous()
    heads = _all_to_all(out, group)
    heads = attention._fused_o_norm_gate.forward(heads, o_gate).view(world, rows, -1).transpose(0, 1)
    return F.linear(heads.reshape(rows, -1), attention.ulysses_o_weight)
