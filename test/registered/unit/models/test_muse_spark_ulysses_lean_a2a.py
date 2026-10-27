"""Muse Spark Ulysses prefill attention with the lean QKV all-to-all
(SGLANG_MUSE_ULYSSES_LEAN_A2A) against the default all-to-all on 4 GPUs: the
layer output and the FP8 KV cache must be bitwise equal."""

import functools
import multiprocessing
import os
import socket
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-c-test-large-8-gpu-amd-mi35x")

WORLD, HIDDEN, Q_HEADS, KV_HEADS, HEAD_DIM, PAGE = 4, 1024, 8, 2, 64, 16
Q_SIZE, KV_SIZE = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
FP8 = torch.float8_e4m3fn
# (new tokens, cached prefix) per sequence; totals not divisible by 4 pad the shards.
CASES = {
    "one sequence with a prefix": [(2051, 700)],
    "two sequences": [(1030, 0), (1023, 333)],
}


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class _CachedAttention:
    """RadixAttention through the AiterAttnBackend extend branch the lean path
    reproduces: the FP8 cache write (launch_reshape_and_cache_flash at the backend's
    scales, through the SWA slot map on sliding layers), then attention over the
    cached K/V of each sequence."""

    k_scale = v_scale = None
    layer_id = 0

    def __init__(self, backend, seqs, window: int):
        self.backend, self.seqs = backend, seqs
        self.sliding_window_size = window

    def slots(self, slots: torch.Tensor) -> torch.Tensor:
        if self.sliding_window_size <= 0:
            return slots
        return self.backend.swa_kv_pool.full_to_swa_index_mapping[slots].long()

    def __call__(self, q, k, v, forward_batch, save_kv_cache=True):
        from sglang.kernels.ops.kvcache.cache_ops import launch_reshape_and_cache_flash

        b = self.backend
        k_cache, v_cache = b.token_to_kv_pool.get_kv_buffer(self.layer_id)
        if save_kv_cache:
            launch_reshape_and_cache_flash(
                k.view(-1, KV_HEADS, HEAD_DIM),
                v.view(-1, KV_HEADS, HEAD_DIM),
                k_cache.view(-1, PAGE, KV_HEADS, HEAD_DIM),
                v_cache.view(-1, PAGE, KV_HEADS, HEAD_DIM),
                forward_batch.out_cache_loc,
                (
                    b.swa_kv_pool.full_to_swa_index_mapping.long()
                    if self.sliding_window_size > 0
                    else None
                ),
                k_scale=b.k_scale,
                v_scale=b.v_scale,
            )
        q = q.view(-1, Q_HEADS, HEAD_DIM).float()
        out, start = [], 0
        for base, new, prefix in self.seqs:
            slots = self.slots(torch.arange(base, base + prefix + new, device=q.device))
            keys = k_cache[slots].float() * b.k_scale
            values = v_cache[slots].float() * b.v_scale
            keys = keys.repeat_interleave(Q_HEADS // KV_HEADS, dim=1)
            values = values.repeat_interleave(Q_HEADS // KV_HEADS, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q[start : start + new], keys)
            query_pos = prefix + torch.arange(new, device=q.device)[:, None]
            key_pos = torch.arange(prefix + new, device=q.device)[None, :]
            visible = key_pos <= query_pos
            if self.sliding_window_size > 0:
                visible &= key_pos > query_pos - self.sliding_window_size
            scores = scores.masked_fill(~visible, float("-inf")) * HEAD_DIM**-0.5
            out.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values))
            start += new
        return torch.cat(out).to(torch.bfloat16).view(-1, Q_SIZE)


def _layer(rank: int, seqs, sliding: bool):
    """One attention layer on this rank: shard, gathered weights and an FP8 cache
    whose prefix slots hold random K/V."""
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.rocm_muse_spark.fused_norm_gate import FusedRMSNormGate
    from sglang.srt.rocm_muse_spark.layers import MuseSpark1xRMSNorm
    from sglang.srt.rocm_muse_spark.model import MuseSpark1xAttention
    from sglang.srt.rocm_muse_spark.sequence_parallel import SPShape

    g = torch.Generator(device="cuda").manual_seed(0)
    tokens = sum(new for new, _ in seqs)
    sp = SPShape(tokens, -(-tokens // WORLD) * WORLD)
    rows = sp.padded // WORLD
    stream = torch.randn(sp.padded, HIDDEN, device="cuda", generator=g).bfloat16()
    stream[tokens:] = 0
    group_width = 2 * Q_SIZE + 2 * KV_SIZE
    qkv_weight = torch.randn(WORLD * group_width, HIDDEN, device="cuda", generator=g)
    o_weight = torch.randn(HIDDEN, WORLD * Q_SIZE, device="cuda", generator=g)

    used = sum(new + prefix for new, prefix in seqs)
    slots = (used // PAGE + 1) * PAGE  # whole pages, one spare
    cache_rng = torch.Generator(device="cuda").manual_seed(1 + rank)
    k_cache = torch.randn(slots, KV_HEADS, HEAD_DIM, device="cuda", generator=cache_rng)
    v_cache = torch.randn(slots, KV_HEADS, HEAD_DIM, device="cuda", generator=cache_rng)
    caches = (k_cache.to(FP8), v_cache.to(FP8))
    backend = object.__new__(AiterAttnBackend)
    backend.__dict__.update(
        use_mla=False,
        kv_cache_is_vectorized_5d=False,
        use_triton_unified_attention=True,
        use_sliding_window_kv_pool=True,
        kv_cache_dtype=FP8,
        page_size=PAGE,
        k_scale=torch.tensor([0.75], device="cuda"),
        v_scale=torch.tensor([1.25], device="cuda"),
        token_to_kv_pool=SimpleNamespace(get_kv_buffer=lambda layer_id: caches),
        swa_kv_pool=SimpleNamespace(
            full_to_swa_index_mapping=torch.arange(
                slots - 1, -1, -1, dtype=torch.int32, device="cuda"
            )
        ),
    )
    locations, positions, layout, base = [], [], [], 0
    for new, prefix in seqs:
        locations.append(base + prefix + torch.arange(new, device="cuda"))
        positions.append(prefix + torch.arange(new, device="cuda"))
        layout.append((base, new, prefix))
        base += new + prefix

    norm = SimpleNamespace(
        use_qk_norm=True,
        head_dim=HEAD_DIM,
        norm_eps=1e-5,
        scale_query_by=0.5,
        use_attn_temperature_tuning=False,
        rotary_emb=(
            get_rope(
                HEAD_DIM,
                rotary_dim=HEAD_DIM,
                max_position=1 << 16,
                base=500000,
                is_neox_style=False,
                dtype=torch.float32,
            ).cuda()
            if sliding
            else None
        ),
    )
    norm._per_head_norm = functools.partial(MuseSpark1xAttention._per_head_norm, norm)
    attention = SimpleNamespace(
        ulysses_qkv_weight=qkv_weight.mul_(0.02).bfloat16(),
        ulysses_o_weight=o_weight.mul_(0.02).bfloat16(),
        q_size=Q_SIZE,
        kv_size=KV_SIZE,
        head_dim=HEAD_DIM,
        qk_norm_rope=functools.partial(MuseSpark1xAttention.qk_norm_rope, norm),
        attn=_CachedAttention(backend, layout, 127 if sliding else -1),
        _fused_o_norm_gate=FusedRMSNormGate(
            MuseSpark1xRMSNorm(HEAD_DIM, eps=1e-5, has_weight=False), True, HEAD_DIM
        ),
    )
    forward_batch = SimpleNamespace(
        attn_attend_prefix_cache=False, out_cache_loc=torch.cat(locations)
    )
    shard = stream[rank * rows : (rank + 1) * rows].contiguous()
    return attention, shard, torch.cat(positions), forward_batch, sp, backend, caches


def _run_rank(rank: int, port: int) -> None:
    # Single node: RCCL's bootstrap must use loopback like the 127.0.0.1 rendezvous,
    # not the first of the host's other interfaces (it hangs there).
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD
    )
    from sglang.srt.rocm_muse_spark import sequence_parallel

    group = SimpleNamespace(
        world_size=WORLD, rank_in_group=rank, device_group=dist.group.WORLD
    )
    try:
        for name, seqs in CASES.items():
            for sliding in (True, False):
                attention, x, positions, forward_batch, sp, backend, caches = _layer(
                    rank, seqs, sliding
                )
                initial = [c.clone() for c in caches]
                results = {}
                for lean in (False, True):
                    for cache, start in zip(caches, initial):
                        cache.copy_(start)
                    with (
                        patch.object(
                            sequence_parallel, "get_tp_group", return_value=group
                        ),
                        patch.object(sequence_parallel, "_LEAN_ALL_TO_ALL", lean),
                        patch(
                            "sglang.srt.model_executor.forward_context.get_attn_backend",
                            return_value=backend,
                        ),
                    ):
                        # The comparison is only meaningful if the lean path runs.
                        assert (
                            sequence_parallel._lean_cache_backend(
                                attention, forward_batch
                            )
                            is backend
                        )
                        out = sequence_parallel.ulysses_attention(
                            attention, x, positions, forward_batch, sp
                        )
                    results[lean] = (
                        out,
                        *(c.view(torch.uint8).clone() for c in caches),
                    )
                for label, a, b in zip(
                    ("output", "K cache", "V cache"), *results.values()
                ):
                    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
                        f"{name}, sliding={sliding}: {label} differs on rank {rank}"
                    )
                written = forward_batch.out_cache_loc
                if sliding:
                    written = backend.swa_kv_pool.full_to_swa_index_mapping[
                        written
                    ].long()
                assert not torch.equal(
                    results[True][1][written], initial[0].view(torch.uint8)[written]
                )
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    is_hip() and is_gfx95_supported() and torch.cuda.device_count() >= WORLD,
    "4 gfx950 GPUs",
)
class TestMuseSparkUlyssesLeanAllToAll(CustomTestCase):
    def test_lean_all_to_all_is_bitwise_equal(self):
        """Sliding-window and full layers, one and two sequences, cached prefixes
        and padded shards: the lean path writes the same FP8 K/V bytes and returns
        the same output bytes on every rank."""
        port = _open_port()
        ctx = multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=_run_rank, args=(r, port)) for r in range(WORLD)]
        for p in procs:
            p.start()
        deadline = time.monotonic() + 600
        for p in procs:
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        for p in procs:
            if p.is_alive():
                p.kill()
        self.assertEqual([p.exitcode for p in procs], [0] * WORLD)


if __name__ == "__main__":
    unittest.main()
