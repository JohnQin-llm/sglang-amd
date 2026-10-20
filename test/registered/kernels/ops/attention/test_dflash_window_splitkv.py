# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Window boundaries, cache addressing and replay for DFlash split-KV."""

import math

import pytest
import torch

from sglang.kernels.ops.attention.dflash_window_splitkv import DraftWindowSplitKV
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gfx95_supported(),
    reason="requires gfx95 ROCm",
)


def _reference(q, k, v, keys, values, length, *, k_scale=1.0, v_scale=1.0):
    # The cached prefix uses the cache's query precision; fresh tokens retain
    # BF16. DFlash's current block is noncausal, including its last token.
    scale = 1.0 / math.sqrt(128)
    prefix_scores = (
        q.to(keys.dtype).float().transpose(0, 1) @ keys[:length, 0].float().T
    ) * (scale * k_scale)
    rows = torch.arange(8, device=q.device)[:, None]
    cols = torch.arange(length, device=q.device)[None, :]
    prefix_scores.masked_fill_(length + rows > cols + 2047, -torch.inf)
    current_scores = (q.float().transpose(0, 1) @ k[:, 0].float().T) * scale
    probabilities = torch.cat((prefix_scores, current_scores), dim=-1).softmax(-1)
    all_values = torch.cat(
        (values[:length, 0].float() * v_scale, v[:, 0].float()), dim=0
    )
    return (probabilities @ all_values).transpose(0, 1)


def _buffers(length, dtype):
    # Nonzero indptr and permuted, strided cache slots catch assumptions that
    # physical cache order equals the logical attention window.
    slots = length + 7
    key_storage = torch.empty((2 * slots, 1, 128), device="cuda", dtype=dtype)
    value_storage = torch.empty_like(key_storage)
    keys, values = key_storage[::2], value_storage[::2]
    logical_slots = torch.randperm(slots, device="cuda")[:length]
    indices = torch.zeros(length + 3, device="cuda", dtype=torch.int64)
    indices[3:] = logical_slots
    indptr = torch.tensor([3, 3 + length], device="cuda", dtype=torch.int32)
    qo_indptr = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    return keys, values, logical_slots, indices, indptr, qo_indptr


@pytest.mark.parametrize("length", [0, 1, 65, 2047])
def test_fp8_window_and_future_token_visibility(length):
    torch.manual_seed(7)
    launcher = DraftWindowSplitKV()
    keys, values, slots, indices, indptr, qo = _buffers(length, torch.float8_e4m3fn)
    keys.zero_()
    values.zero_()
    logical_values = torch.zeros((length, 1, 128), dtype=torch.bfloat16, device="cuda")
    logical_values[: min(7, length)] = 64
    # Assign bytes because advanced indexing into FP8 is not supported by all
    # ROCm PyTorch releases.
    values.view(torch.uint8)[slots] = logical_values.to(values.dtype).view(torch.uint8)
    q = torch.zeros((8, 8, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((8, 1, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    v[-1] = 8
    guard = torch.full((8, 10, 128), 87, device="cuda", dtype=q.dtype)
    out = guard[:, :8]
    launcher(q, k, v, out, keys, values, qo, indptr, indices, sm_scale=128**-0.5)
    expected = _reference(
        q,
        k,
        v,
        torch.zeros_like(logical_values).to(keys.dtype),
        logical_values.to(values.dtype),
        length,
    )
    torch.testing.assert_close(out.float(), expected, atol=0.008, rtol=0.004)
    assert bool((guard[:, 8:] == 87).all())


def test_graph_replay_reads_live_metadata_and_strided_queries():
    torch.manual_seed(11)
    launcher = DraftWindowSplitKV()
    keys, values, slots, indices, indptr, qo = _buffers(2047, torch.bfloat16)
    logical_keys = torch.randn((2047, 1, 128), device="cuda", dtype=torch.bfloat16)
    logical_values = torch.randn_like(logical_keys)
    keys[slots] = logical_keys
    values[slots] = logical_values
    q_storage = torch.randn((8, 10, 128), device="cuda", dtype=torch.bfloat16)
    q = q_storage[:, :8]
    original_q = q.clone()
    k = torch.randn((8, 1, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)

    def launch():
        launcher(
            q,
            k,
            v,
            out,
            keys,
            values,
            qo,
            indptr,
            indices,
            sm_scale=128**-0.5,
            k_scale=0.5,
            v_scale=2.0,
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    original_out = out.clone()
    for length in (0, 65, 2047):
        indptr[1].fill_(3 + length)
        q.copy_(original_q * 0.5)
        graph.replay()
        expected = _reference(
            q,
            k,
            v,
            logical_keys,
            logical_values,
            length,
            k_scale=0.5,
            v_scale=2.0,
        )
        error = (out.float() - expected).square().sum()
        signal = expected.square().sum()
        assert bool(torch.isfinite(out).all())
        assert float(10 * torch.log10(signal / error.clamp_min(1e-30))) >= 35
    q.copy_(original_q)
    graph.replay()
    torch.testing.assert_close(out, original_out, rtol=0, atol=0)
