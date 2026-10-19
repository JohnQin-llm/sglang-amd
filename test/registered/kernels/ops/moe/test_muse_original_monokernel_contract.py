# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the explicit Muse original TP8/W4A4 port."""

from types import SimpleNamespace

import torch
from sglang.srt.mem_cache.memory_pool import MuseMonoMHATokenToKVPool
from sglang.srt.rocm_muse_spark.layers import (
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.rocm_muse_spark.model import MuseSpark1xRouter, _tp_expert_stack_slice
from sglang.srt.rocm_muse_spark.mono.cache import (
    MonoAttentionInputs,
    validate_attention_inputs,
)
from sglang.srt.rocm_muse_spark.mono.sglang_spec import config_disable_reasons
from sglang.srt.rocm_muse_spark.mono.spec import kernel_layers
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")


def test_original_router_layout_matches_k2_and_native_math(monkeypatch):
    monkeypatch.setenv("SGLANG_MUSE_ORIGINAL_NATIVE_ONLY", "1")
    router = MuseSpark1xRouter(16, 4, torch.float32)
    assert router.weight.shape == (4, 16)
    weight = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    router.weight.data.copy_(weight)
    x = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    torch.testing.assert_close(router(x), x @ weight.t(), rtol=0, atol=0)


def test_tp8_gate_up_and_down_sharding():
    experts, tp = 2, 8
    w13 = torch.arange(experts * 32 * 3).reshape(experts, 32, 3)
    shard = _tp_expert_stack_slice(w13, torch.Size((experts, 4, 3)), "w13_weight", 3, tp)
    expected = torch.cat((w13[:, 6:8], w13[:, 22:24]), dim=1)
    torch.testing.assert_close(shard, expected, rtol=0, atol=0)

    w2 = torch.arange(experts * 5 * 32).reshape(experts, 5, 32)
    shard = _tp_expert_stack_slice(w2, torch.Size((experts, 5, 4)), "w2_weight", 3, tp)
    torch.testing.assert_close(shard, w2[:, :, 12:16], rtol=0, atol=0)


def test_original_norms_keep_fp32_arithmetic_until_store():
    x = torch.tensor([[0.5, -1.25, 2.0, -0.75]], dtype=torch.bfloat16)
    norm = MuseSpark1xRMSNorm(4, eps=1e-5, zero_centered_gamma=True)
    norm.original_precision = True
    norm.weight.data.copy_(torch.tensor([0.5, -0.25, 0.125, 0.0]))
    got = norm(x, out_dtype=torch.float32)
    xf = x.float()
    expected = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
    expected = expected * (norm.weight.float() + 1.0)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    post = MuseSpark1xRMSNormWithInputScale(
        4, eps=1e-8, zero_centered_gamma=True
    )
    post.original_precision = True
    post.weight.data.copy_(norm.weight)
    got = post(x, out_dtype=torch.float32)
    scaled = xf * (post.weight.float() + 1.0)
    expected = torch.nn.functional.rms_norm(scaled, (4,), None, 1e-8)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_combined_cache_native_views_share_kv_storage():
    backing = torch.arange(3 * 2 * 16 * 128, dtype=torch.float32).reshape(
        3, 2, 16, 128
    )
    pool = object.__new__(MuseMonoMHATokenToKVPool)
    pool.start_layer = 0
    pool._muse_kv_buffer = [backing]
    pool.k_buffer = [backing[..., :64]]
    pool.v_buffer = [backing[..., 64:]]
    key = pool.get_key_buffer(0)
    value = pool.get_value_buffer(0)
    assert key.shape == value.shape == (3, 16, 2, 64)
    key[1, 2, 0, 3] = -7
    value[1, 2, 0, 3] = -9
    assert backing[1, 0, 2, 3] == -7
    assert backing[1, 0, 2, 67] == -9


def test_cache_metadata_contract_and_q8_guard():
    values = MonoAttentionInputs(
        kv_cache=torch.empty(4, 2, 16, 128, dtype=torch.bfloat16),
        block_table=torch.zeros(2, 8, dtype=torch.int32),
        seq_lens=torch.tensor([20, 18], dtype=torch.int32),
        slot_mapping=torch.tensor([20, 18], dtype=torch.int64),
    )
    validate_attention_inputs(values)

    text = SimpleNamespace(
        hidden_size=8192,
        num_attention_heads=128,
        num_key_value_heads=16,
        head_dim=64,
        num_local_experts=256,
        num_experts_per_tok=8,
        moe_hidden_size=4096,
        moe_intermediate_size=4096,
        sliding_window=2048,
    )
    cfg = SimpleNamespace(
        page_size=16,
        kv_cache_dtype="bfloat16",
        speculative_algorithm="EAGLE",
    )
    assert "rows=8 target-verify is not production-qualified" in config_disable_reasons(
        cfg, text
    )


def test_kernel_layers_exclude_first_last_and_full_attention():
    config = SimpleNamespace(
        num_hidden_layers=6,
        mlp_layer_types=["sparse"] * 6,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
        ],
    )
    assert kernel_layers(config) == [1, 3, 4]
