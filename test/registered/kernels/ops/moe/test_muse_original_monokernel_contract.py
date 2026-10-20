# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the explicit Muse original TP8/W4A4 port."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import MuseMonoMHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.rocm_muse_spark.layers import (
    MuseSpark1xResidualStream,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.rocm_muse_spark.model import (
    MuseSpark1xRouter,
    _original_boundary,
    _tp_expert_stack_slice,
)
from sglang.srt.rocm_muse_spark.mono.cache import (
    MonoAttentionInputs,
    prepare_seq_lens,
    validate_attention_inputs,
)
from sglang.srt.rocm_muse_spark.mono.sglang_spec import config_disable_reasons
from sglang.srt.rocm_muse_spark.mono.spec import kernel_layers
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")


def test_dflash_kv_lengths_include_verify_block_and_reuse_storage():
    # The block crosses a 16-token KV page. SGLang supplies the accepted
    # prefix, whereas K1 derives query i's boundary as total - q_len + i.
    batch = SimpleNamespace(
        seq_lens=torch.tensor([15], dtype=torch.int64),
        forward_mode=ForwardMode.TARGET_VERIFY,
    )
    storage = torch.zeros(8, dtype=torch.int32)
    lengths = prepare_seq_lens(batch, storage, 8)
    assert lengths.tolist() == [23]
    assert lengths.data_ptr() == storage.data_ptr()
    assert (lengths[0] - 8 + torch.arange(8)).tolist() == list(range(15, 23))
    assert batch.seq_lens.tolist() == [15]
    batch.seq_lens.fill_(31)
    replay_lengths = prepare_seq_lens(batch, storage, 8)
    assert replay_lengths.tolist() == [39]
    assert replay_lengths.data_ptr() == lengths.data_ptr()

    batch.forward_mode = ForwardMode.DECODE
    assert prepare_seq_lens(batch, storage, 1).tolist() == [31]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph")
def test_dflash_kv_lengths_read_live_prefix_during_graph_replay():
    batch = SimpleNamespace(
        seq_lens=torch.tensor([15], device="cuda", dtype=torch.int64),
        forward_mode=ForwardMode.TARGET_VERIFY,
    )
    storage = torch.zeros(8, device="cuda", dtype=torch.int32)
    prepare_seq_lens(batch, storage, 8)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = prepare_seq_lens(batch, storage, 8)
    batch.seq_lens.fill_(31)
    graph.replay()
    assert result.tolist() == [39]
    assert result.data_ptr() == storage.data_ptr()
    graph.reset()


def test_original_router_layout_matches_k2_and_native_math(monkeypatch):
    monkeypatch.setenv("SGLANG_MUSE_ORIGINAL_NATIVE_ONLY", "1")
    router = MuseSpark1xRouter(16, 4, torch.float32)
    assert router.weight.shape == (4, 16)
    weight = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    router.weight.data.copy_(weight)
    x = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    torch.testing.assert_close(router(x), x @ weight.t(), rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires GPU native boundary"
)
def test_weightless_post_norm_gpu_boundary():
    post = MuseSpark1xRMSNorm(
        8192, eps=1e-8, has_weight=False, zero_centered_gamma=True
    )
    pre = MuseSpark1xRMSNorm(8192, eps=1e-5).cuda().to(torch.bfloat16)
    pre.weight.data.fill_(1.0)
    branch = torch.randn(32, 8192, device="cuda", dtype=torch.bfloat16)
    stream = torch.randn(32, 8192, device="cuda", dtype=torch.float32)
    connection = SimpleNamespace(
        _coefficients_ready=True,
        alpha=torch.full((8192,), 0.6, device="cuda"),
        beta=torch.full((8192,), 0.8, device="cuda"),
    )
    got, got_stream = _original_boundary(
        MuseSpark1xResidualStream(stream, branch, post, connection), pre
    )
    post_ref = torch.nn.functional.rms_norm(branch.float(), (8192,), None, 1e-8)
    expected_stream = 0.6 * stream + 0.8 * post_ref
    expected = torch.nn.functional.rms_norm(expected_stream, (8192,), None, 1e-5)
    torch.testing.assert_close(got_stream, expected_stream, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(got.float(), expected, atol=0.032, rtol=0.004)


def test_tp8_gate_up_and_down_sharding():
    experts, tp = 2, 8
    w13 = torch.arange(experts * 32 * 3).reshape(experts, 32, 3)
    shard = _tp_expert_stack_slice(
        w13, torch.Size((experts, 4, 3)), "w13_weight", 3, tp
    )
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

    post = MuseSpark1xRMSNormWithInputScale(4, eps=1e-8, zero_centered_gamma=True)
    post.original_precision = True
    post.weight.data.copy_(norm.weight)
    got = post(x, out_dtype=torch.float32)
    scaled = xf * (post.weight.float() + 1.0)
    expected = torch.nn.functional.rms_norm(scaled, (4,), None, 1e-8)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_combined_cache_native_views_share_kv_storage():
    backing = torch.arange(3 * 2 * 16 * 128, dtype=torch.float32).reshape(3, 2, 16, 128)
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


def test_full_attention_k2_selection_is_opt_in_and_keeps_boundary_owners(monkeypatch):
    from sglang.srt.rocm_muse_spark.mono.sglang_spec import _layer_selection

    config = SimpleNamespace(
        num_hidden_layers=6,
        mlp_layer_types=["sparse", "sparse", "sparse", "dense", "sparse", "sparse"],
        layer_types=[
            "full_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    monkeypatch.delenv("SGLANG_MUSE_ORIGINAL_FULL_K2", raising=False)
    monkeypatch.delenv("SGLANG_MUSE_ORIGINAL_NATIVE_ATTENTION", raising=False)
    assert _layer_selection(config) == ([2, 4], set())
    monkeypatch.setenv("SGLANG_MUSE_ORIGINAL_FULL_K2", "1")
    # Sparse boundary layers can retain BF16 experts in a Quark checkpoint.
    assert _layer_selection(config) == ([1, 2, 4], {1})
    monkeypatch.setenv("SGLANG_MUSE_ORIGINAL_NATIVE_ATTENTION", "1")
    assert _layer_selection(config) == ([1, 2, 4], {1, 2, 4})
    monkeypatch.delenv("SGLANG_MUSE_ORIGINAL_FULL_K2")
    assert _layer_selection(config) == ([2, 4], {2, 4})


def test_full_attention_keeps_native_cache_and_only_launches_k2(monkeypatch):
    from unittest.mock import Mock

    from sglang.srt.rocm_muse_spark import model
    from sglang.srt.rocm_muse_spark.mono import cache
    from sglang.srt.rocm_muse_spark.mono.kernel import MuseLayerKernel
    from sglang.srt.rocm_muse_spark.mono.ops import k1, k2

    rows, hidden = 8, 16
    stream = torch.zeros(rows, hidden)
    branch = torch.ones(rows, hidden, dtype=torch.bfloat16)
    norm = SimpleNamespace(variance_epsilon=1e-8)
    connection = SimpleNamespace(alpha=torch.ones(hidden), beta=torch.ones(hidden))
    x = MuseSpark1xResidualStream(stream, branch, norm, connection)
    core = torch.full((rows, 1024), 2.0, dtype=torch.bfloat16)
    attn_input = branch + 1
    native_stream = stream + 3
    boundary = Mock(return_value=(attn_input, native_stream))
    native_core = Mock(return_value=core)
    weight = SimpleNamespace(weight=torch.empty(1))
    attention = SimpleNamespace(
        pre_qkv_proj_norm=weight, forward_attention_core=native_core, o_proj=weight
    )
    experts = SimpleNamespace(
        **{
            name: torch.empty(1)
            for name in (
                "w13_weight",
                "w13_weight_scale",
                "w2_weight",
                "w2_weight_scale",
            )
        }
    )
    moe = SimpleNamespace(
        experts=SimpleNamespace(routed_experts=experts),
        router=weight,
        expert_bias=torch.empty(1),
        pooler=weight,
        pre_expert_norm=weight,
        post_expert_norm=weight,
        unpooler=weight,
    )
    runner = SimpleNamespace(
        seq_lens=torch.zeros(rows, dtype=torch.int32),
        q_len=8,
        rank=0,
        stream_a=torch.empty_like(stream),
        core=torch.empty_like(core),
        step_begin=Mock(),
        launch_args=dict,
        peer_args=dict,
    )
    kernel = object.__new__(MuseLayerKernel)
    kernel.layer = SimpleNamespace(
        self_attn=attention,
        feed_forward=moe,
        layer_idx=5,
        add_attn=connection,
        pre_ffn_norm=weight,
        post_ffn_norm=norm,
        add_ffn=connection,
    )
    kernel.runner, kernel.width, kernel.native_attention = runner, rows, True
    kernel.k1_key = SimpleNamespace(post_eps=1e-8)
    kernel.k2_key = object()
    metadata = Mock(
        side_effect=AssertionError("native attention must own its KV metadata")
    )
    launch_k1 = Mock(side_effect=AssertionError("full attention must not launch K1"))
    launch_k2 = Mock()
    monkeypatch.setattr(model, "_original_boundary", boundary)
    monkeypatch.setattr(cache, "attention_inputs", metadata)
    monkeypatch.setattr(k1, "k1_launch", launch_k1)
    monkeypatch.setattr(k2, "k2_launch", launch_k2)
    batch = SimpleNamespace(
        seq_lens=torch.tensor([15]), forward_mode=ForwardMode.TARGET_VERIFY
    )
    positions = torch.arange(15, 23)
    result = kernel.forward(positions, x, batch, new_step=True)
    assert runner.seq_lens[0] == 23
    runner.step_begin.assert_called_once_with()
    boundary.assert_called_once_with(x, weight)
    native_core.assert_called_once_with(positions, attn_input, batch)
    metadata.assert_not_called()
    launch_k1.assert_not_called()
    assert kernel.warmup_keys() == [kernel.k2_key]
    assert launch_k2.call_args.kwargs["core"] is core
    assert launch_k2.call_args.kwargs["stream_a"] is native_stream
    assert result.stream is launch_k2.call_args.kwargs["stream_out"]
    assert result.branch is launch_k2.call_args.kwargs["branch_out"]
    assert result.post_norm is norm and result.connection is connection


@pytest.mark.parametrize("multiplier", [None, 1.0, 32**-0.5])
@pytest.mark.parametrize("rows,expert_stack", [(1, False), (8, False), (8, True)])
def test_layer_keys_preserve_native_router_multiplier(multiplier, rows, expert_stack):
    from sglang.srt.rocm_muse_spark.mono.kernel import layer_keys

    norm = SimpleNamespace(variance_epsilon=1e-5)
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            attn=SimpleNamespace(sliding_window_size=2047),
            use_rope=True,
            scaling=0.125,
            pre_qkv_proj_norm=norm,
            rms_norm_eps=1e-5,
        ),
        feed_forward=SimpleNamespace(
            pre_expert_norm=norm,
            post_expert_norm=norm,
            router_multiplier=multiplier,
        ),
        post_ffn_norm=norm,
        post_attn_norm=norm,
        pre_ffn_norm=norm,
    )
    attention_key, key = layer_keys(
        layer, width=rows, q_len=rows, experimental_two_cta=expert_stack
    )
    assert attention_key.window == 2047
    assert key.router_output_multiplier == (1.0 if multiplier is None else multiplier)
    assert key.experimental_two_cta == expert_stack
    # The original vLLM adapter uses already-scaled router weights.
    del layer.feed_forward.router_multiplier
    _, original_key = layer_keys(
        layer, width=rows, q_len=rows, experimental_two_cta=expert_stack
    )
    assert original_key.router_output_multiplier == 1.0


@pytest.mark.parametrize("expert_stack", [False, True])
def test_compile_checked_requires_the_entire_selected_grid(monkeypatch, expert_stack):
    from unittest.mock import Mock

    from sglang.srt.rocm_muse_spark.mono.kernel import compile_checked
    from sglang.srt.rocm_muse_spark.mono.ops import k2
    from sglang.srt.rocm_muse_spark.mono.ops.common import residency

    key = k2.K2Build(tokens=8, experimental_two_cta=expert_stack)
    # A one-CTA/CU code object is safe for the default, unsafe for the stack.
    proof = Mock()
    proof.problems.side_effect = lambda blocks, cus: (
        ["grid is not fully resident"] if blocks > cus else []
    )
    monkeypatch.setattr(k2, "compile_only", Mock())
    monkeypatch.setattr(k2, "build", Mock(return_value=object()))
    monkeypatch.setattr(residency, "residency", lambda kernel, threads: [proof])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda: SimpleNamespace(multi_processor_count=256),
    )
    expected = ["grid is not fully resident"] if expert_stack else []
    assert compile_checked(key) == expected
    proof.problems.assert_called_once_with(512 if expert_stack else 256, 256)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8_e4m3", "fp8_e5m2"])
def test_native_attention_k2_kv_dtype_gate(monkeypatch, native, kv_dtype):
    from sglang.srt.rocm_muse_spark.mono.sglang_spec import MONO_SPEC

    monkeypatch.setenv("SGLANG_MUSE_ORIGINAL_NATIVE_ATTENTION", str(int(native)))
    monkeypatch.setenv("SGLANG_MUSE_ORIGINAL_UNQUALIFIED_DFLASH", "1")
    monkeypatch.delenv("SGLANG_MUSE_ORIGINAL_NATIVE_ONLY", raising=False)
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
        kv_cache_dtype=kv_dtype,
        speculative_algorithm="DFLASH",
        speculative_num_draft_tokens=8,
    )
    accepted = kv_dtype == "bf16" or (native and kv_dtype == "fp8_e4m3")
    reasons = config_disable_reasons(cfg, text)
    assert (not reasons) == accepted
    if accepted:
        assert kv_dtype in MONO_SPEC.requires.server_args["kv_cache_dtype"]
    else:
        assert any("KV" in reason for reason in reasons)
    monkeypatch.delenv("SGLANG_MUSE_ORIGINAL_UNQUALIFIED_DFLASH")
    assert "rows=8 target-verify is not production-qualified" in config_disable_reasons(
        cfg, text
    )


@pytest.mark.parametrize(
    "architecture,draft,native,page_major,expected",
    [
        (
            "MuseSparkForConditionalGeneration",
            False,
            False,
            False,
            "MuseMonoMHATokenToKVPool",
        ),
        ("MuseSparkForConditionalGeneration", False, True, False, "MHATokenToKVPool"),
        (
            "MuseSparkForConditionalGeneration",
            False,
            True,
            True,
            "PageMajorMHATokenToKVPool",
        ),
        ("MuseSparkForConditionalGeneration", True, False, False, "MHATokenToKVPool"),
        ("OtherModel", False, False, False, "MHATokenToKVPool"),
        ("OtherModel", False, True, False, "MHATokenToKVPool"),
    ],
)
def test_native_attention_uses_native_kv_pool(
    monkeypatch, architecture, draft, native, page_major, expected
):
    from sglang.srt.mem_cache import kv_cache_configurator as module

    monkeypatch.setenv("SGLANG_MUSE_ORIGINAL_NATIVE_ATTENTION", str(int(native)))
    monkeypatch.setenv("SGLANG_MUSE_ORIGINAL_NATIVE_ONLY", "1")
    configurator = object.__new__(module.KVCacheConfigurator)
    configurator.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture])
    )
    configurator.is_draft_worker = draft
    monkeypatch.setattr(
        module.KVCacheConfigurator, "_page_major_enabled", lambda self: page_major
    )

    class PoolSelected(Exception):
        pass

    def resolve(self, *, kind, default):
        assert kind == "mha"
        assert default.__name__ == expected
        raise PoolSelected

    monkeypatch.setattr(module.KVCacheConfigurator, "_resolve_kv_pool_class", resolve)
    # Stop after the real dispatch chooses its class, before any GPU allocation.
    with pytest.raises(PoolSelected):
        configurator._build_token_to_kv_pool(
            sizes=None, is_dsa_model=False, is_dsv4_model=False, req_to_token_pool=None
        )
