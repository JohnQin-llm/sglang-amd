# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the actual Muse TP8/EP8 model admission and lifecycle."""

import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")
from sglang.srt.rocm_muse_spark import model
from sglang.srt.rocm_muse_spark.mono import ep8_runner as runner_module
from sglang.srt.rocm_muse_spark.mono import lifecycle


def fake_model():
    layer = NS(
        post_attention_residual_gate=object(),
        post_feedforward_residual_gate=object(),
        mlp=NS(experts=object()),
        _fused_attn_boundary=NS(),
        is_moe_layer=True,
    )
    decoder = NS(layers=[layer])
    wrapped = NS(
        layers=[layer],
        language_model=decoder,
        modules=lambda: [],
        embed_tokens=NS(weight=NS(device=NS(type="cuda"))),
    )
    config = NS(
        hidden_size=8192,
        moe_hidden_size=4096,
        moe_intermediate_size=4096,
        num_local_experts=256,
        num_experts_per_tok=8,
        num_shared_experts=0,
    )
    return NS(model=wrapped, config=config, mxfp4_checkpoint=True)


class StartupScope(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "SGLANG_MUSE_EP8_FULL_K2": "1",
                "SGLANG_MUSE_MOE_PREPACKED": "1",
                "MUSE_MOE_BACKEND": "flydsl",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.parallel = NS(
            tp_size=8,
            attn_tp_size=8,
            moe_ep_size=8,
            moe_tp_size=1,
            tp_group=NS(cpu_group=None),
        )
        self.spec = NS(speculative_algorithm="DFLASH", speculative_num_draft_tokens=8)
        self.overlap = NS(
            enable_two_batch_overlap=False, enable_single_batch_overlap=False
        )
        self.disagg = NS(enable_pdmux=False)
        for target, value in [
            ("get_disagg", self.disagg),
            ("get_exec", NS(overlap=self.overlap)),
        ]:
            p = patch.object(model, target, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        for target, value in [("get_parallel", self.parallel), ("get_spec", self.spec)]:
            p = patch.object(model, target, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(model, "_original_layout_selected", return_value=False)
        self.original = p.start()
        self.addCleanup(p.stop)
        p = patch.object(torch.version, "hip", "mock-rocm")
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(
            torch.cuda, "get_device_properties", return_value=NS(gcnArchName="gfx950")
        )
        self.props = p.start()
        self.addCleanup(p.stop)
        p = patch.object(runner_module, "Ep8FullK2Runner")
        self.constructor = p.start()
        self.addCleanup(p.stop)
        self.constructor.return_value = NS(layers={31: None})
        self.value = fake_model()

    def run_load(self):
        model.MuseSpark1xForCausalLM.post_load_weights(self.value)

    def test_default_off_allocates_nothing(self):
        with patch.dict(os.environ, {"SGLANG_MUSE_EP8_FULL_K2": "0"}):
            self.run_load()
        self.constructor.assert_not_called()
        self.props.assert_not_called()

    def test_supported_initializes_once_then_refreshes(self):
        runner = NS(layers={31: None}, refresh=Mock())
        self.constructor.return_value = runner
        self.run_load()
        self.run_load()
        self.constructor.assert_called_once()
        runner.refresh.assert_called_once()

    def test_unsupported_contracts_decline(self):
        cases = [
            (self.parallel, "tp_size", 4),
            (self.parallel, "moe_ep_size", 1),
            (self.parallel, "attn_tp_size", 1),
            (self.parallel, "moe_tp_size", 2),
            (self.spec, "speculative_algorithm", "EAGLE"),
            (self.spec, "speculative_num_draft_tokens", 4),
            (self.value, "mxfp4_checkpoint", False),
            (self.value.config, "hidden_size", 4096),
            (self.value.config, "num_shared_experts", 1),
        ]
        for obj, name, value in cases:
            with self.subTest(name=name, value=value):
                old = getattr(obj, name)
                setattr(obj, name, value)
                self.run_load()
                self.constructor.assert_not_called()
                setattr(obj, name, old)

    def test_concurrent_model_consumers_decline_before_compile(self):
        for owner, name in (
            (self.disagg, "enable_pdmux"),
            (self.overlap, "enable_two_batch_overlap"),
            (self.overlap, "enable_single_batch_overlap"),
        ):
            with self.subTest(name=name):
                setattr(owner, name, True)
                self.run_load()
                self.constructor.assert_not_called()
                self.props.assert_not_called()
                setattr(owner, name, False)

    def test_original_layout_declines(self):
        self.original.return_value = True
        self.run_load()
        self.constructor.assert_not_called()

    def test_other_gpu_declines(self):
        self.props.return_value = NS(gcnArchName="gfx942")
        self.run_load()
        self.constructor.assert_not_called()

    def test_prepacked_off_declines(self):
        with patch.dict(os.environ, {"SGLANG_MUSE_MOE_PREPACKED": "0"}):
            self.run_load()
        self.constructor.assert_not_called()

    def test_layer_contract_fallback_only(self):
        self.constructor.side_effect = runner_module.Ep8FullK2Unsupported(
            "unsupported norm"
        )
        self.run_load()
        self.assertFalse(hasattr(self.value.model.language_model, "_ep8_full_k2"))
        self.constructor.side_effect = RuntimeError("compiler failed")
        with self.assertRaisesRegex(RuntimeError, "compiler failed"):
            self.run_load()


class RouterVersion(unittest.TestCase):
    def test_inference_tensor_without_version_counter(self):
        with torch.inference_mode():
            tensor = torch.empty((2, 3))
        self.assertIsNone(runner_module._tensor_version(tensor))


class RunnerAdmission(unittest.TestCase):
    def setUp(self):
        self.tensor = torch.empty(1)
        self.layer = NS(
            layer_idx=31,
            mlp=NS(experts=NS(_prepacked_moe=object())),
            parameters=lambda: iter([self.tensor]),
        )
        self.parallel = NS(
            tp_size=8,
            attn_tp_size=8,
            moe_ep_size=8,
            moe_tp_size=1,
            moe_ep_rank=0,
            tp_rank=0,
            tp_group=NS(cpu_group=None, rank_in_group=0, world_size=8),
        )

        def add(p):
            result = p.start()
            self.addCleanup(p.stop)
            return result

        add(
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=NS(gcnArchName="gfx950", multi_processor_count=256),
            )
        )
        add(patch.object(torch.cuda, "synchronize"))
        add(patch.object(runner_module.dist, "monitored_barrier"))
        self.params = add(
            patch.object(
                runner_module,
                "_layer_parameters",
                return_value={"test": self.tensor, "w_router": torch.empty((2, 3))},
            )
        )
        add(patch.object(runner_module, "_layer_key", return_value="key"))
        self.compile = add(patch.object(runner_module.ep8_k2, "compile_only"))
        add(patch.object(runner_module.ep8_k2, "build", return_value=object()))
        add(patch.object(runner_module.ep8_k2, "scratch_bytes", return_value=16))
        add(
            patch.object(
                runner_module, "residency", return_value=[NS(problems=lambda *args: [])]
            )
        )

        def gather(out, value, group):
            out[:] = [value] * 8

        self.gather = add(
            patch.object(runner_module.dist, "all_gather_object", side_effect=gather)
        )
        self.peer = add(patch.object(runner_module, "PeerBuffer"))
        self.peer.return_value = NS(
            addresses=torch.arange(8, dtype=torch.int64) * 256 + 4096, close=Mock()
        )

    def make(self):
        return runner_module.Ep8FullK2Runner([self.layer], self.parallel)

    def test_valid_admission_and_close(self):
        result = self.make()
        self.compile.assert_called_once()
        self.peer.assert_called_once()
        result.close()
        self.peer.return_value.close.assert_called_once()

    def test_wrong_rank_declines_before_compile_or_mapping(self):
        self.parallel.moe_ep_rank = 7
        with self.assertRaises(runner_module.Ep8FullK2Unsupported):
            self.make()
        self.compile.assert_not_called()
        self.peer.assert_not_called()

    def test_layer_disagreement_declines_before_compile_or_mapping(self):
        def gather(out, value, group):
            out[:] = [value] * 7 + [("different",)]

        self.gather.side_effect = gather
        with self.assertRaises(runner_module.Ep8FullK2Unsupported):
            self.make()
        self.compile.assert_not_called()
        self.peer.assert_not_called()

    def test_unsupported_layout_declines(self):
        self.params.side_effect = runner_module.Ep8FullK2Unsupported("bad dtype")
        with self.assertRaises(runner_module.Ep8FullK2Unsupported):
            self.make()
        self.compile.assert_not_called()
        self.peer.assert_not_called()

    def test_compiler_failure_is_not_unsupported(self):
        self.compile.side_effect = RuntimeError("compiler failed")
        with self.assertRaisesRegex(RuntimeError, "compiler failed"):
            self.make()
        self.peer.assert_not_called()


@pytest.mark.parametrize(
    "mode,batch_size,rows,width,expected",
    [
        (ForwardMode.TARGET_VERIFY, 1, 8, 8, True),
        (ForwardMode.TARGET_VERIFY, 2, 16, 8, False),
        (ForwardMode.EXTEND, 1, 8, 8, False),
        (ForwardMode.DECODE, 1, 1, 8, False),
        (ForwardMode.TARGET_VERIFY, 1, 8, 4, False),
        (ForwardMode.TARGET_VERIFY, 1, 1, 8, False),
        (ForwardMode.TARGET_VERIFY, 1, 16, 8, False),
    ],
)
def test_runtime_batch_admission(mode, batch_size, rows, width, expected):
    batch = NS(
        forward_mode=mode,
        batch_size=batch_size,
        seq_lens=torch.empty(batch_size, dtype=torch.int64),
        spec_info=NS(draft_token_num=width),
    )
    assert runner_module.Ep8FullK2Runner.eligible(batch, torch.empty(rows)) is expected


def test_close_waits_for_all_ranks_before_freeing_and_is_idempotent(monkeypatch):
    calls = []
    value = object.__new__(runner_module.Ep8FullK2Runner)
    value.device = torch.device("cpu")
    value.group = NS(cpu_group=object())
    value.peers = NS(close=lambda: calls.append("free"))
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: calls.append("device_done")
    )

    def barrier(*, group, timeout, wait_all_ranks):
        assert timeout.total_seconds() == 30 and wait_all_ranks
        assert group is value.group.cpu_group
        calls.append("all_ranks_done")

    monkeypatch.setattr(runner_module.dist, "monitored_barrier", barrier)
    monkeypatch.setattr(runner_module, "tp_agree", lambda ok, group: ok)
    value.close()
    value.close()
    assert calls == ["device_done", "all_ranks_done", "free"]
    assert value.peers is None


def test_failed_synchronize_keeps_peer_allocation_owned(monkeypatch):
    value = object.__new__(runner_module.Ep8FullK2Runner)
    value.device = torch.device("cpu")
    value.group = NS(cpu_group=object())
    value.peers = NS(close=Mock())
    peer = value.peers

    def fail(device):
        raise RuntimeError("pending device work")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    monkeypatch.setattr(runner_module.dist, "monitored_barrier", Mock())
    monkeypatch.setattr(runner_module, "tp_agree", lambda ok, group: ok)
    with pytest.raises(RuntimeError, match="could not quiesce every rank"):
        value.close()
    assert value.peers is peer
    peer.close.assert_not_called()


def test_wrapped_model_close_finds_decoder_and_deduplicates_aliases():
    runner = NS(close=Mock())
    decoder = torch.nn.Module()
    decoder._ep8_full_k2 = runner
    text = torch.nn.Module()
    text.language_model = decoder
    outer = torch.nn.Module()
    outer.model = text
    # A second owner alias must not close the same peer mapping twice.
    text._ep8_full_k2 = runner
    lifecycle.close_model_runners(outer)
    runner.close.assert_called_once()
    lifecycle.close_model_runners(torch.nn.Linear(2, 2))


def test_missing_rank_does_not_release_peers(monkeypatch):
    value = object.__new__(runner_module.Ep8FullK2Runner)
    value.device = torch.device("cpu")
    value.group = NS(cpu_group=object())
    value.peers = NS(close=Mock())
    peer = value.peers
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())

    def missing(**kwargs):
        raise RuntimeError("rank did not reach shutdown")

    monkeypatch.setattr(runner_module.dist, "monitored_barrier", missing)
    with pytest.raises(RuntimeError, match="rank did not reach shutdown"):
        value.close()
    assert value.peers is peer
    peer.close.assert_not_called()


def test_actual_model_forward_dispatch_and_five_aux_lifetimes(monkeypatch):
    """Actual model/layer methods; CPU fakes isolate dispatch and ownership."""
    monkeypatch.setattr(model, "_original_layout_selected", lambda: False)
    calls = {"core": 0, "attention": 0, "mlp": 0, "begin": 0, "end": 0}
    shape = (8, 8192)

    class Attention:
        def __call__(self, positions, value, batch):
            calls["attention"] += 1
            return torch.ones(shape, dtype=torch.bfloat16)

        def forward_attention_core(self, positions, value, batch):
            calls["core"] += 1
            return torch.ones((8, 1024), dtype=torch.bfloat16)

    class Boundary:
        def forward(self, stream, branch, **kwargs):
            result = stream + branch.float()
            return result.bfloat16(), result, None

    class Layer:
        def __init__(self, index):
            self.layer_idx = index
            self.self_attn = Attention()
            self.is_moe_layer = True
            self._fused_attn_boundary = Boundary()

        def self_attn_input_norm(self, stream):
            return stream.bfloat16()

        def mlp(self, value, forward_batch):
            calls["mlp"] += 1
            return torch.ones(shape, dtype=torch.bfloat16)

        def __call__(self, *args, **kwargs):
            return model.MuseSpark1xDecoderLayer.forward(self, *args, **kwargs)

    class Runner:
        eligible = staticmethod(runner_module.Ep8FullK2Runner.eligible)
        layers = dict.fromkeys(range(1, 61))
        stream_out = torch.empty(shape, dtype=torch.float32)
        branch_out = torch.ones(shape, dtype=torch.bfloat16)

        def step_begin(self):
            calls["begin"] += 1

        def step_end(self):
            calls["end"] += 1

        def forward(self, layer, core, stream):
            self.stream_out.copy_(stream + 1)
            return self.stream_out, self.branch_out

    runner = Runner()
    captures = [1, 16, 30, 44, 59]
    decoder = NS(
        embed_norm=None,
        residual_dtype=torch.float32,
        layers=[Layer(i) for i in range(62)],
        layers_to_capture=captures,
        _ffn_boundaries=[Boundary() for _ in range(62)],
        _ep8_full_k2=runner,
    )
    batch = NS(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=1,
        seq_lens=torch.ones(1, dtype=torch.int64),
        spec_info=NS(draft_token_num=8),
    )
    ids = torch.arange(8)
    stream = torch.zeros(shape)
    _, aux = model.MuseSpark1xModel.forward(
        decoder, ids, ids, batch, input_embeds=stream
    )
    assert calls == {"core": 60, "attention": 2, "mlp": 2, "begin": 1, "end": 1}
    assert len(aux) == 5 and len({x.data_ptr() for x in aux}) == 5
    assert all(torch.all(x == 2 * (index + 1)) for x, index in zip(aux, captures))
    assert all(
        x.data_ptr() not in (runner.stream_out.data_ptr(), runner.branch_out.data_ptr())
        for x in aux
    )
    saved = [x.clone() for x in aux]
    model.MuseSpark1xModel.forward(decoder, ids, ids, batch, input_embeds=stream + 10)
    assert all(torch.equal(a, b) for a, b in zip(aux, saved))
    for unsupported in (ForwardMode.EXTEND, ForwardMode.DECODE):
        before = dict(calls)
        batch.forward_mode = unsupported
        model.MuseSpark1xModel.forward(decoder, ids, ids, batch, input_embeds=stream)
        assert calls["core"] == before["core"]
        assert calls["begin"] == before["begin"] and calls["end"] == before["end"]
        assert calls["mlp"] - before["mlp"] == 62


def test_flag_on_unsupported_shutdown_never_imports_device_modules(monkeypatch):
    import builtins
    import importlib

    from sglang.srt.environ import envs

    original_import = builtins.__import__

    def no_device_import(name, *args, **kwargs):
        if name.startswith(
            (
                "flydsl",
                "aiter",
                "sglang.srt.rocm_muse_spark.mono.ep8_runner",
                "sglang.srt.rocm_muse_spark.mono.ops",
            )
        ):
            raise AssertionError(f"Unsupported shutdown imported {name}")
        return original_import(name, *args, **kwargs)

    with envs.SGLANG_MUSE_EP8_FULL_K2.override(True):
        monkeypatch.setattr(builtins, "__import__", no_device_import)
        # Reexecute the complete host module, then call the scheduler's helper
        # on a wrapper which successfully declined runner construction.
        imported = importlib.reload(lifecycle)
        imported.close_model_runners(torch.nn.Linear(2, 2))
