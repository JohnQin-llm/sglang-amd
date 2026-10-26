"""TP4 QKV/cache opt-in: production dispatch, numerical and graph contracts."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.kernels.ops.attention.qk_norm_rope_cache import (
    qk_norm_rope_cache,
    qkv_proj_qk_norm_rope_cache,
)
from sglang.kernels.ops.attention.utils import launch_reshape_and_cache_flash
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.rocm_muse_spark import attention_kv_write as dispatch
from sglang.srt.rocm_muse_spark.muse_qk_norm import fused_qk_norm
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestMuseTP4QKVCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not torch.cuda.get_device_properties(
            0
        ).gcnArchName.startswith("gfx950"):
            raise unittest.SkipTest("requires gfx950")
        cls.rope = RotaryEmbedding(64, 64, 32768, 500000, False, torch.float32).cuda()
        cls.eps = 1e-5
        cls.scale = 5.4730064863875

    def check_numeric(self, actual, expected):
        self.assertTrue(bool(torch.isfinite(actual.float()).all()))
        error = torch.linalg.vector_norm(actual.float() - expected.float())
        norm = torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)
        # Existing component gate, including projection accumulation and BF16
        # midpoint differences; this is not a bitwise-equivalence assertion.
        self.assertLessEqual(float(error / norm), 0.005)

    def test_production_dispatch(self):
        for hq, hk in ((16, 2), (32, 4), (8, 1), (32, 2)):
            attention = SimpleNamespace(
                head_dim=64,
                num_heads=hq,
                num_kv_heads=hk,
                use_qk_norm=True,
                use_attn_temperature_tuning=False,
                attn_attend_prefix_cache=False,
            )
            for mode in (
                ForwardMode.TARGET_VERIFY,
                ForwardMode.DECODE,
                ForwardMode.EXTEND,
            ):
                batch = SimpleNamespace(
                    forward_mode=mode, dcp_kv_mask=None, attn_attend_prefix_cache=False
                )
                for projection in (False, True):
                    for qk_enabled, qkv_enabled in (
                        (False, False),
                        (True, False),
                        (False, True),
                        (True, True),
                    ):
                        with (
                            patch.object(dispatch, "TP4_QK_ROPE_CACHE", qk_enabled),
                            patch.object(dispatch, "TP4_FUSED_QKV_CACHE", qkv_enabled),
                        ):
                            enabled = qkv_enabled if projection else qk_enabled
                            expected = mode != ForwardMode.EXTEND and (
                                (hq, hk) == (16, 2) or ((hq, hk) == (32, 4) and enabled)
                            )
                            self.assertEqual(
                                dispatch._attention_shape_ok(
                                    attention, batch, projection=projection
                                ),
                                expected,
                            )
        attention.num_heads, attention.num_kv_heads = 32, 4
        attention.use_o_gate = True
        attention.norm_eps, attention.scale_query_by = self.eps, self.scale
        attention.qkv_proj = SimpleNamespace(
            weight=torch.empty((4608, 8192), device="cuda", dtype=torch.bfloat16),
            bias=None,
            gather_output=False,
        )
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        target = SimpleNamespace(
            cos_sin=None,
            k_cache=None,
            v_cache=None,
            locations=None,
            swa_mapping=None,
            k_scale=None,
            v_scale=None,
        )
        with (
            patch.object(dispatch, "TP4_QK_ROPE_CACHE", True),
            patch.object(dispatch, "TP4_FUSED_QKV_CACHE", True),
            patch.object(dispatch, "_kv_write_target", return_value=target),
            patch.object(
                dispatch, "qkv_proj_qk_norm_rope_cache", return_value="projected"
            ) as project,
            patch.object(
                dispatch, "qk_norm_rope_cache", return_value="normalized"
            ) as norm,
        ):
            for m in (1, 16, 17, 32, 33, 64, 65, 96, 128, 129):
                x = torch.empty((m, 8192), device="cuda", dtype=torch.bfloat16)
                positions = torch.zeros(m, device="cuda", dtype=torch.int64)
                result = dispatch.qkv_proj_norm_rope_kv_write(
                    attention, positions, x, batch
                )
                self.assertEqual(result, "projected" if m <= 16 else None)
                if m <= 16:
                    self.assertEqual(
                        project.call_args.kwargs, {"num_q_heads": 32, "num_kv_heads": 4}
                    )
                q = x[:, :2048]
                k, v = x[:, :256], x[:, 256:512]
                result = dispatch.qk_norm_rope_kv_write(
                    attention, positions, q, k, v, batch
                )
                self.assertEqual(result, "normalized" if m <= 128 else None)
            self.assertEqual(project.call_count, 2)
            self.assertEqual(norm.call_count, 9)
            attention.num_heads, attention.num_kv_heads = 16, 2
            attention.qkv_proj.weight = attention.qkv_proj.weight[:2304]
            for m in (32, 33, 64, 65):
                x = torch.empty((m, 8192), device="cuda", dtype=torch.bfloat16)
                positions = torch.zeros(m, device="cuda", dtype=torch.int64)
                result = dispatch.qkv_proj_norm_rope_kv_write(
                    attention, positions, x, batch
                )
                self.assertEqual(result, "projected" if m <= 32 else None)
                q, k, v = x[:, :1024], x[:, :128], x[:, 128:256]
                result = dispatch.qk_norm_rope_kv_write(
                    attention, positions, q, k, v, batch
                )
                self.assertEqual(result, "normalized" if m <= 64 else None)

    def test_qk_cache_graph_and_untouched_slots(self):
        for seed, m, use_rope, use_map in (
            (5, 8, False, False),
            (16, 32, True, False),
            (1023, 64, True, True),
            (1024, 128, True, True),
        ):
            torch.manual_seed(seed)
            raw = torch.randn((m, 4608), device="cuda", dtype=torch.bfloat16)
            q, k, v, _ = raw.split((2048, 256, 256, 2048), -1)
            positions = torch.arange(m, device="cuda", dtype=torch.int64) * 127 + 13
            locations = torch.arange(m, device="cuda", dtype=torch.int64) * 3 + 5
            mapping = (
                torch.randperm(512, device="cuda", dtype=torch.int64)
                if use_map
                else None
            )
            ks = torch.tensor([0.7], device="cuda", dtype=torch.float32)
            vs = torch.tensor([0.9], device="cuda", dtype=torch.float32)
            kc = torch.full(
                (32, 16, 4, 64), 0.5, device="cuda", dtype=torch.float8_e4m3fn
            )
            vc, rc, rv = kc.clone(), kc.clone(), kc.clone()
            cs = self.rope.cos_sin_cache if use_rope else None

            candidate = partial(
                qk_norm_rope_cache,
                q,
                k,
                v,
                positions,
                cs,
                kc,
                vc,
                locations,
                mapping,
                self.eps,
                self.scale,
                ks,
                vs,
            )

            candidate()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = candidate()
            saved = raw.clone()
            for state in ("original", "changed", "restored"):
                if state == "changed":
                    raw.mul_(1.125)
                elif state == "restored":
                    raw.copy_(saved)
                expected = fused_qk_norm(q, k, 64, self.eps, self.scale)
                if use_rope:
                    expected = self.rope.forward_cuda(positions, *expected)
                launch_reshape_and_cache_flash(
                    expected[1].view(m, 4, 64),
                    v.view(m, 4, 64),
                    rc,
                    rv,
                    locations,
                    mapping,
                    k_scale=ks,
                    v_scale=vs,
                )
                graph.replay()
                for a, b in zip(actual, expected):
                    self.assertTrue(torch.equal(a, b))
                slots = mapping[locations] if use_map else locations
                for a, b in ((kc, rc), (vc, rv)):
                    self.check_numeric(
                        a.float().view(512, 4, 64)[slots],
                        b.float().view(512, 4, 64)[slots],
                    )
                    untouched = torch.ones(512, dtype=torch.bool, device="cuda")
                    untouched[slots] = False
                    self.assertTrue(
                        torch.equal(
                            a.view(torch.uint8).view(512, -1)[untouched],
                            b.view(torch.uint8).view(512, -1)[untouched],
                        )
                    )
            graph.reset()

    def test_projection_shapes(self):
        torch.manual_seed(1023)
        for hq, hk in ((16, 2), (32, 4)):
            weight = (
                torch.randn(
                    (2 * (hq + hk) * 64, 8192), device="cuda", dtype=torch.bfloat16
                )
                * 0.01
            )
            for m in (1, 16, 24, 32):
                x = torch.randn((m, 8192), device="cuda", dtype=torch.bfloat16)
                positions = torch.arange(m, device="cuda", dtype=torch.int64) * 131 + 17
                locations = torch.arange(m, device="cuda", dtype=torch.int64)
                ks = torch.ones(1, device="cuda", dtype=torch.float32)
                cache = torch.zeros(
                    (4, 16, hk, 64), device="cuda", dtype=torch.float8_e4m3fn
                )
                raw = torch.nn.functional.linear(x, weight)
                q, k, v, gate = raw.split((hq * 64, hk * 64, hk * 64, hq * 64), -1)
                q, k = fused_qk_norm(q, k, 64, self.eps, self.scale)
                q, k = self.rope.forward_cuda(positions, q, k)
                actual = qkv_proj_qk_norm_rope_cache(
                    x,
                    weight,
                    positions,
                    self.rope.cos_sin_cache,
                    cache,
                    cache.clone(),
                    locations,
                    None,
                    self.eps,
                    self.scale,
                    ks,
                    ks,
                    num_q_heads=hq,
                    num_kv_heads=hk,
                )
                for a, b in zip(actual, (q, k, v, gate)):
                    self.check_numeric(a, b)


if __name__ == "__main__":
    unittest.main()
