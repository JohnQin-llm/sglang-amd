"""Numerics, graph replay and dispatch boundaries for the gfx950 Muse pooler."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "Requires MI355X")
class TestMusePooler(CustomTestCase):
    def test_token_sharding_is_prefill_only(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.rocm_muse_spark.model import MuseSpark1xMoE

        x = torch.empty(2048, 8192, device="cuda", dtype=torch.bfloat16)
        moe = SimpleNamespace(
            use_token_sharded_pooler=True,
            pre_expert_proj=Mock(),
            pre_expert_norm=None,
            tp_size=8,
        )
        module = "sglang.srt.rocm_muse_spark.model"
        with (
            patch(
                module + ".get_parallel",
                return_value=SimpleNamespace(attn_tp_size=8, tp_group=Mock()),
            ),
            patch(module + "._dense_linear", return_value=x) as fallback,
            patch(
                "sglang.kernels.ops.muse_spark.pooler.try_token_sharded_pooler",
                return_value=x,
            ) as sharded,
        ):
            prefill = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
            MuseSpark1xMoE._pool_inputs(moe, x, prefill)
            sharded.assert_called_once()
            fallback.assert_not_called()
            sharded.reset_mock()
            for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                fallback.reset_mock()
                MuseSpark1xMoE._pool_inputs(moe, x, SimpleNamespace(forward_mode=mode))
                sharded.assert_not_called()
                fallback.assert_called_once()
            moe.use_token_sharded_pooler = False
            MuseSpark1xMoE._pool_inputs(moe, x, prefill)
            sharded.assert_not_called()

    @torch.no_grad()
    def test_splitk_reference_and_graph_replay(self):
        from sglang.kernels.ops.muse_spark.pooler import try_splitk_pooler

        torch.manual_seed(123)
        weight = torch.randn(4096, 8192, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(8, 8192, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            try_splitk_pooler(x, weight)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_splitk_pooler(x, weight)
        for _ in range(3):
            x.normal_()
            graph.replay()
            reference = F.linear(x.float(), weight.float())
            relative_error = torch.linalg.vector_norm(
                actual.float() - reference
            ) / torch.linalg.vector_norm(reference)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertLess(relative_error.item(), 0.006)
            torch.testing.assert_close(
                actual, try_splitk_pooler(x, weight), atol=0, rtol=0
            )
        for tokens in (0, 1, 7, 9, 32, 256):
            with self.subTest(tokens=tokens):
                other = torch.empty(tokens, 8192, device="cuda", dtype=torch.bfloat16)
                self.assertIsNone(try_splitk_pooler(other, weight))
        self.assertIsNone(try_splitk_pooler(x.float(), weight))
        self.assertIsNone(try_splitk_pooler(x, weight.t()))
        self.assertIsNone(try_splitk_pooler(x.cpu(), weight.cpu()))

    @torch.no_grad()
    def test_prefill_shard_and_fallback(self):
        from sglang.kernels.ops.muse_spark.pooler import try_token_sharded_pooler

        torch.manual_seed(123)
        linear = torch.nn.Linear(
            8192, 4096, bias=False, device="cuda", dtype=torch.bfloat16
        )
        norm = torch.nn.RMSNorm(4096, eps=1e-5, elementwise_affine=False).cuda()
        # Check the production slice and gather axis without a distributed launch.
        # Eight-rank collective/graph execution is validated separately.
        group = SimpleNamespace(
            world_size=8, rank_in_group=3, all_gather=Mock(side_effect=lambda x, dim: x)
        )
        for tokens in (2048, 2049, 3859, 6805, 8192, 8193, 16383, 16384):
            x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16)
            for rank in (0, 7):
                group.rank_in_group = rank
                actual = try_token_sharded_pooler(x, linear, norm, group)
                rows = (tokens + 7) // 8
                start, end = rank * rows, min((rank + 1) * rows, tokens)
                # Independent unpadded projection checks which original rows
                # reach the collective, including the last rank's padded tail.
                reference = norm(F.linear(x[start:end], linear.weight))
                torch.testing.assert_close(
                    actual[: end - start], reference, atol=0.04, rtol=0.02
                )
                self.assertEqual(actual.shape, (rows, 4096))
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(torch.count_nonzero(actual[end - start :]), 0)
                self.assertEqual(group.all_gather.call_args.kwargs, {"dim": 0})
        group.all_gather.reset_mock()
        for tokens in (0, 8, 2047, 16385):
            x = torch.empty(tokens, 8192, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(try_token_sharded_pooler(x, linear, norm, group))
        x = torch.empty(8192, 8192, device="cuda", dtype=torch.bfloat16)
        group.world_size = 4
        self.assertIsNone(try_token_sharded_pooler(x, linear, norm, group))
        group.all_gather.assert_not_called()


if __name__ == "__main__":
    unittest.main()
