import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "Requires MI35x")
class TestMuseMoePrepacked(CustomTestCase):
    def test_model_dispatch_reference_and_graph(self):
        # Match the model loader's custom-op registration order.
        from aiter.ops.flydsl.kernels import moe_2stage_a16wmix as aiter_moe

        from sglang.kernels.ops.muse_spark import (  # noqa: F401
            fused_muse_spark_routing,
            moe_prepacked,
        )
        from sglang.srt.rocm_muse_spark import model

        torch.manual_seed(904)
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.device("meta"):
            experts = model.MuseSpark1xMxfp4Experts(32, 4096, 4096)
        for name, shape in (
            ("w13_weight", (32, 8192, 2048)),
            ("w2_weight", (32, 4096, 2048)),
        ):
            setattr(
                experts,
                name,
                torch.nn.Parameter(
                    torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda"),
                    requires_grad=False,
                ),
            )
        for name, shape in (
            ("w13_weight_scale", (32, 8192, 128)),
            ("w2_weight_scale", (32, 4096, 128)),
        ):
            setattr(
                experts,
                name,
                torch.nn.Parameter(
                    torch.full(shape, 119, dtype=torch.uint8, device="cuda"),
                    requires_grad=False,
                ),
            )
        norm = SimpleNamespace(
            weight=torch.randn(4096, dtype=torch.bfloat16, device="cuda") * 0.1,
            zero_centered_gamma=True,
            gain_center=1.0,
            eps=1e-8,
        )
        with patch.dict(os.environ, {"MUSE_MOE_BACKEND": "flydsl"}):
            with patch.dict(os.environ, {"SGLANG_MUSE_MOE_PREPACKED": "0"}):
                experts.prepare_prepacked_weights()
                self.assertIsNone(experts._prepacked_moe)
            with patch.dict(os.environ, {"SGLANG_MUSE_MOE_PREPACKED": "1"}):
                with patch.dict(os.environ, {"MUSE_MOE_BACKEND": "triton"}):
                    experts.prepare_prepacked_weights()
                    self.assertIsNone(experts._prepacked_moe)
                with patch.object(
                    torch.cuda,
                    "get_device_properties",
                    return_value=SimpleNamespace(gcnArchName="gfx942"),
                ):
                    experts.prepare_prepacked_weights()
                    self.assertIsNone(experts._prepacked_moe)
                experts.prepare_prepacked_weights()
            self.assertIsNotNone(experts._prepacked_moe)
            self.assertFalse(experts._prepacked_moe.state_dict())
            self.assertEqual(len(list(experts._prepacked_moe.buffers())), 4)

            for rows in (8, 15, 16, 17, 32, 56, 63, 64, 65, 128, 144, 248, 256, 264):
                x = torch.randn(rows, 4096, dtype=torch.bfloat16, device="cuda")
                ids = torch.full((rows, 8), 96, dtype=torch.int32, device="cuda")
                scores = torch.full((rows, 8), 0.125, device="cuda")
                with (
                    patch.object(
                        moe_prepacked, "align_local", wraps=moe_prepacked.align_local
                    ) as align,
                    patch.object(
                        aiter_moe,
                        "flydsl_a16w4_gemm1",
                        wraps=aiter_moe.flydsl_a16w4_gemm1,
                    ) as gate,
                    patch.object(
                        aiter_moe,
                        "flydsl_a16w4_gemm2",
                        wraps=aiter_moe.flydsl_a16w4_gemm2,
                    ) as down,
                    patch.object(
                        model, "_run_fused_moe", wraps=model._run_fused_moe
                    ) as fallback,
                    patch.object(
                        experts._prepacked_moe,
                        "forward",
                        wraps=experts._prepacked_moe.forward,
                    ) as candidate,
                ):
                    experts(x, scores, ids, 96, norm)
                    selected = 16 <= rows <= 256 and rows % 8 == 0
                    self.assertEqual(candidate.call_count, int(selected))
                    self.assertEqual(fallback.call_count, int(not selected))
                    if selected:
                        block_m = 16 if rows < 64 else 32
                        self.assertEqual(align.call_args.args[-1], block_m)
                        self.assertEqual(gate.call_args.kwargs["tile_m"], block_m)
                        self.assertEqual(down.call_args.kwargs["tile_m"], block_m)

            # A cached layout does not override an explicit backend selection
            # or a different normalization contract.
            x = torch.randn(256, 4096, dtype=torch.bfloat16, device="cuda")
            ids = torch.full((256, 8), 96, dtype=torch.int32, device="cuda")
            scores = torch.full((256, 8), 0.125, device="cuda")
            with patch.object(model, "_run_fused_moe", return_value=x) as fallback:
                with patch.dict(os.environ, {"MUSE_MOE_BACKEND": "triton"}):
                    experts(x, scores, ids, 96, norm)
                experts(x, scores, ids, 96, None)
                experts(
                    x, scores, ids, 96, SimpleNamespace(**{**vars(norm), "eps": 1e-5})
                )
                self.assertEqual(fallback.call_count, 3)

            for num_rows in (16, 56, 64, 144, 256):
                with self.subTest(num_rows=num_rows):
                    self._assert_graph_reference(experts, norm, num_rows)
            # Disabling/reloading removes derived storage rather than retaining
            # a stale cache. Checkpoint state remains native and unchanged.
            with patch.dict(os.environ, {"SGLANG_MUSE_MOE_PREPACKED": "0"}):
                experts.prepare_prepacked_weights()
                self.assertIsNone(experts._prepacked_moe)
            self.assertEqual(len(experts.state_dict()), 4)

    def _assert_graph_reference(self, experts, norm, num_rows):
        x = torch.randn(num_rows, 4096, dtype=torch.bfloat16, device="cuda")
        ids = torch.full((num_rows, 8), 96, dtype=torch.int32, device="cuda")
        scores = torch.rand(num_rows, 8, device="cuda")
        scores /= scores.sum(-1, keepdim=True)
        for _ in range(3):
            experts(x, scores, ids, 96, norm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = experts(x, scores, ids, 96, norm)

        # Independent FP4 decode and FP32 GEMMs. Retain the model's BF16
        # SiLU/down/scaled-norm boundaries, then combine route scores.
        lookup = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )

        def decode(weight, scale):
            codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2)
            return lookup[codes.long()] * torch.exp2(
                scale.float() - 127
            ).repeat_interleave(32, -1)

        weights = {
            e: (
                decode(experts.w13_weight[e], experts.w13_weight_scale[e]),
                decode(experts.w2_weight[e], experts.w2_weight_scale[e]),
            )
            for e in (0, 31)
        }
        for case in ("hot", "mixed", "empty", "changed"):
            x.normal_()
            if case == "empty":
                ids.fill_(-1)
            elif case == "hot":
                ids.fill_(96)
            else:
                ids.copy_(
                    torch.tensor([95, 96, 127, 128, -1, 255, 96, 127], device="cuda")
                )
            scores.uniform_()
            scores /= scores.sum(-1, keepdim=True)
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.zeros_like(x, dtype=torch.float32)
            for e, (gate_up, down) in weights.items():
                rows, slots = torch.where(ids == 96 + e)
                if not rows.numel():
                    continue
                gu = F.linear(x[rows].float(), gate_up)
                gate, up = gu.chunk(2, -1)
                inter = (F.silu(gate) * up).bfloat16()
                y = F.linear(inter.float(), down).bfloat16()
                y = (y * (norm.weight + 1.0)).float()
                y *= torch.rsqrt(y.square().mean(-1, keepdim=True) + norm.eps)
                expected.index_add_(0, rows, y * scores[rows, slots, None])
            error = torch.linalg.vector_norm(output.float() - expected)
            scale = torch.linalg.vector_norm(expected).clamp_min(1e-20)
            self.assertTrue(torch.isfinite(output).all())
            self.assertLess((error / scale).item(), 0.007)
        # Flip every FP4 sign bit, then rebuild after a weight update.
        # A stale shuffled cache would produce the opposite answer.
        addresses = [b.data_ptr() for b in experts._prepacked_moe.buffers()]
        experts.w2_weight.bitwise_xor_(0x88)
        with patch.dict(os.environ, {"SGLANG_MUSE_MOE_PREPACKED": "1"}):
            experts.prepare_prepacked_weights()
        self.assertEqual(
            addresses, [b.data_ptr() for b in experts._prepacked_moe.buffers()]
        )
        graph.replay()
        error = torch.linalg.vector_norm(output.float() + expected)
        self.assertLess((error / scale).item(), 0.007)
        graph.reset()


if __name__ == "__main__":
    unittest.main()
