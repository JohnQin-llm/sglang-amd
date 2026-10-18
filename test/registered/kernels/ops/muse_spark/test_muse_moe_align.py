import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "Requires MI35x")
class TestMuseMoeAlign(CustomTestCase):
    def test_moe_graph_and_shape_fallback(self):
        from sglang.srt.rocm_muse_spark import muse_moe_flydsl as module

        torch.manual_seed(425)
        w13 = torch.randint(0, 256, (32, 8192, 2048), dtype=torch.uint8, device="cuda")
        w2 = torch.randint(0, 256, (32, 4096, 2048), dtype=torch.uint8, device="cuda")
        s13 = torch.full((32, 8192, 128), 119, dtype=torch.uint8, device="cuda")
        s2 = torch.full((32, 4096, 128), 119, dtype=torch.uint8, device="cuda")
        weight = torch.zeros(4096, dtype=torch.bfloat16, device="cuda")
        for rows in (8, 248, 256, 264):
            x = torch.randn(rows, 4096, dtype=torch.bfloat16, device="cuda")
            ids = torch.randint(80, 144, (rows, 8), dtype=torch.int32, device="cuda")
            scores = torch.rand(rows, 8, device="cuda")
            scores /= scores.sum(-1, keepdim=True)

            def run(x=x, scores=scores, ids=ids):
                return module.muse_moe_flydsl(
                    x,
                    scores,
                    ids,
                    w13,
                    w2,
                    96,
                    weight,
                    post_norm_gain=1.0,
                    w13_scale=s13,
                    w2_scale=s2,
                )

            captures = []
            flags = [(False, False), (True, True)]
            if rows == 256:
                flags.extend([(True, False), (False, True)])
            for use_m32, use_compact in flags:
                with (
                    patch.object(module, "_USE_VERIFY_M32", use_m32),
                    patch.object(module, "_USE_COMPACT_ALIGN", use_compact),
                    patch.object(
                        module,
                        "moe_align_block_size",
                        wraps=module.moe_align_block_size,
                    ) as fallback,
                    patch.object(
                        module, "_launch_gemm", wraps=module._launch_gemm
                    ) as launch,
                ):
                    for _ in range(3):
                        run()
                    expected_block_m = module._pick_flydsl_config(rows, 8, 32, True)[
                        "BLOCK_M"
                    ]
                    if use_m32 and rows == 256:
                        expected_block_m = 32
                    self.assertEqual(launch.call_count, 6)
                    for call in launch.call_args_list:
                        self.assertEqual(call.args[14]["BLOCK_M"], expected_block_m)
                    if use_compact and rows == 256:
                        fallback.assert_not_called()
                    else:
                        self.assertEqual(fallback.call_count, 3)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = run()
                    captures.append((graph, output))
            for repeat in range(3):
                x.normal_()
                ids.random_(80, 144)
                if repeat == 2:
                    ids.fill_(-1)
                for graph, _ in captures:
                    graph.replay()
                torch.cuda.synchronize()
                for _, output in captures[1:]:
                    torch.testing.assert_close(output, captures[0][1], atol=0, rtol=0)

    def test_changed_graph_routing_and_padding(self):
        from sglang.kernels.ops.muse_spark.moe_align import align_local

        torch.manual_seed(318)
        for rows in (8, 256):
            for block_m in (16, 32):
                ids = torch.zeros((rows, 8), dtype=torch.int32, device="cuda")
                for _ in range(3):
                    align_local(ids, 32, 96, block_m)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    sorted_ids, experts, total = align_local(ids, 32, 96, block_m)
                # Include entirely remote routes, padding, all-local traffic,
                # a hot expert and both edges of the local expert interval.
                cases = [
                    torch.full_like(ids, -1),
                    torch.randint(0, 256, ids.shape, device="cuda", dtype=ids.dtype),
                    torch.randint(96, 128, ids.shape, device="cuda", dtype=ids.dtype),
                    torch.full_like(ids, 96),
                    torch.full_like(ids, 127),
                ]
                cases[1][0] = torch.tensor(
                    [-1, 95, 96, 127, 128, 0, 255, -1], device="cuda"
                )
                for case in cases:
                    ids.copy_(case)
                    for _ in range(3):
                        graph.replay()
                    flattened = ids.cpu().flatten()
                    expected_slots, expected_experts = [], []
                    for expert in range(32):
                        slots = torch.where(flattened == expert + 96)[0].tolist()
                        padding = (-len(slots)) % block_m
                        expected_slots.extend(slots + [ids.numel()] * padding)
                        expected_experts.extend(
                            [expert] * ((len(slots) + padding) // block_m)
                        )
                    self.assertEqual(total.item(), len(expected_slots))
                    self.assertEqual(
                        sorted_ids[: len(expected_slots)].cpu().tolist(), expected_slots
                    )
                    actual_experts = experts.cpu().tolist()
                    self.assertEqual(
                        actual_experts[: len(expected_experts)], expected_experts
                    )
                    self.assertTrue(
                        all(e == -1 for e in actual_experts[len(expected_experts) :])
                    )


if __name__ == "__main__":
    unittest.main()
