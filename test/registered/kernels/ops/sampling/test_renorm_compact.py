import os
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.sampling.renorm_compact import top_p_renorm_probs_compact
from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")


def reference(probs, top_ps):
    values = probs.float().double()
    sorted_values = values.sort(-1).values
    target = (1.0 - top_ps.float()).double().expand(values.shape[0])
    cut = torch.searchsorted(sorted_values.cumsum(-1), target[:, None].contiguous())
    pivot = sorted_values.gather(1, cut.clamp(max=values.shape[1] - 1))
    kept = torch.where(values >= pivot, values, 0.0)
    return kept / kept.sum(-1, keepdim=True)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "Requires MI35x")
class TestCompactRenorm(CustomTestCase):
    def assert_reference(self, actual, probs, top_ps):
        expected = reference(probs, top_ps)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue(torch.equal(actual > 0, expected > 0))
        self.assertLess((actual.double() - expected).abs().sum(-1).max().item(), 2e-6)

    def test_serving_shapes_and_graph_transitions(self):
        torch.manual_seed(43)
        for rows in (8, 256):
            probs = torch.zeros((rows, 202048), device="cuda")
            probs[:, 0] = 1.0
            top_ps = torch.full((rows,), 0.95, device="cuda")
            top_p_renorm_probs_compact(probs, top_ps)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = top_p_renorm_probs_compact(probs, top_ps)
            # The same graph switches between bounded sort and radix paths,
            # mixed rows, and p=1; no host decision can specialize the data.
            for kind in ("sparse", "uniform", "mixed", "full"):
                with self.subTest(rows=rows, kind=kind):
                    if kind == "sparse":
                        probs.zero_()
                        probs[:, :3] = torch.tensor([0.125, 0.25, 0.625], device="cuda")
                    elif kind == "uniform":
                        probs.fill_(1.0 / probs.shape[1])
                    else:
                        probs.copy_(torch.randn_like(probs).mul_(5).softmax(-1))
                        probs[::2].fill_(1.0 / probs.shape[1])
                    top_ps.fill_(1.0 if kind == "full" else 0.95)
                    graph.replay()
                    self.assert_reference(output, probs, top_ps)
            graph.reset()

    def test_exact_boundaries_and_input_layout(self):
        for nonzero in (3, 1023, 1024, 1025, 2049):
            probs = torch.zeros((8, 202048), device="cuda")
            if nonzero == 3:
                probs[:, :3] = torch.tensor([0.125, 0.25, 0.625], device="cuda")
                top_ps = torch.full((8,), 0.875, device="cuda")
            elif nonzero < 2049:
                probs[:, : nonzero - 1] = 1.0 / 4096
                probs[:, nonzero - 1] = 1.0 - (nonzero - 1) / 4096
                top_ps = torch.full((8,), 1.0 - (nonzero - 1) / 4096, device="cuda")
            else:
                probs[:, :1024] = 1.0 / 4096
                probs[:, 1024:2048] = 1.0 / 2048
                probs[:, 2048] = 0.25
                top_ps = torch.full((8,), 0.75, device="cuda")
            for direction in (0.0, None, 1.0):
                threshold = (
                    top_ps
                    if direction is None
                    else torch.nextafter(top_ps, torch.full_like(top_ps, direction))
                )
                actual = top_p_renorm_probs_compact(probs, threshold)
                self.assert_reference(actual, probs, threshold)
                expected = top_p_renorm_probs_triton(probs, threshold)
                self.assertTrue(torch.equal(actual > 0, expected > 0))
            # A non-contiguous probability view and broadcast threshold use
            # the same preparation contract as the existing implementation.
            view = probs.to(torch.bfloat16)[:, ::2]
            actual = top_p_renorm_probs_compact(view, top_ps[:1])
            self.assert_reference(actual, view, top_ps[:1])

    def test_concurrent_graphs(self):
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        inputs = [torch.randn((n, 202048), device="cuda").softmax(-1) for n in (8, 256)]
        thresholds = [torch.full((n,), 0.95, device="cuda") for n in (8, 256)]
        outputs = []
        for stream, graph, probs, top_ps in zip(streams, graphs, inputs, thresholds):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                top_p_renorm_probs_compact(probs, top_ps)
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                outputs.append(top_p_renorm_probs_compact(probs, top_ps))
        for kind in ("dense", "sparse", "dense"):
            for probs in inputs:
                if kind == "dense":
                    probs.copy_(torch.randn_like(probs).softmax(-1))
                else:
                    probs.zero_()
                    probs[:, 17] = 1.0
            for stream, graph in zip(streams, graphs):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph.replay()
            for stream in streams:
                stream.synchronize()
            for actual, probs, top_ps in zip(outputs, inputs, thresholds):
                self.assert_reference(actual, probs, top_ps)
        for graph in graphs:
            graph.reset()

    def test_dflash_dispatch_scope(self):
        from sglang.kernels.ops.sampling import renorm_compact
        from sglang.srt.speculative import dflash_utils

        probs = torch.empty((8, 202048), device="cuda")
        top_ps = torch.ones(8, device="cuda")
        old, new = object(), object()
        with (
            patch.object(dflash_utils, "top_p_renorm_prob", return_value=old),
            patch.object(
                renorm_compact, "top_p_renorm_probs_compact", return_value=new
            ),
        ):
            with patch.dict(os.environ, {"SGLANG_MUSE_COMPACT_TOP_P": "0"}):
                self.assertIs(
                    dflash_utils._dflash_top_p_renorm_prob(probs, top_ps), old
                )
            with patch.dict(os.environ, {"SGLANG_MUSE_COMPACT_TOP_P": "1"}):
                self.assertIs(
                    dflash_utils._dflash_top_p_renorm_prob(probs, top_ps), new
                )
                for guard in ("is_hip", "is_gfx95_supported"):
                    with patch.object(dflash_utils, guard, return_value=False):
                        self.assertIs(
                            dflash_utils._dflash_top_p_renorm_prob(probs, top_ps), old
                        )
                for unsupported in (
                    probs[:, :16],
                    probs.bfloat16(),
                    torch.empty((257, 202048), device="cuda"),
                    torch.empty((8, 202048 * 2), device="cuda")[:, ::2],
                ):
                    thresholds = torch.ones(unsupported.shape[0], device="cuda")
                    self.assertIs(
                        dflash_utils._dflash_top_p_renorm_prob(unsupported, thresholds),
                        old,
                    )

    def test_empty_and_invalid_arguments(self):
        for shape in ((0, 202048), (8, 0)):
            probs = torch.empty(shape, device="cuda")
            self.assertEqual(top_p_renorm_probs_compact(probs, 0.95).shape, shape)
        for mass in (0.0, 1e-11):
            probs = torch.zeros((8, 202048), device="cuda")
            probs[:, :3] = torch.tensor([0.1, 0.2, 0.7], device="cuda") * mass
            for threshold in (0.95, 1.0):
                actual = top_p_renorm_probs_compact(probs, threshold)
                expected = top_p_renorm_probs_triton(probs, threshold)
                torch.testing.assert_close(actual, expected, equal_nan=True)
        probs = torch.ones((8, 111), device="cuda") / 111
        self.assert_reference(
            top_p_renorm_probs_compact(probs, 0.5),
            probs,
            torch.tensor([0.5], device="cuda"),
        )
        for invalid in (0.0, -1.0, 1.1):
            with self.assertRaises(ValueError):
                top_p_renorm_probs_compact(probs, invalid)
        with self.assertRaises(ValueError):
            top_p_renorm_probs_compact(probs, torch.ones(3, device="cuda"))


if __name__ == "__main__":
    unittest.main()
