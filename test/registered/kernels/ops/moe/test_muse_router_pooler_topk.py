"""Muse Spark joint router/pooler (joint_router_pooler_topk) at every verify width
against an FP32 reference of the unfused routing front."""

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

HIDDEN, EXPERTS, POOLED, TOP_K = 8192, 256, 4096, 8
MULTIPLIER, ROUTE_SCALE, NORM_EPS = 1.5, 2.5, 1e-5
VERIFY_WIDTHS = range(8, 129, 8)
# Rows whose 8th and 9th biased scores are closer than this may swap experts
# between FP32 summation orders; the joint form's score error is below 1e-6.
ROUTE_MARGIN = 1e-5


class _MoE(nn.Module):
    """The weights prepare_joint_router_pooler and the joint route read from a
    Muse Spark MoE layer."""

    def __init__(self, router, pooler, expert_bias, norm_weight):
        super().__init__()
        self.router = nn.Module()
        self.router.weight = nn.Parameter(router, requires_grad=False)
        self.pre_expert_proj = nn.Module()
        self.pre_expert_proj.weight = nn.Parameter(pooler, requires_grad=False)
        self.expert_bias = expert_bias
        self.norm_weight = norm_weight


def random_moe(seed: int) -> _MoE:
    g = torch.Generator().manual_seed(seed)
    return _MoE(
        router=torch.randn(HIDDEN, EXPERTS, generator=g).mul_(0.005).cuda(),
        pooler=torch.randn(POOLED, HIDDEN, generator=g).mul_(0.01).bfloat16().cuda(),
        expert_bias=torch.randn(EXPERTS, generator=g).mul_(0.01).cuda(),
        norm_weight=(1 + 0.1 * torch.randn(POOLED, generator=g)).bfloat16().cuda(),
    )


def inputs(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, HIDDEN, generator=g).bfloat16().cuda()


def joint(moe: _MoE, x: torch.Tensor):
    from sglang.kernels.ops.moe.router_pooler_topk import joint_router_pooler_topk

    return joint_router_pooler_topk(
        x,
        moe.router_pooler_packed_weight,
        moe.expert_bias,
        moe.norm_weight,
        MULTIPLIER,
        ROUTE_SCALE,
        NORM_EPS,
    )


def capture(fn):
    """Warm ``fn`` up on a side stream and capture it: (graph, its outputs)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    return graph, outputs


def reference(x, moe: _MoE):
    """FP32 router logits, sigmoid top-8 on the biased scores, and the pooler
    projection rounded to BF16 before the RMSNorm."""
    scores = torch.sigmoid((x.float() @ moe.router.weight) * MULTIPLIER)
    order = (scores + moe.expert_bias).topk(TOP_K + 1, dim=-1)
    ids = order.indices[:, :TOP_K]
    margin = order.values[:, TOP_K - 1] - order.values[:, TOP_K]
    selected = scores.gather(-1, ids)
    weights = selected * (ROUTE_SCALE / (selected.sum(-1, keepdim=True) + 1e-15))
    pooled = F.linear(x.float(), moe.pre_expert_proj.weight.float())
    pooled = pooled.bfloat16().float()
    rstd = torch.rsqrt(pooled.pow(2).mean(-1, keepdim=True) + NORM_EPS)
    return ids, weights, pooled * rstd * moe.norm_weight.float(), margin


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "tuned for gfx950")
class TestMuseJointRouterPooler(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.rocm_muse_spark.router_pooler import (
            prepare_joint_router_pooler,
        )

        cls.moe = random_moe(seed=0)
        prepare_joint_router_pooler(cls.moe)

    def test_routes_and_expert_input_match_fp32_reference(self):
        """Every verify width routes like the FP32 unfused front and normalizes
        the BF16-rounded pooler output."""
        for rows in VERIFY_WIDTHS:
            with self.subTest(rows=rows):
                x = inputs(rows, seed=rows)
                scores, ids, expert_input = joint(self.moe, x)
                ref_ids, ref_weights, ref_input, margin = reference(x, self.moe)
                clear = margin > ROUTE_MARGIN
                self.assertGreaterEqual(int(clear.sum()), 0.95 * rows)
                # Same expert set per row; near-equal scores may swap slots.
                ids, order = ids.long().sort(dim=-1)
                ref_ids, ref_order = ref_ids.sort(dim=-1)
                torch.testing.assert_close(ids[clear], ref_ids[clear], rtol=0, atol=0)
                torch.testing.assert_close(
                    scores.gather(-1, order)[clear],
                    ref_weights.gather(-1, ref_order)[clear],
                    rtol=0,
                    atol=2e-5,
                )
                cosine = F.cosine_similarity(expert_input.float(), ref_input, dim=-1)
                self.assertGreater(float(cosine.min()), 0.99999)
                torch.testing.assert_close(
                    expert_input.float(), ref_input, rtol=1.6e-2, atol=1.6e-2
                )

    def test_graph_replay_matches_eager(self):
        """The joint route is deterministic and a captured graph replays it on new
        inputs bit for bit, at the split-K and the hipBLASLt widths."""
        for rows in (40, 48, 96, 128):
            with self.subTest(rows=rows):
                static = inputs(rows, seed=1)
                first = joint(self.moe, static)
                for a, b in zip(first, joint(self.moe, static)):
                    self.assertTrue(torch.equal(a, b))
                graph, captured = capture(lambda: joint(self.moe, static))
                fresh = inputs(rows, seed=2)
                static.copy_(fresh)
                graph.replay()
                for a, b in zip(captured, joint(self.moe, fresh)):
                    self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
