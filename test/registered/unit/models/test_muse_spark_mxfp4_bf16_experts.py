"""Load-time MXFP4 rounding of the BF16 routed experts that a Quark MXFP4 export
leaves unquantized (SGLANG_MUSE_MXFP4_BF16_EXPERTS)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")

HIDDEN, INTERMEDIATE, GROUP = 256, 128, 32
# OCP MX E2M1 magnitudes by the low three bits of a nibble; bit 3 is the sign.
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequantize(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quark layout: two E2M1 values per byte, low nibble first, one E8M0 scale per 32 columns."""
    lut = torch.tensor(E2M1 + tuple(-x for x in E2M1), device=packed.device)
    values = torch.stack(
        (lut[(packed & 0xF).long()], lut[(packed >> 4).long()]), dim=-1
    ).flatten(-2)
    return values * torch.exp2(scale.float() - 127).repeat_interleave(GROUP, dim=-1)


def random_mxfp4(rows: int, cols: int, generator: torch.Generator):
    """Packed weights and scales whose BF16 dequantization is exact and re-quantizes
    to the same bytes: every group holds a +-4 or +-6, and no negative zero."""
    nibbles = torch.randint(0, 15, (rows, cols), generator=generator)
    nibbles = torch.where(nibbles >= 8, nibbles + 1, nibbles)  # skip 8, i.e. -0.0
    group_max = torch.tensor([6, 7, 14, 15])[
        torch.randint(0, 4, (rows, cols // GROUP), generator=generator)
    ]
    nibbles.view(rows, cols // GROUP, GROUP)[:, :, 0] = group_max
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
    scale = torch.randint(
        117, 138, (rows, cols // GROUP), generator=generator, dtype=torch.uint8
    )
    return packed.cuda(), scale.cuda()


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "MXFP4 experts need gfx950")
class TestMuseSparkMxfp4Bf16Experts(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.rocm_muse_spark.model import (
            MuseSpark1xMxfp4Experts,
            MuseSparkTextForCausalLM,
        )

        cls.experts_cls = MuseSpark1xMxfp4Experts
        cls.model_cls = MuseSparkTextForCausalLM

    def test_round_trip_reproduces_quark_bytes(self):
        """MXFP4 values written out as BF16 round back to the same packed bytes and
        scales, which pins the nibble order, the scale layout and the rounding."""
        packed, scale = random_mxfp4(
            2 * INTERMEDIATE, HIDDEN, torch.Generator().manual_seed(0)
        )
        weight = torch.empty_like(packed)
        weight_scale = torch.empty_like(scale)
        self.model_cls._round_expert_to_mxfp4(
            "w", dequantize(packed, scale).to(torch.bfloat16), weight, weight_scale
        )
        torch.testing.assert_close(weight, packed, rtol=0, atol=0)
        torch.testing.assert_close(weight_scale, scale, rtol=0, atol=0)

    def test_loader_writes_each_projection_to_its_rows(self):
        """gate_proj and up_proj fill the two halves of w13 and of its scales,
        down_proj fills w2, and other experts are left alone."""
        prefix = "model.layers.0.mlp.experts"
        experts = self.experts_cls(2, HIDDEN, INTERMEDIATE).cuda()
        params = {f"{prefix}.{n}": p for n, p in experts.named_parameters()}
        for p in params.values():
            p.data.fill_(0xAA)
        generator = torch.Generator().manual_seed(1)
        expected = {
            "gate_proj": random_mxfp4(INTERMEDIATE, HIDDEN, generator),
            "up_proj": random_mxfp4(INTERMEDIATE, HIDDEN, generator),
            "down_proj": random_mxfp4(HIDDEN, INTERMEDIATE, generator),
        }
        loader = SimpleNamespace(
            config=SimpleNamespace(num_local_experts=2),
            _round_expert_to_mxfp4=self.model_cls._round_expert_to_mxfp4,
        )
        parallel = SimpleNamespace(moe_ep_size=1, moe_ep_rank=0)
        with patch(
            "sglang.srt.rocm_muse_spark.model.get_parallel", return_value=parallel
        ):
            for projection, (packed, scale) in expected.items():
                bf16 = dequantize(packed, scale).to(torch.bfloat16).cpu()
                name = f"{prefix}.1.{projection}.weight"
                self.assertTrue(
                    self.model_cls._load_single_expert(loader, params, name, bf16)
                )

        w13, w13_scale = experts.w13_weight[1], experts.w13_weight_scale[1]
        rows = slice(0, INTERMEDIATE), slice(INTERMEDIATE, 2 * INTERMEDIATE)
        for projection, half in zip(("gate_proj", "up_proj"), rows):
            packed, scale = expected[projection]
            torch.testing.assert_close(w13[half], packed, rtol=0, atol=0)
            torch.testing.assert_close(w13_scale[half], scale, rtol=0, atol=0)
        packed, scale = expected["down_proj"]
        torch.testing.assert_close(experts.w2_weight[1], packed, rtol=0, atol=0)
        torch.testing.assert_close(experts.w2_weight_scale[1], scale, rtol=0, atol=0)
        for p in experts.parameters():
            self.assertTrue(bool((p[0] == 0xAA).all()))


if __name__ == "__main__":
    unittest.main()
