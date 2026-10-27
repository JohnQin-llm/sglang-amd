"""EP4 expert placement of the Muse Spark routed experts (SGLANG_MUSE_EP4_PLACEMENT):
the shipped table and the load-time relabelling of the expert weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

EP, EXPERTS, MOE_LAYERS, LAYER = 4, 256, 62, 5
LOCAL = EXPERTS // EP
PREFIX = f"model.layers.{LAYER}.mlp.experts"
# Tiny packed shapes: gate/up [2, 4] each, down [4, 2]; every byte holds its expert id.
GATE, DOWN = (2, 4), (4, 2)


def expert_bytes(shape, expert):
    return torch.full(shape, expert, dtype=torch.uint8)


@unittest.skipUnless(is_hip(), "the ROCm Muse Spark model")
class TestMuseSparkEp4Placement(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.rocm_muse_spark.ep4_placement import layer_placements
        from sglang.srt.rocm_muse_spark.model import MuseSparkTextForCausalLM

        cls.layer_placements = staticmethod(layer_placements)
        cls.model_cls = MuseSparkTextForCausalLM

    def test_table_places_every_moe_layer(self):
        """The table covers all 62 MoE layers at EP4, and other EP sizes keep the
        checkpoint order."""
        placements = self.layer_placements(EP, EXPERTS)
        self.assertEqual(sorted(placements), list(range(MOE_LAYERS)))
        self.assertEqual(self.layer_placements(8, EXPERTS), {})

    def test_each_rank_loads_the_experts_of_its_slots(self):
        """Per-expert and stacked checkpoint tensors land in the slots the table
        gives their experts on every rank: slot p holds ``expert[p]``, the logical
        expert whose router column the loader moves to column p."""
        placement = self.layer_placements(EP, EXPERTS)[LAYER]
        stack = torch.stack([expert_bytes(GATE, e) for e in range(EXPERTS)])
        for rank in range(EP):
            with self.subTest(rank=rank):
                loader = SimpleNamespace(
                    config=SimpleNamespace(num_local_experts=EXPERTS),
                    expert_placements={LAYER: placement},
                )
                params = {
                    f"{PREFIX}.w13_weight": torch.zeros(
                        LOCAL, 2 * GATE[0], GATE[1], dtype=torch.uint8
                    ),
                    f"{PREFIX}.w2_weight": torch.zeros(LOCAL, *DOWN, dtype=torch.uint8),
                }
                stacked = torch.zeros(LOCAL, *GATE, dtype=torch.uint8)
                parallel = SimpleNamespace(moe_ep_size=EP, moe_ep_rank=rank)
                with patch(
                    "sglang.srt.rocm_muse_spark.model.get_parallel",
                    return_value=parallel,
                ):
                    for e in range(EXPERTS):
                        for projection, shape in (
                            ("gate_proj", GATE),
                            ("up_proj", GATE),
                            ("down_proj", DOWN),
                        ):
                            self.model_cls._load_single_expert(
                                loader,
                                params,
                                f"{PREFIX}.{e}.{projection}.weight",
                                expert_bytes(shape, e),
                            )
                    self.model_cls._copy_expert_stack(
                        loader, stacked, stack, f"{PREFIX}.w2_weight"
                    )
                expected = placement.expert[rank * LOCAL : (rank + 1) * LOCAL].to(
                    torch.uint8
                )
                for loaded in (
                    params[f"{PREFIX}.w13_weight"],
                    params[f"{PREFIX}.w2_weight"],
                    stacked,
                ):
                    self.assertTrue(torch.equal(loaded.flatten(1).amin(1), expected))
                    self.assertTrue(torch.equal(loaded.flatten(1).amax(1), expected))


if __name__ == "__main__":
    unittest.main()
