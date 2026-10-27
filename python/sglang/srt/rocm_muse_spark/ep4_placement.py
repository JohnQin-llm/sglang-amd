"""Static per-layer expert placement for Muse Spark under EP4 (``SGLANG_MUSE_EP4_PLACEMENT``).

Each MoE layer ends in an all-reduce that waits for the slowest rank. At verify widths a rank's expert stage
costs a fixed part plus one weight stream per distinct local expert it is routed to. In checkpoint order rank r
owns experts 64r..64r+63, and some layers' popular experts cluster on one rank. ``ep4_expert_placement.json``
lists, per MoE layer, the 64 logical experts each rank holds instead.

The placement is a relabelling at load time. Logical expert e moves to physical slot ``slot[e]``, and rank r
still owns slots 64r..64r+63. The router columns and ``expert_bias`` load in physical order, so top-k returns
physical ids, and each expert's weights load into its slot. Every consumer of routes and expert weights keeps the
contiguous layout it already handles. Each row routes to the same experts (every logit is the same dot product in
a moved column). The MoE output is not bitwise: moving experts between ranks re-associates the all-reduce sum.

Provenance: 24,198 target-verify steps (batch 1..13, 8..112 rows) with the routes of every MoE layer, recorded
from a 600 s AgentX C16 serving run at TP4/EP4 on MI355X. Per layer, a local search over cross-rank expert swaps
minimizes the sum over steps of the largest per-rank count of distinct routed experts, and the table is the fit
on all recorded steps. Priced on held-out steps (fit on alternating 2000-step blocks, scored on the others), it
lowers that count, summed over the 62 layers, by 53 experts per step at 8 rows and by 73-91 at 16..112 rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import torch

_TABLE = Path(__file__).with_name("ep4_expert_placement.json")


class LayerPlacement(NamedTuple):
    """int64 CPU permutations: ``slot[e]`` is logical expert e's physical slot, ``expert[p]`` the logical expert
    in physical slot p."""

    slot: torch.Tensor
    expert: torch.Tensor


def layer_placements(ep_size: int, num_experts: int) -> dict[int, LayerPlacement]:
    """Placement per MoE layer index; empty unless the table was fitted for this EP size and expert count."""
    table = json.loads(_TABLE.read_text())
    if (ep_size, num_experts) != (table["ep_size"], table["num_experts"]):
        return {}
    placements = {}
    for layer, ranks in table["layers"].items():
        assert len(ranks) == ep_size, f"layer {layer}: {len(ranks)} ranks"
        assert all(len(experts) == num_experts // ep_size for experts in ranks)
        expert = torch.tensor([e for experts in ranks for e in experts])
        assert torch.equal(expert.sort().values, torch.arange(num_experts)), layer
        slot = torch.empty_like(expert)
        slot[expert] = torch.arange(num_experts)
        placements[int(layer)] = LayerPlacement(slot=slot, expert=expert)
    return placements
