"""The DFLASH tree builder must pick the best-first prefixes under predecessor-conditional scores."""

import heapq
import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dflash_tree import (
    build_dflash_sampled_tree_proposal,
    build_dflash_tree_proposal,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")


def _reference(candidate_ids, cond_log_probs, max_nodes):
    """Best-first expansion where a child's score depends on its parent's rank."""
    num_depths, k = candidate_ids.shape
    nodes = [(0, 0, 0.0)]  # (depth, rank, mass) of the root
    edges, tokens, heap = [], [], []

    def push_children(node):
        depth, rank, mass = nodes[node]
        if depth < num_depths:
            for child_rank in range(k):
                child_mass = mass + cond_log_probs[depth, rank, child_rank].item()
                heapq.heappush(heap, (-child_mass, node, child_rank))

    push_children(0)
    while len(nodes) < max_nodes:
        neg_mass, parent, rank = heapq.heappop(heap)
        depth = nodes[parent][0]
        edges.append(parent * k + rank)
        tokens.append(candidate_ids[depth, rank].item())
        nodes.append((depth + 1, rank, -neg_mass))
        push_children(len(nodes) - 1)
    return edges, tokens


@pytest.mark.parametrize(
    "num_depths, k, max_nodes", [(7, 16, 16), (7, 16, 64), (7, 8, 33), (3, 4, 10)]
)
def test_matches_best_first_reference(num_depths, k, max_nodes):
    gen = torch.Generator().manual_seed(num_depths * 100 + k + max_nodes)
    bs = 6
    candidate_ids = torch.randint(0, 50000, (bs, num_depths, k), generator=gen)
    scores = torch.randn(bs, num_depths, k, k, generator=gen) * 2
    scores[:, 0] = scores[:, 0, :1]  # depth 0 does not depend on the predecessor
    cond_log_probs = torch.log_softmax(scores, dim=-1)
    proposal = build_dflash_tree_proposal(
        candidate_ids.cuda(), cond_log_probs.cuda(), max_nodes=max_nodes
    )
    for b in range(bs):
        edges, tokens = _reference(candidate_ids[b], cond_log_probs[b], max_nodes)
        assert proposal.top_scores_index[b].tolist() == edges, f"row {b}"
        assert proposal.draft_tokens[b].tolist() == tokens, f"row {b}"
        assert proposal.parent_list[b, 0].item() == -1
        assert (
            proposal.parent_list[b, 1 : max_nodes - 1].tolist()
            == edges[: max_nodes - 2]
        )


def test_sampled_tree_draws_children_from_q():
    # The first node is the root's first draw, so its token must follow q at depth 0.
    rows, num_depths, k = 40000, 3, 4
    gen = torch.Generator().manual_seed(3)
    log_q = torch.log_softmax(
        torch.randn(num_depths, k, k, generator=gen) * 1.5, dim=-1
    )
    candidate_ids = (
        torch.arange(num_depths * k).view(1, num_depths, k).repeat(rows, 1, 1)
    )
    proposal, node_tokens, node_probs = build_dflash_sampled_tree_proposal(
        candidate_ids.cuda(), log_q.repeat(rows, 1, 1, 1).cuda(), max_nodes=6
    )
    first = proposal.draft_tokens[:, 0].cpu()
    freq = torch.bincount(first, minlength=k).double() / rows
    assert (freq - log_q[0, 0].exp().double()).abs().max().item() < 0.012
    assert torch.allclose(node_probs[:, 0].cpu(), log_q[0, 0].exp().expand(rows, k))
    assert torch.equal(node_tokens[:, 0].cpu(), candidate_ids[:, 0])
    # Siblings are listed in drawing order, so a node's children never repeat a token.
    edges = proposal.top_scores_index.cpu()
    for b in range(64):
        parents = (edges[b] // k).tolist()
        tokens = proposal.draft_tokens[b].cpu().tolist()
        for parent in set(parents):
            kids = [t for p, t in zip(parents, tokens) if p == parent]
            assert len(kids) == len(set(kids))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
