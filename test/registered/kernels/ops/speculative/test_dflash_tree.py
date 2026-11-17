"""The DFLASH tree builder must pick the best-first prefixes under predecessor-conditional scores."""

import heapq
import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dflash_tree import (
    build_dflash_sampled_tree_proposal,
    build_dflash_tree_proposal,
    update_draw_acceptance,
)
from sglang.kernels.ops.speculative.multi_draft_sampling import (
    tree_speculative_sampling_multi_draft,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")


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


@pytest.mark.parametrize("accept_prior", [None, (0.6, 0.3)])
def test_sampled_tree_verification_commits_the_target(accept_prior):
    """The tree builder once let a drawn child's own q decide whether it entered the tree,
    so the children kept were biased toward high q and multi-draft verification committed
    tokens off the target distribution (max error 0.033 here, about 20 sigma)."""
    rows, num_depths, k, max_nodes = 100000, 3, 4, 6
    gen = torch.Generator().manual_seed(7)
    log_q = torch.log_softmax(
        torch.randn(num_depths, k, k, generator=gen) * 2.0, dim=-1
    )
    vocab = num_depths * k
    target = torch.softmax(torch.randn(vocab, generator=gen) * 1.5, dim=-1)
    candidate_ids = torch.arange(vocab).view(1, num_depths, k).repeat(rows, 1, 1).cuda()
    proposal, node_tokens, node_probs = build_dflash_sampled_tree_proposal(
        candidate_ids,
        log_q.repeat(rows, 1, 1, 1).cuda(),
        max_nodes=max_nodes,
        accept_prior=accept_prior,
    )
    seq_lens = torch.full((rows,), 10, dtype=torch.long, device="cuda")
    _, _, retrieve_index, next_token, next_sibling, tokens = (
        build_tree_kernel_efficient(
            torch.zeros(rows, dtype=torch.long, device="cuda"),
            proposal.parent_list,
            proposal.top_scores_index,
            proposal.draft_tokens,
            seq_lens,
            int(seq_lens.sum()),
            k,
            num_depths,
            max_nodes,
            TreeMaskMode.FULL_MASK,
        )
    )
    predicts = torch.full((rows * max_nodes,), -1, dtype=torch.int32, device="cuda")
    tree_speculative_sampling_multi_draft(
        predicts=predicts,
        accept_index=torch.full(
            (rows, num_depths + 1), -1, dtype=torch.int32, device="cuda"
        ),
        accept_token_num=torch.zeros(rows, dtype=torch.int32, device="cuda"),
        candidates=tokens.view(rows, max_nodes),
        retrive_index=retrieve_index.view(rows, max_nodes),
        retrive_next_token=next_token.view(rows, max_nodes),
        retrive_next_sibling=next_sibling.view(rows, max_nodes),
        uniform_samples=torch.rand(rows, max_nodes, device="cuda"),
        uniform_samples_for_final_sampling=torch.rand(rows, device="cuda"),
        target_probs=target.cuda().expand(rows, max_nodes, vocab).contiguous(),
        node_draft_tokens=node_tokens,
        node_draft_probs=node_probs,
    )
    first = predicts.view(rows, max_nodes)[:, 0].long().cpu()
    freq = torch.bincount(first, minlength=vocab).double() / rows
    assert (freq - target.double()).abs().max().item() < 0.008


def test_accept_prior_grows_depth_before_siblings():
    # With a likely first draw and an unlikely second one, the tree is the full-depth
    # chain before any node takes a second child.
    rows, num_depths, k = 64, 4, 8
    log_q = torch.log_softmax(torch.zeros(rows, num_depths, k, k), dim=-1)
    candidate_ids = (
        torch.arange(num_depths * k).view(1, num_depths, k).repeat(rows, 1, 1)
    )
    proposal, _, _ = build_dflash_sampled_tree_proposal(
        candidate_ids.cuda(),
        log_q.cuda(),
        max_nodes=num_depths + 3,
        accept_prior=(0.9, 0.05),
    )
    parents = (proposal.top_scores_index.cpu() // k).tolist()
    for row in parents:
        assert row[:num_depths] == list(range(num_depths))
        assert row[num_depths:] == [0, 0]


def test_online_draw_acceptance_counts():
    # Root draws node 1 then 3; node 1 draws node 2 then 4. Paths: root's second draw
    # accepted; node 1 then its second draw; nothing accepted at the root.
    q, k = 5, 4
    edges = torch.tensor([[0, 4, 1, 5]] * 3)
    accept_index = torch.tensor([[0, 3, -1], [q, q + 1, q + 4], [2 * q, -1, -1]])
    stats, log_priors = torch.zeros(4), torch.zeros(4)
    update_draw_acceptance(
        stats=stats,
        log_priors=log_priors,
        top_scores_index=edges,
        accept_index=accept_index,
        accept_lens=torch.tensor([2, 3, 1]),
        num_nodes=q,
        topk=k,
    )
    assert stats.tolist() == [1.0, 4.0, 2.0, 3.0]
    assert torch.allclose(log_priors.exp()[[0, 2]], torch.tensor([2 / 6, 3 / 5]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
