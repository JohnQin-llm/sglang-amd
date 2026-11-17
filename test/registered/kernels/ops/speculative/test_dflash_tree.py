"""The DFLASH tree builder must pick the best-first prefixes under predecessor-conditional scores."""

import heapq
import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dflash_tree import (
    NUM_CALIBRATION_BINS,
    build_dflash_sampled_tree_proposal,
    build_dflash_tree_proposal,
    tree_depth,
    tree_prefix,
    update_draw_acceptance,
    update_tree_calibration,
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


def _expected_accept_lens(proposal, cond_log_probs, calibration, prior):
    """Reference survival: each edge's probability, scaled by its (parent depth,
    probability bin) factor and capped at 1, multiplied down the path."""
    bs, num_edges = proposal.top_scores_index.shape
    k = cond_log_probs.shape[-1]
    out = []
    for b in range(bs):
        depth, rank, log_surv, total, row = [0], [0], [0.0], 1.0, [1.0]
        for i in range(num_edges):
            edge = proposal.top_scores_index[b, i].item()
            parent, child_rank = divmod(edge, k)
            lp = cond_log_probs[b, depth[parent], rank[parent], child_rank].item()
            if calibration is not None:
                bin_ = min(int(4 * (-lp) ** 0.5), NUM_CALIBRATION_BINS - 1)
                slot = depth[parent] * NUM_CALIBRATION_BINS + bin_
                hits, _, raw = calibration[:, slot].tolist()
                lp = min(
                    lp + torch.tensor((hits + prior) / (raw + prior)).log().item(), 0.0
                )
            depth.append(depth[parent] + 1)
            rank.append(child_rank)
            log_surv.append(log_surv[parent] + lp)
            total += torch.tensor(log_surv[-1]).exp().item()
            row.append(total)
        out.append(row)
    return torch.tensor(out)


@pytest.mark.parametrize("calibrated", [False, True])
def test_prefix_accept_len_matches_reference(calibrated):
    gen = torch.Generator().manual_seed(11)
    bs, num_depths, k, max_nodes = 4, 7, 8, 24
    candidate_ids = torch.randint(0, 50000, (bs, num_depths, k), generator=gen)
    scores = torch.randn(bs, num_depths, k, k, generator=gen) * 2
    cond_log_probs = torch.log_softmax(scores, dim=-1)
    calibration = None
    if calibrated:
        raw = torch.rand(num_depths * NUM_CALIBRATION_BINS, generator=gen) * 20
        hits = raw * (0.3 + 1.2 * torch.rand(raw.shape, generator=gen))
        calibration = torch.stack([hits, raw * 2, raw])
    proposal = build_dflash_tree_proposal(
        candidate_ids.cuda(),
        cond_log_probs.cuda(),
        max_nodes=max_nodes,
        calibration=None if calibration is None else calibration.cuda(),
        calibration_prior=2.0,
    )
    expected = _expected_accept_lens(proposal, cond_log_probs, calibration, prior=2.0)
    torch.testing.assert_close(
        proposal.survival.prefix_accept_len.cpu(), expected, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("width", [8, 16, 32])
def test_tree_prefix_is_the_narrower_tree(width):
    # The adaptive width verifies the first n nodes of the Q_max build; best-first order
    # makes that the n-node build exactly, which the EAGLE tree kernel must accept, also
    # when n nodes cannot reach the lattice depth.
    gen = torch.Generator().manual_seed(5)
    bs, num_depths, k = 3, 15, 16
    candidate_ids = torch.randint(0, 50000, (bs, num_depths, k), generator=gen).cuda()
    cond_log_probs = torch.log_softmax(
        torch.randn(bs, num_depths, k, k, generator=gen) * 2, dim=-1
    ).cuda()
    wide = build_dflash_tree_proposal(candidate_ids, cond_log_probs, max_nodes=48)
    narrow = build_dflash_tree_proposal(candidate_ids, cond_log_probs, max_nodes=width)
    prefix = tree_prefix(wide, width, topk=k)
    assert torch.equal(prefix.draft_tokens, narrow.draft_tokens)
    assert torch.equal(prefix.top_scores_index, narrow.top_scores_index)
    depth = tree_depth(wide, width, topk=k)
    assert depth == min(num_depths, width - 1)
    seq_lens = torch.full((bs,), 20, dtype=torch.long, device="cuda")
    bonus = torch.zeros(bs, dtype=torch.long, device="cuda")
    outs = [
        build_tree_kernel_efficient(
            bonus,
            p.parent_list,
            p.top_scores_index,
            p.draft_tokens,
            seq_lens,
            int(seq_lens.sum()),
            k,
            steps,
            width,
            TreeMaskMode.FULL_MASK,
        )
        for p, steps in ((prefix, depth), (narrow, num_depths))
    ]
    for got, want in zip(*outs):
        assert torch.equal(got, want)


def test_calibration_update_counts_reached_and_accepted_nodes():
    # Root -> node 1 (rank 0) -> node 2 (rank 0); root -> node 3 (rank 1); node 1 -> node
    # 4 (rank 1). The path root, 1, 4 reaches nodes 1, 3 (root's children) and 2, 4
    # (node 1's children) and accepts 1 and 4; a 4-node prefix never sees node 4.
    k, max_nodes = 4, 5
    num_depths = 3
    candidate_ids = torch.arange(num_depths * k).view(1, num_depths, k).cuda()
    log_p = torch.full((1, num_depths, k, k), -10.0)
    log_p[0, 0, 0, 0], log_p[0, 0, 0, 1] = -0.1, -1.0
    log_p[0, 1, 0, 0], log_p[0, 1, 0, 1] = -0.2, -1.5
    proposal = build_dflash_tree_proposal(
        candidate_ids, log_p.cuda(), max_nodes=max_nodes
    )
    assert proposal.top_scores_index[0].tolist() == [0, 4, 1, 5]
    num_slots = num_depths * NUM_CALIBRATION_BINS
    for width, path, length, want_hits, want_trials in (
        (5, [0, 1, 4], 3, 2.0, 4.0),
        (4, [0, 1], 2, 1.0, 3.0),
    ):
        stats = torch.ones(3, num_slots, device="cuda")
        update_tree_calibration(
            stats=stats,
            survival=proposal.survival,
            top_scores_index=proposal.top_scores_index,
            accept_index=torch.tensor([path + [-1] * (4 - len(path))]).cuda(),
            accept_lens=torch.tensor([length]).cuda(),
            num_nodes=width,
            topk=k,
            decay=0.5,
        )
        decayed = 0.5 * num_slots
        assert stats[0].sum().item() == pytest.approx(decayed + want_hits)
        assert stats[1].sum().item() == pytest.approx(decayed + want_trials)
        probs = proposal.survival.node_edge_prob[0, 1:width].sum().item()
        assert stats[2].sum().item() == pytest.approx(decayed + probs, rel=1e-5)
        # Depth-1 nodes (2 and 4) land in parent-depth row 1.
        row1 = stats[1, NUM_CALIBRATION_BINS : 2 * NUM_CALIBRATION_BINS].sum().item()
        assert row1 == pytest.approx(0.5 * NUM_CALIBRATION_BINS + width - 3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
