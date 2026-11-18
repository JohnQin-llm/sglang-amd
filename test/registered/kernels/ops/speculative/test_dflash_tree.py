"""DFLASH tree builders: best-first prefixes under predecessor-conditional scores, sampled
trees that verify losslessly, and batches mixing greedy and sampling rows."""

import collections
import heapq
import math
import os
import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dflash_tree import (
    NUM_CALIBRATION_BINS,
    NUM_DRAW_CLASSES,
    build_dflash_greedy_tree_proposal,
    build_dflash_sampled_tree,
    build_dflash_sampled_tree_proposal,
    build_dflash_tree_proposal,
    log_draw_priors,
    tree_depth,
    tree_prefix,
    update_draw_acceptance,
    update_rank_table,
    sampled_node_drafts,
    update_tree_calibration,
)
from sglang.kernels.ops.speculative.multi_draft_sampling import (
    tree_speculative_sampling_multi_draft,
)
from sglang.kernels.ops.speculative.traversal_verify import (
    tree_speculative_sampling_traversal,
    tree_speculative_sampling_traversal_reference,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=40, stage="jit-kernel-unit", runner_config="amd")

# The mixed-batch cases also run on CPU: Triton kernels then need TRITON_INTERPRET=1.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
needs_triton = pytest.mark.skipif(
    DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
    reason="Triton kernels need a GPU or TRITON_INTERPRET=1",
)


def _reference(candidate_ids, cond_log_probs, max_nodes):
    """Best-first expansion where a child's score depends on its parent's rank; returns
    the edges, the draft tokens and each node's (depth, rank, mass)."""
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
    return edges, tokens, nodes


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
        edges, tokens, _ = _reference(candidate_ids[b], cond_log_probs[b], max_nodes)
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


@pytest.mark.parametrize(
    "accept_prior, verify_nodes", [(None, 6), ((0.6, 0.3), 6), ((0.6, 0.3), 4)]
)
def test_sampled_tree_verification_commits_the_target(accept_prior, verify_nodes):
    """The tree builder once let a drawn child's own q decide whether it entered the tree,
    so the children kept were biased toward high q and multi-draft verification committed
    tokens off the target distribution (max error 0.033 here, about 20 sigma). A prefix
    of the tree (the adaptive width) keeps each node's first draws and stays lossless."""
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
    proposal = tree_prefix(proposal, verify_nodes, topk=k)
    depth = tree_depth(proposal, verify_nodes, topk=k)
    node_tokens = node_tokens[:, :verify_nodes].contiguous()
    node_probs = node_probs[:, :verify_nodes].contiguous()
    max_nodes = verify_nodes
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
            depth,
            max_nodes,
            TreeMaskMode.FULL_MASK,
        )
    )
    predicts = torch.full((rows * max_nodes,), -1, dtype=torch.int32, device="cuda")
    tree_speculative_sampling_multi_draft(
        predicts=predicts,
        accept_index=torch.full(
            (rows, depth + 1), -1, dtype=torch.int32, device="cuda"
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
                octave = math.floor(math.log2(max(-lp, 1e-12))) + 11
                bin_ = min(max(octave, 0), NUM_CALIBRATION_BINS - 1)
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
    # A server whose own width is below the lattice depth verifies its full build.
    whole = tree_prefix(narrow, width, topk=k)
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
        for p, steps in ((prefix, depth), (whole, depth), (narrow, num_depths))
    ]
    for got, whole_got, want in zip(*outs):
        assert torch.equal(got, want)
        assert torch.equal(whole_got, want)


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


def test_draw_acceptance_counts_only_sampled_rows():
    # The rows above with the middle one greedy: its best-first edges are not draws.
    q, k = 5, 4
    edges = torch.tensor([[0, 4, 1, 5]] * 3)
    local = torch.tensor([[0, 3, -1], [0, 1, 4], [0, -1, -1]])
    accept_lens = torch.tensor([2, 3, 1])

    def update(rows, sampled_rows=None):
        offsets = torch.arange(len(rows))[:, None] * q
        accept_index = torch.where(local[rows] >= 0, local[rows] + offsets, -1)
        stats, log_priors = torch.zeros(4), torch.zeros(4)
        update_draw_acceptance(
            stats=stats,
            log_priors=log_priors,
            top_scores_index=edges[rows],
            accept_index=accept_index,
            accept_lens=accept_lens[rows],
            num_nodes=q,
            topk=k,
            sampled_rows=sampled_rows,
        )
        return stats

    assert update([0, 2]).tolist() == [0.0, 2.0, 1.0, 2.0]
    assert torch.equal(
        update([0, 1, 2], torch.tensor([True, False, True])), update([0, 2])
    )


def _lattice(rows, num_depths, k, vocab, seed):
    """Selector lattice; every third row is so peaked that much of its q underflows."""
    gen = torch.Generator().manual_seed(seed)
    candidate_ids = torch.stack(
        [torch.randperm(vocab, generator=gen)[: num_depths * k] for _ in range(rows)]
    ).view(rows, num_depths, k)
    scores = torch.randn(rows, num_depths, k, k, generator=gen) * 2
    scores[::3] *= 60
    scores[:, 0] = scores[:, 0, :1]
    return candidate_ids, torch.log_softmax(scores, dim=-1)


def _tree_links(edges, k, max_nodes):
    """EAGLE's retrieve links of one row, siblings in node order as build_tree_kernel."""
    next_token, next_sibling = [-1] * max_nodes, [-1] * max_nodes
    for node in range(max_nodes - 1, 0, -1):
        parent = edges[node - 1] // k
        next_sibling[node] = next_token[parent]
        next_token[parent] = node
    return next_token, next_sibling


def _one_hot_argmax(next_token, next_sibling, candidates, vocab, gen):
    """Per node, the argmax of a one-hot target: mostly one of its children's tokens."""
    argmax = []
    for child in next_token:
        children = []
        while child != -1:
            children.append(candidates[child])
            child = next_sibling[child]
        pick = torch.randint(0, 5 * max(len(children), 1), (), generator=gen).item()
        if pick < 4 * len(children):
            argmax.append(children[pick % len(children)])
        else:
            argmax.append(torch.randint(0, vocab, (), generator=gen).item())
    return argmax


def _greedy_walk(next_token, next_sibling, candidates, argmax):
    """Greedy tree verification: the path of argmax children and the bonus token."""
    path = [0]
    while True:
        child = next_token[path[-1]]
        while child != -1 and candidates[child] != argmax[path[-1]]:
            child = next_sibling[child]
        if child == -1:
            return path, argmax[path[-1]]
        path.append(child)


def _rrs_reference(tree, target, coins, final_coin, num_spec):
    """Host recursive rejection sampling of one row in float64, as the RRS kernel."""
    candidates, next_token, next_sibling, node_tokens, node_probs = tree
    path = [0]
    while True:
        node = path[-1]
        p = target[node].double().clone()
        tokens, q = node_tokens[node], node_probs[node].double()
        drawn = torch.zeros_like(q, dtype=torch.bool)
        child = next_token[node] if len(path) < num_spec else -1
        while child != -1:
            q_k = torch.where(drawn, 0.0, q) / max(1 - q[drawn].sum().item(), 1e-20)
            q_full = torch.zeros_like(p).index_add_(0, tokens, q_k)
            token = candidates[child]
            if q_full[token] > 0 and coins[child] * q_full[token] < p[token]:
                path.append(child)
                break
            p = (p - q_full).clamp(min=0)
            p = p / p.sum()
            drawn |= tokens == token
            child = next_sibling[child]
        if path[-1] == node:
            break
    hit = ((p.cumsum(0) > final_coin * p.sum()) & (p > 0)).nonzero()
    return path, (hit[0] if len(hit) else (p > 0).nonzero()[-1]).item()


def test_one_hot_target_verifies_best_first_tree_like_greedy():
    """A greedy row's target is one-hot, so RRS and Traversal Verification of its
    best-first tree, drafted with a uniform q, accept greedy verification's path and
    bonus whatever the coins, also on peaked rows that put deep siblings in the tree."""
    rows, num_depths, k, max_nodes, vocab = 60, 7, 16, 24, 300
    candidate_ids, cond_log_probs = _lattice(rows, num_depths, k, vocab, seed=3)
    gen = torch.Generator().manual_seed(4)
    for b in range(rows):
        edges, tokens, nodes = _reference(
            candidate_ids[b], cond_log_probs[b], max_nodes
        )
        next_token, next_sibling = _tree_links(edges, k, max_nodes)
        candidates = [0] + tokens
        depths = torch.tensor([depth for depth, _, _ in nodes])
        node_tokens = candidate_ids[b, depths.clamp(max=num_depths - 1)]
        node_probs = torch.full((max_nodes, k), 1 / k) * (depths < num_depths)[:, None]
        argmax = _one_hot_argmax(next_token, next_sibling, candidates, vocab, gen)
        target = torch.nn.functional.one_hot(torch.tensor(argmax), vocab).float()
        coins = torch.rand(max_nodes, generator=gen)
        final_coin = torch.rand((), generator=gen).item()
        expected = _greedy_walk(next_token, next_sibling, candidates, argmax)

        tree = (candidates, next_token, next_sibling, node_tokens, node_probs)
        assert _rrs_reference(tree, target, coins, final_coin, num_depths + 1) == (
            expected
        ), f"row {b}"
        predicts = torch.full((max_nodes,), -1, dtype=torch.int32)
        accept_index = torch.full((1, num_depths + 1), -1, dtype=torch.int32)
        accept_num = torch.zeros(1, dtype=torch.int32)
        tree_speculative_sampling_traversal_reference(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_num,
            candidates=torch.tensor([candidates]),
            retrive_index=torch.arange(max_nodes)[None],
            retrive_next_token=torch.tensor([next_token]),
            retrive_next_sibling=torch.tensor([next_sibling]),
            uniform_samples=coins[None],
            uniform_samples_for_final_sampling=torch.tensor([final_coin]),
            target_probs=target[None],
            node_draft_tokens=node_tokens[None],
            node_draft_probs=node_probs[None],
        )
        path = accept_index[0, : accept_num.item() + 1].tolist()
        assert (path, predicts[path[-1]].item()) == expected, f"row {b}"


def _device_tree(proposal, node_tokens, node_probs, k):
    rows, num_edges = proposal.top_scores_index.shape
    max_nodes = num_edges + 1
    links = [
        _tree_links(edges, k, max_nodes)
        for edges in proposal.top_scores_index.cpu().tolist()
    ]
    candidates = torch.cat(
        [torch.zeros(rows, 1, dtype=torch.long), proposal.draft_tokens.cpu()], dim=1
    )
    return dict(
        candidates=candidates.to(DEVICE),
        retrive_index=torch.arange(rows * max_nodes).view(rows, max_nodes).to(DEVICE),
        retrive_next_token=torch.tensor([n for n, _ in links]).to(DEVICE),
        retrive_next_sibling=torch.tensor([s for _, s in links]).to(DEVICE),
        node_draft_tokens=node_tokens,
        node_draft_probs=node_probs,
    )


def _run_verify(verify_fn, tree, target_probs, coins, final_coins, num_spec):
    rows, max_nodes = tree["candidates"].shape
    predicts = torch.full((rows * max_nodes,), -1, dtype=torch.int32, device=DEVICE)
    accept_index = torch.full((rows, num_spec), -1, dtype=torch.int32, device=DEVICE)
    accept_num = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
    verify_fn(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_num,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=final_coins,
        target_probs=target_probs,
        **tree,
    )
    return predicts.cpu(), accept_index.cpu(), accept_num.cpu()


VERIFY_FNS = [
    tree_speculative_sampling_multi_draft,
    tree_speculative_sampling_traversal,
]


@needs_triton
@pytest.mark.parametrize("verify_fn", VERIFY_FNS)
def test_greedy_rows_verify_like_greedy(verify_fn):
    """The kernels on the builder's best-first tree and node drafts, one-hot targets."""
    rows, num_depths, k, max_nodes, vocab = 48, 7, 16, 24, 4099
    candidate_ids, cond_log_probs = _lattice(rows, num_depths, k, vocab, seed=8)
    tree = _device_tree(
        *build_dflash_greedy_tree_proposal(
            candidate_ids.to(DEVICE), cond_log_probs.to(DEVICE), max_nodes=max_nodes
        ),
        k,
    )
    host = {name: value.cpu().tolist() for name, value in tree.items()}
    gen = torch.Generator().manual_seed(9)
    argmax = [
        _one_hot_argmax(
            host["retrive_next_token"][b],
            host["retrive_next_sibling"][b],
            host["candidates"][b],
            vocab,
            gen,
        )
        for b in range(rows)
    ]
    predicts, accept_index, accept_num = _run_verify(
        verify_fn,
        tree,
        torch.nn.functional.one_hot(torch.tensor(argmax), vocab).float().to(DEVICE),
        torch.rand(rows, max_nodes, generator=gen).to(DEVICE),
        torch.rand(rows, generator=gen).to(DEVICE),
        num_depths + 1,
    )
    for b in range(rows):
        path, bonus = _greedy_walk(
            host["retrive_next_token"][b],
            host["retrive_next_sibling"][b],
            host["candidates"][b],
            argmax[b],
        )
        slots = [b * max_nodes + node for node in path]
        assert accept_num[b].item() == len(path) - 1, f"row {b}"
        assert accept_index[b, : len(path)].tolist() == slots, f"row {b}"
        committed = [host["candidates"][b][node] for node in path[1:]] + [bonus]
        assert predicts[slots].tolist() == committed, f"row {b}"
    assert len(set(accept_num.tolist())) >= 3


@needs_triton
@pytest.mark.parametrize("verify_fn", VERIFY_FNS)
def test_mixed_batch_keeps_sampled_rows(verify_fn):
    """Greedy rows take the best-first tree; sampled rows keep the tree, node drafts and
    verdicts of the all-sampling batch drawn from the same random state."""
    from sglang.srt.speculative.dflash_tree_verify import (
        propose_dflash_mixed_tree,
        propose_dflash_sampled_tree,
        propose_dflash_tree,
    )

    rows, num_depths, k, max_nodes, vocab = 24, 7, 16, 16, 4099
    candidate_ids, scores = (
        x.to(DEVICE) for x in _lattice(rows, num_depths, k, vocab, seed=12)
    )
    greedy_rows = (torch.arange(rows) % 3 == 1).to(DEVICE)
    kwargs = dict(
        candidate_ids=candidate_ids,
        scores=scores,
        temperatures=torch.where(greedy_rows, 1.0, 0.7),
        topk=k,
        num_nodes=max_nodes,
    )
    torch.manual_seed(13)
    mixed = propose_dflash_mixed_tree(greedy_rows=greedy_rows, workspace={}, **kwargs)
    torch.manual_seed(13)
    sampled = propose_dflash_sampled_tree(workspace={}, **kwargs)
    best_first = propose_dflash_tree(
        candidate_ids=candidate_ids,
        scores=scores,
        topk=k,
        num_nodes=max_nodes,
        workspace={},
    )
    keep = ~greedy_rows
    for name in ("draft_tokens", "parent_list", "top_scores_index"):
        # Only the first Q - 1 parent-list entries are written and read.
        mixed_value, sampled_value, best_first_value = (
            getattr(tree, name)[:, : max_nodes - 1]
            for tree in (mixed[0], sampled[0], best_first)
        )
        assert torch.equal(mixed_value[keep], sampled_value[keep])
        assert torch.equal(mixed_value[greedy_rows], best_first_value[greedy_rows])
    for mixed_value, sampled_value in zip(mixed[1:], sampled[1:]):
        assert torch.equal(mixed_value[keep], sampled_value[keep])

    gen = torch.Generator().manual_seed(14)
    logits = torch.randn(rows, max_nodes, vocab, generator=gen).to(DEVICE)
    logits.scatter_add_(2, sampled[1], sampled[2].clamp(min=1e-30).log() * 0.7 + 9)
    target = torch.softmax(logits, dim=-1)
    target[greedy_rows] = torch.nn.functional.one_hot(
        target[greedy_rows].argmax(-1), vocab
    ).float()
    coins = torch.rand(rows, max_nodes, generator=gen).to(DEVICE)
    final_coins = torch.rand(rows, generator=gen).to(DEVICE)
    (mixed_predicts, mixed_index, mixed_num), (all_predicts, all_index, all_num) = (
        _run_verify(
            verify_fn,
            _device_tree(*tree, k),
            target,
            coins,
            final_coins,
            num_depths + 1,
        )
        for tree in (mixed, sampled)
    )
    keep = keep.cpu()
    assert torch.equal(mixed_num[keep], all_num[keep])
    assert torch.equal(mixed_index[keep], all_index[keep])
    used = all_index[keep][all_index[keep] >= 0].long()
    assert torch.equal(mixed_predicts[used], all_predicts[used])
    assert all_num[keep].max().item() >= 2


def test_chain_is_every_nodes_first_draw():
    rows, num_depths, k = 64, 7, 8
    gen = torch.Generator().manual_seed(4)
    log_q = torch.log_softmax(torch.randn(rows, num_depths, k, k, generator=gen), -1)
    candidate_ids = torch.randint(0, 50000, (rows, num_depths, k), generator=gen)
    tree = build_dflash_sampled_tree(
        candidate_ids.cuda(), log_q.cuda(), max_nodes=num_depths + 1, chain=True
    )
    nodes = torch.arange(1, num_depths + 1)
    assert torch.equal(
        tree.proposal.top_scores_index.cpu(), ((nodes - 1) * k).expand(rows, -1)
    )
    assert torch.equal(tree.node_depths[:, 1:].cpu(), nodes.int().expand(rows, -1))
    ranks = tree.node_ranks[:, 1:].long().cpu()
    tokens = candidate_ids.gather(2, ranks.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(tree.proposal.draft_tokens.cpu(), tokens)
    with pytest.raises(ValueError):
        build_dflash_sampled_tree(
            candidate_ids.cuda(), log_q.cuda(), max_nodes=num_depths + 2, chain=True
        )


@pytest.mark.parametrize("deep_first", [True, False])
def test_rank_table_orders_slots_by_depth(deep_first):
    # A table that makes deep first draws likely grows the chain before any sibling; one
    # that makes the root's second draw likelier than any depth-1 child fills the root
    # first.
    rows, num_depths, k = 32, 4, 8
    log_q = torch.log_softmax(torch.zeros(rows, num_depths, k, k), dim=-1).cuda()
    candidate_ids = (
        torch.arange(num_depths * k).view(1, num_depths, k).repeat(rows, 1, 1).cuda()
    )
    rates = torch.full((num_depths, NUM_DRAW_CLASSES), 1e-3)
    if deep_first:
        rates[:, 0] = 0.9
    else:
        rates[0] = torch.tensor([0.5, 0.4, 0.3, 0.2])
        rates[1:, 0] = 0.01
    tree = build_dflash_sampled_tree(
        candidate_ids,
        log_q,
        max_nodes=num_depths + 1,
        accept_prior=(0.7, 0.3),
        rank_table=rates.log().view(-1).cuda(),
    )
    parents = (tree.proposal.top_scores_index.cpu() // k).tolist()
    want = list(range(num_depths)) if deep_first else [0] * num_depths
    assert all(row == want for row in parents)


def test_rank_table_is_the_slot_rate_by_depth_and_draw():
    num_depths = 3
    bins = NUM_CALIBRATION_BINS
    stats = torch.zeros(3, num_depths * NUM_DRAW_CLASSES * bins)
    view = stats[:2].view(2, num_depths, NUM_DRAW_CLASSES, bins)
    view[1, 1, 0, 3] = 6.0  # depth 1, first draws: 6 reached over two bins, 5 hit
    view[1, 1, 0, 7] = 2.0
    view[0, 1, 0, 3] = 5.0
    log_priors = torch.tensor(log_draw_priors((0.7, 0.3)))
    table = torch.zeros(num_depths * NUM_DRAW_CLASSES)
    update_rank_table(rank_table=table, stats=stats, log_priors=log_priors, prior=2.0)
    rates = table.exp().view(num_depths, NUM_DRAW_CLASSES)
    assert rates[1, 0].item() == pytest.approx((5 + 2 * 0.7) / (8 + 2))
    # Without data each class keeps its draw prior: a, then (1 - a) (1 - a')^(c - 1) a'.
    assert rates[0, 0].item() == pytest.approx(0.7)
    assert rates[2, 2].item() == pytest.approx(0.3 * 0.7 * 0.3)


def _sampled_survival_reference(tree, log_q, stats, log_priors, prior):
    """Calibrated survival of a sampled build: per node, its slot's rate shrunk toward
    the draw prior, multiplied down."""
    edges = tree.proposal.top_scores_index.cpu()
    depths, ranks = tree.node_depths.cpu(), tree.node_ranks.cpu()
    bs, num_edges = edges.shape
    k = log_q.shape[-1]
    first, miss_first, sibling, miss_sibling = log_priors
    slots, out = [], []
    for b in range(bs):
        log_surv, total, row, slot_row = [0.0], 1.0, [1.0], [-1]
        for i in range(num_edges):
            parent, draw = divmod(edges[b, i].item(), k)
            d, r = depths[b, parent].item(), ranks[b, parent].item()
            lq = log_q[b, d, r, ranks[b, i + 1]].item()
            octave = math.floor(math.log2(max(-lq, 1e-12))) + 11
            slot = (d * NUM_DRAW_CLASSES + min(draw, NUM_DRAW_CLASSES - 1)) * (
                NUM_CALIBRATION_BINS
            ) + min(max(octave, 0), NUM_CALIBRATION_BINS - 1)
            r0 = math.exp(
                first if draw == 0 else miss_first + (draw - 1) * miss_sibling + sibling
            )
            hits, reached = stats[0, slot].item(), stats[1, slot].item()
            rate = min((hits + prior * r0) / (reached + prior), 1.0)
            log_surv.append(log_surv[parent] + math.log(rate))
            total += math.exp(log_surv[-1])
            row.append(total)
            slot_row.append(slot)
        out.append(row)
        slots.append(slot_row)
    return torch.tensor(out), torch.tensor(slots, dtype=torch.int32)


@pytest.mark.parametrize("chain", [False, True])
def test_calibrated_sampled_survival_matches_reference(chain):
    gen = torch.Generator().manual_seed(13)
    bs, num_depths, k = 4, 7, 8
    max_nodes = num_depths + 1 if chain else 24
    candidate_ids = torch.randint(0, 50000, (bs, num_depths, k), generator=gen)
    log_q = torch.log_softmax(torch.randn(bs, num_depths, k, k, generator=gen) * 2, -1)
    num_slots = num_depths * NUM_DRAW_CLASSES * NUM_CALIBRATION_BINS

    def random_stats():
        reached = torch.rand(num_slots, generator=gen) * 20
        hits = reached * torch.rand(num_slots, generator=gen)
        return torch.stack([hits, reached, reached * 0.5])

    stats = random_stats()
    accept_prior = (0.7, 0.3)
    tree = build_dflash_sampled_tree(
        candidate_ids.cuda(),
        log_q.cuda(),
        max_nodes=max_nodes,
        accept_prior=accept_prior,
        chain=chain,
        calibration=stats.cuda(),
        calibration_prior=2.0,
    )
    expected, slots = _sampled_survival_reference(
        tree, log_q, stats, log_draw_priors(accept_prior), prior=2.0
    )
    survival = tree.proposal.survival
    assert torch.equal(survival.node_cal_slot.cpu(), slots)
    torch.testing.assert_close(
        survival.prefix_accept_len.cpu(), expected, rtol=1e-4, atol=1e-4
    )


def _path_log_q(log_q, path, k):
    """Sum of q's log-probs along ``path`` (tokens from the root); candidate ids are
    arange, so token t sits at lattice depth t // k, rank t % k."""
    return sum(
        log_q[d, 0 if d == 0 else path[d - 1] % k, path[d] % k].item()
        for d in range(len(path))
    )


def _node_q(log_q, path, k, log_tau):
    """float64 q over the vocabulary of the children of the node at ``path``, truncated
    to the candidates whose path log q clears ``log_tau`` (if any) and renormalized."""
    num_depths = log_q.shape[0]
    depth = len(path)
    q = torch.zeros(num_depths * k, dtype=torch.float64)
    if depth == num_depths:
        return q
    row = log_q[depth, 0 if depth == 0 else path[-1] % k]
    keep = torch.ones(k, dtype=torch.bool)
    if log_tau is not None:
        keep = _path_log_q(log_q, path, k) + row >= log_tau
    q[depth * k : (depth + 1) * k] = torch.where(keep, row.exp(), 0.0)
    return q / q.sum() if q.sum() > 0 else q


def _traversal_exact(children, tokens, node_q, target):
    """Traversal Verification's output distribution on one tree, in float64: committed
    sequence -> probability. Node n's children were drawn from ``node_q[n]``."""
    out = collections.defaultdict(float)
    undecided = [1.0]

    def visit(node, p_node, path):
        r = p_node * target(path)
        q = node_q[node].clone()
        for child in children[node]:
            t = tokens[child]
            p_child = min(1.0, (r[t] / q[t]).item()) if q[t] > 0 else 0.0
            visit(child, p_child, path + (t,))
            norm = max(1.0 - torch.minimum(r, q).sum().item(), 1e-300)
            r = (r - q).clamp(min=0) / norm
            q[t] = 0.0
            q = q / q.sum() if q.sum() > 0 else q
        accept = 1.0 if node == 0 else r.sum().item()
        if accept > 0:
            for y in torch.nonzero(r > 0).flatten().tolist():
                out[path + (y,)] += undecided[0] * accept * (r[y] / r.sum()).item()
        undecided[0] *= 1.0 - accept

    visit(0, 1.0, ())
    return out


def _extend(dist, target, length):
    """Each sequence continued by target draws to ``length`` tokens."""
    full = collections.defaultdict(float)
    stack = list(dist.items())
    while stack:
        seq, prob = stack.pop()
        if len(seq) == length:
            full[seq] += prob
            continue
        nxt = target(seq)
        stack.extend((seq + (y,), prob * nxt[y].item()) for y in range(len(nxt)))
    return full


@pytest.mark.parametrize(
    "seed, log_tau, padded", [(7, -3.0, False), (9, -5.0, False), (5, -2.0, True)]
)
def test_truncated_sampled_tree_is_exact(seed, log_tau, padded):
    """Every node draws its children from q truncated to the candidates whose path log q
    clears tau (q'); Traversal Verification with q' commits exactly the target. The
    builder's trees over a million rows are every tree there is (their q' draw
    probabilities sum to one), each appears at that probability, padding comes only
    once no node has support left, and verifying them with the untruncated q fails.
    Padding depends on the lattice alone: whether fewer paths clear tau than slots."""
    num_depths, k, max_nodes, rows = 2, 4, 5, 1 << 20
    vocab = num_depths * k
    gen = torch.Generator().manual_seed(seed)
    scores = torch.randn(num_depths, k, k, generator=gen)
    scores[0] = scores[0, :1]
    log_q = torch.log_softmax(scores, dim=-1)
    log_q64 = log_q.double()
    # No path within rounding of tau, so float32 (kernel) and float64 supports agree.
    margins = [
        abs(_path_log_q(log_q64, path, k) - log_tau)
        for path in [(a,) for a in range(k)]
        + [(a, k + b) for a in range(k) for b in range(k)]
    ]
    assert min(margins) > 1e-3
    favored = torch.zeros(vocab + 1, vocab)
    for prev in range(vocab + 1):
        depth = 0 if prev == vocab else min(prev // k + 1, num_depths - 1)
        favored[prev, depth * k : (depth + 1) * k] = 2.0
    next_target = torch.softmax(
        torch.randn(vocab + 1, vocab, generator=gen).double() + favored, -1
    )

    def target(path):
        return next_target[path[-1] if path else vocab]

    torch.manual_seed(6)
    candidate_ids = torch.arange(vocab).view(1, num_depths, k).repeat(rows, 1, 1)
    lattice = (candidate_ids.cuda(), log_q.repeat(rows, 1, 1, 1).cuda())
    tree = build_dflash_sampled_tree(
        *lattice, max_nodes=max_nodes, accept_prior=(0.7, 0.3), log_tau=log_tau
    )
    node_probs = sampled_node_drafts(*lattice, tree, max_nodes)[1][:256].cpu()
    keys = torch.cat(
        [
            tree.proposal.top_scores_index.cpu() // k,
            tree.proposal.draft_tokens.cpu(),
            tree.proposal.node_count.cpu()[:, None].long(),
        ],
        dim=1,
    )
    trees, counts = torch.unique(keys, dim=0, return_counts=True)
    first_row = {
        tuple(key): row for row, key in reversed(list(enumerate(keys[:256].tolist())))
    }

    output = collections.defaultdict(float)
    control = collections.defaultdict(float)
    total, z_max, padded_rows = 0.0, 0.0, 0
    for key, count in zip(trees.tolist(), counts.tolist()):
        parents = key[: max_nodes - 1]
        tokens = [0] + key[max_nodes - 1 : 2 * (max_nodes - 1)]
        num_real = key[-1]
        children = {node: [] for node in range(max_nodes)}
        paths, node_q, prob = [()], [_node_q(log_q64, (), k, log_tau)], 1.0
        drawn_mass = collections.defaultdict(float)
        for node in range(1, max_nodes):
            parent, token = parents[node - 1], tokens[node]
            children[parent].append(node)
            paths.append(paths[parent] + (token,))
            if node < num_real:
                q = node_q[parent]
                prob *= q[token].item() / (1.0 - drawn_mass[parent])
                drawn_mass[parent] += q[token].item()
                node_q.append(_node_q(log_q64, paths[node], k, log_tau))
            else:
                node_q.append(torch.zeros(vocab, dtype=torch.float64))
        if num_real < max_nodes:
            padded_rows += count
            for node in range(num_real):
                left = node_q[node].clone()
                left[[tokens[c] for c in children[node] if c < num_real]] = 0.0
                assert left.sum().item() == 0.0, key
        if tuple(key) in first_row:
            row = first_row[tuple(key)]
            kernel_q = node_probs[row].double()
            kernel_q = kernel_q / kernel_q.sum(-1, keepdim=True).clamp(min=1e-30)
            for node in range(max_nodes):
                depth = len(paths[node])
                if depth < num_depths:
                    expected = node_q[node][depth * k : (depth + 1) * k]
                    assert torch.allclose(kernel_q[node], expected, atol=1e-6)
        total += prob
        sigma = math.sqrt(rows * prob * (1 - prob))
        z_max = max(z_max, abs(count - rows * prob) / sigma)
        for dist, qs in (
            (output, node_q),
            (control, [_node_q(log_q64, path, k, None) for path in paths]),
        ):
            for seq, p in _extend(
                _traversal_exact(children, tokens, qs, target), target, num_depths + 1
            ).items():
                dist[seq] += prob * p
    reference = _extend({(): 1.0}, target, num_depths + 1)
    error = max(abs(output[seq] - p) for seq, p in reference.items())
    control_error = max(abs(control[seq] - p) for seq, p in reference.items())
    assert abs(total - 1.0) < 1e-9
    assert z_max < 5.5
    assert padded_rows == (rows if padded else 0)
    assert error < 1e-12
    assert control_error > 1e-3


@pytest.mark.parametrize("accept_prior", [None, (0.7, 0.3), "tensor"])
def test_truncation_at_minus_infinity_is_bitwise(accept_prior):
    """A threshold of -inf keeps every candidate: the tree and its drafts are bit for bit
    those of the untruncated builder. (Where q underflows to zero on every undrawn
    candidate, the untruncated builder still adds nodes, with no q mass, where the
    truncated one pads; ``_lattice``'s peaked rows are left out for that.)"""
    candidate_ids, log_q = _lattice(72, 15, 16, 151000, seed=31)
    peaked = torch.arange(72) % 3 == 0
    candidate_ids, log_q = candidate_ids[~peaked], log_q[~peaked]
    if accept_prior == "tensor":
        accept_prior = torch.tensor(log_draw_priors((0.8, 0.2))).cuda()
    outputs = []
    for log_tau in (None, -math.inf):
        torch.manual_seed(32)
        proposal, node_tokens, node_probs = build_dflash_sampled_tree_proposal(
            candidate_ids.cuda(),
            log_q.cuda(),
            max_nodes=32,
            accept_prior=accept_prior,
            log_tau=log_tau,
        )
        fields = [proposal.draft_tokens, proposal.top_scores_index]
        # Parent-list entries past the nodes are never written nor read.
        fields += [proposal.parent_list[:, :32], node_tokens, node_probs]
        if proposal.survival is not None:
            fields += [proposal.survival.node_log_survival]
            fields += [proposal.survival.prefix_accept_len]
        outputs.append([field.clone() for field in fields])
    for plain, truncated in zip(*outputs):
        assert torch.equal(plain, truncated)


def test_draw_acceptance_skips_padding():
    # Root draws node 1; nodes 2 and 3 are padding under the root, then nothing is
    # accepted: one first draw rejected, no later draws.
    q, k = 4, 4
    stats, log_priors = torch.zeros(4), torch.zeros(4)
    update_draw_acceptance(
        stats=stats,
        log_priors=log_priors,
        top_scores_index=torch.tensor([[0, k - 1, k - 1]]),
        accept_index=torch.tensor([[0, -1, -1]]),
        accept_lens=torch.tensor([1]),
        num_nodes=q,
        topk=k,
        node_count=torch.tensor([2], dtype=torch.int32),
    )
    assert stats.tolist() == [0.0, 1.0, 0.0, 0.0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
