"""DFLASH tree builders: best-first prefixes under predecessor-conditional scores, sampled
trees that verify losslessly, and batches mixing greedy and sampling rows."""

import heapq
import os
import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dflash_tree import (
    build_dflash_greedy_tree_proposal,
    build_dflash_sampled_tree_proposal,
    build_dflash_tree_proposal,
    update_draw_acceptance,
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
