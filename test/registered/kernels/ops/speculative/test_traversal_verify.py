"""Traversal Verification on sampled DFLASH trees must commit exactly the target."""

import math
import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dflash_tree import (
    build_dflash_sampled_tree_proposal,
)
from sglang.kernels.ops.speculative.traversal_verify import (
    tree_speculative_sampling_traversal,
    tree_speculative_sampling_traversal_reference,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


def _sampled_tree(candidate_ids, log_q, max_nodes):
    rows, num_depths, k = candidate_ids.shape
    proposal, node_tokens, node_probs = build_dflash_sampled_tree_proposal(
        candidate_ids, log_q, max_nodes=max_nodes
    )
    seq_lens = torch.full((rows,), 10, dtype=torch.long, device="cuda")
    _, _, retrive_index, next_token, next_sibling, tokens = build_tree_kernel_efficient(
        bonus_tokens=torch.zeros(rows, dtype=torch.long, device="cuda"),
        parent_list=proposal.parent_list,
        top_scores_index=proposal.top_scores_index,
        draft_tokens=proposal.draft_tokens,
        seq_lens=seq_lens,
        seq_lens_sum=int(seq_lens.sum()),
        topk=k,
        spec_steps=num_depths,
        num_verify_tokens=max_nodes,
        tree_mask_mode=TreeMaskMode.FULL_MASK,
    )
    return dict(
        candidates=tokens.view(rows, max_nodes),
        retrive_index=retrive_index.view(rows, max_nodes),
        retrive_next_token=next_token.view(rows, max_nodes),
        retrive_next_sibling=next_sibling.view(rows, max_nodes),
        node_draft_tokens=node_tokens,
        node_draft_probs=node_probs,
    )


def _verify(verify_fn, tree, *, target_probs, coins, final_coins, num_depths):
    rows, max_nodes = tree["candidates"].shape
    device = tree["candidates"].device
    predicts = torch.full((rows * max_nodes,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full(
        (rows, num_depths + 1), -1, dtype=torch.int32, device=device
    )
    accept_token_num = torch.zeros(rows, dtype=torch.int32, device=device)
    result = verify_fn(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=final_coins,
        target_probs=target_probs,
        **tree,
    )
    return predicts, accept_index, accept_token_num, result


def _committed(predicts, accept_index, accept_token_num, num_tokens):
    """First ``num_tokens`` committed tokens per row, -1 past the accepted length."""
    slots = accept_index[:, :num_tokens].long()
    tokens = predicts[slots.clamp(min=0)].long()
    in_round = (
        torch.arange(num_tokens, device=slots.device) <= accept_token_num[:, None]
    )
    return torch.where(in_round, tokens, -1)


@pytest.mark.parametrize(
    "num_depths, k, max_nodes, vocab",
    [(7, 16, 16, 4099), (7, 16, 8, 131), (3, 4, 10, 37)],
)
def test_kernel_matches_reference(num_depths, k, max_nodes, vocab):
    torch.manual_seed(num_depths * 1000 + k * 10 + max_nodes)
    rows = 96
    candidate_ids = torch.stack(
        [torch.randperm(vocab)[: num_depths * k] for _ in range(rows)]
    ).view(rows, num_depths, k)
    scores = torch.randn(rows, num_depths, k, k) * 2
    scores[:, 0] = scores[:, 0, :1]
    log_q = torch.log_softmax(scores, dim=-1)
    tree = _sampled_tree(candidate_ids.cuda(), log_q.cuda(), max_nodes)
    # A target near each node's draft, so rows accept paths of every length.
    logits = torch.randn(rows, max_nodes, vocab, device="cuda")
    boost = math.log(vocab) + 1 + tree["node_draft_probs"].log().clamp(min=-30) * 0.7
    logits.scatter_add_(2, tree["node_draft_tokens"], boost)
    target_probs = torch.softmax(logits, dim=-1)
    coins = torch.rand(rows, max_nodes, device="cuda")
    final_coins = torch.rand(rows, device="cuda")
    kwargs = dict(
        target_probs=target_probs,
        coins=coins,
        final_coins=final_coins,
        num_depths=num_depths,
    )

    predicts, accept_index, accept_num, _ = _verify(
        tree_speculative_sampling_traversal, tree, **kwargs
    )
    cpu_tree = {name: value.cpu() for name, value in tree.items()}
    cpu_kwargs = {
        name: value.cpu() if torch.is_tensor(value) else value
        for name, value in kwargs.items()
    }
    ref_predicts, ref_accept_index, ref_accept_num, margins = _verify(
        tree_speculative_sampling_traversal_reference, cpu_tree, **cpu_kwargs
    )
    # FP32 may flip a coin or a bonus draw that lands within rounding of its threshold.
    clear = margins[:, 0] > 1e-5
    assert clear.sum() >= rows - 2
    assert torch.equal(accept_num.cpu()[clear], ref_accept_num[clear])
    assert torch.equal(accept_index.cpu()[clear], ref_accept_index[clear])
    clear &= margins[:, 1] > 2e-6
    assert clear.sum() >= rows - 6
    used = ref_accept_index[clear].long()
    used = used[used >= 0]
    assert torch.equal(predicts.cpu()[used], ref_predicts[used])
    assert len(set(ref_accept_num.tolist())) >= 3


def test_sampled_tree_traversal_commits_the_target():
    """The first two committed tokens must follow the target jointly; a round that commits
    one token is extended by a target draw (Lemma A.1 of arXiv 2505.12398)."""
    rows, num_depths, k, max_nodes = 200000, 3, 4, 8
    gen = torch.Generator().manual_seed(11)
    log_q = torch.log_softmax(
        torch.randn(num_depths, k, k, generator=gen) * 2.0, dim=-1
    )
    vocab = num_depths * k
    # Bigram target favoring the next depth's candidates, so long paths get accepted;
    # row ``vocab`` is the root's.
    favored = torch.zeros(vocab + 1, vocab)
    for prev in range(vocab + 1):
        depth = 0 if prev == vocab else prev // k + 1
        favored[prev, depth * k : (depth + 1) * k] = 4.0
    probs = torch.softmax(
        torch.randn(vocab + 1, vocab, generator=gen) * 1.5 + favored, -1
    )
    next_target, root_target = probs[:vocab], probs[vocab]
    torch.manual_seed(12)
    candidate_ids = torch.arange(vocab).view(1, num_depths, k).repeat(rows, 1, 1)
    tree = _sampled_tree(
        candidate_ids.cuda(), log_q.repeat(rows, 1, 1, 1).cuda(), max_nodes
    )
    # The target after a node depends on the node's token (a bigram model).
    target_probs = next_target.cuda()[tree["candidates"]]
    target_probs[:, 0] = root_target.cuda()
    predicts, accept_index, accept_num, _ = _verify(
        tree_speculative_sampling_traversal,
        tree,
        target_probs=target_probs,
        coins=torch.rand(rows, max_nodes, device="cuda"),
        final_coins=torch.rand(rows, device="cuda"),
        num_depths=num_depths,
    )
    first, second = _committed(predicts, accept_index, accept_num, 2).unbind(-1)
    extension = torch.multinomial(next_target.cuda()[first], 1).squeeze(-1)
    second = torch.where(second >= 0, second, extension)
    assert (accept_num >= 2).float().mean() > 0.05

    joint = torch.bincount(first * vocab + second, minlength=vocab * vocab)
    freq = joint.double().cpu() / rows
    expected = (root_target[:, None] * next_target).flatten().double()
    sigma = (expected * (1 - expected) / rows).sqrt().clamp(min=1e-12)
    assert ((freq - expected).abs() / sigma).max().item() < 5
    first_freq = freq.view(vocab, vocab).sum(-1)
    first_sigma = (root_target * (1 - root_target) / rows).sqrt().double()
    assert ((first_freq - root_target.double()).abs() / first_sigma).max().item() < 5


def _block_verification(target, draft, tokens, coins, final_coins):
    """Block verification (arXiv 2403.10444, Algorithm 2) on a chain in float64.

    ``target`` is [rows, L + 1, V], ``draft`` [rows, L, V], ``tokens`` [rows, L + 1]
    (``tokens[:, i]`` drafted from ``draft[:, i - 1]``), ``coins[:, i]`` decides step i.
    """
    rows, steps = tokens.shape[0], tokens.shape[1] - 1
    row = torch.arange(rows)
    prefix = [torch.ones(rows, dtype=torch.float64)]
    for i in range(1, steps + 1):
        ratio = target[row, i - 1, tokens[:, i]] / draft[row, i - 1, tokens[:, i]]
        prefix.append((prefix[-1] * ratio).clamp(max=1))
    length = torch.zeros(rows, dtype=torch.long)
    decided = torch.zeros(rows, dtype=torch.bool)
    for i in range(steps, 0, -1):
        if i == steps:
            h = prefix[i]
        else:
            mass = (
                (prefix[i][:, None] * target[:, i] - draft[:, i]).clamp(min=0).sum(-1)
            )
            h = mass / (mass + 1 - prefix[i])
        take = ~decided & (coins[:, i] < h)
        length = torch.where(take, i, length)
        decided |= take
    scale = torch.stack(prefix, dim=1)[row, length]
    weights = target[row, length].clone()
    short = length < steps
    weights[short] = (
        scale[short, None] * weights[short] - draft[row[short], length[short]]
    ).clamp(min=0)
    cdf = weights.cumsum(-1) / weights.sum(-1, keepdim=True)
    bonus = (cdf <= final_coins[:, None]).sum(-1).clamp(max=target.shape[-1] - 1)
    return length, bonus


def test_chain_matches_block_verification():
    torch.manual_seed(5)
    rows, steps, k, vocab = 4096, 7, 16, 257
    nodes = steps + 1
    node_tokens = torch.stack(
        [torch.randperm(vocab)[:k] for _ in range(rows * nodes)]
    ).view(rows, nodes, k)
    node_probs = torch.softmax(torch.randn(rows, nodes, k) * 2, dim=-1)
    drawn = torch.multinomial(node_probs.view(-1, k), 1).view(rows, nodes)
    candidates = torch.zeros(rows, nodes, dtype=torch.long)
    candidates[:, 1:] = node_tokens[:, :-1].gather(2, drawn[:, :-1, None]).squeeze(-1)
    logits = torch.randn(rows, nodes, vocab)
    logits.scatter_add_(2, node_tokens, math.log(vocab) + 1 + node_probs.log() * 0.7)
    target = torch.softmax(logits, dim=-1)
    coins = torch.rand(rows, nodes)
    final_coins = torch.rand(rows)
    local = torch.arange(nodes)
    tree = dict(
        candidates=candidates,
        retrive_index=(torch.arange(rows)[:, None] * nodes + local).contiguous(),
        retrive_next_token=torch.where(local < steps, local + 1, -1).repeat(rows, 1),
        retrive_next_sibling=torch.full((rows, nodes), -1),
        node_draft_tokens=node_tokens,
        node_draft_probs=node_probs,
    )
    predicts, accept_index, accept_num, _ = _verify(
        tree_speculative_sampling_traversal,
        {name: value.cuda() for name, value in tree.items()},
        target_probs=target.cuda(),
        coins=coins.cuda(),
        final_coins=final_coins.cuda(),
        num_depths=steps,
    )
    draft = torch.zeros(rows, steps, vocab, dtype=torch.float64)
    draft.scatter_(2, node_tokens[:, :-1], node_probs[:, :-1].double())
    length, bonus = _block_verification(
        target.double(), draft, candidates, coins.double(), final_coins.double()
    )
    assert torch.equal(accept_num.cpu().long(), length)
    committed = _committed(predicts, accept_index, accept_num, nodes).cpu()
    accepted = local[None, :] < length[:, None]
    assert torch.equal(committed[accepted], candidates[:, 1:][accepted[:, :-1]])
    final = committed[torch.arange(rows), length]
    assert (final == bonus).float().mean() > 0.999
    assert length.float().std() > 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
