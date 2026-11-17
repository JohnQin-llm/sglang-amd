"""Multi-draft tree verification must commit tokens distributed exactly as the target."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.multi_draft_sampling import (
    tree_speculative_sampling_multi_draft,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"


def _two_level_trees(rows, vocab, k, children, gen):
    """Per row: root draws `children` tokens without replacement from q_root over k candidates,
    and every child draws `children` grandchildren from its own q. Node order is BFS."""
    cand = torch.stack([torch.randperm(vocab, generator=gen)[:k] for _ in range(rows)])
    q_root = F.softmax(torch.randn(rows, k, generator=gen) * 1.5, -1)
    order = (
        q_root.log() - torch.log(-torch.log(torch.rand(rows, k, generator=gen)))
    ).argsort(-1, descending=True)
    first = order[:, :children]
    num_nodes = 1 + children + children * children
    tokens = torch.zeros(rows, num_nodes, dtype=torch.long)
    node_tokens = torch.zeros(rows, num_nodes, k, dtype=torch.long)
    node_probs = torch.zeros(rows, num_nodes, k)
    next_token = torch.full((rows, num_nodes), -1, dtype=torch.long)
    next_sibling = torch.full((rows, num_nodes), -1, dtype=torch.long)
    node_tokens[:, 0], node_probs[:, 0] = cand, q_root
    next_token[:, 0] = 1
    for c in range(children):
        node = 1 + c
        tokens[:, node] = cand.gather(1, first[:, c : c + 1])[:, 0]
        if c + 1 < children:
            next_sibling[:, node] = node + 1
        child_cand = torch.stack(
            [torch.randperm(vocab, generator=gen)[:k] for _ in range(rows)]
        )
        q_child = F.softmax(torch.randn(rows, k, generator=gen) * 1.5, -1)
        child_order = (
            q_child.log() - torch.log(-torch.log(torch.rand(rows, k, generator=gen)))
        ).argsort(-1, descending=True)
        node_tokens[:, node], node_probs[:, node] = child_cand, q_child
        base = 1 + children + c * children
        next_token[:, node] = base
        for g in range(children):
            tokens[:, base + g] = child_cand.gather(1, child_order[:, g : g + 1])[:, 0]
            if g + 1 < children:
                next_sibling[:, base + g] = base + g + 1
    return tokens, node_tokens, node_probs, next_token, next_sibling


def _run(
    tokens,
    node_tokens,
    node_probs,
    next_token,
    next_sibling,
    target_probs,
    coins,
    final,
    num_spec,
):
    rows, num_nodes = tokens.shape
    predicts = torch.full((rows * num_nodes,), -1, dtype=torch.int32, device=DEVICE)
    accept_index = torch.full((rows, num_spec), -1, dtype=torch.int32, device=DEVICE)
    accept_num = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
    retrive_index = torch.arange(rows * num_nodes, dtype=torch.long).view(
        rows, num_nodes
    )
    tree_speculative_sampling_multi_draft(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_num,
        candidates=tokens.to(DEVICE),
        retrive_index=retrive_index.to(DEVICE),
        retrive_next_token=next_token.to(DEVICE),
        retrive_next_sibling=next_sibling.to(DEVICE),
        uniform_samples=coins.to(DEVICE),
        uniform_samples_for_final_sampling=final.to(DEVICE),
        target_probs=target_probs.to(DEVICE),
        node_draft_tokens=node_tokens.to(DEVICE),
        node_draft_probs=node_probs.to(DEVICE),
    )
    return predicts.view(rows, num_nodes).cpu(), accept_index.cpu(), accept_num.cpu()


def _reference(
    tokens,
    node_tokens,
    node_probs,
    next_token,
    next_sibling,
    target_probs,
    coins,
    final,
    num_spec,
):
    """Host recursive rejection sampling, the algorithm the kernel implements."""
    rows, num_nodes = tokens.shape
    out = torch.full((rows, num_nodes), -1, dtype=torch.int32)
    out_num = torch.zeros(rows, dtype=torch.int32)
    for b in range(rows):
        node, slot, num = 0, 0, 0
        while True:
            p = target_probs[b, node].double().clone()
            toks, q = node_tokens[b, node], node_probs[b, node].double()
            drawn, accepted = torch.zeros_like(q, dtype=torch.bool), False
            child = next_token[b, node].item() if num < num_spec - 1 else -1
            while child != -1:
                q_k = torch.where(drawn, 0.0, q) / max(1 - q[drawn].sum().item(), 1e-20)
                q_full = torch.zeros_like(p).index_add_(0, toks, q_k)
                tok = tokens[b, child].item()
                if (
                    q_full[tok] > 0
                    and coins[b, child].item() * q_full[tok].item() < p[tok].item()
                ):
                    out[b, slot], num, slot, node, accepted = (
                        tok,
                        num + 1,
                        child,
                        child,
                        True,
                    )
                    break
                p = (p - q_full).clamp(min=0)
                p = p / p.sum()
                drawn |= toks == tok
                child = next_sibling[b, child].item()
            if not accepted:
                break
        cdf = p.cumsum(0)
        hit = ((cdf > final[b].item() * p.sum()) & (p > 0)).nonzero()
        out[b, slot] = hit[0].item() if len(hit) else (p > 0).nonzero()[-1].item()
        out_num[b] = num
    return out, out_num


def test_matches_host_reference():
    gen = torch.Generator().manual_seed(11)
    rows, vocab, k = 64, 300, 16
    tree = _two_level_trees(rows, vocab, k, children=3, gen=gen)
    num_nodes = tree[0].shape[1]
    target = F.softmax(torch.randn(rows, num_nodes, vocab, generator=gen) * 3, -1)
    coins = torch.rand(rows, num_nodes, generator=gen)
    final = torch.rand(rows, generator=gen)
    predicts, _, accept_num = _run(*tree, target, coins, final, num_spec=3)
    ref, ref_num = _reference(*tree, target, coins, final, num_spec=3)
    assert torch.equal(accept_num, ref_num)
    assert torch.equal(predicts.masked_fill(ref == -1, -1), ref)


def test_committed_tokens_follow_the_target():
    gen = torch.Generator().manual_seed(5)
    rows, vocab, k = 60000, 12, 4
    tree = _two_level_trees(rows, vocab, k, children=2, gen=gen)
    num_nodes = tree[0].shape[1]
    target = F.softmax(torch.randn(num_nodes, vocab, generator=gen) * 1.5, -1).repeat(
        rows, 1, 1
    )
    predicts, _, accept_num = _run(
        *tree,
        target,
        torch.rand(rows, num_nodes, generator=gen),
        torch.rand(rows, generator=gen),
        num_spec=3,
    )
    first = predicts[:, 0].long()
    freq = torch.bincount(first, minlength=vocab).double() / rows
    assert (freq - target[0, 0].double()).abs().max().item() < 0.012
    # Given the first child (node 1) was accepted, the next token must follow node 1's target.
    tokens = tree[0]
    took_node1 = (accept_num >= 1) & (first == tokens[:, 1])
    second = predicts[took_node1, 1].long()
    freq2 = torch.bincount(second, minlength=vocab).double() / second.numel()
    assert second.numel() > 5000
    assert (freq2 - target[0, 1].double()).abs().max().item() < 0.03


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
