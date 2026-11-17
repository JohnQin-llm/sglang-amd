"""The Triton target-only tree sampler must match the CUDA kernel's decisions and stay lossless."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.target_only_sampling import (
    tree_speculative_sampling_target_only_triton,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"


def _run(
    candidates,
    retrive_index,
    next_token,
    next_sibling,
    target_probs,
    coins,
    final_coins,
    num_spec_tokens,
    threshold_single=1.0,
    threshold_acc=1.0,
):
    bs, num_draft_tokens = candidates.shape
    predicts = torch.full(
        (bs * num_draft_tokens,), -1, dtype=torch.int32, device=DEVICE
    )
    accept_index = torch.full(
        (bs, num_spec_tokens), -1, dtype=torch.int32, device=DEVICE
    )
    accept_token_num = torch.zeros((bs,), dtype=torch.int32, device=DEVICE)
    tree_speculative_sampling_target_only_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=next_token,
        retrive_next_sibling=next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=final_coins,
        target_probs=target_probs,
        draft_probs=torch.zeros_like(target_probs),
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
    )
    return predicts, accept_index, accept_token_num


def _reference(
    candidates,
    retrive_index,
    next_token,
    next_sibling,
    target_probs,
    coins,
    final_coins,
    num_spec_tokens,
    threshold_single=1.0,
    threshold_acc=1.0,
):
    """Line-by-line port of TreeSpeculativeSamplingTargetOnly (sampling.cuh) on the host."""
    bs, num_draft_tokens = candidates.shape
    vocab = target_probs.shape[-1]
    tp = target_probs.double().cpu()
    cand, ridx = candidates.cpu(), retrive_index.cpu()
    nt, ns = next_token.cpu(), next_sibling.cpu()
    predicts = torch.full((bs * num_draft_tokens,), -1, dtype=torch.int32)
    accept_index = torch.full((bs, num_spec_tokens), -1, dtype=torch.int32)
    accept_token_num = torch.zeros((bs,), dtype=torch.int32)
    for b in range(bs):
        dp = torch.zeros(num_draft_tokens, vocab, dtype=torch.float64)
        prob_node, coin, prob_acc = 0, coins[b, 0].item(), 0.0
        last = ridx[b, 0].item()
        accept_index[b, 0] = last
        num_correct, node = 0, 0
        for _ in range(1, num_spec_tokens):
            node = nt[b, node].item()
            while node != -1:
                token = cand[b, node].item()
                p = tp[b, prob_node, token].item()
                prob_acc += p
                if p > 0 and (
                    coin < prob_acc / max(threshold_acc, 1e-9) or p >= threshold_single
                ):
                    prob_acc, prob_node, coin = 0.0, node, coins[b, node].item()
                    predicts[last] = token
                    num_correct += 1
                    last = ridx[b, node].item()
                    accept_index[b, num_correct] = last
                    break
                dp[prob_node, token] = p
                node = ns[b, node].item()
            if node == -1:
                break
        accept_token_num[b] = num_correct
        q = tp[b, prob_node]
        p = (
            dp[prob_node]
            if num_correct != num_spec_tokens - 1
            else torch.zeros(vocab, dtype=torch.float64)
        )
        residual = (q - p).clamp(min=0)
        u = final_coins[b].item() * residual.sum().item()
        hit = ((residual.cumsum(0) > u) & (residual > 0)).nonzero()
        if len(hit):
            sampled = hit[0].item()
        else:
            valid = (residual > 0).nonzero()
            sampled = valid[-1].item() if len(valid) else vocab - 1
        predicts[last] = sampled
    return predicts, accept_index, accept_token_num


def _random_trees(bs, num_draft_tokens, generator):
    """EAGLE-shaped trees: node i > 0 hangs under a random earlier node, siblings in index order."""
    parent = torch.zeros(bs, num_draft_tokens, dtype=torch.int64)
    for i in range(1, num_draft_tokens):
        parent[:, i] = torch.randint(0, i, (bs,), generator=generator)
    next_token = torch.full((bs, num_draft_tokens), -1, dtype=torch.int64)
    next_sibling = torch.full((bs, num_draft_tokens), -1, dtype=torch.int64)
    for b in range(bs):
        last_child = {}
        for i in range(1, num_draft_tokens):
            par = parent[b, i].item()
            if par not in last_child:
                next_token[b, par] = i
            else:
                next_sibling[b, last_child[par]] = i
            last_child[par] = i
    return next_token, next_sibling


@pytest.mark.parametrize("threshold_single, threshold_acc", [(1.0, 1.0), (0.0, 0.0)])
def test_matches_cuda_expected_outputs(threshold_single, threshold_acc):
    # Same tree and logits as test_speculative_sampling.py; near-one-hot targets make it deterministic.
    candidates = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]], dtype=torch.int64, device=DEVICE
    )
    retrive_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=torch.int64, device=DEVICE
    )
    next_token = torch.tensor(
        [[1, 2, -1, 4, 5, -1], [4, 2, 3, -1, 5, -1]], dtype=torch.int64, device=DEVICE
    )
    next_sibling = torch.tensor(
        [[-1, 3, -1, -1, -1, -1], [-1, -1, -1, -1, 1, -1]],
        dtype=torch.int64,
        device=DEVICE,
    )
    logits = torch.full((2, 6, 20), 1.0, device=DEVICE)
    logits[0, 0, 3] = logits[0, 3, 4] = logits[0, 4, 5] = 10
    logits[1, 0, 11] = logits[1, 4, 12] = 10
    for i in range(2):
        for j in range(6):
            if logits[i, j].max() < 10:
                logits[i, j, 18] = 10
    target_probs = F.softmax(logits / 0.01, dim=-1)
    predicts, accept_index, accept_token_num = _run(
        candidates,
        retrive_index,
        next_token,
        next_sibling,
        target_probs,
        torch.rand(2, 6, device=DEVICE),
        torch.rand(2, device=DEVICE),
        num_spec_tokens=4,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
    )
    assert predicts.tolist() == [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18]
    assert accept_index.tolist() == [[0, 3, 4, 5], [6, 10, 11, -1]]
    assert accept_token_num.tolist() == [3, 2]


@pytest.mark.parametrize(
    "candidate_tokens, target_probabilities, coin, threshold_single, expected_token, expected_correct",
    [
        ([2], [0.0, 1.0, 0.0], 0.0, 0.0, 1, 0),
        ([2], [0.0, 1.0, 0.0], 0.0, 1.0, 1, 0),
        ([1, 2, 3], [0.0, 0.25, 0.0, 0.75], 0.25, 1.0, 3, 1),
        ([1], [0.0, 0.25, 0.75], 0.0, 1.0, 1, 1),
        ([1], [0.0, 1.0], 1.0 - 2**-24, 1.0, 1, 1),
    ],
    ids=[
        "zero-mass-zero-threshold",
        "zero-mass-default-threshold",
        "interior-boundary",
        "lower-endpoint",
        "upper-endpoint",
    ],
)
def test_cdf_boundaries(
    candidate_tokens,
    target_probabilities,
    coin,
    threshold_single,
    expected_token,
    expected_correct,
):
    num_draft_tokens = len(candidate_tokens) + 1
    candidates = torch.tensor(
        [[0, *candidate_tokens]], dtype=torch.int64, device=DEVICE
    )
    retrive_index = torch.arange(num_draft_tokens, dtype=torch.int64, device=DEVICE)[
        None
    ]
    next_token = torch.full_like(retrive_index, -1)
    next_token[0, 0] = 1
    next_sibling = torch.full_like(retrive_index, -1)
    next_sibling[0, 1 : num_draft_tokens - 1] = torch.arange(
        2, num_draft_tokens, device=DEVICE
    )
    target_probs = torch.zeros(
        (1, num_draft_tokens, len(target_probabilities)), device=DEVICE
    )
    target_probs[0, 0] = torch.tensor(target_probabilities, device=DEVICE)
    target_probs[0, 1:, 0] = 1.0
    coins = torch.zeros((1, num_draft_tokens), device=DEVICE)
    coins[0, 0] = coin
    predicts, _, accept_token_num = _run(
        candidates,
        retrive_index,
        next_token,
        next_sibling,
        target_probs,
        coins,
        torch.zeros((1,), device=DEVICE),
        num_spec_tokens=2,
        threshold_single=threshold_single,
    )
    assert predicts[0].item() == expected_token
    assert accept_token_num.item() == expected_correct


@pytest.mark.parametrize(
    "num_draft_tokens, num_spec_tokens, vocab",
    [(8, 8, 50), (16, 6, 4100), (33, 8, 9000)],
)
def test_matches_reference_on_random_trees(num_draft_tokens, num_spec_tokens, vocab):
    gen = torch.Generator().manual_seed(num_draft_tokens * 1000 + vocab)
    bs = 24
    next_token, next_sibling = _random_trees(bs, num_draft_tokens, generator=gen)
    # Peaked targets so that many drafts are accepted; candidates often hit the mode.
    logits = torch.randn(bs, num_draft_tokens, vocab, generator=gen) * 3
    target_probs = F.softmax(logits, dim=-1)
    top = target_probs.topk(4, dim=-1).indices
    pick = torch.randint(0, 4, (bs, num_draft_tokens, 1), generator=gen)
    candidates = top.gather(-1, pick)[..., 0]
    retrive_index = torch.arange(bs * num_draft_tokens, dtype=torch.int64).view(
        bs, num_draft_tokens
    )
    coins = torch.rand(bs, num_draft_tokens, generator=gen)
    final_coins = torch.rand(bs, generator=gen)
    args = (
        candidates,
        retrive_index,
        next_token,
        next_sibling,
        target_probs,
        coins,
        final_coins,
    )
    got = _run(*(a.to(DEVICE) for a in args), num_spec_tokens=num_spec_tokens)
    expected = _reference(*args, num_spec_tokens=num_spec_tokens)
    assert torch.equal(got[2].cpu(), expected[2])
    assert torch.equal(got[1].cpu(), expected[1])
    for b in range(bs):
        slots = expected[1][b, : expected[2][b] + 1].tolist()
        assert got[0].cpu()[slots].tolist() == expected[0][slots].tolist(), f"row {b}"


def test_first_token_is_distributed_as_target():
    # Root with three children: the token committed at the root slot must follow the target exactly.
    vocab, rows = 12, 40000
    gen = torch.Generator().manual_seed(7)
    p_root = F.softmax(torch.randn(vocab, generator=gen) * 1.5, dim=-1)
    candidates = torch.tensor([0, 4, 1, 9], dtype=torch.int64).repeat(rows, 1)
    next_token = torch.tensor([1, -1, -1, -1], dtype=torch.int64).repeat(rows, 1)
    next_sibling = torch.tensor([-1, 2, 3, -1], dtype=torch.int64).repeat(rows, 1)
    retrive_index = torch.arange(rows * 4, dtype=torch.int64).view(rows, 4)
    target_probs = torch.full((rows, 4, vocab), 1.0 / vocab)
    target_probs[:, 0] = p_root
    predicts, accept_index, _ = _run(
        candidates.to(DEVICE),
        retrive_index.to(DEVICE),
        next_token.to(DEVICE),
        next_sibling.to(DEVICE),
        target_probs.to(DEVICE),
        torch.rand(rows, 4, generator=gen).to(DEVICE),
        torch.rand(rows, generator=gen).to(DEVICE),
        num_spec_tokens=2,
    )
    first = predicts.view(rows, 4)[:, 0].cpu().long()
    freq = torch.bincount(first, minlength=vocab).double() / rows
    assert (freq - p_root.double()).abs().max().item() < 0.012


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
