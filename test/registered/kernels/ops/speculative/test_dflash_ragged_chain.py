"""DFLASH ragged chain lengths: the selector collision mass, the split of a batch's
verify rows over its requests, and the packed row layout."""

import itertools
import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dflash import selector_collision_mass
from sglang.srt.speculative.dflash_ragged_chain import (
    RaggedChainPolicy,
    RaggedChainRows,
    RaggedChainTable,
    fill_padded_chain_rows,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")


def _enumerated_collision_mass(scores, temperature):
    """Sum over depths of sum over paths of q(path)^2, by enumerating the paths."""
    slots, top_k = scores.shape[0], scores.shape[-1]
    q = torch.softmax(scores.double() / temperature, dim=-1)
    total = 0.0
    for depth in range(1, slots + 1):
        for path in itertools.product(range(top_k), repeat=depth):
            prob = q[0, 0, path[0]]
            for slot in range(1, depth):
                prob = prob * q[slot, path[slot - 1], path[slot]]
            total += prob.item() ** 2
    return total


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_collision_mass_enumerates_path_agreement(device):
    gen = torch.Generator().manual_seed(0)
    bs, slots, top_k = 3, 4, 4
    scores = 2 * torch.randn(bs, slots, top_k, top_k, generator=gen)
    temperatures = torch.tensor([1.0, 0.7, 1.3])
    mass = selector_collision_mass(
        scores=scores.to(device), temperatures=temperatures.to(device)
    ).cpu()
    for b in range(bs):
        expected = _enumerated_collision_mass(scores[b], temperatures[b].item())
        assert abs(mass[b].item() - expected) < 1e-4 * max(1.0, expected)


def test_collision_mass_kernel_matches_torch_at_serving_shape():
    gen = torch.Generator().manual_seed(1)
    scores = 3 * torch.randn(16, 15, 16, 16, generator=gen)
    temperatures = torch.rand(16, generator=gen) + 0.5
    reference = selector_collision_mass(scores=scores, temperatures=temperatures)
    kernel = selector_collision_mass(
        scores=scores.cuda().bfloat16(), temperatures=temperatures.cuda()
    ).cpu()
    reference_bf16 = selector_collision_mass(
        scores=scores.bfloat16(), temperatures=temperatures
    )
    torch.testing.assert_close(kernel, reference_bf16, rtol=1e-4, atol=1e-4)
    assert (reference > 0).all() and (reference <= 15).all()


def _policy(curves, rows_per_request=((4096, 2),)):
    block = len(curves[0])
    table = RaggedChainTable(
        block=block,
        bin_edges=[float(j) + 0.5 for j in range(len(curves) - 1)],
        accept_lens=curves,
        rows_per_request=[list(pair) for pair in rows_per_request],
    )
    return RaggedChainPolicy(table=table, device=torch.device("cpu"))


def test_verify_lens_maximize_expected_accept_length():
    """With concave curves the split is the best allocation of the total rows."""
    curves = [
        [1.0, 1.2, 1.3, 1.35],
        [1.0, 1.8, 2.4, 2.8],
        [1.0, 1.9, 2.7, 3.4],
    ]
    policy = _policy(curves)
    mass = torch.tensor([0.0, 1.0, 2.0, 1.0])  # bins 0, 1, 2, 1
    for rows in (1, 2, 3, 4):
        lens = policy.verify_lens(collision_mass=mass, rows_per_request=rows)
        assert lens.sum().item() == rows * len(mass)
        assert ((lens >= 1) & (lens <= 4)).all()
        value = sum(curves[int(m)][k - 1] for m, k in zip(mass.tolist(), lens.tolist()))
        best = max(
            sum(curves[int(m)][k - 1] for m, k in zip(mass.tolist(), split))
            for split in itertools.product(range(1, 5), repeat=len(mass))
            if sum(split) == rows * len(mass)
        )
        assert abs(value - best) < 1e-6, (rows, lens.tolist())


def test_rows_per_request_by_batch_size():
    policy = _policy([[1.0, 2.0]], rows_per_request=((2, 2), (12, 1)))
    assert [policy.rows_per_request(bs) for bs in (1, 2, 3, 12, 40)] == [2, 2, 1, 1, 1]
    assert policy.widths == [1, 2]


def test_packed_rows_and_graph_padding():
    rows = RaggedChainRows.build(
        verify_lens=torch.tensor([3, 1, 2], dtype=torch.int32),
        rows_per_request=2,
        block=4,
    )
    assert rows.qo_indptr.tolist() == [0, 3, 4, 6]
    assert rows.token_reqs.tolist() == [0, 0, 0, 1, 2, 2]
    assert rows.block_index.tolist() == [0, 1, 2, 4, 8, 9]
    # Two graph padding requests take the uniform rows per request.
    qo_indptr = torch.full((6,), -1, dtype=torch.int32)
    token_reqs = torch.full((10,), -1, dtype=torch.int32)
    fill_padded_chain_rows(
        rows=rows, bs=5, rows_per_request=2, qo_indptr_out=qo_indptr,
        token_reqs_out=token_reqs,
    )  # fmt: skip
    assert qo_indptr.tolist() == [0, 3, 4, 6, 8, 10]
    assert token_reqs.tolist() == [0, 0, 0, 1, 2, 2, 3, 3, 4, 4]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))


@pytest.mark.parametrize(
    "bs,rows_per_request", [(16, 8), (13, 8), (8, 10), (3, 12), (2, 16), (32, 1)]
)
def test_fused_rows_match_the_torch_split(bs, rows_per_request):
    """The one-kernel split and layout equal the torch ops (distinct gains: no ties)."""
    gen = torch.Generator().manual_seed(bs * 100 + rows_per_request)
    block, bins = 16, 20
    gains = (
        torch.rand(bins, block - 1, generator=gen).sort(dim=1, descending=True).values
    )
    curves = torch.cat([torch.ones(bins, 1), 1 + gains.cumsum(1)], 1)
    policy = _policy(curves.tolist())
    mass = torch.rand(bs, generator=gen) * bins
    draft_tokens = torch.randint(0, 1000, (bs, block), generator=gen)
    positions = torch.randint(0, 5000, (bs, block), generator=gen)
    expected, tokens, packed = policy.plan(
        collision_mass=mass, rows_per_request=rows_per_request,
        draft_tokens=draft_tokens, positions=positions,
    )  # fmt: skip
    gpu = _policy(curves.tolist())
    gpu._bin_edges, gpu._gains = gpu._bin_edges.cuda(), gpu._gains.cuda()
    rows, gpu_tokens, gpu_packed = gpu.plan(
        collision_mass=mass.cuda(), rows_per_request=rows_per_request,
        draft_tokens=draft_tokens.cuda(), positions=positions.cuda(),
    )  # fmt: skip
    for name in ("verify_lens", "qo_indptr", "token_reqs", "block_index"):
        assert getattr(rows, name).tolist() == getattr(expected, name).tolist(), name
    assert gpu_tokens.tolist() == tokens.tolist()
    assert gpu_packed.tolist() == packed.tolist()


def test_fused_rows_split_ties_to_the_earlier_request():
    policy = _policy([[1.0, 2.0, 2.5, 3.0]] * 2)
    policy._bin_edges, policy._gains = policy._bin_edges.cuda(), policy._gains.cuda()
    rows, _, _ = policy.plan(
        collision_mass=torch.zeros(3, device="cuda"), rows_per_request=2,
        draft_tokens=torch.zeros(3, 4, dtype=torch.long, device="cuda"),
        positions=torch.zeros(3, 4, dtype=torch.long, device="cuda"),
    )  # fmt: skip
    # Equal curves: the three extra rows take each request's 1.0 gain, in order.
    assert rows.verify_lens.tolist() == [2, 2, 2]
    rows, _, _ = policy.plan(
        collision_mass=torch.zeros(2, device="cuda"), rows_per_request=3,
        draft_tokens=torch.zeros(2, 4, dtype=torch.long, device="cuda"),
        positions=torch.zeros(2, 4, dtype=torch.long, device="cuda"),
    )  # fmt: skip
    # Gains 1, .5, .5 each, four extra rows: both 1s, then the earlier request's .5s.
    assert rows.verify_lens.tolist() == [4, 2]


@pytest.mark.parametrize("live_bs", [0, 3, 5])
def test_graph_padding_kernel_matches_torch(live_bs):
    bs, rows_per_request = 5, 4
    rows = None
    if live_bs:
        lens = torch.tensor([1, 7, 4, 4, 4][:live_bs], dtype=torch.int32)
        lens[-1] += rows_per_request * live_bs - int(lens.sum())
        rows = RaggedChainRows.build(
            verify_lens=lens, rows_per_request=rows_per_request, block=16
        )
    outs = []
    for device in ("cpu", "cuda"):
        live = None
        if rows is not None:
            live = RaggedChainRows(
                verify_lens=rows.verify_lens.to(device),
                qo_indptr=rows.qo_indptr.to(device),
                token_reqs=rows.token_reqs.to(device),
                block_index=rows.block_index.to(device),
                rows_per_request=rows_per_request,
            )
        qo_indptr = torch.full((bs + 2,), -1, dtype=torch.int32, device=device)
        token_reqs = torch.full(
            (bs * rows_per_request + 3,), -1, dtype=torch.int32, device=device
        )
        fill_padded_chain_rows(
            rows=live, bs=bs, rows_per_request=rows_per_request,
            qo_indptr_out=qo_indptr, token_reqs_out=token_reqs,
        )  # fmt: skip
        outs.append((qo_indptr.tolist(), token_reqs.tolist()))
    assert outs[0] == outs[1]
