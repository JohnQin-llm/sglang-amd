"""tree_verify_attention against dense masked attention in fp32: each tree node sees
the prefix inside the sliding window measured from its own position, and its
ancestors-or-self among the tree nodes."""

import math

import pytest
import torch

from sglang.kernels.ops.attention.tree_verify_splitkv import (
    tree_ancestor_bits,
    tree_verify_attention,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")


def _random_tree(nodes: int, gen: torch.Generator) -> torch.Tensor:
    """QLEN_ONLY mask [nodes, nodes]: row i marks node i and its ancestors."""
    mask = torch.eye(nodes, dtype=torch.bool)
    for j in range(1, nodes):
        parent = int(torch.randint(max(0, j - 4), j, (1,), generator=gen))
        mask[j] |= mask[parent]
    return mask


def _slots(num_reqs, kv_lens, page_size, gen):
    """Request table of scattered slots, page-contiguous when page_size > 1."""
    pages = [math.ceil(n / page_size) for n in kv_lens]
    perm = torch.randperm(sum(pages) + 3, generator=gen) + 1
    table = torch.zeros(num_reqs + 1, max(pages) * page_size + 8, dtype=torch.int32)
    used = 0
    for r, n in enumerate(pages):
        page_ids = perm[used : used + n]
        used += n
        row = (page_ids[:, None] * page_size + torch.arange(page_size)).reshape(-1)
        table[r + 1, : n * page_size] = row.to(torch.int32)
    return table, (int(perm.max()) + 1) * page_size


def _reference(q, k, v, table, pool_rows, prefixes, masks, positions, **kw):
    nodes = masks[0].shape[0]
    heads, kv_heads = q.shape[1], k.shape[1]
    out = torch.empty(q.shape, dtype=torch.float64)
    for r, prefix in enumerate(prefixes):
        n = prefix + nodes
        slots = table[pool_rows[r], :n].long()
        if kw["slot_map"] is not None:
            slots = kw["slot_map"][slots].long()
        kk = k[slots].double().repeat_interleave(heads // kv_heads, 1)
        vv = v[slots].double().repeat_interleave(heads // kv_heads, 1)
        rows = slice(r * nodes, (r + 1) * nodes)
        key = torch.arange(n)
        visible = (key[None, :] < prefix).repeat(nodes, 1)
        if kw["window"] > 0:
            visible &= positions[rows, None] - key[None, :] < kw["window"]
        visible[:, prefix:] = masks[r]
        scores = torch.einsum("thd,khd->htk", q[rows].double(), kk)
        scores *= kw["sm_scale"] * kw["k_scale"]
        if kw["logit_cap"] > 0:
            scores = kw["logit_cap"] * torch.tanh(scores / kw["logit_cap"])
        scores = scores.masked_fill(~visible[None], float("-inf"))
        if kw["sinks"] is not None:
            sink = kw["sinks"].double()[:, None, None].expand(-1, nodes, 1)
            probs = torch.softmax(torch.cat([scores, sink], -1), -1)[..., :-1]
        else:
            probs = torch.softmax(scores, -1)
        out[rows] = torch.einsum("htk,khd->thd", probs, vv) * kw["v_scale"]
    return out


def run_case(
    *,
    prefixes,
    nodes,
    heads,
    kv_heads,
    head_dim,
    window=0,
    kv_dtype=torch.bfloat16,
    q_dtype=torch.bfloat16,
    page_size=1,
    k_scale=1.0,
    v_scale=1.0,
    logit_cap=0.0,
    with_sinks=False,
    near_tie=False,
    splits=None,
    min_split_keys=None,
    chain=False,
    device="cuda",
    seed=0,
):
    """Max abs error of the kernel against the fp64 reference. ``chain`` verifies
    a linear block: no ancestor masks or positions reach the kernel."""
    gen = torch.Generator().manual_seed(seed)
    num_reqs = len(prefixes)
    kv_lens = [p + nodes for p in prefixes]
    table, num_slots = _slots(num_reqs, kv_lens, page_size, gen)
    pool_rows = torch.arange(1, num_reqs + 1)
    slot_map = None
    if window > 0:
        # A window pool with its own slot order, as the SWA KV pool keeps.
        slot_map = torch.randperm(num_slots, generator=gen).to(torch.int64)
    k = torch.randn(num_slots, kv_heads, head_dim, generator=gen)
    v = torch.randn(num_slots, kv_heads, head_dim, generator=gen)
    q = torch.randn(num_reqs * nodes, heads, head_dim, generator=gen)
    if near_tie:
        # Every key close to one direction the queries share: large logits that
        # differ in the third significant digit across thousands of keys.
        base = torch.randn(head_dim, generator=gen)
        k = base + 1e-2 * k
        q = 4.0 * base + 1e-2 * q
    k, v, q = k.to(kv_dtype), v.to(kv_dtype), q.to(q_dtype)
    if chain:
        masks = [torch.ones(nodes, nodes, dtype=torch.bool).tril() for _ in prefixes]
    else:
        masks = [_random_tree(nodes, gen) for _ in prefixes]
    depth = torch.cat([m.sum(1) - 1 for m in masks])
    positions = torch.tensor(prefixes).repeat_interleave(nodes) + depth
    sinks = torch.randn(heads, generator=gen) if with_sinks else None
    sm_scale = head_dim**-0.5

    to = lambda t: None if t is None else t.to(device)  # noqa: E731
    ancestors = tree_ancestor_bits(
        tree_mask=to(torch.stack(masks).reshape(-1)),
        out=torch.empty(num_reqs * nodes, dtype=torch.int64, device=device),
        num_nodes=nodes,
        num_masked_reqs=num_reqs,
    )
    out = torch.empty(q.shape, dtype=q_dtype, device=device)
    tree_verify_attention(
        q=to(q),
        k_cache=to(k),
        v_cache=to(v),
        out=out,
        req_to_token=to(table),
        req_pool_indices=to(pool_rows),
        seq_lens=to(torch.tensor(prefixes)),
        positions=None if chain else to(positions),
        ancestors=None if chain else ancestors,
        sm_scale=sm_scale,
        window=window,
        slot_map=to(slot_map),
        k_scale=to(torch.tensor([k_scale])),
        v_scale=to(torch.tensor([v_scale])),
        logit_cap=logit_cap,
        sinks=to(sinks),
        splits=splits,
        min_split_keys=min_split_keys,
    )
    ref = _reference(
        q.double(), k.double(), v.double(), table, pool_rows, prefixes, masks,
        positions, slot_map=slot_map, window=window, sm_scale=sm_scale,
        k_scale=k_scale, v_scale=v_scale, logit_cap=logit_cap, sinks=sinks,
    )  # fmt: skip
    return (out.cpu().double() - ref).abs().max().item()


# (prefixes, nodes, heads, kv_heads, head_dim, window): Muse-Glimmer full and
# sliding-window layers, Qwen3.8-27B (GQA 6), Muse Spark (head_dim 64, GQA 8).
SHAPES = [
    ([1000, 37, 4100], 16, 32, 2, 128, 0),
    ([3000, 100, 2052], 32, 32, 2, 128, 2047),
    ([0, 1, 2046], 8, 32, 2, 128, 2047),
    ([1500, 9], 16, 24, 4, 256, 0),
    ([700, 2200], 64, 24, 4, 256, 0),
    ([2100, 300, 1], 32, 32, 4, 64, 2048),
]


@pytest.mark.parametrize("prefixes,nodes,heads,kv_heads,head_dim,window", SHAPES)
def test_matches_dense_reference(prefixes, nodes, heads, kv_heads, head_dim, window):
    err = run_case(
        prefixes=prefixes, nodes=nodes, heads=heads, kv_heads=kv_heads,
        head_dim=head_dim, window=window,
    )  # fmt: skip
    assert err < 2e-2, err


@pytest.mark.parametrize("window", [0, 2047])
def test_fp8_cache_scales(window):
    err = run_case(
        prefixes=[1800, 33], nodes=16, heads=32, kv_heads=2, head_dim=128,
        window=window, kv_dtype=torch.float8_e4m3fn, k_scale=0.5, v_scale=2.0,
        page_size=16,
    )  # fmt: skip
    assert err < 5e-2, err


@pytest.mark.parametrize("splits", [1, 4, 32])
def test_near_tie_logits_across_splits(splits):
    """Large near-equal logits merged across key ranges stay within bf16 rounding."""
    err = run_case(
        prefixes=[5000, 64], nodes=32, heads=32, kv_heads=2, head_dim=128,
        near_tie=True, splits=splits, min_split_keys=64,
    )  # fmt: skip
    assert err < 2e-2, err


# Chain blocks: Muse-Glimmer full and sliding-window layers (prefixes past the
# window), Qwen3.8-27B at its block of 8.
CHAIN_SHAPES = [
    ([1000, 37, 4100], 16, 32, 2, 128, 0),
    ([3000, 5, 2047], 16, 32, 2, 128, 2047),
    ([1500, 9, 6000], 8, 24, 4, 256, 0),
]


@pytest.mark.parametrize("prefixes,nodes,heads,kv_heads,head_dim,window", CHAIN_SHAPES)
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_chain_matches_dense_reference(
    prefixes, nodes, heads, kv_heads, head_dim, window, kv_dtype
):
    err = run_case(
        prefixes=prefixes, nodes=nodes, heads=heads, kv_heads=kv_heads,
        head_dim=head_dim, window=window, kv_dtype=kv_dtype, k_scale=0.5,
        v_scale=2.0, chain=True,
    )  # fmt: skip
    assert err < 5e-2, err


def test_logit_cap_and_sinks():
    err = run_case(
        prefixes=[900, 2], nodes=16, heads=24, kv_heads=4, head_dim=256,
        window=512, logit_cap=5.0, with_sinks=True, splits=4,
    )  # fmt: skip
    assert err < 2e-2, err


def test_padded_requests_see_only_themselves():
    """Graph padding rows past the masked requests get a self-only mask, so their
    attention stays finite whatever the stale mask cells hold."""
    nodes = 8
    mask = torch.ones(3, nodes, nodes, dtype=torch.bool, device="cuda")
    bits = tree_ancestor_bits(
        tree_mask=mask.reshape(-1),
        out=torch.empty(3 * nodes, dtype=torch.int64, device="cuda"),
        num_nodes=nodes,
        num_masked_reqs=1,
    ).cpu()
    assert (bits[:nodes] == (1 << nodes) - 1).all()
    assert (bits[nodes:] == (1 << torch.arange(nodes)).repeat(2)).all()


def test_64_node_mask_uses_sign_bit():
    nodes = 64
    mask = torch.tril(torch.ones(nodes, nodes, dtype=torch.bool, device="cuda"))
    bits = tree_ancestor_bits(
        tree_mask=mask.reshape(-1),
        out=torch.empty(nodes, dtype=torch.int64, device="cuda"),
        num_nodes=nodes,
        num_masked_reqs=1,
    ).cpu()
    expected = [((1 << (i + 1)) - 1) for i in range(nodes)]
    expected = [x - (1 << 64) if x >= 1 << 63 else x for x in expected]
    assert bits.tolist() == expected
