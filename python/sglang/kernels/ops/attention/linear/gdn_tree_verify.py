# SPDX-License-Identifier: Apache-2.0
"""GDN tree verify without per-node state snapshots.

One program replays one root-to-leaf path of the draft tree from the committed
state, so no node state leaves registers. A node on several paths is recomputed
by each of them, but only the path that reaches it through first children
stores its output and its ReplaySSM ring record. A small kernel first turns the
parent links into per-leaf node lists, built once per verify, so the replay
needs no tree walk. Each step copies the GDN branch of
``fused_sigmoid_gating_delta_rule_update_kernel`` (same tile and op order), so
outputs are bitwise those of the snapshot tree verify. The commit replays the
accepted path from the ring (``gdn_replayssm_spec_fold``).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Shared with the commit fold: both must reduce over K in the same tree.
TREE_VERIFY_BV = 32
TREE_VERIFY_NUM_WARPS = 1


@triton.jit(do_not_specialize=["T"])
def _gdn_tree_paths_kernel(
    retrieve_parent_token,
    paths,
    path_lens,
    owned_froms,
    leaves,
    num_leaves,
    T,
    stride_parent_seq,
    stride_parent_token,
    NP2_T: tl.constexpr,
):
    """Per request: its leaves in index order and, per leaf, the root-to-leaf
    node list, its length, and the path position from which the leaf owns the
    nodes (those it reaches through first, lowest-index, children)."""
    i_n = tl.program_id(0)
    o_t = tl.arange(0, NP2_T)
    parents = tl.load(
        retrieve_parent_token + i_n * stride_parent_seq + o_t * stride_parent_token,
        mask=o_t < T,
        other=0,
    ).to(tl.int32)
    # Node 0 is the root. A link that does not point to an earlier node (CUDA
    # graph pad rows) is read as the previous node, so every row is a tree.
    parents = tl.where((parents >= 0) & (parents < o_t), parents, o_t - 1)
    parents = tl.where(o_t < T, parents, -2)

    # Parents precede children, so one pass in index order sees each parent
    # first. top[i]: the highest node reaching i through first children only.
    ancestors = tl.zeros([NP2_T], dtype=tl.int64)
    top = tl.zeros([NP2_T], dtype=tl.int32)
    for i in range(0, T):
        parent = tl.sum(tl.where(o_t == i, parents, 0))
        parent_ancestors = tl.sum(tl.where(o_t == parent, ancestors, 0))
        first_child = tl.min(tl.where(parents == parent, o_t, NP2_T))
        parent_top = tl.sum(tl.where(o_t == parent, top, 0))
        node_top = tl.where((i > 0) & (first_child == i), parent_top, i)
        node_bit = tl.full((), 1, tl.int64) << i.to(tl.int64)
        ancestors = tl.where(o_t == i, parent_ancestors | node_bit, ancestors)
        top = tl.where(o_t == i, node_top, top)

    top_parent = tl.sum(
        tl.where(o_t[None, :] == top[:, None], parents[None, :], 0), axis=1
    )
    above_top = tl.sum(
        tl.where(o_t[None, :] == top_parent[:, None], ancestors[None, :], 0), axis=1
    )
    num_children = tl.sum((parents[None, :] == o_t[:, None]).to(tl.int32), axis=1)
    is_leaf = (num_children == 0) & (o_t < T)

    # Row: leaf, column: node. A path lists its ancestors in index order.
    on_path = ((ancestors[:, None] >> o_t[None, :].to(tl.int64)) & 1).to(tl.int32)
    owned = (((ancestors & ~above_top)[:, None] >> o_t[None, :].to(tl.int64)) & 1).to(
        tl.int32
    )
    position = tl.cumsum(on_path, axis=1) - 1
    path_len = tl.sum(on_path, axis=1)
    row = i_n * T + o_t
    tl.store(
        paths + row[:, None] * T + position,
        o_t[None, :] + tl.zeros([NP2_T, NP2_T], dtype=tl.int32),
        mask=(on_path != 0) & is_leaf[:, None],
    )
    tl.store(path_lens + row, path_len, mask=o_t < T)
    leaf_rank = tl.cumsum(is_leaf.to(tl.int32), axis=0) - 1
    tl.store(leaves + i_n * T + leaf_rank, o_t, mask=is_leaf)
    tl.store(num_leaves + i_n, tl.sum(is_leaf.to(tl.int32)))
    tl.store(owned_froms + row, path_len - tl.sum(owned, axis=1), mask=o_t < T)


@triton.jit(do_not_specialize=["T"])
def _gdn_tree_path_verify_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    paths,
    path_lens,
    owned_froms,
    leaves,
    num_leaves,
    rawv_ring,
    rawk_ring,
    g_ring,
    beta_ring,
    ring_indices,
    scale,
    softplus_beta,
    softplus_threshold,
    N,
    T,
    stride_h0_slot,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_rawv_slot,
    stride_rawk_slot,
    stride_g_slot,
    stride_beta_slot,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    CACHE_RING: tl.constexpr,
):
    # Leaf slots are the slowest grid axis, so every real path is dispatched
    # before the empty slots of requests with fewer leaves.
    i_v = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_sn = tl.program_id(2)
    i_n = i_sn % N
    slot = i_sn // N
    if slot >= tl.load(num_leaves + i_n):
        return
    i_h = i_hv // (HV // H)
    bos = i_n.to(tl.int64) * T
    i_nl = i_n * T + tl.load(leaves + i_n * T + slot)
    path_len = tl.load(path_lens + i_nl)
    owned_from = tl.load(owned_froms + i_nl)
    p_path = paths + i_nl.to(tl.int64) * T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    state_idx = tl.load(h0_indices + i_n).to(tl.int64)
    if state_idx >= 0:
        p_h0 = (
            h0_source
            + state_idx * stride_h0_slot
            + i_hv * K * V
            + o_v[None, :] * K
            + o_k[:, None]
        )
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    if CACHE_RING:
        ring_slot = tl.load(ring_indices + i_n).to(tl.int64)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

    node = tl.load(p_path)
    for t in range(0, path_len):
        # The next node index is loaded a step ahead, off the state chain.
        next_node = tl.load(p_path + t + 1, mask=t + 1 < path_len, other=0)
        owned = t >= owned_from
        tok = bos + node
        b_k = tl.load(k + tok * stride_k + i_h * K + o_k, mask=mask_k, other=0).to(
            tl.float32
        )
        b_v = tl.load(v + tok * stride_v + i_hv * V + o_v, mask=mask_v, other=0).to(
            tl.float32
        )
        b_b = tl.load(b + tok * stride_b + i_hv).to(tl.float32)
        b_a = tl.load(a + tok * stride_a + i_hv).to(tl.float32)

        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Raw inputs (pre-norm k, pre-delta v) and the formed gates: what the
        # commit fold replays. Ring layout matches the recurrent kernel's.
        if CACHE_RING:
            if owned & (ring_slot >= 0):
                tl.store(
                    rawv_ring
                    + ring_slot * stride_rawv_slot
                    + (i_hv * MAX_CACHE_LEN + node) * V
                    + o_v,
                    b_v.to(rawv_ring.dtype.element_ty),
                    mask=mask_v,
                )
                if i_v == 0:
                    if i_hv % (HV // H) == 0:
                        tl.store(
                            rawk_ring
                            + ring_slot * stride_rawk_slot
                            + (i_h * MAX_CACHE_LEN + node) * K
                            + o_k,
                            b_k.to(rawk_ring.dtype.element_ty),
                            mask=mask_k,
                        )
                    tl.store(
                        g_ring
                        + ring_slot * stride_g_slot
                        + i_hv * MAX_CACHE_LEN
                        + node,
                        b_g,
                    )
                    tl.store(
                        beta_ring
                        + ring_slot * stride_beta_slot
                        + i_hv * MAX_CACHE_LEN
                        + node,
                        b_beta,
                    )

        if USE_QK_L2NORM_IN_KERNEL:
            k_sum_sq = tl.sum(b_k * b_k)
            k_rsqrt = 1.0 / tl.sqrt(k_sum_sq + 1e-6)
            b_k *= k_rsqrt

        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

        if owned:
            b_q = tl.load(q + tok * stride_q + i_h * K + o_k, mask=mask_k, other=0).to(
                tl.float32
            )
            if USE_QK_L2NORM_IN_KERNEL:
                q_sum_sq = tl.sum(b_q * b_q)
                q_rsqrt = 1.0 / tl.sqrt(q_sum_sq + 1e-6)
                b_q *= q_rsqrt
            b_q = b_q * scale
            b_o = tl.sum(b_h * b_q[:, None], 0)
            tl.store(
                o + (tok * HV + i_hv) * V + o_v,
                b_o.to(o.dtype.element_ty),
                mask=mask_v,
            )
        node = next_node


def gdn_tree_paths(
    retrieve_parent_token: torch.Tensor, num_draft_tokens: int
) -> tuple[torch.Tensor, ...]:
    """``(paths, path_lens, owned_froms, leaves, num_leaves)`` for
    ``gdn_tree_verify``: per request and node, ``[N, T, T]`` root-first node
    lists, ``[N, T]`` lengths and ``[N, T]`` first owned path positions (read
    for leaves only); per request, ``[N, T]`` leaf nodes in index order and
    ``[N]`` leaf counts. All layers of one verify share them."""
    T = num_draft_tokens
    assert T <= 64, "ancestor sets are int64 bitmasks"
    N = retrieve_parent_token.shape[0]
    device = retrieve_parent_token.device
    paths = torch.empty(N, T, T, dtype=torch.int32, device=device)
    path_lens = torch.empty(N, T, dtype=torch.int32, device=device)
    owned_froms = torch.empty(N, T, dtype=torch.int32, device=device)
    leaves = torch.empty(N, T, dtype=torch.int32, device=device)
    num_leaves = torch.empty(N, dtype=torch.int32, device=device)
    _gdn_tree_paths_kernel[(N,)](
        retrieve_parent_token,
        paths,
        path_lens,
        owned_froms,
        leaves,
        num_leaves,
        T,
        retrieve_parent_token.stride(0),
        retrieve_parent_token.stride(1),
        NP2_T=triton.next_power_of_2(T),
        num_warps=1,
    )
    return paths, path_lens, owned_froms, leaves, num_leaves


def gdn_tree_verify(
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    retrieve_parent_token: torch.Tensor,
    num_draft_tokens: int,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    rawv_ring: Optional[torch.Tensor] = None,
    rawk_ring: Optional[torch.Tensor] = None,
    g_ring: Optional[torch.Tensor] = None,
    beta_ring: Optional[torch.Tensor] = None,
    ring_indices: Optional[torch.Tensor] = None,
    tree_paths: Optional[tuple[torch.Tensor, ...]] = None,
) -> torch.Tensor:
    """Tree verify output, ``[1, num_tokens, HV, V]``.

    q/k: ``[(1,) num_tokens, H, K]`` and v: ``[(1,) num_tokens, HV, V]``, token
    strided (packed QKV views are fine); a/b: ``[num_tokens, HV]``. Tokens are
    request-major, ``num_draft_tokens`` per request, node 0 the root.
    ``initial_state_source``: ``[slots, HV, V, K]``, never written. The rings
    (per-layer ``[ring_slots, heads, L, dim]`` views) get each node's record at
    step = node index, row ``ring_indices[i]``. ``tree_paths`` is
    ``gdn_tree_paths(retrieve_parent_token, num_draft_tokens)``, built here
    when not given.
    """
    H, K = k.shape[-2:]
    HV, V = v.shape[-2:]
    num_tokens = k.shape[-3]
    T = num_draft_tokens
    N = num_tokens // T
    assert N * T == num_tokens, f"{num_tokens=} is not a multiple of {T=}"
    for name, x, heads, dim in (("q", q, H, K), ("k", k, H, K), ("v", v, HV, V)):
        assert x.stride(-1) == 1 and x.stride(-2) == dim, f"{name} heads not packed"
        assert x.shape[-2] == heads, name
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    assert initial_state_source.stride(-1) == 1
    BK = triton.next_power_of_2(K)
    assert BK == K, "head_k_dim must be a power of two"
    BV = min(triton.next_power_of_2(V), TREE_VERIFY_BV)
    if scale is None:
        scale = K**-0.5

    cache_ring = rawv_ring is not None
    if cache_ring:
        assert ring_indices is not None
        assert rawv_ring.dim() == 4 and rawk_ring.dim() == 4
        assert g_ring.dim() == 3 and beta_ring.dim() == 3
        max_cache_len = rawv_ring.shape[-2]
        assert max_cache_len >= T, f"ring holds {max_cache_len} < {T} nodes"
        ring_strides = (
            rawv_ring.stride(0),
            rawk_ring.stride(0),
            g_ring.stride(0),
            beta_ring.stride(0),
        )
    else:
        max_cache_len = 0
        ring_strides = (0, 0, 0, 0)

    if tree_paths is None:
        tree_paths = gdn_tree_paths(retrieve_parent_token, T)
    paths, path_lens, owned_froms, leaves, num_leaves = tree_paths

    o = q.new_empty(1, num_tokens, HV, V)
    grid = (triton.cdiv(V, BV), HV, N * T)
    _gdn_tree_path_verify_kernel[grid](
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        initial_state_source,
        initial_state_indices,
        paths,
        path_lens,
        owned_froms,
        leaves,
        num_leaves,
        rawv_ring,
        rawk_ring,
        g_ring,
        beta_ring,
        ring_indices,
        scale,
        softplus_beta,
        softplus_threshold,
        N,
        T,
        initial_state_source.stride(0),
        q.stride(-3),
        k.stride(-3),
        v.stride(-3),
        a.stride(-2),
        b.stride(-2),
        *ring_strides,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        MAX_CACHE_LEN=max_cache_len,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        CACHE_RING=cache_ring,
        num_warps=TREE_VERIFY_NUM_WARPS,
        num_stages=3,
    )
    return o
