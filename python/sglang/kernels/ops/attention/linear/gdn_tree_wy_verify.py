# SPDX-License-Identifier: Apache-2.0
"""GDN tree verify in the chunked (UT/WY) form.

One program covers every node of one request's draft tree for one value head
and V block, so the committed state is read once per request instead of once
per root-to-leaf path (``gdn_tree_verify``). Along a path the gated delta rule
is the chain chunk algorithm of ``gdn_replayssm_spec_circular_kernel``; on a
tree the gate prefix sums run over each node's ancestors (G_t = G_parent + g_t)
and the strictly-lower causal mask becomes the strict-ancestor mask:

    u_t = beta_t (v_t - e^{G_t} S k_t - sum_{j in anc(t)} e^{G_t - G_j} (k_j.k_t) u_j)
    o_t = e^{G_t} S q_t + sum_{j in anc(t) + t} e^{G_t - G_j} (k_j.q_t) u_j

Parents precede children, so the u system is lower-triangular in node order.
The ring takes each node's raw inputs exactly as ``gdn_tree_verify`` stores
them, so the accepted-path commit fold (``gdn_replayssm_spec_fold``) and the
committed state are unchanged; only the verify outputs differ, by fp32
rounding.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _gdn_tree_ancestors_kernel(
    retrieve_parent_token,
    ancestors,
    T,
    stride_parent_seq,
    stride_parent_token,
    NP2_T: tl.constexpr,
):
    i_n = tl.program_id(0)
    o_t = tl.arange(0, NP2_T)
    parents = tl.load(
        retrieve_parent_token + i_n * stride_parent_seq + o_t * stride_parent_token,
        mask=o_t < T,
        other=0,
    ).to(tl.int32)
    # Same reading as gdn_tree_paths: a link that does not point to an earlier
    # node (CUDA graph pad rows) is the previous node, so every row is a tree.
    parents = tl.where((parents >= 0) & (parents < o_t), parents, o_t - 1)
    bits = tl.zeros([NP2_T], dtype=tl.int64)
    for i in range(0, T):
        parent = tl.sum(tl.where(o_t == i, parents, 0))
        parent_bits = tl.sum(tl.where(o_t == parent, bits, 0))
        node_bit = tl.full((), 1, tl.int64) << i.to(tl.int64)
        bits = tl.where(o_t == i, parent_bits | node_bit, bits)
    tl.store(ancestors + i_n * T + o_t, bits, mask=o_t < T)


@triton.jit(do_not_specialize=["T"])
def _gdn_tree_wy_verify_kernel(
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
    ancestors,
    rawv_ring,
    rawk_ring,
    g_ring,
    beta_ring,
    ring_indices,
    scale,
    softplus_beta,
    softplus_threshold,
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
    BV: tl.constexpr,
    BS: tl.constexpr,
    BKT: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    CACHE_RING: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)

    o_v = i_v * BV + tl.arange(0, BV)
    o_s = tl.arange(0, BS)
    mask_v = o_v < V
    mask_s = o_s < T
    # Pad lanes (BS > T) read the request's root; zero gates and an empty
    # ancestor set keep them out of every real node.
    tok = i_n.to(tl.int64) * T + tl.where(mask_s, o_s, 0)

    # Same expressions as gdn_tree_verify, so the ring records match it bitwise.
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    b_a = tl.load(a + tok * stride_a + i_hv).to(tl.float32)
    b_b = tl.load(b + tok * stride_b + i_hv).to(tl.float32)
    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = tl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    b_g = tl.where(mask_s, -tl.exp(b_A_log) * softplus_x, 0.0)
    b_beta = tl.where(mask_s, 1.0 / (1.0 + tl.exp(-b_b)), 0.0)

    # on_path[t, j]: node j is t or an ancestor of t.
    b_anc = tl.load(ancestors + i_n * T + o_s, mask=mask_s, other=0)
    on_path = ((b_anc[:, None] >> o_s[None, :].to(tl.int64)) & 1) != 0
    strict_anc = on_path & (o_s[:, None] != o_s[None, :])
    # fp64 prefix sums: on deep paths |G| reaches ~20, and its fp32 rounding
    # alone gave 7-10x path replay's output error on 24-32 node chains.
    b_G = tl.sum(tl.where(on_path, b_g.to(tl.float64)[None, :], 0.0), axis=1)
    decay = tl.where(on_path, tl.exp((b_G[:, None] - b_G[None, :]).to(tl.float32)), 0.0)
    exp_G = tl.exp(b_G).to(tl.float32)

    state_idx = tl.load(h0_indices + i_n).to(tl.int64)
    if CACHE_RING:
        ring_slot = tl.load(ring_indices + i_n).to(tl.int64)
        write_k = (ring_slot >= 0) & (i_v == 0) & (i_hv % (HV // H) == 0)

    # Raw (pre-norm) dot products; the L2 norms are row scales applied after
    # the K loop, so q and k are read once.
    kk = tl.zeros([BS, BS], dtype=tl.float32)
    qk = tl.zeros([BS, BS], dtype=tl.float32)
    h_q = tl.zeros([BV, BS], dtype=tl.float32)
    h_k = tl.zeros([BV, BS], dtype=tl.float32)
    q_sq = tl.zeros([BS], dtype=tl.float32)
    k_sq = tl.zeros([BS], dtype=tl.float32)
    for i_k in tl.static_range(K // BKT):
        o_k = i_k * BKT + tl.arange(0, BKT)
        b_q = tl.load(q + tok[:, None] * stride_q + i_h * K + o_k[None, :]).to(
            tl.float32
        )
        b_k = tl.load(k + tok[:, None] * stride_k + i_h * K + o_k[None, :]).to(
            tl.float32
        )
        if CACHE_RING:
            if write_k:
                tl.store(
                    rawk_ring
                    + ring_slot * stride_rawk_slot
                    + (i_h * MAX_CACHE_LEN + o_s[:, None]) * K
                    + o_k[None, :],
                    b_k.to(rawk_ring.dtype.element_ty),
                    mask=mask_s[:, None],
                )
        q_sq += tl.sum(b_q * b_q, axis=1)
        k_sq += tl.sum(b_k * b_k, axis=1)
        k_t = tl.trans(b_k)
        kk += tl.dot(b_k, k_t, input_precision="ieee")
        qk += tl.dot(b_q, k_t, input_precision="ieee")
        if state_idx >= 0:
            b_h = tl.load(
                h0_source
                + state_idx * stride_h0_slot
                + i_hv * V * K
                + o_v[:, None] * K
                + o_k[None, :],
                mask=mask_v[:, None],
                other=0.0,
            ).to(tl.float32)
            h_q += tl.dot(b_h, tl.trans(b_q), input_precision="ieee")
            h_k += tl.dot(b_h, k_t, input_precision="ieee")

    if USE_QK_L2NORM_IN_KERNEL:
        q_scale = scale / tl.sqrt(q_sq + 1e-6)
        k_scale = 1.0 / tl.sqrt(k_sq + 1e-6)
    else:
        q_scale = tl.zeros([BS], dtype=tl.float32) + scale
        k_scale = tl.zeros([BS], dtype=tl.float32) + 1.0
    kk = kk * k_scale[:, None] * k_scale[None, :]
    qk = qk * q_scale[:, None] * k_scale[None, :]
    h_q = h_q * q_scale[None, :]
    h_k = h_k * k_scale[None, :]

    # (I + A)^-1 = I + X with X = -A - A X. Row t of X reads only the rows of
    # t's ancestors, so each pass makes one more tree level exact and the tree
    # depth (at most 7 for DFLASH, not BS) of passes solves it.
    A = tl.where(strict_anc, b_beta[:, None] * decay * kk, 0.0)
    depth = tl.max(tl.sum(on_path.to(tl.int32), axis=1)) - 1
    inv = -A
    for _ in range(1, depth):
        inv = -A - tl.dot(A, inv, input_precision="ieee")
    inv += (o_s[:, None] == o_s[None, :]).to(tl.float32)

    # [BV, BS] tiles hold one node per column.
    b_v = tl.load(
        v + tok[None, :] * stride_v + i_hv * V + o_v[:, None],
        mask=mask_v[:, None],
        other=0.0,
    ).to(tl.float32)
    rhs = b_beta[None, :] * (b_v - exp_G[None, :] * h_k)
    u = tl.dot(rhs, tl.trans(inv), input_precision="ieee")
    b_o = exp_G[None, :] * h_q + tl.dot(u, tl.trans(decay * qk), input_precision="ieee")
    tl.store(
        o + (tok[None, :] * HV + i_hv) * V + o_v[:, None],
        b_o.to(o.dtype.element_ty),
        mask=mask_v[:, None] & mask_s[None, :],
    )

    if CACHE_RING:
        if ring_slot >= 0:
            tl.store(
                rawv_ring
                + ring_slot * stride_rawv_slot
                + (i_hv * MAX_CACHE_LEN + o_s[None, :]) * V
                + o_v[:, None],
                b_v.to(rawv_ring.dtype.element_ty),
                mask=mask_v[:, None] & mask_s[None, :],
            )
            if i_v == 0:
                ring_pos = i_hv * MAX_CACHE_LEN + o_s
                tl.store(
                    g_ring + ring_slot * stride_g_slot + ring_pos, b_g, mask=mask_s
                )
                tl.store(
                    beta_ring + ring_slot * stride_beta_slot + ring_pos,
                    b_beta,
                    mask=mask_s,
                )


def _default_block_v(num_programs_per_block: int, V: int) -> int:
    # Measured on MI355X at Qwen3.8 shapes: past ~1024 one-warp programs per
    # launch a second wave starts and the step time jumps, so V is split only
    # while the grid stays under that.
    for block_v in (32, 64):
        if num_programs_per_block * triton.cdiv(V, block_v) <= 1024:
            return block_v
    return triton.next_power_of_2(V)


def gdn_tree_ancestors(
    retrieve_parent_token: torch.Tensor, num_draft_tokens: int
) -> torch.Tensor:
    """``[N, T]`` int64 bitmask per node of itself and its ancestors, read by
    ``gdn_tree_wy_verify``; all layers of one verify share it."""
    T = num_draft_tokens
    assert T <= 64, "ancestor sets are int64 bitmasks"
    N = retrieve_parent_token.shape[0]
    ancestors = torch.empty(
        N, T, dtype=torch.int64, device=retrieve_parent_token.device
    )
    _gdn_tree_ancestors_kernel[(N,)](
        retrieve_parent_token,
        ancestors,
        T,
        retrieve_parent_token.stride(0),
        retrieve_parent_token.stride(1),
        NP2_T=triton.next_power_of_2(T),
        num_warps=1,
    )
    return ancestors


def gdn_tree_wy_verify(
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
    tree_ancestors: Optional[torch.Tensor] = None,
    block_v: Optional[int] = None,
    block_k: int = 64,
    num_warps: int = 1,
) -> torch.Tensor:
    """Drop-in for ``gdn_tree_verify`` (same tensors, ring records and output
    ``[1, num_tokens, HV, V]``). ``tree_ancestors`` is
    ``gdn_tree_ancestors(retrieve_parent_token, num_draft_tokens)``, built here
    when not given."""
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
    assert K % block_k == 0, f"{K=} is not a multiple of {block_k=}"
    if block_v is None:
        block_v = _default_block_v(N * HV, V)
    BV = min(triton.next_power_of_2(V), block_v)
    # tl.dot needs at least 4 rows on gfx950 and sm80+.
    BS = max(4, triton.next_power_of_2(T))
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

    if tree_ancestors is None:
        tree_ancestors = gdn_tree_ancestors(retrieve_parent_token, T)

    o = q.new_empty(1, num_tokens, HV, V)
    grid = (triton.cdiv(V, BV), N, HV)
    _gdn_tree_wy_verify_kernel[grid](
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
        tree_ancestors,
        rawv_ring,
        rawk_ring,
        g_ring,
        beta_ring,
        ring_indices,
        scale,
        softplus_beta,
        softplus_threshold,
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
        BV=BV,
        BS=BS,
        BKT=block_k,
        MAX_CACHE_LEN=max_cache_len,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        CACHE_RING=cache_ring,
        num_warps=num_warps,
        num_stages=1,
    )
    return o
