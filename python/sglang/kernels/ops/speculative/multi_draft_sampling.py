"""Lossless multi-draft tree verification by recursive rejection sampling, in Triton.

Each tree node's children must be drawn one at a time without replacement from the node's
draft distribution q (K candidates). At a node the children are tried in drawing order:
child k is accepted with probability min(1, p_k(x) / q_k(x)), where q_k is q renormalized
over the not-yet-drawn candidates; a rejection replaces p_k by norm(max(p_k - q_k, 0)).
If every child is rejected, the bonus token is drawn from the last residual. This keeps
the committed tokens distributed exactly as the target (recursive rejection sampling).

Only the K candidates of a node change under these updates, so the residual is kept as
their probabilities plus one scale for every other token.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.sparse_reject_sampling import (
    lookup_many,
    row_support,
    target_at,
    walk_in_token_order,
)


@triton.jit
def _node_state(
    node_tokens,
    node_probs,
    tp_node,
    ids_node,
    stride_tp_v,
    node_row,
    K: tl.constexpr,
    KT: tl.constexpr,
    KT_PAD: tl.constexpr,
    SPARSE: tl.constexpr,
):
    ranks = tl.arange(0, K)
    tokens = tl.load(node_tokens + node_row * K + ranks)
    q = tl.load(node_probs + node_row * K + ranks).to(tl.float32)
    p = target_at(tp_node, ids_node, tokens, stride_tp_v, KT, KT_PAD, SPARSE)
    return tokens, q, p, tl.sum(p, axis=0)


@triton.jit
def _tree_multi_draft_kernel(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    retrive_index_ptr,
    retrive_next_token_ptr,
    retrive_next_sibling_ptr,
    uniform_samples_ptr,
    uniform_samples_final_ptr,
    target_probs_ptr,
    target_ids_ptr,
    node_tokens_ptr,
    node_probs_ptr,
    stride_tp_b,
    stride_tp_n,
    stride_tp_v,
    NUM_SPEC_TOKENS: tl.constexpr,
    NUM_DRAFT_TOKENS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    KT: tl.constexpr,
    KT_PAD: tl.constexpr,
    SPARSE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * NUM_DRAFT_TOKENS
    tp_row = target_probs_ptr + pid.to(tl.int64) * stride_tp_b
    # A sparse target's token ids share its probabilities' layout.
    ids_row = target_ids_ptr + pid.to(tl.int64) * stride_tp_b
    ranks = tl.arange(0, K)

    node = 0
    last_accept_slot = tl.load(retrive_index_ptr + row).to(tl.int32)
    tl.store(accept_index_ptr + pid * NUM_SPEC_TOKENS, last_accept_slot)
    num_correct_drafts = 0
    tokens, q, w, cand_mass = _node_state(
        node_tokens_ptr,
        node_probs_ptr,
        tp_row,
        ids_row,
        stride_tp_v,
        row,
        K,
        KT,
        KT_PAD,
        SPARSE,
    )
    # w: residual probability of the candidates; alpha scales every other token's p.
    one = tl.full((), 1.0, tl.float32)
    alpha = one
    drawn = tl.zeros((K,), dtype=tl.int32)
    drawn_mass = one * 0.0
    level = 1
    walking = 1
    while (level < NUM_SPEC_TOKENS) & (walking == 1):
        child = tl.load(retrive_next_token_ptr + row + node).to(tl.int32)
        accept = 0
        while (child != -1) & (accept == 0):
            token = tl.load(candidates_ptr + row + child)
            hit = tokens == token
            q_k = tl.where(drawn == 0, q, 0.0) / tl.maximum(1.0 - drawn_mass, 1e-20)
            q_token = tl.sum(tl.where(hit, q_k, 0.0), axis=0)
            p_token = tl.sum(tl.where(hit, w, 0.0), axis=0)
            coin = tl.load(uniform_samples_ptr + row + child)
            if (q_token > 0.0) & (coin * q_token < p_token):
                tl.store(predicts_ptr + last_accept_slot, token.to(tl.int32))
                num_correct_drafts += 1
                last_accept_slot = tl.load(retrive_index_ptr + row + child).to(tl.int32)
                tl.store(
                    accept_index_ptr + pid * NUM_SPEC_TOKENS + num_correct_drafts,
                    last_accept_slot,
                )
                node = child
                tokens, q, w, cand_mass = _node_state(
                    node_tokens_ptr,
                    node_probs_ptr,
                    tp_row + node.to(tl.int64) * stride_tp_n,
                    ids_row + node.to(tl.int64) * stride_tp_n,
                    stride_tp_v,
                    row + node,
                    K,
                    KT,
                    KT_PAD,
                    SPARSE,
                )
                alpha = one
                drawn = tl.zeros((K,), dtype=tl.int32)
                drawn_mass = one * 0.0
                accept = 1
            else:
                residual = tl.maximum(w - q_k, 0.0)
                norm = tl.sum(residual, axis=0) + alpha * (1.0 - cand_mass)
                norm = tl.maximum(norm, 1e-20)
                w = residual / norm
                alpha = alpha / norm
                drawn = tl.where(hit, 1, drawn)
                drawn_mass += tl.sum(tl.where(hit, q, 0.0), axis=0)
                child = tl.load(retrive_next_sibling_ptr + row + child).to(tl.int32)
        walking = accept
        level += 1
    tl.store(accept_token_num_ptr + pid, num_correct_drafts)

    # Bonus token from the residual at the last node: candidates take w, the rest alpha * p.
    tp_node = tp_row + node.to(tl.int64) * stride_tp_n
    total = tl.sum(w, axis=0) + alpha * (1.0 - cand_mass)
    u = tl.load(uniform_samples_final_ptr + pid) * total
    if SPARSE:
        ids, probs = row_support(
            ids_row + node.to(tl.int64) * stride_tp_n, tp_node, 0, KT, KT_PAD
        )
        is_cand = tl.sum((ids[:, None] == tokens[None, :]).to(tl.int32), axis=1) > 0
        val = tl.where(is_cand, lookup_many(tokens, w, ids), probs * alpha)
        _, sampled, found = walk_in_token_order(ids, val, 0.0, u, BLOCK_V)
        last_valid = tl.max(tl.where(val > 0.0, ids, -1), axis=0)
        sampled = tl.where(found == 1, sampled, last_valid)
        sampled = tl.where(sampled < 0, VOCAB_SIZE - 1, sampled)
    else:
        cum = 0.0
        sampled = VOCAB_SIZE
        last_valid = -1
        for v_start in range(0, VOCAB_SIZE, BLOCK_V):
            if sampled == VOCAB_SIZE:
                offsets = v_start + tl.arange(0, BLOCK_V)
                mask = offsets < VOCAB_SIZE
                val = (
                    tl.load(tp_node + offsets * stride_tp_v, mask=mask, other=0.0)
                    * alpha
                )
                for j in tl.static_range(K):
                    token_j = tl.sum(tl.where(ranks == j, tokens, 0), axis=0)
                    w_j = tl.sum(tl.where(ranks == j, w, 0.0), axis=0)
                    val = tl.where(offsets == token_j, w_j, val)
                hit = ((cum + tl.cumsum(val, axis=0)) > u) & (val > 0.0)
                if tl.max(hit.to(tl.int32), axis=0) > 0:
                    sampled = v_start + tl.argmax(hit.to(tl.int32), axis=0)
                last_valid = tl.maximum(
                    last_valid, tl.max(tl.where(val > 0.0, offsets, -1), axis=0)
                )
                cum += tl.sum(val)
        # u can exceed the summed mass by rounding; fall back to the last positive token.
        if sampled == VOCAB_SIZE:
            sampled = tl.where(last_valid == -1, VOCAB_SIZE - 1, last_valid)
    tl.store(predicts_ptr + last_accept_slot, sampled.to(tl.int32))


def tree_speculative_sampling_multi_draft(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    node_draft_tokens: torch.Tensor,
    node_draft_probs: torch.Tensor,
    *,
    target_token_ids: Optional[torch.Tensor] = None,
    vocab_size: Optional[int] = None,
) -> None:
    """Same outputs as ``tree_speculative_sampling_target_only``; ``node_draft_tokens`` and
    ``node_draft_probs`` are ``[bs, num_draft_tokens, K]``: each node's K child candidates
    and their q, with the node's children listed in the tree in drawing order. A sparse
    target (``target_token_ids``, ``vocab_size``) is as in
    ``tree_speculative_sampling_traversal``."""
    batch_size, num_draft_tokens = candidates.shape
    if batch_size == 0:
        return
    target_probs = target_probs.view(batch_size, num_draft_tokens, -1)
    sparse = target_token_ids is not None
    if sparse:
        target_probs = target_probs.contiguous()
        target_token_ids = target_token_ids.view(target_probs.shape).contiguous()
    else:
        vocab_size = target_probs.shape[-1]
        target_token_ids = target_probs
    support = target_probs.shape[-1] if sparse else 1
    _tree_multi_draft_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates.contiguous(),
        retrive_index.contiguous(),
        retrive_next_token.contiguous(),
        retrive_next_sibling.contiguous(),
        uniform_samples.contiguous(),
        uniform_samples_for_final_sampling,
        target_probs,
        target_token_ids,
        node_draft_tokens.contiguous(),
        node_draft_probs.float().contiguous(),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        NUM_SPEC_TOKENS=accept_index.shape[1],
        NUM_DRAFT_TOKENS=num_draft_tokens,
        VOCAB_SIZE=vocab_size,
        K=node_draft_tokens.shape[-1],
        BLOCK_V=4096,
        KT=support,
        KT_PAD=triton.next_power_of_2(support),
        SPARSE=sparse,
        num_warps=8,
    )
