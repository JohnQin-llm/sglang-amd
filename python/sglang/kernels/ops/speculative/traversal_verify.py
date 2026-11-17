"""Lossless tree verification by Traversal Verification (arXiv 2505.12398), in Triton.

Same tree contract as ``multi_draft_sampling``: each node's children are drawn one at a
time without replacement from the node's draft q over K candidates and are listed in
drawing order. Verification walks the tree depth-first, first child first, and decides
at a node only after all of its children were rejected, so accepting a node accepts its
whole path (sequence-level acceptance; on a chain this is block verification).

Every node keeps an unnormalized residual R = p_node * target, where p_node is the
probability left for accepting a path that ends at or below the node:
  - a child with token t starts with p_child = min(1, R(t) / q(t)), q renormalized over
    the node's not-yet-rejected children (the distribution the child was drawn from);
  - once a child's whole subtree is rejected, R <- max(R - q, 0) / (1 - sum(min(R, q)))
    and t leaves q;
  - with no children left, the node is accepted with probability sum(R) (the root
    always), and the bonus token is drawn from R / sum(R).
At the root sum(R) stays 1 and the update is recursive rejection sampling; below it the
missing 1 - sum(R) is the mass of rejecting the node itself.

Lossless for any builder that picks where the next node goes before drawing it and adds
it as its parent's last child (``build_dflash_sampled_tree_proposal``): averaged over
that draw, adding the node leaves the output distribution unchanged.

Only the K candidates of a node change R, so it is kept as their values plus one scale
for every other token.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _node_draft(
    node_tokens, node_probs, tp_node, stride_tp_v, node_row, K: tl.constexpr
):
    ranks = tl.arange(0, K)
    tokens = tl.load(node_tokens + node_row * K + ranks)
    q = tl.load(node_probs + node_row * K + ranks).to(tl.float32)
    p = tl.load(tp_node + tokens * stride_tp_v).to(tl.float32)
    return tokens, q, p, tl.sum(p, axis=0)


@triton.jit
def _undrawn_q(q, drawn, K: tl.constexpr):
    """q renormalized over the candidates no rejected child has taken."""
    ranks = tl.arange(0, K)
    q_avail = tl.where(((drawn >> ranks) & 1) == 0, q, 0.0)
    return q_avail / tl.maximum(tl.sum(q_avail, axis=0), 1e-20)


@triton.jit
def _traversal_verify_kernel(
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
    node_tokens_ptr,
    node_probs_ptr,
    stride_tp_b,
    stride_tp_n,
    stride_tp_v,
    NUM_SPEC_TOKENS: tl.constexpr,
    NUM_DRAFT_TOKENS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    K: tl.constexpr,
    STACK: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * NUM_DRAFT_TOKENS
    tp_row = target_probs_ptr + pid.to(tl.int64) * stride_tp_b
    ranks = tl.arange(0, K)
    levels = tl.arange(0, STACK)

    # Ancestors of the current node, one level each: the node, the child being
    # verified under it, and the node's residual (w on candidates, alpha elsewhere).
    stack_node = tl.zeros((STACK,), dtype=tl.int32)
    stack_child = tl.zeros((STACK,), dtype=tl.int32)
    stack_w = tl.zeros((STACK, K), dtype=tl.float32)
    stack_alpha = tl.zeros((STACK,), dtype=tl.float32)
    stack_drawn = tl.zeros((STACK,), dtype=tl.int32)
    stack_cand_mass = tl.zeros((STACK,), dtype=tl.float32)

    node = 0
    level = 0
    tokens, q, w, cand_mass = _node_draft(
        node_tokens_ptr, node_probs_ptr, tp_row, stride_tp_v, row, K
    )
    alpha = tl.full((), 1.0, tl.float32)
    drawn = tl.full((), 0, tl.int32)
    child = tl.load(retrive_next_token_ptr + row).to(tl.int32)
    done = 0
    while done == 0:
        if child != -1:
            token = tl.load(candidates_ptr + row + child)
            hit = tokens == token
            q_k = _undrawn_q(q, drawn, K)
            q_token = tl.sum(tl.where(hit, q_k, 0.0), axis=0)
            r_token = tl.sum(tl.where(hit, w, 0.0), axis=0)
            p_child = tl.where(
                q_token > 0.0,
                tl.minimum(r_token / tl.maximum(q_token, 1e-20), 1.0),
                0.0,
            )
            here = levels == level
            stack_node = tl.where(here, node, stack_node)
            stack_child = tl.where(here, child, stack_child)
            stack_w = tl.where(here[:, None], w[None, :], stack_w)
            stack_alpha = tl.where(here, alpha, stack_alpha)
            stack_drawn = tl.where(here, drawn, stack_drawn)
            stack_cand_mass = tl.where(here, cand_mass, stack_cand_mass)
            node = child
            level += 1
            tokens, q, p, cand_mass = _node_draft(
                node_tokens_ptr,
                node_probs_ptr,
                tp_row + node.to(tl.int64) * stride_tp_n,
                stride_tp_v,
                row + node,
                K,
            )
            w = p * p_child
            alpha = p_child
            drawn = tl.full((), 0, tl.int32)
            child = tl.load(retrive_next_token_ptr + row + node).to(tl.int32)
        else:
            accept_prob = tl.sum(w, axis=0) + alpha * (1.0 - cand_mass)
            coin = tl.load(uniform_samples_ptr + row + node)
            if (level == 0) | (coin < accept_prob):
                done = 1
            else:
                level -= 1
                here = levels == level
                rejected = tl.sum(tl.where(here, stack_child, 0), axis=0)
                node = tl.sum(tl.where(here, stack_node, 0), axis=0)
                w = tl.sum(tl.where(here[:, None], stack_w, 0.0), axis=0)
                alpha = tl.sum(tl.where(here, stack_alpha, 0.0), axis=0)
                drawn = tl.sum(tl.where(here, stack_drawn, 0), axis=0)
                cand_mass = tl.sum(tl.where(here, stack_cand_mass, 0.0), axis=0)
                tokens = tl.load(node_tokens_ptr + (row + node) * K + ranks)
                q = tl.load(node_probs_ptr + (row + node) * K + ranks).to(tl.float32)
                q_k = _undrawn_q(q, drawn, K)
                norm = tl.maximum(1.0 - tl.sum(tl.minimum(w, q_k), axis=0), 1e-20)
                w = tl.maximum(w - q_k, 0.0) / norm
                alpha = alpha / norm
                token = tl.load(candidates_ptr + row + rejected)
                drawn = drawn | tl.sum(tl.where(tokens == token, 1 << ranks, 0), axis=0)
                child = tl.load(retrive_next_sibling_ptr + row + rejected).to(tl.int32)

    # The stack now holds the accepted path: level j's child is the token after it.
    for path_level in range(0, level):
        here = levels == path_level
        slot = tl.load(
            retrive_index_ptr + row + tl.sum(tl.where(here, stack_node, 0), axis=0)
        ).to(tl.int32)
        accepted = tl.sum(tl.where(here, stack_child, 0), axis=0)
        tl.store(
            predicts_ptr + slot, tl.load(candidates_ptr + row + accepted).to(tl.int32)
        )
        tl.store(accept_index_ptr + pid * NUM_SPEC_TOKENS + path_level, slot)
    last_slot = tl.load(retrive_index_ptr + row + node).to(tl.int32)
    tl.store(accept_index_ptr + pid * NUM_SPEC_TOKENS + level, last_slot)
    tl.store(accept_token_num_ptr + pid, level)

    # Bonus token from the accepted node's residual: candidates take w, the rest alpha * p.
    # Pass 1 finds the chunk holding u from per-chunk masses; pass 2 scans that chunk.
    tp_node = tp_row + node.to(tl.int64) * stride_tp_n
    p_cand = tl.load(tp_node + tokens * stride_tp_v).to(tl.float32)
    cand_delta = w - alpha * p_cand
    cand_chunk = (tokens // BLOCK_V).to(tl.int32)
    total = tl.sum(w, axis=0) + alpha * (1.0 - cand_mass)
    u = tl.load(uniform_samples_final_ptr + pid) * total
    chunk_ids = tl.arange(0, CHUNKS)
    sums = tl.zeros((CHUNKS,), dtype=tl.float32)
    for c in tl.range(0, NUM_CHUNKS):
        offsets = c * BLOCK_V + tl.arange(0, BLOCK_V)
        vals = tl.load(
            tp_node + offsets * stride_tp_v, mask=offsets < VOCAB_SIZE, other=0.0
        )
        mass = alpha * tl.sum(vals.to(tl.float32), axis=0)
        mass += tl.sum(tl.where(cand_chunk == c, cand_delta, 0.0), axis=0)
        sums = tl.where(chunk_ids == c, mass, sums)
    ends = tl.cumsum(sums, axis=0)
    crossing = (ends > u) & (sums > 0.0)
    last_positive = tl.max(tl.where(sums > 0.0, chunk_ids, -1), axis=0)
    chunk = tl.where(
        tl.max(crossing.to(tl.int32), axis=0) > 0,
        tl.argmax(crossing.to(tl.int32), axis=0),
        last_positive,
    )
    chunk = tl.maximum(chunk, 0)
    base = tl.sum(tl.where(chunk_ids == chunk, ends - sums, 0.0), axis=0)
    offsets = chunk * BLOCK_V + tl.arange(0, BLOCK_V)
    val = tl.load(tp_node + offsets * stride_tp_v, mask=offsets < VOCAB_SIZE, other=0.0)
    val = val.to(tl.float32) * alpha
    for j in tl.static_range(K):
        token_j = tl.sum(tl.where(ranks == j, tokens, 0), axis=0)
        w_j = tl.sum(tl.where(ranks == j, w, 0.0), axis=0)
        val = tl.where(offsets == token_j, w_j, val)
    hit = ((base + tl.cumsum(val, axis=0)) > u) & (val > 0.0)
    # u can exceed the summed mass by rounding; fall back to the last positive token.
    last_valid = tl.max(tl.where(val > 0.0, offsets, -1), axis=0)
    sampled = tl.where(
        tl.max(hit.to(tl.int32), axis=0) > 0,
        chunk * BLOCK_V + tl.argmax(hit.to(tl.int32), axis=0),
        last_valid,
    )
    sampled = tl.where(sampled < 0, VOCAB_SIZE - 1, sampled)
    tl.store(predicts_ptr + last_slot, sampled.to(tl.int32))


def tree_speculative_sampling_traversal(
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
) -> None:
    """Drop-in for ``tree_speculative_sampling_multi_draft``, same inputs and outputs.

    ``uniform_samples[b, i]`` is node ``i``'s acceptance coin; the root's is unused.
    """
    batch_size, num_draft_tokens = candidates.shape
    if batch_size == 0:
        return
    if node_draft_tokens.shape[-1] > 32:
        raise ValueError("traversal verification tracks rejected candidates in 32 bits")
    num_spec_tokens = accept_index.shape[1]
    target_probs = target_probs.view(batch_size, num_draft_tokens, -1)
    vocab_size = target_probs.shape[-1]
    block_v = 4096
    num_chunks = triton.cdiv(vocab_size, block_v)
    _traversal_verify_kernel[(batch_size,)](
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
        node_draft_tokens.contiguous(),
        node_draft_probs.float().contiguous(),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        NUM_SPEC_TOKENS=num_spec_tokens,
        NUM_DRAFT_TOKENS=num_draft_tokens,
        VOCAB_SIZE=vocab_size,
        K=node_draft_tokens.shape[-1],
        STACK=triton.next_power_of_2(num_spec_tokens),
        BLOCK_V=block_v,
        NUM_CHUNKS=num_chunks,
        CHUNKS=triton.next_power_of_2(num_chunks),
        num_warps=8,
    )


def _verify_request_reference(
    *,
    candidates: list[int],
    next_token: list[int],
    next_sibling: list[int],
    coins: list[float],
    target: torch.Tensor,
    node_tokens: torch.Tensor,
    node_probs: torch.Tensor,
) -> tuple[list[int], torch.Tensor, float]:
    """Returns the accepted node path from the root, the residual at its last node, and
    the smallest |coin - acceptance probability| over the coins it compared."""
    vocab = target.shape[-1]
    margin = math.inf

    def draft(node: int) -> torch.Tensor:
        q = torch.zeros(vocab, dtype=torch.float64)
        q.index_add_(0, node_tokens[node], node_probs[node])
        total = q.sum()
        return q / total if total > 0 else q

    def visit(node: int, p_node: float):
        nonlocal margin
        residual = p_node * target[node]
        q = draft(node)
        child = next_token[node]
        while child != -1:
            token = candidates[child]
            q_token = q[token].item()
            p_child = min(1.0, residual[token].item() / q_token) if q_token > 0 else 0.0
            accepted = visit(child, p_child)
            if accepted is not None:
                return [node] + accepted[0], accepted[1]
            norm = max(1.0 - torch.minimum(residual, q).sum().item(), 1e-300)
            residual = (residual - q).clamp(min=0) / norm
            q[token] = 0
            total = q.sum()
            q = q / total if total > 0 else q
            child = next_sibling[child]
        accept_prob = residual.sum().item()
        if node != 0:
            margin = min(margin, abs(coins[node] - accept_prob))
            if not coins[node] < accept_prob:
                return None
        return [node], residual

    path, residual = visit(0, 1.0)
    return path, residual, margin


def tree_speculative_sampling_traversal_reference(
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
) -> torch.Tensor:
    """Float64 replay of the kernel over the full vocabulary, one request at a time.

    Writes the same outputs and returns ``[bs, 2]`` margins per request: the smallest
    |coin - acceptance probability| it compared, and the bonus draw's distance to the
    edge of its token's CDF interval (relative to the residual mass), so callers can skip
    rows where FP32 rounding may legitimately flip a decision.
    """
    batch_size, num_draft_tokens = candidates.shape
    target_probs = target_probs.view(batch_size, num_draft_tokens, -1)
    margins = torch.empty(batch_size, 2, dtype=torch.float64)
    for b in range(batch_size):
        path, residual, margin = _verify_request_reference(
            candidates=candidates[b].tolist(),
            next_token=retrive_next_token[b].tolist(),
            next_sibling=retrive_next_sibling[b].tolist(),
            coins=uniform_samples[b].double().tolist(),
            target=target_probs[b].double().cpu(),
            node_tokens=node_draft_tokens[b].long().cpu(),
            node_probs=node_draft_probs[b].double().cpu(),
        )
        slots = retrive_index[b, path].tolist()
        for parent_slot, child in zip(slots, path[1:]):
            predicts[parent_slot] = int(candidates[b, child])
        accept_index[b, : len(path)] = torch.tensor(
            slots, dtype=accept_index.dtype, device=accept_index.device
        )
        accept_token_num[b] = len(path) - 1

        total = residual.sum()
        u = uniform_samples_for_final_sampling[b].double().item() * total.item()
        cdf = residual.cumsum(0)
        positive = residual > 0
        above = torch.nonzero((cdf > u) & positive)
        bonus = int(above[0]) if above.numel() else int(torch.nonzero(positive)[-1])
        lower = cdf[bonus].item() - residual[bonus].item()
        bonus_margin = min(u - lower, cdf[bonus].item() - u) / total.item()
        predicts[slots[-1]] = bonus
        margins[b, 0] = margin
        margins[b, 1] = bonus_margin
    return margins
