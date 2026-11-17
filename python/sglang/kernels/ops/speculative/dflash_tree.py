"""Best-first DFLASH draft trees from the DFlash 2 selector lattice, in EAGLE's proposal ABI.

The selector already scores every (predecessor, candidate) pair of the per-position top-K
lattice, so a child's log-probability depends on its parent's rank at the previous depth.
Selection, tie-breaking and the output encoding follow `uno_tree.py`, so the result goes
straight into `build_eagle_verify_input` with the bonus tokens as roots.
"""

from __future__ import annotations

import math

import msgspec
import torch
import triton
import triton.language as tl
from torch import Tensor


class DFlashTreeProposal(msgspec.Struct, frozen=True):
    """EAGLE's proposal ABI: for non-root node ``i``, ``top_scores_index[:, i - 1]`` is the
    edge ``parent_node * K + rank``, and ``parent_list`` maps a candidate row to its edge."""

    draft_tokens: Tensor
    parent_list: Tensor
    top_scores_index: Tensor


@triton.jit
def _build_dflash_tree_kernel(
    candidate_ids,
    cond_log_probs,
    draft_tokens,
    selected_edges,
    parent_list,
    search_depths,
    search_ranks,
    search_log_masses,
    CAND_BATCH_STRIDE: tl.constexpr,
    CAND_DEPTH_STRIDE: tl.constexpr,
    CAND_RANK_STRIDE: tl.constexpr,
    LP_BATCH_STRIDE: tl.constexpr,
    LP_DEPTH_STRIDE: tl.constexpr,
    LP_PARENT_STRIDE: tl.constexpr,
    LP_RANK_STRIDE: tl.constexpr,
    PARENT_BATCH_STRIDE: tl.constexpr,
    NUM_DEPTHS: tl.constexpr,
    K: tl.constexpr,
    Q: tl.constexpr,
    BLOCK_CANDIDATES: tl.constexpr,
):
    batch = tl.program_id(0)
    search_offset = batch * Q
    edge_output_offset = batch * (Q - 1)
    parent_output_offset = batch * PARENT_BATCH_STRIDE
    candidate_slots = tl.arange(0, BLOCK_CANDIDATES)
    candidate_parents = candidate_slots // K
    candidate_ranks = candidate_slots % K
    candidate_in_bounds = candidate_slots < Q * K
    used = tl.zeros((BLOCK_CANDIDATES,), dtype=tl.int1)

    tl.store(
        parent_list + parent_output_offset + candidate_slots,
        -1,
        mask=candidate_slots < PARENT_BATCH_STRIDE,
    )
    # The root's rank only indexes depth 0, whose scores do not depend on the predecessor.
    tl.store(search_depths + search_offset, 0)
    tl.store(search_ranks + search_offset, 0)
    tl.store(search_log_masses + search_offset, 0.0)
    tl.debug_barrier()

    for node_index in range(1, Q):
        parent_valid = candidate_in_bounds & (candidate_parents < node_index)
        safe_parent = tl.where(parent_valid, candidate_parents, 0)
        parent_depth = tl.load(
            search_depths + search_offset + safe_parent,
            mask=parent_valid,
            other=NUM_DEPTHS,
        )
        parent_rank = tl.load(
            search_ranks + search_offset + safe_parent, mask=parent_valid, other=0
        )
        parent_mass = tl.load(
            search_log_masses + search_offset + safe_parent,
            mask=parent_valid,
            other=-float("inf"),
        ).to(tl.float32)
        valid = parent_valid & (parent_depth < NUM_DEPTHS) & ~used
        safe_depth = tl.where(valid, parent_depth, 0)
        token = tl.load(
            candidate_ids
            + batch * CAND_BATCH_STRIDE
            + safe_depth * CAND_DEPTH_STRIDE
            + candidate_ranks * CAND_RANK_STRIDE,
            mask=valid,
            other=0,
        )
        log_prob = tl.load(
            cond_log_probs
            + batch * LP_BATCH_STRIDE
            + safe_depth * LP_DEPTH_STRIDE
            + parent_rank * LP_PARENT_STRIDE
            + candidate_ranks * LP_RANK_STRIDE,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        mass = parent_mass + log_prob
        child_depth = parent_depth + 1

        # Deterministic best-first order: mass, shallower depth, lower rank,
        # lower token ID, then lower parent node.
        best_mass = tl.max(tl.where(valid, mass, -float("inf")), axis=0)
        winner = valid & (mass == best_mass)
        best_depth = tl.min(tl.where(winner, child_depth, 1 << 30), axis=0)
        winner &= child_depth == best_depth
        best_rank = tl.min(tl.where(winner, candidate_ranks, 1 << 30), axis=0)
        winner &= candidate_ranks == best_rank
        best_token = tl.min(tl.where(winner, token, 1 << 30), axis=0)
        winner &= token == best_token
        best_parent = tl.min(tl.where(winner, candidate_parents, 1 << 30), axis=0)
        winner &= candidate_parents == best_parent
        best_slot = tl.min(tl.where(winner, candidate_slots, 1 << 30), axis=0)

        selected_parent = best_slot // K
        selected_rank = best_slot % K
        selected_depth = tl.load(search_depths + search_offset + selected_parent)
        selected_parent_rank = tl.load(search_ranks + search_offset + selected_parent)
        selected_mass = tl.load(search_log_masses + search_offset + selected_parent).to(
            tl.float32
        ) + tl.load(
            cond_log_probs
            + batch * LP_BATCH_STRIDE
            + selected_depth * LP_DEPTH_STRIDE
            + selected_parent_rank * LP_PARENT_STRIDE
            + selected_rank * LP_RANK_STRIDE
        ).to(tl.float32)
        selected_token = tl.load(
            candidate_ids
            + batch * CAND_BATCH_STRIDE
            + selected_depth * CAND_DEPTH_STRIDE
            + selected_rank * CAND_RANK_STRIDE
        )

        output_index = edge_output_offset + node_index - 1
        tl.store(draft_tokens + output_index, selected_token)
        tl.store(selected_edges + output_index, best_slot)
        if node_index < Q - 1:
            tl.store(parent_list + parent_output_offset + node_index, best_slot)
        tl.store(search_depths + search_offset + node_index, selected_depth + 1)
        tl.store(search_ranks + search_offset + node_index, selected_rank)
        tl.store(search_log_masses + search_offset + node_index, selected_mass)
        used |= candidate_slots == best_slot
        tl.debug_barrier()


@torch.inference_mode()
def build_dflash_tree_proposal(
    candidate_ids: Tensor,
    cond_log_probs: Tensor,
    *,
    max_nodes: int,
    workspace: dict[str, Tensor] | None = None,
) -> DFlashTreeProposal:
    """Pick the ``max_nodes - 1`` best draft nodes of the selector lattice.

    ``candidate_ids`` is ``[B, L, K]`` and
    ``cond_log_probs`` is ``[B, L, K, K]``: the log-probability of candidate rank ``r`` at
    depth ``d`` after the candidate of rank ``p`` at depth ``d - 1``. Depth 0 ignores ``p``.
    """
    return _build_best_first_tree(
        candidate_ids, cond_log_probs, max_nodes=max_nodes, workspace=workspace
    )[0]


@torch.inference_mode()
def build_dflash_greedy_tree_proposal(
    candidate_ids: Tensor,
    cond_log_probs: Tensor,
    *,
    max_nodes: int,
    workspace: dict[str, Tensor] | None = None,
) -> tuple[DFlashTreeProposal, Tensor, Tensor]:
    """The best-first tree with ``build_dflash_sampled_tree_proposal``'s node drafts, for
    greedy rows in a batch that verifies sampled trees: each node's K child candidates
    with a uniform q.

    Against a greedy row's one-hot target, multi-draft verification accepts a child iff
    it is the argmax, as greedy verification does, for any q > 0 on the drafted children.
    A uniform q keeps the verifiers' without-replacement renormalization exact; the
    selector's q can lose the undrawn mass of a peaked node to float32 cancellation.
    """
    proposal, node_depths = _build_best_first_tree(
        candidate_ids, cond_log_probs, max_nodes=max_nodes, workspace=workspace
    )
    batch_size, num_depths, candidate_top_k = candidate_ids.shape
    rows = torch.arange(batch_size, device=candidate_ids.device)[:, None]
    depths = node_depths.long().clamp(max=num_depths - 1)
    node_draft_tokens = candidate_ids[rows, depths]
    has_children = (node_depths < num_depths).unsqueeze(-1).float()
    node_draft_probs = has_children.expand(-1, -1, candidate_top_k) / candidate_top_k
    return proposal, node_draft_tokens, node_draft_probs


def _build_best_first_tree(
    candidate_ids: Tensor,
    cond_log_probs: Tensor,
    *,
    max_nodes: int,
    workspace: dict[str, Tensor] | None,
) -> tuple[DFlashTreeProposal, Tensor]:
    """The proposal and each node's depth ``[B, Q]``, 0 at the root."""
    batch_size, num_depths, candidate_top_k = candidate_ids.shape
    if cond_log_probs.shape != (
        batch_size,
        num_depths,
        candidate_top_k,
        candidate_top_k,
    ):
        raise ValueError(
            "cond_log_probs must be [B, L, K, K] matching candidate_ids "
            f"{tuple(candidate_ids.shape)}, got {tuple(cond_log_probs.shape)}"
        )
    parent_width = candidate_top_k * max(num_depths - 1, 0) + 1
    if max_nodes < 2 or max_nodes - 1 > parent_width:
        raise ValueError(
            f"EAGLE's parent-list ABI holds 2 <= Q <= K*(depth-1)+2 nodes; got Q={max_nodes}, "
            f"K={candidate_top_k}, depth={num_depths}"
        )
    if max_nodes > 128 or max_nodes * candidate_top_k > 4096:
        raise ValueError(
            f"the single-program builder needs Q <= 128 and Q*K <= 4096; "
            f"got Q={max_nodes}, K={candidate_top_k}"
        )

    def buffer(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        value = None if workspace is None else workspace.get(name)
        if value is None or value.shape != shape or value.dtype != dtype:
            value = torch.empty(shape, dtype=dtype, device=candidate_ids.device)
            if workspace is not None:
                workspace[name] = value
        return value

    edge_shape = (batch_size, max_nodes - 1)
    draft_tokens = buffer("draft_tokens", edge_shape, torch.long)
    selected_edges = buffer("top_scores_index", edge_shape, torch.long)
    parent_list = buffer("parent_list", (batch_size, parent_width), torch.long)
    search_shape = (batch_size, max_nodes)
    search_depths = buffer("search_depths", search_shape, torch.int32)
    _build_dflash_tree_kernel[(batch_size,)](
        candidate_ids,
        cond_log_probs,
        draft_tokens,
        selected_edges,
        parent_list,
        search_depths,
        buffer("search_ranks", search_shape, torch.int32),
        buffer("search_log_masses", search_shape, torch.float32),
        CAND_BATCH_STRIDE=candidate_ids.stride(0),
        CAND_DEPTH_STRIDE=candidate_ids.stride(1),
        CAND_RANK_STRIDE=candidate_ids.stride(2),
        LP_BATCH_STRIDE=cond_log_probs.stride(0),
        LP_DEPTH_STRIDE=cond_log_probs.stride(1),
        LP_PARENT_STRIDE=cond_log_probs.stride(2),
        LP_RANK_STRIDE=cond_log_probs.stride(3),
        PARENT_BATCH_STRIDE=parent_list.stride(0),
        NUM_DEPTHS=num_depths,
        K=candidate_top_k,
        Q=max_nodes,
        BLOCK_CANDIDATES=triton.next_power_of_2(max_nodes * candidate_top_k),
        num_warps=8,
        num_stages=1,
    )
    proposal = DFlashTreeProposal(
        draft_tokens=draft_tokens,
        parent_list=parent_list,
        top_scores_index=selected_edges,
    )
    return proposal, search_depths


def select_tree_rows(
    take_first: Tensor,
    first: tuple[DFlashTreeProposal, Tensor, Tensor],
    second: tuple[DFlashTreeProposal, Tensor, Tensor],
) -> tuple[DFlashTreeProposal, Tensor, Tensor]:
    """Per row, the tree of ``first`` where ``take_first`` ([B] bool) holds, else that of
    ``second``; each is a proposal with its node draft tokens and probs, of equal Q and K."""
    first_proposal, first_tokens, first_probs = first
    second_proposal, second_tokens, second_probs = second
    rows = take_first.view(-1, 1)
    proposal = DFlashTreeProposal(
        draft_tokens=torch.where(
            rows, first_proposal.draft_tokens, second_proposal.draft_tokens
        ),
        parent_list=torch.where(
            rows, first_proposal.parent_list, second_proposal.parent_list
        ),
        top_scores_index=torch.where(
            rows, first_proposal.top_scores_index, second_proposal.top_scores_index
        ),
    )
    node_rows = take_first.view(-1, 1, 1)
    return (
        proposal,
        torch.where(node_rows, first_tokens, second_tokens),
        torch.where(node_rows, first_probs, second_probs),
    )


@triton.jit
def _draw_child(cond_log_q, base, drawn, u, K: tl.constexpr):
    """Draw one not-yet-drawn candidate from q (a row of K log-probs); -1 when none is left."""
    ranks = tl.arange(0, K)
    log_q = tl.load(cond_log_q + base + ranks).to(tl.float32)
    avail = ((drawn >> ranks) & 1) == 0
    q = tl.where(avail, tl.exp(log_q), 0.0)
    total = tl.sum(q, axis=0)
    hit = avail & (tl.cumsum(q, axis=0) > u * total)
    rank = tl.min(tl.where(hit, ranks, K), axis=0)
    # Rounding can leave u * total above the last cumsum: take the last available rank.
    rank = tl.where(rank == K, tl.max(tl.where(avail, ranks, -1), axis=0), rank)
    rank = tl.where(total > 0.0, rank, -1)
    child_log_q = tl.sum(tl.where(ranks == rank, log_q, 0.0), axis=0)
    return rank, tl.where(rank >= 0, child_log_q, -float("inf"))


@triton.jit
def _max_undrawn_log_q(cond_log_q, base, drawn, K: tl.constexpr):
    """Largest log q among a row's not-yet-drawn candidates; -inf when none is left."""
    ranks = tl.arange(0, K)
    log_q = tl.load(cond_log_q + base + ranks).to(tl.float32)
    avail = ((drawn >> ranks) & 1) == 0
    return tl.max(tl.where(avail, log_q, -float("inf")), axis=0)


@triton.jit
def _draw_prior(taken, log_first, log_miss_first, log_sibling, log_miss_sibling):
    """Log-probability that a node's draw number ``taken`` (0-based) is the child
    verification accepts: the first draw is accepted with prob a, each later one with
    prob a' once every earlier sibling was rejected."""
    later = log_miss_first + (taken - 1).to(tl.float32) * log_miss_sibling + log_sibling
    return tl.where(taken == 0, log_first, later)


@triton.jit
def _build_dflash_sampled_tree_kernel(
    candidate_ids,
    cond_log_q,
    uniforms,
    draft_tokens,
    selected_edges,
    parent_list,
    node_depths,
    node_ranks,
    log_priors,
    CAND_BATCH_STRIDE: tl.constexpr,
    CAND_DEPTH_STRIDE: tl.constexpr,
    LQ_BATCH_STRIDE: tl.constexpr,
    LQ_DEPTH_STRIDE: tl.constexpr,
    LQ_PARENT_STRIDE: tl.constexpr,
    PARENT_BATCH_STRIDE: tl.constexpr,
    NUM_DEPTHS: tl.constexpr,
    K: tl.constexpr,
    Q: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    ACCEPT_PRIOR: tl.constexpr,
):
    batch = tl.program_id(0)
    nodes = tl.arange(0, BLOCK_NODES)
    lq_row = batch * LQ_BATCH_STRIDE
    u_row = uniforms + batch * Q * K

    tl.store(
        parent_list + batch * PARENT_BATCH_STRIDE + nodes,
        -1,
        mask=nodes < PARENT_BATCH_STRIDE,
    )
    tl.store(node_depths + batch * Q, 0)
    tl.store(node_ranks + batch * Q, 0)

    # Per-node state; node 0 is the root (the bonus token) at depth 0. A node competes
    # for the next slot with a score fixed before the draw: letting the drawn value
    # decide whether the child enters the tree would bias the children the verifier
    # treats as draws from q. The score is the node's path score plus the value of its
    # next draw: path log-q plus the largest q it could still draw, or with
    # ACCEPT_PRIOR, the log-probabilities that each draw on the path is the accepted one.
    depth = tl.zeros((BLOCK_NODES,), dtype=tl.int32)
    rank = tl.zeros((BLOCK_NODES,), dtype=tl.int32)
    mass = tl.where(nodes == 0, 0.0, -float("inf"))
    taken = tl.zeros((BLOCK_NODES,), dtype=tl.int32)
    drawn = tl.zeros((BLOCK_NODES,), dtype=tl.int32)
    no_draws = tl.full((), 0, tl.int32)
    # log a, log(1 - a), log a', log(1 - a'); read only with ACCEPT_PRIOR.
    log_first = tl.load(log_priors + 0)
    log_miss_first = tl.load(log_priors + 1)
    log_sibling = tl.load(log_priors + 2)
    log_miss_sibling = tl.load(log_priors + 3)
    if ACCEPT_PRIOR:
        root_bound = _draw_prior(
            no_draws, log_first, log_miss_first, log_sibling, log_miss_sibling
        )
    else:
        root_bound = _max_undrawn_log_q(cond_log_q, lq_row, no_draws, K)
    bound = tl.where(nodes == 0, root_bound, -float("inf"))

    for node_index in range(1, Q):
        frontier = mass + bound
        best = tl.max(frontier, axis=0)
        parent = tl.min(tl.where(frontier == best, nodes, BLOCK_NODES), axis=0)
        is_parent = nodes == parent
        parent_mass = tl.sum(tl.where(is_parent, mass, 0.0), axis=0)
        parent_next = tl.sum(tl.where(is_parent, bound, 0.0), axis=0)
        parent_depth = tl.sum(tl.where(is_parent, depth, 0), axis=0)
        parent_rank = tl.sum(tl.where(is_parent, rank, 0), axis=0)
        parent_taken = tl.sum(tl.where(is_parent, taken, 0), axis=0)
        parent_drawn = tl.sum(tl.where(is_parent, drawn, 0), axis=0)
        parent_row = (
            lq_row + parent_depth * LQ_DEPTH_STRIDE + parent_rank * LQ_PARENT_STRIDE
        )
        child_rank, child_log_q = _draw_child(
            cond_log_q,
            parent_row,
            parent_drawn,
            tl.load(u_row + parent * K + parent_taken),
            K,
        )
        token = tl.load(
            candidate_ids
            + batch * CAND_BATCH_STRIDE
            + parent_depth * CAND_DEPTH_STRIDE
            + child_rank
        )
        # The k-th drawn child of a node is its k-th sibling, the order verification replays.
        edge = parent * K + parent_taken
        out = batch * (Q - 1) + node_index - 1
        tl.store(draft_tokens + out, token)
        tl.store(selected_edges + out, edge)
        if node_index < Q - 1:
            tl.store(parent_list + batch * PARENT_BATCH_STRIDE + node_index, edge)
        tl.store(node_depths + batch * Q + node_index, parent_depth + 1)
        tl.store(node_ranks + batch * Q + node_index, child_rank)

        parent_drawn = parent_drawn | (1 << child_rank)
        parent_taken += 1
        child_depth = parent_depth + 1
        if ACCEPT_PRIOR:
            parent_bound = _draw_prior(
                parent_taken, log_first, log_miss_first, log_sibling, log_miss_sibling
            )
            child_bound = _draw_prior(
                no_draws, log_first, log_miss_first, log_sibling, log_miss_sibling
            )
            child_mass = parent_mass + parent_next
        else:
            parent_bound = _max_undrawn_log_q(cond_log_q, parent_row, parent_drawn, K)
            child_bound = _max_undrawn_log_q(
                cond_log_q,
                lq_row
                + tl.minimum(child_depth, NUM_DEPTHS - 1) * LQ_DEPTH_STRIDE
                + child_rank * LQ_PARENT_STRIDE,
                no_draws,
                K,
            )
            child_mass = parent_mass + child_log_q
        taken = tl.where(is_parent, parent_taken, taken)
        drawn = tl.where(is_parent, parent_drawn, drawn)
        bound = tl.where(
            is_parent, tl.where(parent_taken < K, parent_bound, -float("inf")), bound
        )
        is_new = nodes == node_index
        depth = tl.where(is_new, child_depth, depth)
        rank = tl.where(is_new, child_rank, rank)
        mass = tl.where(is_new, child_mass, mass)
        bound = tl.where(
            is_new,
            tl.where(child_depth < NUM_DEPTHS, child_bound, -float("inf")),
            bound,
        )


def log_draw_priors(accept_prior: tuple[float, float] | None) -> tuple[float, ...]:
    if accept_prior is None:
        return (0.0, 0.0, 0.0, 0.0)
    first, sibling = accept_prior
    if not (0.0 < first < 1.0 and 0.0 < sibling < 1.0):
        raise ValueError(f"accept_prior entries must lie in (0, 1); got {accept_prior}")
    return (
        math.log(first),
        math.log1p(-first),
        math.log(sibling),
        math.log1p(-sibling),
    )


@torch.inference_mode()
def build_dflash_sampled_tree_proposal(
    candidate_ids: Tensor,
    cond_log_q: Tensor,
    *,
    max_nodes: int,
    workspace: dict[str, Tensor] | None = None,
    accept_prior: tuple[float, float] | Tensor | None = None,
) -> tuple[DFlashTreeProposal, Tensor, Tensor]:
    """Sampled tree for lossless multi-draft verification at temperature > 0.

    Every node draws its children one at a time without replacement from its own
    ``q = exp(cond_log_q[b, depth, rank])``; the next slot goes to the node with the best
    path log-q plus largest undrawn q, which then draws, and a node's children enter the
    tree in drawing order. ``accept_prior = (a, a')`` ranks slots instead by the chance
    that verification reaches and accepts them, with a node's first draw accepted with
    prob ``a`` and each later draw with prob ``a'`` after its elder siblings were
    rejected; a float32 tensor holds ``(log a, log(1 - a), log a', log(1 - a'))`` on the
    device instead, for priors updated online. Returns the proposal and, per node, the K
    candidate tokens and q of its children's distribution (rank order), which the
    verifier replays.
    """
    batch_size, num_depths, candidate_top_k = candidate_ids.shape
    if candidate_top_k > 32:
        raise ValueError("the sampled tree tracks drawn candidates in a 32-bit mask")
    parent_width = candidate_top_k * max(num_depths - 1, 0) + 1
    if max_nodes < 2 or max_nodes - 1 > parent_width or max_nodes > 128:
        raise ValueError(
            f"EAGLE's parent-list ABI holds 2 <= Q <= min(128, K*(depth-1)+2) nodes; "
            f"got Q={max_nodes}, K={candidate_top_k}, depth={num_depths}"
        )
    device = candidate_ids.device

    def buffer(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        value = None if workspace is None else workspace.get(name)
        if value is None or value.shape != shape or value.dtype != dtype:
            value = torch.empty(shape, dtype=dtype, device=device)
            if workspace is not None:
                workspace[name] = value
        return value

    edge_shape = (batch_size, max_nodes - 1)
    draft_tokens = buffer("sampled_draft_tokens", edge_shape, torch.long)
    selected_edges = buffer("sampled_top_scores_index", edge_shape, torch.long)
    parent_list = buffer("sampled_parent_list", (batch_size, parent_width), torch.long)
    node_depths = buffer("node_depths", (batch_size, max_nodes), torch.int32)
    node_ranks = buffer("node_ranks", (batch_size, max_nodes), torch.int32)
    uniforms = torch.rand(
        (batch_size, max_nodes, candidate_top_k), dtype=torch.float32, device=device
    )
    cond_log_q = cond_log_q.contiguous()
    candidate_ids = candidate_ids.contiguous()
    if isinstance(accept_prior, Tensor):
        log_priors = accept_prior
    else:
        key = f"log_draw_priors{accept_prior}"
        log_priors = None if workspace is None else workspace.get(key)
        if log_priors is None:
            log_priors = torch.tensor(
                log_draw_priors(accept_prior), dtype=torch.float32, device=device
            )
            if workspace is not None:
                workspace[key] = log_priors
    _build_dflash_sampled_tree_kernel[(batch_size,)](
        candidate_ids,
        cond_log_q,
        uniforms,
        draft_tokens,
        selected_edges,
        parent_list,
        node_depths,
        node_ranks,
        log_priors,
        CAND_BATCH_STRIDE=candidate_ids.stride(0),
        CAND_DEPTH_STRIDE=candidate_ids.stride(1),
        LQ_BATCH_STRIDE=cond_log_q.stride(0),
        LQ_DEPTH_STRIDE=cond_log_q.stride(1),
        LQ_PARENT_STRIDE=cond_log_q.stride(2),
        PARENT_BATCH_STRIDE=parent_list.stride(0),
        NUM_DEPTHS=num_depths,
        K=candidate_top_k,
        Q=max_nodes,
        BLOCK_NODES=triton.next_power_of_2(max_nodes),
        ACCEPT_PRIOR=accept_prior is not None,
        num_warps=4,
        num_stages=1,
    )
    rows = torch.arange(batch_size, device=device)[:, None]
    depths = node_depths.long().clamp(max=num_depths - 1)
    node_draft_tokens = candidate_ids[rows, depths]
    node_draft_probs = cond_log_q[rows, depths, node_ranks.long()].exp()
    node_draft_probs = node_draft_probs * (node_depths < num_depths).unsqueeze(-1)
    proposal = DFlashTreeProposal(
        draft_tokens=draft_tokens,
        parent_list=parent_list,
        top_scores_index=selected_edges,
    )
    return proposal, node_draft_tokens, node_draft_probs


@torch.inference_mode()
def update_draw_acceptance(
    *,
    stats: Tensor,
    log_priors: Tensor,
    top_scores_index: Tensor,
    accept_index: Tensor,
    accept_lens: Tensor,
    num_nodes: int,
    topk: int,
    decay: float = 0.99,
    sampled_rows: Tensor | None = None,
) -> None:
    """Online estimate of the draw priors of a sampled tree from one verify step.

    ``stats`` holds decayed counts (first-draw accepts, first draws, later-draw accepts,
    later draws); ``log_priors`` receives ``(log a, log(1 - a), log a', log(1 - a'))``.
    Along each accepted path, the child taken at draw ``s`` means draws ``1..s`` were
    later draws, the last of them accepted; at the path's end every drawn child was
    rejected. ``accept_index`` holds global node ids, root first, -1 past
    ``accept_lens``. ``sampled_rows`` ([bs] bool), when given, limits the counts to the
    rows whose tree was sampled. Runs on the device without a host sync.
    """
    bs = accept_lens.shape[0]
    edges = top_scores_index[:bs]
    draw = edges % topk
    children = torch.zeros(bs, num_nodes, dtype=torch.float32, device=edges.device)
    children.scatter_add_(1, edges // topk, torch.ones_like(edges, dtype=torch.float32))
    offsets = torch.arange(bs, device=edges.device).unsqueeze(1) * num_nodes
    nodes = accept_index[:bs].long() - offsets
    stepped = accept_index[:bs, 1:] >= 0
    if sampled_rows is not None:
        stepped = stepped & sampled_rows[:bs, None]
    taken = draw.gather(1, nodes[:, 1:].clamp(min=1) - 1)
    taken = torch.where(stepped, taken, torch.zeros_like(taken)).float()
    last = nodes.gather(1, (accept_lens.long() - 1).clamp(min=0).unsqueeze(1))
    left = children.gather(1, last.clamp(min=0)).squeeze(1)
    if sampled_rows is not None:
        left = torch.where(sampled_rows[:bs], left, torch.zeros_like(left))
    counts = torch.stack(
        [
            (stepped & (taken == 0)).sum().float(),
            stepped.sum().float() + (left > 0).sum().float(),
            (taken > 0).sum().float(),
            taken.sum() + (left - 1).clamp(min=0).sum(),
        ]
    )
    stats.mul_(decay).add_(counts)
    first = ((stats[0] + 1) / (stats[1] + 2)).clamp(0.05, 0.95)
    later = ((stats[2] + 1) / (stats[3] + 2)).clamp(0.05, 0.95)
    log_priors.copy_(
        torch.stack([first.log(), (-first).log1p(), later.log(), (-later).log1p()])
    )
