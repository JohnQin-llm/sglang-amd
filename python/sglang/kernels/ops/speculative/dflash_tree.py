"""Best-first DFLASH draft trees from the DFlash 2 selector lattice, in EAGLE's proposal ABI.

The selector already scores every (predecessor, candidate) pair of the per-position top-K
lattice, so a child's log-probability depends on its parent's rank at the previous depth.
Selection, tie-breaking and the output encoding follow `uno_tree.py`, so the result goes
straight into `build_eagle_verify_input` with the bonus tokens as roots.
"""

from __future__ import annotations

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
    _build_dflash_tree_kernel[(batch_size,)](
        candidate_ids,
        cond_log_probs,
        draft_tokens,
        selected_edges,
        parent_list,
        buffer("search_depths", search_shape, torch.int32),
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
    return DFlashTreeProposal(
        draft_tokens=draft_tokens,
        parent_list=parent_list,
        top_scores_index=selected_edges,
    )
