"""DFLASH tree verify: one draft block of F, a best-first tree of Q nodes, EAGLE's tree verify.

The selector already scores the per-position top-K lattice, so building the tree costs no
extra drafter compute. The target then verifies the tree through EAGLE's machinery (tree
mask, greedy or target-only sampling, accepted-path KV and hidden-state compaction,
hybrid-state commit), which leaves the accepted path at the front of the request window,
exactly where the chain path leaves it.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from typing import Any

import msgspec
import torch

from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_uniform_func,
)
from sglang.kernels.ops.speculative.dflash_tree import (
    DFlashTreeProposal,
    build_dflash_greedy_tree_proposal,
    build_dflash_sampled_tree,
    build_dflash_sampled_tree_proposal,
    build_dflash_tree_proposal,
    sampled_node_drafts,
    select_tree_rows,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.speculative.dflash_adaptive_width import AdaptiveVerifyWidth, is_chain
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_utils import default_tree_mask_mode
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    run_eagle_verify,
)


def propose_dflash_tree(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
    num_nodes: int,
    workspace: dict[str, torch.Tensor],
    calibration: torch.Tensor | None = None,
) -> DFlashTreeProposal:
    """Best-first tree over the first ``topk`` candidates of the selector lattice.

    ``candidate_ids`` is ``[B, L, K]`` and ``scores`` ``[B, L, K, K]`` (predecessor rank,
    candidate rank), with candidates in unary-logit order. The tree is ranked by
    temperature-1 selector log-probabilities; verification keeps sampling lossless.
    ``calibration`` (``update_tree_calibration``'s statistics) only changes the
    proposal's survival estimate, not the tree.
    """
    _check_topk(candidate_ids, topk)
    return build_dflash_tree_proposal(
        candidate_ids[:, :, :topk].contiguous(),
        _greedy_cond_log_probs(scores, topk),
        max_nodes=num_nodes,
        workspace=workspace,
        calibration=calibration,
    )


def propose_dflash_sampled_tree(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    temperatures: torch.Tensor,
    topk: int,
    num_nodes: int,
    workspace: dict[str, torch.Tensor],
    accept_prior: tuple[float, float] | torch.Tensor | None = None,
) -> tuple[DFlashTreeProposal, torch.Tensor, torch.Tensor]:
    """Tree drawn from the selector's q at each request's temperature, for lossless
    multi-draft verification; also returns each node's child candidates and their q."""
    lattice = sampled_lattice(
        candidate_ids=candidate_ids,
        scores=scores,
        temperatures=temperatures,
        topk=topk,
    )
    return build_dflash_sampled_tree_proposal(
        lattice.candidate_ids,
        lattice.cond_log_q,
        max_nodes=num_nodes,
        workspace=workspace,
        accept_prior=accept_prior,
    )


class SampledLattice(msgspec.Struct, frozen=True):
    """What sampled trees draw from: the first ``topk`` selector candidates ``[B, L, K]``
    and, per (depth, predecessor rank), their q at each request's temperature as
    log-probs ``[B, L, K, K]``."""

    candidate_ids: torch.Tensor
    cond_log_q: torch.Tensor


def sampled_lattice(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    temperatures: torch.Tensor,
    topk: int,
) -> SampledLattice:
    _check_topk(candidate_ids, topk)
    logits = scores[:, :, :topk, :topk].float() / (
        temperatures.view(-1, 1, 1, 1) * _Q_TEMP
    )
    return SampledLattice(
        candidate_ids=candidate_ids[:, :, :topk].contiguous(),
        cond_log_q=torch.log_softmax(logits, dim=-1),
    )


class SampledVerifyPlan(msgspec.Struct, frozen=True):
    """One step's sampled verify under the adaptive width: the chosen member, its verify
    rows, its build (proposal with survival and edges), its node drafts, and every
    member's expected accept length ``[bs, members]`` (None when nothing reads it)."""

    member: int
    rows: int
    proposal: DFlashTreeProposal
    node_draft_tokens: torch.Tensor
    node_draft_probs: torch.Tensor
    predicted: torch.Tensor | None


def propose_adaptive_sampled_tree(
    *,
    adaptive: AdaptiveVerifyWidth,
    member: int,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    temperatures: torch.Tensor,
    topk: int,
    num_nodes: int,
    workspace: dict[str, torch.Tensor],
    accept_prior: tuple[float, float] | torch.Tensor | None,
    bs: int,
) -> SampledVerifyPlan:
    """Build the sampled tree at ``num_nodes`` and, when the chain is a member, the
    chain from the same q; their survival estimates feed later steps. ``member``
    (``adaptive.choose_sampled_member``, from earlier steps only) verifies its own
    build, which keeps every node's first draws in draw order, so it stays lossless."""
    lattice = sampled_lattice(
        candidate_ids=candidate_ids,
        scores=scores,
        temperatures=temperatures,
        topk=topk,
    )
    tree = build_dflash_sampled_tree(
        lattice.candidate_ids,
        lattice.cond_log_q,
        max_nodes=num_nodes,
        workspace=workspace,
        accept_prior=accept_prior,
        calibration=adaptive.sampled_stats,
        rank_table=adaptive.rank_table,
    )
    chain = None
    if adaptive.has_chain:
        chain = build_dflash_sampled_tree(
            lattice.candidate_ids,
            lattice.cond_log_q,
            max_nodes=adaptive.chain_rows,
            workspace=workspace,
            accept_prior=accept_prior,
            chain=True,
            calibration=adaptive.sampled_stats,
        )
    predicted = adaptive.predict_sampled(
        tree_survival=tree.proposal.survival,
        chain_survival=None if chain is None else chain.proposal.survival,
        bs=bs,
    )
    # A shorter chain member verifies the first rows of the full-block chain.
    chosen = chain if is_chain(member) else tree
    rows = adaptive.member_rows(member)
    node_draft_tokens, node_draft_probs = sampled_node_drafts(
        lattice.candidate_ids, lattice.cond_log_q, chosen, rows
    )
    return SampledVerifyPlan(
        member=member,
        rows=rows,
        proposal=chosen.proposal,
        node_draft_tokens=node_draft_tokens,
        node_draft_probs=node_draft_probs,
        predicted=predicted,
    )


def propose_dflash_mixed_tree(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_rows: torch.Tensor,
    topk: int,
    num_nodes: int,
    workspace: dict[str, torch.Tensor],
    accept_prior: tuple[float, float] | torch.Tensor | None = None,
) -> tuple[DFlashTreeProposal, torch.Tensor, torch.Tensor]:
    """``propose_dflash_sampled_tree`` for a batch that mixes greedy and sampling requests:
    rows where ``greedy_rows`` holds take the best-first tree instead. Their target is
    one-hot, so multi-draft verification of that tree accepts what greedy verification
    accepts; the sampled rows keep the tree and draws they get without greedy rows."""
    sampled = propose_dflash_sampled_tree(
        candidate_ids=candidate_ids,
        scores=scores,
        temperatures=temperatures,
        topk=topk,
        num_nodes=num_nodes,
        workspace=workspace,
        accept_prior=accept_prior,
    )
    greedy = build_dflash_greedy_tree_proposal(
        candidate_ids[:, :, :topk].contiguous(),
        _greedy_cond_log_probs(scores, topk),
        max_nodes=num_nodes,
        workspace=workspace,
    )
    return select_tree_rows(greedy_rows, greedy, sampled)


def _check_topk(candidate_ids: torch.Tensor, topk: int) -> None:
    if topk > candidate_ids.shape[-1]:
        raise ValueError(
            f"--speculative-eagle-topk {topk} exceeds the selector's top-k "
            f"{candidate_ids.shape[-1]}"
        )


def _greedy_cond_log_probs(scores: torch.Tensor, topk: int) -> torch.Tensor:
    return torch.log_softmax(scores[:, :, :topk, :topk].float() / _SCORE_TEMP, dim=-1)


# Online draw priors start here and then track verify outcomes.
ONLINE_PRIOR_START = (0.7, 0.3)


def read_accept_prior_setting() -> str | tuple[float, float] | None:
    """Experiment knob SGLANG_DFLASH_TREE_ACCEPT_PRIOR: unset ranks sampled-tree slots
    by q mass, "a,a2" by fixed draw priors, "online" by priors estimated while serving."""
    value = os.environ.get("SGLANG_DFLASH_TREE_ACCEPT_PRIOR")
    if not value:
        return None
    if value == "online":
        return value
    first, sibling = (float(x) for x in value.split(","))
    return first, sibling


# Experiment knobs: the greedy tree ranks by selector log-probs at this temperature, and
# sampled trees draw (and verify) with q at the request temperature times this factor.
_SCORE_TEMP = float(os.environ.get("SGLANG_DFLASH_TREE_SCORE_TEMP") or 1.0)
_Q_TEMP = float(os.environ.get("SGLANG_DFLASH_TREE_Q_TEMP") or 1.0)


def read_tree_verify_window(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """``[bs, num_nodes]``: each request's KV slots from its committed length, the
    tree verify's write window. Read it when the step starts, as the chain reads its
    block: without a WAR barrier (HIP) the scheduler can hand a finished request's row
    to a new request, with its cached prefix, before this step's verify runs."""
    bs = req_pool_indices.shape[0]
    return assign_extend_cache_locs_uniform_func(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        start_offset=seq_lens,
        batch_size=bs,
        draft_token_num=num_nodes,
        device=req_to_token.device,
    ).view(bs, num_nodes)


def verify_dflash_tree(
    *,
    batch: ScheduleBatch,
    bonus_tokens: torch.Tensor,
    proposal: DFlashTreeProposal,
    target_worker: TpModelWorker,
    plan_stream: Any,
    plan_stream_ctx: Any,
    topk: int,
    depth: int,
    num_nodes: int,
    out_cache_loc: torch.Tensor,
    device: str,
    node_draft_tokens: torch.Tensor | None = None,
    node_draft_probs: torch.Tensor | None = None,
    target_logits: LogitsProcessor | None = None,
    argmax_from_shard: bool = False,
    host_seq_lens_bound: torch.Tensor | None = None,
    host_seq_lens_bound_sum: int | None = None,
) -> GenerationBatchResult:
    """Verify the tree rooted at each request's bonus token, which sits at the committed
    length and has no KV yet, so the verify batch keeps ``batch.seq_lens`` as is.
    ``out_cache_loc`` is the nodes' KV slots, ``num_nodes`` per request, from
    ``read_tree_verify_window`` at the start of the step.
    With ``node_draft_*`` set (a sampled tree), sampling rows verify by recursive
    rejection sampling against each node's q. ``target_logits`` is the target's
    LogitsProcessor when it defers the verify vocab gather (see ``run_eagle_verify``)."""
    verify_batch = copy.copy(batch)
    if verify_batch.seq_lens_cpu is None and host_seq_lens_bound is not None:
        # Sync-free scheduling leaves no host lengths; the backend plans the verify
        # from the allocated lengths, an upper bound, as the chain verify does.
        verify_batch.seq_lens_cpu = host_seq_lens_bound
        verify_batch.seq_lens_sum = host_seq_lens_bound_sum
    verify_input = build_eagle_verify_input(
        verify_batch,
        EagleDraftInput(bonus_tokens=bonus_tokens),
        proposal.parent_list,
        proposal.top_scores_index,
        proposal.draft_tokens,
        None,
        target_worker=target_worker,
        topk=topk,
        num_steps=depth,
        num_draft_tokens=num_nodes,
        tree_mask_mode=default_tree_mask_mode(),
        device=device,
    )
    verify_input = dataclasses.replace(
        verify_input,
        prepared_out_cache_loc=out_cache_loc,
        tree_draft_tokens=node_draft_tokens,
        tree_draft_probs=node_draft_probs,
    )
    verify_batch.spec_info = verify_input
    model_runner = target_worker.model_runner
    return run_eagle_verify(
        verify_batch,
        target_worker=target_worker,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        plan_stream=plan_stream,
        plan_stream_ctx=plan_stream_ctx,
        topk=topk,
        num_draft_tokens=num_nodes,
        device=device,
        metadata_ready_pre_pad=False,
        finalize_tree_path=True,
        deferred_logits=target_logits,
        argmax_from_shard=argmax_from_shard,
    )
