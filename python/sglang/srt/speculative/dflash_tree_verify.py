"""DFLASH tree verify: one draft block of F, a best-first tree of Q nodes, EAGLE's tree verify.

The selector already scores the per-position top-K lattice, so building the tree costs no
extra drafter compute. The target then verifies the tree through EAGLE's machinery (tree
mask, greedy or target-only sampling, accepted-path KV and hidden-state compaction,
hybrid-state commit), which leaves the accepted path at the front of the request window,
exactly where the chain path leaves it.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from sglang.kernels.ops.speculative.dflash_tree import (
    DFlashTreeProposal,
    build_dflash_tree_proposal,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
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
) -> DFlashTreeProposal:
    """Best-first tree over the first ``topk`` candidates of the selector lattice.

    ``candidate_ids`` is ``[B, L, K]`` and ``scores`` ``[B, L, K, K]`` (predecessor rank,
    candidate rank), with candidates in unary-logit order. The tree is ranked by
    temperature-1 selector log-probabilities; verification keeps sampling lossless.
    """
    if topk > candidate_ids.shape[-1]:
        raise ValueError(
            f"--speculative-eagle-topk {topk} exceeds the selector's top-k "
            f"{candidate_ids.shape[-1]}"
        )
    cond_log_probs = torch.log_softmax(scores[:, :, :topk, :topk].float(), dim=-1)
    return build_dflash_tree_proposal(
        candidate_ids[:, :, :topk].contiguous(),
        cond_log_probs,
        max_nodes=num_nodes,
        workspace=workspace,
    )


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
    device: str,
) -> GenerationBatchResult:
    """Verify the tree rooted at each request's bonus token, which sits at the committed
    length and has no KV yet, so the verify batch keeps ``batch.seq_lens`` as is."""
    verify_batch = copy.copy(batch)
    verify_batch.spec_info = build_eagle_verify_input(
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
    )
