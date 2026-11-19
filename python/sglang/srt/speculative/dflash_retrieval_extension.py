"""Agreement-gated retrieval extension of DFLASH tree verify (SGLANG_DFLASH_RETRIEVAL_EXTENSION).

Each step looks up the longest earlier occurrence of the request's context suffix (prompt
and generated tokens, ending at the bonus token) and takes the tokens that followed it as
a proposal. When the drafted path, the first child of every node from the root down to the
draft block's last depth, equals the proposal's first F - 1 tokens, the verify tree's last
nodes become a chain of the proposal's next tokens below the path's leaf
(``kernels/ops/speculative/dflash_retrieval.py``). The gate reads only the drafts and the
context, the chain's q is a point mass, and the replaced nodes are the tree's last ones, so
verification (greedy or Traversal Verification) stays lossless; the verify keeps its width,
so no step changes shape and nothing waits for the host. Measured reason: on copy spans the
drafter accepts 14.2 of 15 drafts and retrieval 44.6 tokens (explore/sysloop), so the block
cap, not the drafter, ends the step.
"""

from __future__ import annotations

import logging

import torch
from sglang.kernels.ops.speculative.dflash_retrieval import (
    MAX_CONTEXT,
    ExtendedTree,
    extend_tree_with_retrieval,
    find_suffix_matches,
    history_context,
)
from sglang.kernels.ops.speculative.dflash_tree import DFlashTreeProposal
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class RetrievalExtension:
    """Token history of every request slot, and the per-step match and tree rewrite."""

    def __init__(
        self,
        *,
        max_extension: int,
        min_match: int,
        path_depth: int,
        topk: int,
        device: torch.device,
    ):
        self.max_extension = max_extension
        self.min_match = min_match
        self.path_depth = path_depth
        self.topk = topk
        self.device = device
        self._history: torch.Tensor | None = None

    @classmethod
    def maybe_create(
        cls, *, block_size: int, topk: int, device: torch.device
    ) -> RetrievalExtension | None:
        max_extension = envs.SGLANG_DFLASH_RETRIEVAL_EXTENSION.get()
        if max_extension <= 0:
            return None
        min_match = envs.SGLANG_DFLASH_RETRIEVAL_MIN_MATCH.get()
        logger.info(
            "DFLASH retrieval extension: up to %d tokens past the block, match >= %d",
            max_extension,
            min_match,
        )
        return cls(
            max_extension=max_extension,
            min_match=min_match,
            path_depth=block_size - 1,
            topk=topk,
            device=device,
        )

    def extends(self, width: int) -> bool:
        """Whether a tree of ``width`` nodes has room for a chain below the drafted path
        (the path takes at least ``path_depth`` + 1 nodes)."""
        return width > self.path_depth + 1

    def path_width(self, width: int) -> int:
        """Longest accepted path, root included, of an extended tree of ``width`` nodes."""
        return min(width, self.path_depth + 1 + self.max_extension)

    # -- token history ---------------------------------------------------------

    def _history_for(self, req_to_token: torch.Tensor) -> torch.Tensor:
        if self._history is None:
            if req_to_token.shape[1] > MAX_CONTEXT:
                raise ValueError(
                    f"the DFLASH retrieval extension packs positions in "
                    f"{MAX_CONTEXT.bit_length() - 1} bits; context "
                    f"{req_to_token.shape[1]} is longer"
                )
            # One more column than the context: the sink of masked writes.
            self._history = torch.full(
                (req_to_token.shape[0], req_to_token.shape[1] + 1),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
        return self._history

    def record_extend(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        extend_lens: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """A prefill's tokens. A cached prefix (radix cache) is not recorded: its slots
        keep older tokens, which can only weaken the proposals, never the outputs."""
        history = self._history_for(req_to_token)
        rows = torch.repeat_interleave(
            req_pool_indices.long(), extend_lens.long(), output_size=num_tokens
        )
        history[rows, positions.long()] = input_ids[:num_tokens].to(torch.int32)

    def record_commit(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        tokens: torch.Tensor,
        accept_lens: torch.Tensor,
    ) -> None:
        """A decode step's committed tokens ``tokens[:, :accept_lens]``, which follow the
        bonus token at ``seq_lens`` (the last one is the next bonus)."""
        history = self._history_for(req_to_token)
        context = history_context(history)
        cols = torch.arange(tokens.shape[1], device=tokens.device)
        positions = seq_lens.long().unsqueeze(1) + 1 + cols
        keep = (cols < accept_lens.unsqueeze(1)) & (positions < context)
        positions = torch.where(keep, positions, context)
        rows = req_pool_indices.long().unsqueeze(1).expand_as(positions)
        history[rows, positions] = tokens.to(torch.int32)

    # -- one decode step ---------------------------------------------------------

    def match(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Records each bonus token at ``seq_lens`` (a prefill's sampled token has no
        other record) and returns the packed suffix matches ending at it."""
        history = self._history_for(req_to_token)
        history[
            req_pool_indices.long(), seq_lens.long().clamp(max=history_context(history))
        ] = bonus_tokens.to(torch.int32)
        return find_suffix_matches(
            history=history, req_pool_indices=req_pool_indices, seq_lens=seq_lens
        )

    def extend(
        self,
        *,
        matches: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        proposal: DFlashTreeProposal,
        width: int,
        node_draft_tokens: torch.Tensor | None,
        node_draft_probs: torch.Tensor | None,
    ) -> ExtendedTree:
        assert self._history is not None
        return extend_tree_with_retrieval(
            history=self._history,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            matches=matches,
            draft_tokens=proposal.draft_tokens,
            parent_list=proposal.parent_list,
            top_scores_index=proposal.top_scores_index,
            node_count=proposal.node_count,
            node_draft_tokens=node_draft_tokens,
            node_draft_probs=node_draft_probs,
            width=width,
            topk=self.topk,
            path_depth=self.path_depth,
            max_extension=self.max_extension,
            min_match=self.min_match,
        )

    def block_accepts(
        self, *, accept_index: torch.Tensor, accept_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The accepts as the unextended tree saw them, for the width calibration: the
        path up to the block's last depth (the chain is never part of its builds)."""
        block = self.path_depth + 1
        return accept_index[:, :block], accept_lens.clamp(max=block)
