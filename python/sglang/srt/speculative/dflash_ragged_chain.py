"""Ragged per-request chain lengths for the DFLASH chain verify.

With SGLANG_DFLASH_RAGGED_CHAIN_TABLE set, a decode step of batch size bs verifies
``rows_per_request(bs) * bs`` rows in total (one target verify graph set per
rows-per-request value), split over the batch: request i verifies the first k_i
rows of its drafted block (its first k_i - 1 drafts), 1 <= k_i <= block. The
split depends only on the selector lattice, before any draw: each request's
self-collision path mass picks one of the table's bins, each bin holds the mean
block-verification accept length of a chain cut to k rows (measured on served
T=1 runs), and the batch's extra rows go to the largest marginal accept lengths.
Block verification of a prefix whose length is fixed before verification is
lossless, so sampled outputs keep the target distribution.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Optional

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class RaggedChainTable(msgspec.Struct, frozen=True):
    block: int
    # Ascending upper edges of the collision-mass bins (len(accept_lens) - 1).
    bin_edges: list[float]
    # [bins][block]: expected accept length (drafts + bonus) of a chain of k rows.
    accept_lens: list[list[float]]
    # Ascending [largest batch size, verify rows per request] pairs.
    rows_per_request: list[list[int]]
    feature: str = ""
    source: str = ""


@functools.lru_cache(maxsize=1)
def read_ragged_chain_table() -> Optional[RaggedChainTable]:
    path = envs.SGLANG_DFLASH_RAGGED_CHAIN_TABLE.get()
    if not path:
        return None
    with open(path, "rb") as f:
        table = msgspec.json.decode(f.read(), type=RaggedChainTable)
    if len(table.accept_lens) != len(table.bin_edges) + 1 or any(
        len(curve) != table.block for curve in table.accept_lens
    ):
        raise ValueError(f"{path}: needs bins + 1 accept-length curves of block rows")
    if not table.rows_per_request or any(
        not 1 <= rows <= table.block for _, rows in table.rows_per_request
    ):
        raise ValueError(f"{path}: rows per request must lie in 1..block")
    return table


class RaggedChainRows(msgspec.Struct, frozen=True):
    """One verify's packed rows: request i's are [qo_indptr[i], qo_indptr[i + 1]),
    the first verify_lens[i] columns of its [block] row of the drafted block."""

    verify_lens: torch.Tensor  # [bs] int32
    qo_indptr: torch.Tensor  # [bs + 1] int32
    token_reqs: torch.Tensor  # [rows] int32, each row's request
    block_index: torch.Tensor  # [rows] int64, flat index into [bs, block]
    rows_per_request: int

    @classmethod
    def build(
        cls, *, verify_lens: torch.Tensor, rows_per_request: int, block: int
    ) -> RaggedChainRows:
        bs = verify_lens.numel()
        num_rows = rows_per_request * bs
        device = verify_lens.device
        verify_lens = verify_lens.to(torch.int32)
        qo_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        torch.cumsum(verify_lens, dim=0, dtype=torch.int32, out=qo_indptr[1:])
        token_reqs = torch.repeat_interleave(
            torch.arange(bs, dtype=torch.int32, device=device),
            verify_lens,
            output_size=num_rows,
        )
        offsets = torch.arange(num_rows, device=device) - qo_indptr[token_reqs]
        return cls(
            verify_lens=verify_lens,
            qo_indptr=qo_indptr,
            token_reqs=token_reqs,
            block_index=token_reqs.to(torch.int64) * block + offsets,
            rows_per_request=rows_per_request,
        )


def fill_padded_chain_rows(
    *,
    rows: Optional[RaggedChainRows],
    bs: int,
    rows_per_request: int,
    qo_indptr_out: torch.Tensor,
    token_reqs_out: torch.Tensor,
) -> None:
    """Write a verify's row layout for ``bs`` requests into attention buffers: the
    live requests' ragged rows, then rows_per_request rows for each graph padding
    request (all of them with no ragged rows: graph capture, a uniform chain)."""
    device = qo_indptr_out.device
    verify_lens = torch.full((bs,), rows_per_request, dtype=torch.int32, device=device)
    if rows is not None:
        verify_lens[: rows.verify_lens.numel()] = rows.verify_lens
    qo_indptr_out[0] = 0
    torch.cumsum(verify_lens, dim=0, dtype=torch.int32, out=qo_indptr_out[1 : bs + 1])
    token_reqs_out[: bs * rows_per_request] = torch.repeat_interleave(
        torch.arange(bs, dtype=torch.int32, device=device),
        verify_lens,
        output_size=bs * rows_per_request,
    )


class RaggedChainPolicy:
    def __init__(self, *, table: RaggedChainTable, device: torch.device):
        self.block = table.block
        self._rows_per_request = [tuple(pair) for pair in table.rows_per_request]
        self._bin_edges = torch.tensor(
            table.bin_edges, dtype=torch.float32, device=device
        )
        curves = torch.tensor(table.accept_lens, dtype=torch.float32)
        # Made non-increasing in k so the largest gains of a request form a prefix;
        # the measured curves break it by at most 0.01 (sampling noise).
        gains = curves.diff(dim=1).clamp(min=0).cummin(dim=1).values
        self._gains = gains.to(device)

    @property
    def widths(self) -> list[int]:
        return sorted({rows for _, rows in self._rows_per_request})

    def rows_per_request(self, bs: int) -> int:
        for max_bs, rows in self._rows_per_request:
            if bs <= max_bs:
                return rows
        return self._rows_per_request[-1][1]

    def verify_lens(
        self, *, collision_mass: torch.Tensor, rows_per_request: int
    ) -> torch.Tensor:
        """[bs] int32 rows per request, summing to rows_per_request * bs."""
        bs = collision_mass.numel()
        device = collision_mass.device
        if rows_per_request >= self.block:
            return torch.full((bs,), self.block, dtype=torch.int32, device=device)
        extra = (rows_per_request - 1) * bs
        verify_lens = torch.ones(bs, dtype=torch.int32, device=device)
        if extra == 0:
            return verify_lens
        gains = self._gains[torch.bucketize(collision_mass, self._bin_edges)]
        picked = gains.view(-1).topk(extra, sorted=False).indices
        return verify_lens.index_add_(
            0,
            picked // (self.block - 1),
            torch.ones(extra, dtype=torch.int32, device=device),
        )


class _VerifyState(msgspec.Struct, frozen=True):
    attn_backend: object
    graph_runner: object


class RaggedChainVerify:
    """The policy plus one target verify backend and graph set per rows-per-request
    value; the startup ones (built for the whole block) serve the block value."""

    def __init__(self, *, table: RaggedChainTable, device: torch.device):
        self.policy = RaggedChainPolicy(table=table, device=device)
        self.block = table.block
        self._states: dict[int, _VerifyState] = {}
        self._active = self.block

    @classmethod
    def maybe_create(
        cls, *, block_size: int, device: torch.device
    ) -> Optional[RaggedChainVerify]:
        table = read_ragged_chain_table()
        if table is None:
            return None
        if table.block != block_size:
            raise ValueError(
                f"SGLANG_DFLASH_RAGGED_CHAIN_TABLE is for block {table.block}, "
                f"the drafter verifies blocks of {block_size}"
            )
        return cls(table=table, device=device)

    def capture(self, model_runner) -> None:
        from sglang.srt.runtime_context import get_spec

        self._states[self.block] = _VerifyState(
            attn_backend=model_runner.attn_backend,
            graph_runner=model_runner.decode_cuda_graph_runner,
        )
        default_runner = model_runner.decode_cuda_graph_runner
        captures_graphs = (
            default_runner is not None
            and default_runner is not model_runner.eager_runner
        )
        free_before, _ = torch.cuda.mem_get_info(model_runner.device)
        tic = time.perf_counter()
        for rows in self.policy.widths:
            if rows == self.block:
                continue
            with get_spec().override(speculative_num_draft_tokens=rows):
                saved_flag = model_runner.init_new_workspace
                try:
                    attn_backend = model_runner._get_attention_backend(
                        init_new_workspace=True
                    )
                finally:
                    model_runner.init_new_workspace = saved_flag
                graph_runner = default_runner
                if captures_graphs:
                    graph_runner = type(default_runner)(
                        model_runner,
                        attn_backend=attn_backend,
                        speculative_num_draft_tokens=rows,
                    )
            self._states[rows] = _VerifyState(
                attn_backend=attn_backend, graph_runner=graph_runner
            )
        free_after, _ = torch.cuda.mem_get_info(model_runner.device)
        logger.info(
            "DFLASH ragged chain: verify rows per request %s; extra verify "
            "backends and graphs took %.2f s and %.2f GB",
            self.policy.widths,
            time.perf_counter() - tic,
            (free_before - free_after) / (1 << 30),
        )

    def activate(self, rows_per_request: int, model_runner) -> None:
        if rows_per_request == self._active:
            return
        state = self._states[rows_per_request]
        model_runner.attn_backend = state.attn_backend
        model_runner.decode_cuda_graph_runner = state.graph_runner
        self._active = rows_per_request

    def plan(self, *, collision_mass: torch.Tensor) -> RaggedChainRows:
        rows_per_request = self.policy.rows_per_request(collision_mass.numel())
        return RaggedChainRows.build(
            verify_lens=self.policy.verify_lens(
                collision_mass=collision_mass, rows_per_request=rows_per_request
            ),
            rows_per_request=rows_per_request,
            block=self.block,
        )


def check_ragged_chain_batch(*, sampling_info, has_grammar: bool, return_logprob: bool):
    """Raise for what a ragged chain verify cannot serve yet: per-request logit
    adjustments are broadcast over a uniform block, logprobs and grammars read one."""
    unsupported = []
    if has_grammar:
        unsupported.append("grammar")
    if return_logprob:
        unsupported.append("return_logprob")
    if sampling_info is not None:
        if sampling_info.has_custom_logit_processor:
            unsupported.append("custom logit processors")
        penalizer = sampling_info.penalizer_orchestrator
        if penalizer is not None and penalizer.is_required:
            unsupported.append("penalties")
        if sampling_info.logit_bias is not None:
            unsupported.append("logit_bias")
        if sampling_info.sampling_mask_batch_indices is not None:
            unsupported.append("sampling masks")
    if unsupported:
        raise NotImplementedError(
            "DFLASH ragged chain verify (SGLANG_DFLASH_RAGGED_CHAIN_TABLE) does not "
            "support " + ", ".join(unsupported)
        )


def unpack_greedy_rows(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    rows: RaggedChainRows,
) -> tuple[torch.Tensor, torch.Tensor]:
    """[bs, block] candidates and target argmax for the greedy accept kernel: a
    request's columns past its verified rows never match (-1 against -2), so it
    accepts at most verify_lens - 1 drafts and its bonus comes from a verified row."""
    bs, block = candidates.shape
    unpacked = torch.full(
        (bs * block,), -2, dtype=target_predict.dtype, device=candidates.device
    )
    unpacked[rows.block_index] = target_predict
    columns = torch.arange(block, device=candidates.device)
    verified = columns[None, :] < rows.verify_lens[:, None]
    return torch.where(verified, candidates, -1), unpacked.view(bs, block)


def accept_ragged_chain_sampling(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input,
    rows: RaggedChainRows,
    block_verification: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(correct drafts, bonus tokens) of each request's first verify_lens - 1 drafts.
    candidates [bs, block]; target_logits [rows, vocab], packed; draft_probs
    [bs, block - 1, vocab]."""
    from sglang.kernels.ops.speculative.dspark.dspark_accept import (
        gather_two_level_bonus_triton,
    )
    from sglang.kernels.ops.speculative.reject_sampling import (
        chain_speculative_sampling_triton,
    )
    from sglang.srt.speculative.dflash_utils import (
        _get_or_create_chain_verify_buffers,
        build_speculative_verify_target_probs,
    )

    bs, block = candidates.shape
    device = candidates.device
    target_probs = build_speculative_verify_target_probs(
        next_token_logits=target_logits,
        sampling_info=sampling_info,
        draft_token_num=rows.rows_per_request,
        bs=bs,
        max_top_k=draft_input.max_top_k,
        uniform_top_k_value=draft_input.uniform_top_k_value,
        row_requests=rows.token_reqs,
    )
    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(bs=bs, draft_token_num=block, device=device)
    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=torch.rand((bs, block - 1), dtype=torch.float32, device=device),
        uniform_samples_for_final_sampling=torch.rand(
            (bs,), dtype=torch.float32, device=device
        ),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
        block_verification=block_verification,
        verify_lens=rows.verify_lens,
        target_row_starts=rows.qo_indptr,
    )
    bonus = gather_two_level_bonus_triton(
        accept_index=accept_index, predicts=predicts, correct_len=accept_token_num
    )
    return accept_token_num, bonus
