"""Adaptive verify width for DFLASH tree verify.

Each decode step builds one best-first tree at the largest captured width ``Q_max``, so
any prefix of ``n`` nodes is itself a tree; with calibrated per-node survival
probabilities ``s_i`` its expected accept length is ``1 + sum_{0 < i < n} s_i``. The host
picks the verify width ``Q`` from a small captured set to maximize the batch's expected
tokens per second, ``bs * E[AL](Q) / c(bs, Q)``, where ``c`` is a measured step-time
table. The host never waits for the GPU: it reads the expected lengths of an earlier
step from pinned memory once that step's copy has landed.

Batches with sampling rows build the sampled tree and, when the chain is a member, the
chain (every node's first draw, as deep as the draft block) from the same q. Their
survival estimates are calibrated online per (depth, draw class, drawn-q bin) and may
depend on the draws, so the member is chosen from earlier steps' estimates only. The
chain under traversal verification is block verification, so the policy can fall back
to the DFlash chain with block verification wherever it is predicted best.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import time
from typing import Optional

import msgspec
import torch

logger = logging.getLogger(__name__)


class StepTimer:
    """Debug instrumentation: appends ``bs member interval_s`` for consecutive decode steps.

    With overlap scheduling the host-side interval between consecutive decode calls is
    the pipeline period, i.e. the step cost, whether the GPU or the host is the bound.
    ``member`` is the verify width per request, "chain" for the chain member of a tree
    worker, and 0 for the DFLASH chain path."""

    def __init__(self, path: str):
        self._file = open(path, "a", buffering=1 << 16)
        self._prev: Optional[list] = None
        self._lines = 0

    def begin(self, *, is_decode: bool, bs: int, member: str) -> None:
        now = time.perf_counter()
        prev = self._prev
        if prev is not None and prev[1] and is_decode:
            self._file.write(f"{prev[2]} {prev[3]} {now - prev[0]:.6f}\n")
            self._lines += 1
            if self._lines % 32 == 0:
                self._file.flush()
        self._prev = [now, is_decode, bs, member]

    def set_member(self, member: str) -> None:
        if self._prev is not None:
            self._prev[3] = member


def make_step_timer() -> Optional[StepTimer]:
    path = os.environ.get("SGLANG_DFLASH_STEP_TIMES")
    return StepTimer(path) if path else None


# Member id of the chain in a sampled width set: every node's first draw, one path as
# deep as the draft block, whose traversal verification is block verification.
CHAIN = 0


def member_label(member: int) -> str:
    return "chain" if member == CHAIN else str(member)


def _parse_members(value: str) -> tuple[int, ...]:
    """ "chain,16,32" -> (CHAIN, 16, 32): the chain first, then widths ascending."""
    members = {CHAIN if x.strip() == "chain" else int(x) for x in value.split(",")}
    return tuple(sorted(members))


class AdaptiveWidthConfig(msgspec.Struct, frozen=True):
    """Experiment knobs: SGLANG_DFLASH_TREE_WIDTHS ("16,32") for greedy batches and
    SGLANG_DFLASH_TREE_SAMPLED_WIDTHS ("chain,16,32"; default the greedy widths) for
    batches with sampling rows; the largest tree width must be
    --speculative-num-draft-tokens. SGLANG_DFLASH_TREE_COST_TABLE is a JSON of
    ``{"cost_ms": {"<Q>": {"<bs>": ms}}, "sampled_cost_ms": {"chain" | "<Q>": ...}}``
    (sampled batches fall back to ``cost_ms``). SGLANG_DFLASH_TREE_FORCE_WIDTH verifies
    at this member every step where it is one (for measurements), and
    SGLANG_DFLASH_TREE_CALIBRATION_DUMP appends per-step predicted and actual accept
    lengths (one host sync every few dozen steps). SGLANG_DFLASH_TREE_RANK=depth ranks
    sampled-tree slots by accept rates learned per (depth, draw class) instead of the
    two online draw priors."""

    widths: tuple[int, ...]
    sampled_members: tuple[int, ...]
    cost_ms: dict[int, dict[int, float]]
    sampled_cost_ms: dict[int, dict[int, float]]
    force_width: Optional[int]
    calibration_dump: Optional[str]
    rank_by_depth: bool = False


def read_adaptive_width_config(max_width: int) -> Optional[AdaptiveWidthConfig]:
    value = os.environ.get("SGLANG_DFLASH_TREE_WIDTHS")
    sampled_value = os.environ.get("SGLANG_DFLASH_TREE_SAMPLED_WIDTHS")
    calibration_dump = os.environ.get("SGLANG_DFLASH_TREE_CALIBRATION_DUMP") or None
    force = os.environ.get("SGLANG_DFLASH_TREE_FORCE_WIDTH")
    if not (value or sampled_value or calibration_dump or force):
        return None
    widths = _parse_members(value) if value else (max_width,)
    sampled_members = _parse_members(sampled_value) if sampled_value else widths
    if CHAIN in widths:
        raise ValueError("the chain member is for sampled batches only")
    trees = sorted((set(widths) | set(sampled_members)) - {CHAIN})
    if not trees or trees[-1] != max_width:
        raise ValueError(
            f"the largest tree width of {widths} and {sampled_members} must be "
            f"--speculative-num-draft-tokens={max_width}"
        )
    if trees[0] < 2:
        raise ValueError(f"tree verify widths must be >= 2, got {trees}")
    force_width = _parse_members(force)[0] if force else None
    if force_width is not None and force_width not in widths + sampled_members:
        raise ValueError(
            f"forced width {force} is not in {widths} or {sampled_members}"
        )
    table_path = os.environ.get("SGLANG_DFLASH_TREE_COST_TABLE")
    cost_ms, sampled_cost_ms = load_cost_table(table_path) if table_path else ({}, {})
    for name, members, costs in (
        ("greedy", widths, cost_ms),
        ("sampled", sampled_members, sampled_cost_ms),
    ):
        if len(members) > 1 and force_width not in members:
            missing = [member_label(m) for m in members if m not in costs]
            if missing:
                raise ValueError(
                    f"SGLANG_DFLASH_TREE_COST_TABLE has no {name} step costs for {missing}"
                )
    return AdaptiveWidthConfig(
        widths=widths,
        sampled_members=sampled_members,
        cost_ms=cost_ms,
        sampled_cost_ms=sampled_cost_ms,
        force_width=force_width,
        calibration_dump=calibration_dump,
        rank_by_depth=os.environ.get("SGLANG_DFLASH_TREE_RANK") == "depth",
    )


def load_cost_table(
    path: str,
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    """The greedy and sampled step-cost tables, ``{member: {bs: ms}}``; the sampled one
    is the greedy one where the file has no ``sampled_cost_ms``."""
    with open(path) as f:
        raw = json.load(f)

    def table(section: dict) -> dict[int, dict[int, float]]:
        return {
            _parse_members(q)[0]: {int(b): float(ms) for b, ms in row.items()}
            for q, row in section.items()
        }

    greedy = table(raw["cost_ms"])
    return greedy, table(raw["sampled_cost_ms"]) if "sampled_cost_ms" in raw else greedy


def interpolate_cost(row: dict[int, float], bs: int) -> float:
    """Step cost at ``bs``, linear between measured batch sizes, flat outside them."""
    sizes = sorted(row)
    if bs <= sizes[0]:
        return row[sizes[0]]
    if bs >= sizes[-1]:
        return row[sizes[-1]]
    hi = bisect.bisect_left(sizes, bs)
    lo_bs, hi_bs = sizes[hi - 1], sizes[hi]
    frac = (bs - lo_bs) / (hi_bs - lo_bs)
    return row[lo_bs] + frac * (row[hi_bs] - row[lo_bs])


class VerifyWidthPolicy:
    """Picks the width (or member) with the most expected tokens per second at this batch
    size."""

    def __init__(
        self, *, widths: tuple[int, ...], cost_ms: dict[int, dict[int, float]]
    ):
        self.widths = widths
        self._cost_ms = cost_ms
        self._cache: dict[int, list[float]] = {}

    def costs(self, bs: int) -> list[float]:
        cost = self._cache.get(bs)
        if cost is None:
            cost = [interpolate_cost(self._cost_ms[q], bs) for q in self.widths]
            self._cache[bs] = cost
        return cost

    def choose(self, *, bs: int, expected_accept_lens: list[float]) -> int:
        """``expected_accept_lens[j]``: mean expected accept length per request at
        ``widths[j]``; ties go to the narrower width."""
        best, best_rate = self.widths[0], -1.0
        for width, accept_len, cost in zip(
            self.widths, expected_accept_lens, self.costs(bs)
        ):
            rate = accept_len / cost
            if rate > best_rate * (1.0 + 1e-6):
                best, best_rate = width, rate
        return best


class StaleDeviceStats:
    """Hands the host a recent device vector without a sync.

    ``push`` copies a small device vector into a pinned slot on the current stream and
    records an event; ``latest`` returns the newest pushed vector whose copy has
    finished, or the initial value before any has."""

    def __init__(
        self,
        *,
        initial: list[float],
        device: torch.device,
        slots: int = 4,
        event_factory=None,
    ):
        host = torch.zeros((slots, len(initial)), dtype=torch.float32)
        self._host = host if device.type == "cpu" else host.pin_memory()
        if event_factory is None:
            event_factory = torch.get_device_module(device).Event
        self._events = [event_factory() for _ in range(slots)]
        self._pending: list[int] = []
        self._next = 0
        self._latest = list(initial)

    def push(self, values: torch.Tensor) -> None:
        slot = self._next
        self._next = (slot + 1) % len(self._events)
        if slot in self._pending:
            self._pending.remove(slot)
        self._host[slot].copy_(values, non_blocking=True)
        self._events[slot].record()
        self._pending.append(slot)

    def latest(self) -> list[float]:
        while self._pending and self._events[self._pending[0]].query():
            self._latest = self._host[self._pending.pop(0)].tolist()
        return self._latest


class VerifyWidthState(msgspec.Struct, frozen=True):
    """The target attention backend and verify graph runner captured for one width."""

    attn_backend: object
    graph_runner: object


# Calibration dump lines are queued on the device and written with one host sync per
# this many steps; arbitrary.
_DUMP_FLUSH_STEPS = 64
# Every this many sampled steps verify another member in turn, so the calibration of
# members the policy does not pick stays current; arbitrary.
_EXPLORE_EVERY = 32
# Per-member accept-length scale: decay per verify of the member, and the prior weight
# in accepted tokens that holds it at 1 until data arrives. A weaker prior let a dozen
# exploration steps at batch size 1 (accept length sd ~3 tokens each) move a member's
# scale by 25% on Glimmer at T=1; otherwise arbitrary.
_MEMBER_SCALE_DECAY = 0.995
_MEMBER_SCALE_PRIOR = 1000.0


class AdaptiveVerifyWidth:
    """Per-step verify width for one DFLASH tree worker: the online survival
    calibrations (greedy trees, sampled trees, the sampled chain), the expected accept
    lengths handed to the host, the policies, and one target verify backend and graph
    set per verify row count."""

    def __init__(
        self,
        *,
        config: AdaptiveWidthConfig,
        num_depths: int,
        chain_rows: int,
        device: torch.device,
    ):
        from sglang.kernels.ops.speculative.dflash_tree import (
            NUM_CALIBRATION_BINS,
            NUM_DRAW_CLASSES,
            sampled_calibration_slots,
        )

        self.widths = config.widths
        self.sampled_members = config.sampled_members
        self.max_width = max(config.widths + config.sampled_members)
        self.chain_rows = chain_rows
        self.has_chain = CHAIN in config.sampled_members
        if self.has_chain and chain_rows > self.max_width:
            raise ValueError(
                f"the chain verifies {chain_rows} rows, more than the server width "
                f"{self.max_width}"
            )
        self._config = config
        self._device = device

        def stats(slots: int) -> torch.Tensor:
            return torch.zeros((3, slots), dtype=torch.float32, device=device)

        self.calibration_stats = stats(num_depths * NUM_CALIBRATION_BINS)
        self.sampled_stats = stats(sampled_calibration_slots(num_depths))
        self.chain_stats = (
            stats(sampled_calibration_slots(num_depths)) if self.has_chain else None
        )
        # Filled by refresh_rank_table before the first step.
        self.rank_table = (
            torch.zeros(num_depths * NUM_DRAW_CLASSES, device=device)
            if config.rank_by_depth
            else None
        )
        self._width_columns = torch.tensor(
            [width - 1 for width in config.widths], dtype=torch.long, device=device
        )
        self._sampled_tree_columns = torch.tensor(
            [m - 1 for m in config.sampled_members if m != CHAIN],
            dtype=torch.long,
            device=device,
        )
        force = config.force_width
        self._policy = self._stats = None
        if len(config.widths) > 1 and force not in config.widths:
            self._policy = VerifyWidthPolicy(
                widths=config.widths, cost_ms=config.cost_ms
            )
            # Before the first copy lands, rank widths by cost alone.
            self._stats = StaleDeviceStats(
                initial=[1.0] * len(config.widths), device=device
            )
        self._sampled_policy = self._sampled_stats = None
        self._sampled_steps = 0
        # Decayed sums of actual and predicted (per-node model) accept length over each
        # member's verifies; their ratio rescales the member's prediction.
        self._member_totals = torch.zeros(
            (2, len(config.sampled_members)), dtype=torch.float32, device=device
        )
        if len(config.sampled_members) > 1 and force not in config.sampled_members:
            self._sampled_policy = VerifyWidthPolicy(
                widths=config.sampled_members, cost_ms=config.sampled_cost_ms
            )
            self._sampled_stats = StaleDeviceStats(
                initial=[1.0] * len(config.sampled_members), device=device
            )
        self._states: dict[int, VerifyWidthState] = {}
        self._active: Optional[int] = None
        self._dump = (
            open(config.calibration_dump, "a", buffering=1)
            if config.calibration_dump
            else None
        )
        self._dump_queue: list[tuple[dict, torch.Tensor]] = []

    @classmethod
    def maybe_create(
        cls,
        *,
        max_width: int,
        num_depths: int,
        chain_rows: int,
        device: torch.device,
    ) -> Optional["AdaptiveVerifyWidth"]:
        config = read_adaptive_width_config(max_width)
        if config is None:
            return None
        return cls(
            config=config, num_depths=num_depths, chain_rows=chain_rows, device=device
        )

    def member_rows(self, member: int) -> int:
        return self.chain_rows if member == CHAIN else member

    def _captured_rows(self) -> list[int]:
        force = self._config.force_width
        greedy = (force,) if force in self.widths else self.widths
        sampled = (force,) if force in self.sampled_members else self.sampled_members
        rows = {self.member_rows(m) for m in greedy + sampled}
        return sorted(rows - {self.max_width})

    def capture(self, model_runner) -> None:
        """Build a target verify backend and graph set per row count below the startup
        one, whose backend and graphs serve the largest width."""
        from sglang.srt.runtime_context import get_spec

        self._states[self.max_width] = VerifyWidthState(
            attn_backend=model_runner.attn_backend,
            graph_runner=model_runner.decode_cuda_graph_runner,
        )
        default_runner = model_runner.decode_cuda_graph_runner
        free_before, _ = torch.cuda.mem_get_info(self._device)
        tic = time.perf_counter()
        captures_graphs = (
            default_runner is not None
            and default_runner is not model_runner.eager_runner
        )
        for width in self._captured_rows():
            with get_spec().override(speculative_num_draft_tokens=width):
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
                        speculative_num_draft_tokens=width,
                    )
            self._states[width] = VerifyWidthState(
                attn_backend=attn_backend, graph_runner=graph_runner
            )
        free_after, _ = torch.cuda.mem_get_info(self._device)
        logger.info(
            "DFLASH adaptive tree width: greedy %s, sampled %s, verify rows %s; extra "
            "verify backends and graphs took %.2f s and %.2f GB",
            list(self.widths),
            [member_label(m) for m in self.sampled_members],
            sorted(self._states),
            time.perf_counter() - tic,
            (free_before - free_after) / (1 << 30),
        )

    def select_width(self, *, survival, bs: int) -> int:
        """Greedy batches: hand this step's expected accept lengths to a later step's
        host decision and pick this step's width from an earlier one's. ``survival`` is
        None for a tree without an acceptance estimate, verified at the largest width."""
        force = self._config.force_width
        if force is not None and force in self.widths:
            return force
        if survival is None or self._policy is None:
            return self.max_width
        self._stats.push(
            survival.prefix_accept_len[:bs, self._width_columns].mean(dim=0)
        )
        return self._policy.choose(bs=bs, expected_accept_lens=self._stats.latest())

    def choose_sampled_member(self, *, bs: int) -> int:
        """Batches with sampling rows: the member to verify, from earlier steps'
        expected accept lengths only. A sampled tree's estimate depends on its draws, so
        reading this step's would make the verified prefix depend on them."""
        force = self._config.force_width
        if force is not None and force in self.sampled_members:
            return force
        if self._sampled_policy is None:
            return self.sampled_members[-1]
        choice = self._sampled_policy.choose(
            bs=bs, expected_accept_lens=self._sampled_stats.latest()
        )
        self._sampled_steps += 1
        if self._sampled_steps % _EXPLORE_EVERY == 0:
            others = [m for m in self.sampled_members if m != choice]
            choice = others[(self._sampled_steps // _EXPLORE_EVERY) % len(others)]
        return choice

    def predict_sampled(
        self, *, tree_survival, chain_survival, bs: int
    ) -> Optional[torch.Tensor]:
        """Expected accept length of every sampled member, ``[bs, 2 * members]``: the
        per-node model's rescaled by the member's measured ratio, then unscaled. The
        rescaled batch mean is pushed for later steps' decisions. None when neither a
        policy nor the dump reads it."""
        if self._sampled_policy is None and self._dump is None:
            return None
        columns = []
        if self.has_chain:
            columns.append(
                chain_survival.prefix_accept_len[
                    :bs, self.chain_rows - 1 : self.chain_rows
                ]
            )
        if self._sampled_tree_columns.numel():
            columns.append(
                tree_survival.prefix_accept_len[:bs, self._sampled_tree_columns]
            )
        raw = torch.cat(columns, dim=1)
        scale = (self._member_totals[0] + _MEMBER_SCALE_PRIOR) / (
            self._member_totals[1] + _MEMBER_SCALE_PRIOR
        )
        predicted = raw * scale
        if self._sampled_policy is not None:
            self._sampled_stats.push(predicted.mean(dim=0))
        return torch.cat([predicted, raw], dim=1)

    def activate(self, width: int, model_runner) -> None:
        if width == self._active:
            return
        state = self._states[width]
        model_runner.attn_backend = state.attn_backend
        model_runner.decode_cuda_graph_runner = state.graph_runner
        self._active = width

    def after_verify(
        self,
        *,
        proposal,
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
        width: int,
        topk: int,
        raw_log_mass: Optional[torch.Tensor],
    ) -> None:
        from sglang.kernels.ops.speculative.dflash_tree import update_tree_calibration

        if self._dump is not None:
            self._queue_greedy_dump(
                survival=proposal.survival,
                accept_index=accept_index,
                accept_lens=accept_lens,
                width=width,
                raw_log_mass=raw_log_mass,
            )
        update_tree_calibration(
            stats=self.calibration_stats,
            survival=proposal.survival,
            top_scores_index=proposal.top_scores_index,
            accept_index=accept_index,
            accept_lens=accept_lens,
            num_nodes=width,
            topk=topk,
        )

    def after_sampled_verify(
        self,
        *,
        member: int,
        proposal,
        predicted: Optional[torch.Tensor],
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
        topk: int,
    ) -> None:
        """Update the verified member's calibration (the chain's, or the sampled
        trees'), whose survival and edges ``proposal`` (that member's build) holds."""
        from sglang.kernels.ops.speculative.dflash_tree import update_tree_calibration

        if predicted is not None:
            j = self.sampled_members.index(member)
            n = len(self.sampled_members)
            totals = self._member_totals[:, j]
            totals.mul_(_MEMBER_SCALE_DECAY).add_(
                torch.stack([accept_lens.float().sum(), predicted[:, n + j].sum()])
            )
        if self._dump is not None and predicted is not None:
            self._queue_dump(
                {
                    "kind": "sampled",
                    "member": member_label(member),
                    "members": [member_label(m) for m in self.sampled_members],
                },
                torch.cat([accept_lens.view(-1, 1).float(), predicted], dim=1),
            )
        update_tree_calibration(
            stats=self.chain_stats if member == CHAIN else self.sampled_stats,
            survival=proposal.survival,
            top_scores_index=proposal.top_scores_index,
            accept_index=accept_index,
            accept_lens=accept_lens,
            num_nodes=self.member_rows(member),
            topk=topk,
        )

    def refresh_rank_table(self, log_priors: torch.Tensor) -> None:
        """Recompute the sampled trees' ranking from their calibration statistics; the
        next step's tree reads it, so it never sees that step's draws."""
        from sglang.kernels.ops.speculative.dflash_tree import update_rank_table

        if self.rank_table is not None:
            update_rank_table(
                rank_table=self.rank_table,
                stats=self.sampled_stats,
                log_priors=log_priors,
            )

    def _queue_dump(self, header: dict, rows: torch.Tensor) -> None:
        self._dump_queue.append((header, rows))
        if len(self._dump_queue) < _DUMP_FLUSH_STEPS:
            return
        flat = torch.cat([r.reshape(-1) for _, r in self._dump_queue]).tolist()
        offset = 0
        for header, r in self._dump_queue:
            n, m = r.shape
            values = flat[offset : offset + n * m]
            offset += n * m
            line = dict(header, rows=[values[i * m : (i + 1) * m] for i in range(n)])
            self._dump.write(json.dumps(line) + "\n")
        self._dump_queue = []

    def _queue_greedy_dump(
        self, *, survival, accept_index, accept_lens, width, raw_log_mass
    ) -> None:
        """One line per step: width, then per request the accept length, the predicted
        (calibrated, then raw) expected accept length at every width, and the accept
        length each narrower width would have had (-1 above the verified width): the
        accepted path's leading nodes inside that prefix, since greedy verification
        accepts the target's one path wherever the tree holds it."""
        bs = accept_lens.shape[0]
        offsets = torch.arange(bs, device=accept_index.device).unsqueeze(1) * width
        path = accept_index[:bs].long() - offsets
        on_path = torch.arange(path.shape[1], device=path.device) < accept_lens.view(
            -1, 1
        )
        counterfactual = []
        for w in self.widths:
            if w > width:
                counterfactual.append(torch.full_like(accept_lens, -1).float())
                continue
            inside = (path < w) & on_path
            counterfactual.append(torch.cumprod(inside.int(), dim=1).sum(dim=1).float())
        predicted = survival.prefix_accept_len[:bs, self._width_columns]
        if raw_log_mass is None:
            raw = torch.zeros_like(predicted)
        else:
            raw = 1.0 + torch.cumsum(raw_log_mass[:bs, 1:].exp(), dim=1)
            raw = raw[:, self._width_columns - 1]
        rows = torch.cat(
            [
                accept_lens.view(-1, 1).float(),
                predicted,
                raw,
                torch.stack(counterfactual, dim=1),
            ],
            dim=1,
        )
        self._queue_dump({"width": width}, rows)
