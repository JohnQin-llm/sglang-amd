"""Adaptive verify width for DFLASH tree verify.

Each decode step builds one best-first tree at the largest captured width ``Q_max``, so
any prefix of ``n`` nodes is itself a tree; with calibrated per-node survival
probabilities ``s_i`` its expected accept length is ``1 + sum_{0 < i < n} s_i``. The host
picks the verify width ``Q`` from a small captured set to maximize the batch's expected
tokens per second, ``bs * E[AL](Q) / c(bs, Q)``, where ``c`` is a measured step-time
table. The host never waits for the GPU: it reads the expected lengths of an earlier
step from pinned memory once that step's copy has landed.
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
    """Debug instrumentation: appends ``bs rows interval_s`` for consecutive decode steps.

    With overlap scheduling the host-side interval between consecutive decode calls is
    the pipeline period, i.e. the step cost, whether the GPU or the host is the bound.
    ``rows`` is the verify width per request (0 for the chain path)."""

    def __init__(self, path: str):
        self._file = open(path, "a", buffering=1 << 16)
        self._prev: Optional[list] = None
        self._lines = 0

    def begin(self, *, is_decode: bool, bs: int, rows: int) -> None:
        now = time.perf_counter()
        prev = self._prev
        if prev is not None and prev[1] and is_decode:
            self._file.write(f"{prev[2]} {prev[3]} {now - prev[0]:.6f}\n")
            self._lines += 1
            if self._lines % 32 == 0:
                self._file.flush()
        self._prev = [now, is_decode, bs, rows]

    def set_rows(self, rows: int) -> None:
        if self._prev is not None:
            self._prev[3] = rows


def make_step_timer() -> Optional[StepTimer]:
    path = os.environ.get("SGLANG_DFLASH_STEP_TIMES")
    return StepTimer(path) if path else None


class AdaptiveWidthConfig(msgspec.Struct, frozen=True):
    """Experiment knobs: SGLANG_DFLASH_TREE_WIDTHS ("16,32"; its largest entry must be
    --speculative-num-draft-tokens), SGLANG_DFLASH_TREE_COST_TABLE (JSON of
    ``{"cost_ms": {"<Q>": {"<bs>": ms}}}``), SGLANG_DFLASH_TREE_FORCE_WIDTH (verify at
    this width every step; for measurements), SGLANG_DFLASH_TREE_CALIBRATION_DUMP
    (append per-step predicted and actual accept lengths; syncs every step)."""

    widths: tuple[int, ...]
    cost_ms: dict[int, dict[int, float]]
    force_width: Optional[int]
    calibration_dump: Optional[str]


def read_adaptive_width_config(max_width: int) -> Optional[AdaptiveWidthConfig]:
    value = os.environ.get("SGLANG_DFLASH_TREE_WIDTHS")
    calibration_dump = os.environ.get("SGLANG_DFLASH_TREE_CALIBRATION_DUMP") or None
    if not value:
        if calibration_dump is None:
            return None
        value = str(max_width)
    widths = tuple(sorted({int(x) for x in value.split(",")}))
    if widths[-1] != max_width:
        raise ValueError(
            f"SGLANG_DFLASH_TREE_WIDTHS={value} must end at "
            f"--speculative-num-draft-tokens={max_width}"
        )
    if widths[0] < 2:
        raise ValueError(f"tree verify widths must be >= 2, got {widths}")
    force = os.environ.get("SGLANG_DFLASH_TREE_FORCE_WIDTH")
    force_width = int(force) if force else None
    if force_width is not None and force_width not in widths:
        raise ValueError(f"forced width {force_width} is not in {widths}")
    table_path = os.environ.get("SGLANG_DFLASH_TREE_COST_TABLE")
    cost_ms = load_cost_table(table_path) if table_path else {}
    if len(widths) > 1 and force_width is None:
        missing = [q for q in widths if q not in cost_ms]
        if missing:
            raise ValueError(
                f"SGLANG_DFLASH_TREE_COST_TABLE has no step costs for widths {missing}"
            )
    return AdaptiveWidthConfig(
        widths=widths,
        cost_ms=cost_ms,
        force_width=force_width,
        calibration_dump=calibration_dump,
    )


def load_cost_table(path: str) -> dict[int, dict[int, float]]:
    """``{"cost_ms": {"<Q>": {"<bs>": ms}}}`` -> ``{Q: {bs: ms}}``."""
    with open(path) as f:
        raw = json.load(f)["cost_ms"]
    return {
        int(q): {int(b): float(ms) for b, ms in row.items()} for q, row in raw.items()
    }


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
    """Picks the width with the most expected tokens per second at this batch size."""

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


class AdaptiveVerifyWidth:
    """Per-step verify width for one DFLASH tree worker: the online survival calibration,
    the expected accept lengths handed to the host, the policy, and one target verify
    backend and graph set per width."""

    def __init__(
        self,
        *,
        config: AdaptiveWidthConfig,
        num_depths: int,
        device: torch.device,
    ):
        from sglang.kernels.ops.speculative.dflash_tree import NUM_CALIBRATION_BINS

        self.widths = config.widths
        self.max_width = config.widths[-1]
        self._config = config
        self._device = device
        self.calibration_stats = torch.zeros(
            (3, num_depths * NUM_CALIBRATION_BINS), dtype=torch.float32, device=device
        )
        self._width_columns = torch.tensor(
            [width - 1 for width in config.widths], dtype=torch.long, device=device
        )
        self._policy = None
        self._stats = None
        if len(config.widths) > 1 and config.force_width is None:
            self._policy = VerifyWidthPolicy(
                widths=config.widths, cost_ms=config.cost_ms
            )
            # Before the first copy lands, rank widths by cost alone.
            self._stats = StaleDeviceStats(
                initial=[1.0] * len(config.widths), device=device
            )
        self._states: dict[int, VerifyWidthState] = {}
        self._active: Optional[int] = None
        self._dump = (
            open(config.calibration_dump, "a", buffering=1)
            if config.calibration_dump
            else None
        )

    @classmethod
    def maybe_create(
        cls, *, max_width: int, num_depths: int, device: torch.device
    ) -> Optional["AdaptiveVerifyWidth"]:
        config = read_adaptive_width_config(max_width)
        if config is None:
            return None
        return cls(config=config, num_depths=num_depths, device=device)

    def capture(self, model_runner) -> None:
        """Build a target verify backend and graph set per width below the startup one,
        whose backend and graphs serve the largest width."""
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
        for width in self.widths[:-1]:
            if (
                self._config.force_width is not None
                and width != self._config.force_width
            ):
                continue
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
            "DFLASH adaptive tree width: widths=%s, extra verify backends and graphs "
            "took %.2f s and %.2f GB",
            list(self._states),
            time.perf_counter() - tic,
            (free_before - free_after) / (1 << 30),
        )

    def select_width(self, *, survival, bs: int) -> int:
        """Hand this step's expected accept lengths to a later step's host decision and
        pick this step's width from an earlier one's. ``survival`` is None for a tree
        without an acceptance estimate (a sampled tree ranked by q), verified at the
        largest width."""
        if self._config.force_width is not None:
            return self._config.force_width
        if survival is None:
            return self.max_width
        if self._policy is None:
            return self.max_width
        self._stats.push(
            survival.prefix_accept_len[:bs, self._width_columns].mean(dim=0)
        )
        return self._policy.choose(bs=bs, expected_accept_lens=self._stats.latest())

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
            self._write_dump(
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

    def _write_dump(
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
            inside = (path < w) & on_path
            lead = torch.cumprod(inside.int(), dim=1).sum(dim=1)
            counterfactual.append(
                torch.where(
                    torch.tensor(w <= width, device=path.device),
                    lead.float(),
                    torch.full_like(lead, -1).float(),
                )
            )
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
        ).tolist()
        self._dump.write(json.dumps({"width": width, "rows": rows}) + "\n")
