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
to the DFlash chain with block verification wherever it is predicted best. The ragged
chain member verifies each request's own prefix of a chain draft, at lengths the ragged
chain table sets from the selector lattice (dflash_ragged_chain.py), on the chain verify
path.
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

from sglang.srt.speculative.dflash_ragged_chain import (
    RaggedChainPolicy,
    read_ragged_chain_table,
)

logger = logging.getLogger(__name__)


class StepTimer:
    """Debug instrumentation: appends ``bs member interval_s boottime_ns`` for consecutive
    decode steps.

    With overlap scheduling the host-side interval between consecutive decode calls is
    the pipeline period, i.e. the step cost, whether the GPU or the host is the bound.
    ``member`` is the verify width per request, "chain" for the chain member of a tree
    worker, and 0 for the DFLASH chain path; ``boottime_ns`` (CLOCK_BOOTTIME, the clock of
    rocprofv3 timestamps) is when the step began."""

    def __init__(self, path: str):
        self._file = open(path, "a", buffering=1 << 16)
        self._prev: Optional[list] = None
        self._lines = 0

    def begin(self, *, is_decode: bool, bs: int, member: str) -> None:
        now = time.perf_counter()
        prev = self._prev
        if prev is not None and prev[1] and is_decode:
            self._file.write(f"{prev[2]} {prev[3]} {now - prev[0]:.6f} {prev[4]}\n")
            self._lines += 1
            if self._lines % 32 == 0:
                self._file.flush()
        self._prev = [
            now,
            is_decode,
            bs,
            member,
            time.clock_gettime_ns(time.CLOCK_BOOTTIME),
        ]

    def set_member(self, member: str) -> None:
        if self._prev is not None:
            self._prev[3] = member


# Tensor-parallel rank of this process; the debug files are written by rank 0 only, since
# every rank of a TP group shares their paths.
_DEBUG_RANK = 0


def set_debug_rank(rank: int) -> None:
    global _DEBUG_RANK
    _DEBUG_RANK = rank


def make_step_timer() -> Optional[StepTimer]:
    path = os.environ.get("SGLANG_DFLASH_STEP_TIMES")
    return StepTimer(path) if path and _DEBUG_RANK == 0 else None


# Member id of the chain in a sampled width set: every node's first draw, one path as
# deep as the draft block, whose traversal verification is block verification. A
# shorter chain, the first k nodes of that path ("chain<k>"), is -k.
CHAIN = 0
# Member id of the ragged chain ("ragged"): a selector chain draft of which request i
# verifies its first k_i rows, the k_i set by the ragged chain table before any draw
# and summing to the table's rows per request times the batch size. Below every chain
# id, so it sorts first.
RAGGED = -(1 << 16)


def is_chain(member: int) -> bool:
    """A chain member verified as a tree (traversal verification); not the ragged
    chain, which verifies on the chain path."""
    return RAGGED < member <= CHAIN


def member_label(member: int) -> str:
    if member == RAGGED:
        return "ragged"
    if member == CHAIN:
        return "chain"
    return f"chain{-member}" if member < 0 else str(member)


def _parse_member(value: str) -> int:
    value = value.strip()
    if value == "ragged":
        return RAGGED
    if value == "chain":
        return CHAIN
    if value.startswith("chain"):
        rows = int(value[len("chain") :])
        if rows < 2:
            raise ValueError(f"a chain member needs >= 2 rows, got {value}")
        return -rows
    return int(value)


def _parse_members(value: str) -> tuple[int, ...]:
    """ "chain,chain10,16,32" -> (-10, CHAIN, 16, 32): the ragged chain, then the
    chains, then widths ascending."""
    return tuple(sorted({_parse_member(x) for x in value.split(",")}))


class StepPhases:
    """Debug instrumentation: GPU time between named marks of each decode step, from
    timing events on the current stream, one JSON line per step (``{"bs": n, "<mark>":
    ms since the previous mark}``), written with one host sync every 64 steps."""

    def __init__(self, path: str):
        self._file = open(path, "a", buffering=1)
        self._done: list[tuple[int, list]] = []
        self._current: Optional[tuple[int, list]] = None

    def begin(self, bs: int) -> None:
        self._current = (bs, [])
        self.mark("begin")

    def mark(self, name: str) -> None:
        if self._current is None:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._current[1].append((name, event))

    def end(self) -> None:
        if self._current is None:
            return
        self.mark("end")
        self._done.append(self._current)
        self._current = None
        if len(self._done) < 64:
            return
        self._done[-1][1][-1][1].synchronize()
        for bs, marks in self._done:
            line = {"bs": bs}
            for (_, prev), (name, event) in zip(marks, marks[1:]):
                line[name] = round(prev.elapsed_time(event), 4)
            self._file.write(json.dumps(line) + "\n")
        self._done = []


_STEP_PHASES: Optional[StepPhases] = None


def step_phases() -> Optional[StepPhases]:
    """The process's StepPhases when SGLANG_DFLASH_STEP_PHASES names a file, else None."""
    global _STEP_PHASES
    path = os.environ.get("SGLANG_DFLASH_STEP_PHASES")
    if path and _STEP_PHASES is None and _DEBUG_RANK == 0:
        _STEP_PHASES = StepPhases(path)
    return _STEP_PHASES


class AdaptiveWidthConfig(msgspec.Struct, frozen=True):
    """Experiment knobs: SGLANG_DFLASH_TREE_WIDTHS ("16,32") for greedy batches and
    SGLANG_DFLASH_TREE_SAMPLED_WIDTHS ("chain,16,32"; default the greedy widths) for
    batches with sampling rows; the largest tree width must be
    --speculative-num-draft-tokens. SGLANG_DFLASH_TREE_COST_TABLE is a JSON of
    ``{"cost_ms": {"<Q>": {"<bs>": ms}}, "sampled_cost_ms": {"chain" | "<Q>": ...}}``
    (sampled batches fall back to ``cost_ms``); a member's measured accept length
    (``member_accept_len`` for greedy widths, ``sampled_member_accept_len`` for sampled
    members) is one number, or ``{"<bs>": AL}`` for the ragged chain, whose rows per
    request change with the batch size. SGLANG_DFLASH_TREE_FORCE_WIDTH verifies at this
    member every step where it is one (for measurements), and
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
    sampled_accept_len: dict[int, dict[int, float]] = {}
    content_weight: float = 0.5
    accept_len: dict[int, dict[int, float]] = {}


def read_adaptive_width_config(max_width: int) -> Optional[AdaptiveWidthConfig]:
    value = os.environ.get("SGLANG_DFLASH_TREE_WIDTHS")
    sampled_value = os.environ.get("SGLANG_DFLASH_TREE_SAMPLED_WIDTHS")
    calibration_dump = os.environ.get("SGLANG_DFLASH_TREE_CALIBRATION_DUMP") or None
    force = os.environ.get("SGLANG_DFLASH_TREE_FORCE_WIDTH")
    if not (value or sampled_value or calibration_dump or force):
        return None
    widths = _parse_members(value) if value else (max_width,)
    sampled_members = _parse_members(sampled_value) if sampled_value else widths
    if any(is_chain(m) for m in widths):
        raise ValueError("tree-verified chain members are for sampled batches only")
    trees = sorted(
        m for m in set(widths) | set(sampled_members) if not is_chain(m) and m != RAGGED
    )
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
    cost_ms, sampled_cost_ms, accept_len, sampled_accept_len = (
        load_cost_table(table_path) if table_path else ({}, {}, {}, {})
    )
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
        sampled_accept_len=sampled_accept_len,
        content_weight=float(
            os.environ.get("SGLANG_DFLASH_TREE_CONTENT_WEIGHT") or 0.5
        ),
        accept_len=accept_len,
    )


def load_cost_table(
    path: str,
) -> tuple[
    dict[int, dict[int, float]],
    dict[int, dict[int, float]],
    dict[int, dict[int, float]],
    dict[int, dict[int, float]],
]:
    """The greedy and sampled step-cost tables, ``{member: {bs: ms}}`` (the sampled one
    is the greedy one where the file has no ``sampled_cost_ms``), and the greedy and
    sampled members' measured accept lengths, ``{member: {bs: AL}}`` from
    ``member_accept_len`` and ``sampled_member_accept_len`` (a single number is the
    same at every batch size; empty when absent)."""
    with open(path) as f:
        raw = json.load(f)

    def table(section: dict) -> dict[int, dict[int, float]]:
        return {
            _parse_members(q)[0]: {int(b): float(ms) for b, ms in row.items()}
            for q, row in section.items()
        }

    greedy = table(raw["cost_ms"])
    sampled = table(raw["sampled_cost_ms"]) if "sampled_cost_ms" in raw else greedy

    def levels(section: dict) -> dict[int, dict[int, float]]:
        return {
            _parse_members(q)[0]: (
                {int(b): float(x) for b, x in al.items()}
                if isinstance(al, dict)
                else {1: float(al)}
            )
            for q, al in section.items()
        }

    return (
        greedy,
        sampled,
        levels(raw.get("member_accept_len", {})),
        levels(raw.get("sampled_member_accept_len", {})),
    )


def interpolate_cost(row: dict[int, float], bs: int) -> float:
    """Step cost (or accept length) at ``bs``, linear between measured batch sizes,
    flat outside them."""
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
        return self.rank(bs=bs, expected_accept_lens=expected_accept_lens)[0]

    def rank(self, *, bs: int, expected_accept_lens: list[float]) -> list[int]:
        """The widths by expected tokens per second, best first; ties keep the narrower
        width first."""
        rates = [
            accept_len / cost
            for accept_len, cost in zip(expected_accept_lens, self.costs(bs))
        ]
        order = sorted(range(len(self.widths)), key=lambda j: -rates[j])
        return [self.widths[j] for j in order]


class StaleDeviceStats:
    """Hands the host a recent device vector without a sync.

    ``push`` copies a small device vector into a pinned slot on the current stream and
    records an event; ``latest`` returns the newest pushed vector whose copy has
    finished, or the initial value before any has. With ``lag``, ``latest`` instead
    returns the vector pushed exactly ``lag`` pushes before the newest one (the initial
    value before there is one), waiting for its copy. The copy of a step ``lag`` back
    has normally landed, as the host runs about one step ahead of the device.

    A tensor-parallel group must make the same choice on every rank: ``share`` (e.g. a
    broadcast from rank 0, in place) makes the pushed vectors bitwise equal, since the
    calibration sums use float atomics, and ``lag`` makes every rank read the same push,
    where the newest finished copy depends on each rank's timing."""

    def __init__(
        self,
        *,
        initial: list[float],
        device: torch.device,
        slots: int = 4,
        event_factory=None,
        lag: Optional[int] = None,
        share=None,
    ):
        if lag is not None and not 0 < lag < slots:
            raise ValueError(f"lag {lag} must be in [1, slots={slots})")
        self._lag = lag
        self._share = share
        self._pushes = 0
        self._lagged: Optional[tuple[int, list[float]]] = None
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
        if self._share is not None:
            values = values.float().contiguous().clone()
            self._share(values)
        self._host[slot].copy_(values, non_blocking=True)
        self._events[slot].record()
        self._pending.append(slot)
        self._pushes += 1

    def latest(self) -> list[float]:
        if self._lag is not None:
            push = self._pushes - 1 - self._lag
            if push < 0:
                return self._latest
            if self._lagged is None or self._lagged[0] != push:
                slot = push % len(self._events)
                self._events[slot].synchronize()
                self._lagged = (push, self._host[slot].tolist())
            return self._lagged[1]
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
# Every this many sampled steps verify the runner-up member, and every this many of
# those the members in turn, so the members the policy does not pick stay calibrated.
# Exploring all members in turn every 32 steps was estimated from the cost table to
# cost about 1% of throughput at C16 on Qwen; otherwise arbitrary. Not used when the
# cost table carries measured member accept lengths.
_EXPLORE_EVERY = 32
_EXPLORE_ROUND_ROBIN = 8
# Per-member accept-length scale: decay per verify of the member, and the prior weight
# in accepted tokens that holds it at 1 until data arrives. A weaker prior let a dozen
# exploration steps at batch size 1 (accept length sd ~3 tokens each) move a member's
# scale by 25% on Glimmer at T=1; otherwise arbitrary.
_MEMBER_SCALE_DECAY = 0.995
_MEMBER_SCALE_PRIOR = 1000.0
# Decisions read an exponential average of the per-step predictions, weight per step.
# Sampled: per-step choices on one request's noisy prediction picked the chain on steps
# that looked easy, trained the chain's calibration on them, and lost 5-6% at C1 to the
# best fixed tree on Glimmer and Qwen. The window length is otherwise arbitrary.
_PREDICTION_AVERAGE = 1.0 / 32
# Greedy, replaying forced-32 greedy sweeps' calibration dumps on the combined stack
# (explore/combined/replay_greedy.py, choices one step late): per-step choices on the
# calibrated predictions lost 2.2% to the best fixed width at bs 1-3 on Qwen (32 picked
# too often). Averaged over 32 steps they were within 0.2% of it on Qwen but lost 2% at
# bs 24-32 on Glimmer, where per-step content choices gained 4-9%. Measured levels times
# the content factor, averaged over about two steps, were within 0.2% on Qwen and kept
# +4% / +8% at bs 24 / 32 on Glimmer.
_GREEDY_PREDICTION_AVERAGE = 1.0 / 32
_GREEDY_LEVELS_AVERAGE = 1.0 / 2
# With measured member accept lengths, a member's prediction is its measured length
# times (its per-node estimate / that estimate's long-run average) ** content_weight:
# the per-node model only says how this content differs from the average, since its
# level per member was 5-10% off in relative terms (Glimmer C16: chain vs tree8) under
# traversal verification. The long-run window is arbitrary.
_RAW_AVERAGE = 1.0 / 256


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
        sync_lag: Optional[int] = None,
        share=None,
        ragged: Optional[RaggedChainPolicy] = None,
    ):
        """``sync_lag`` and ``share`` (``StaleDeviceStats``) keep the choices of a
        tensor-parallel group equal on every rank. ``ragged`` plans the ragged chain
        member's rows; it is needed exactly when that member is configured."""
        from sglang.kernels.ops.speculative.dflash_tree import (
            NUM_CALIBRATION_BINS,
            NUM_DRAW_CLASSES,
            sampled_calibration_slots,
        )

        self.widths = config.widths
        self.sampled_members = config.sampled_members
        self.max_width = max(config.widths + config.sampled_members)
        self.chain_rows = chain_rows
        self.has_chain = any(is_chain(m) for m in config.sampled_members)
        self.has_ragged = RAGGED in config.widths + config.sampled_members
        if self.has_ragged != (ragged is not None):
            raise ValueError(
                "the member 'ragged' needs SGLANG_DFLASH_RAGGED_CHAIN_TABLE, and the "
                "table needs that member"
            )
        if ragged is not None and ragged.block != chain_rows:
            raise ValueError(
                f"SGLANG_DFLASH_RAGGED_CHAIN_TABLE is for block {ragged.block}, the "
                f"drafter's block is {chain_rows}"
            )
        self.ragged = ragged
        if self.has_chain and chain_rows > self.max_width:
            raise ValueError(
                f"the chain verifies {chain_rows} rows, more than the server width "
                f"{self.max_width}"
            )
        short = [
            m
            for m in config.sampled_members
            if is_chain(m) and m < CHAIN and -m >= chain_rows
        ]
        if short:
            raise ValueError(
                f"chain members {[member_label(m) for m in short]} need fewer rows than "
                f"the draft block {chain_rows} (the full block is 'chain')"
            )
        self._config = config
        self._device = device

        def stats(slots: int) -> torch.Tensor:
            return torch.zeros((3, slots), dtype=torch.float32, device=device)

        self.calibration_stats = stats(num_depths * NUM_CALIBRATION_BINS)
        # One calibration for every sampled member, the chain included: a separate one
        # per member learned only on the steps the policy picked it for, which made it
        # optimistic for the member picked on easy steps.
        self.sampled_stats = stats(sampled_calibration_slots(num_depths))
        # Filled by refresh_rank_table before the first step.
        self.rank_table = (
            torch.zeros(num_depths * NUM_DRAW_CLASSES, device=device)
            if config.rank_by_depth
            else None
        )
        # Tree widths; the ragged chain (first in widths) takes a column of ones.
        self._width_columns = torch.tensor(
            [width - 1 for width in config.widths if width > CHAIN],
            dtype=torch.long,
            device=device,
        )
        self._sampled_tree_columns = torch.tensor(
            [m - 1 for m in config.sampled_members if m > CHAIN],
            dtype=torch.long,
            device=device,
        )
        # The ragged chain, then the chain members sort before the trees in
        # sampled_members, so their columns lead.
        self._sampled_chain_columns = torch.tensor(
            [self.member_rows(m) - 1 for m in config.sampled_members if is_chain(m)],
            dtype=torch.long,
            device=device,
        )
        force = config.force_width
        self._policy = self._stats = None
        self._greedy_average: Optional[torch.Tensor] = None
        self._greedy_raw_average: Optional[torch.Tensor] = None
        # Host-side {bs: AL} per greedy width, when the table measured every one.
        self._accept_len: Optional[list[dict[int, float]]] = None
        if config.accept_len and all(m in config.accept_len for m in config.widths):
            self._accept_len = [config.accept_len[m] for m in config.widths]
        if len(config.widths) > 1 and force not in config.widths:
            if RAGGED in config.widths and self._accept_len is None:
                raise ValueError(
                    "choosing the ragged chain for greedy batches needs measured "
                    "accept lengths (member_accept_len) for every greedy width"
                )
            self._policy = VerifyWidthPolicy(
                widths=config.widths, cost_ms=config.cost_ms
            )
            # Before the first copy lands, rank widths by cost alone.
            self._stats = StaleDeviceStats(
                initial=[1.0] * len(config.widths),
                device=device,
                lag=sync_lag,
                share=share,
            )
        self._sampled_policy = self._sampled_stats = None
        self._sampled_average: Optional[torch.Tensor] = None
        self._raw_average: Optional[torch.Tensor] = None
        # Host-side {bs: AL} per sampled member, when the table measured every one.
        self._static_accept_len: Optional[list[dict[int, float]]] = None
        if config.sampled_accept_len and all(
            m in config.sampled_accept_len for m in config.sampled_members
        ):
            self._static_accept_len = [
                config.sampled_accept_len[m] for m in config.sampled_members
            ]
        self._sampled_steps = 0
        # Decayed sums of actual and predicted (per-node model) accept length over each
        # member's verifies; their ratio rescales the member's prediction.
        self._member_totals = torch.zeros(
            (2, len(config.sampled_members)), dtype=torch.float32, device=device
        )
        if len(config.sampled_members) > 1 and force not in config.sampled_members:
            if self.has_ragged and self._static_accept_len is None:
                raise ValueError(
                    "choosing the ragged chain member needs measured accept lengths "
                    "(sampled_member_accept_len) for every sampled member"
                )
            self._sampled_policy = VerifyWidthPolicy(
                widths=config.sampled_members, cost_ms=config.sampled_cost_ms
            )
            self._sampled_stats = StaleDeviceStats(
                initial=[1.0] * len(config.sampled_members),
                device=device,
                lag=sync_lag,
                share=share,
            )
        self._states: dict[int, VerifyWidthState] = {}
        self._active: Optional[int] = None
        self._dump = (
            open(config.calibration_dump, "a", buffering=1)
            if config.calibration_dump and _DEBUG_RANK == 0
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
        sync_lag: Optional[int] = None,
        share=None,
    ) -> Optional["AdaptiveVerifyWidth"]:
        config = read_adaptive_width_config(max_width)
        if config is None:
            return None
        ragged = None
        if RAGGED in config.widths + config.sampled_members:
            table = read_ragged_chain_table()
            if table is not None:
                ragged = RaggedChainPolicy(table=table, device=device)
        return cls(
            config=config,
            num_depths=num_depths,
            chain_rows=chain_rows,
            device=device,
            sync_lag=sync_lag,
            share=share,
            ragged=ragged,
        )

    def member_rows(self, member: int) -> int:
        if member == CHAIN:
            return self.chain_rows
        return -member if member < 0 else member

    def ragged_key(self, rows: int) -> int:
        """The chain member whose verify state (backend and graphs) serves a ragged
        chain of ``rows`` rows per request in all."""
        return CHAIN if rows == self.chain_rows else -rows

    def _captured_keys(self) -> list[int]:
        """Verify states to build besides the startup one (the largest width): the
        chain members (with those serving the ragged chain's rows per request), then
        tree widths by row count."""
        force = self._config.force_width
        greedy = (force,) if force in self.widths else self.widths
        sampled = (force,) if force in self.sampled_members else self.sampled_members
        rows = {m for m in greedy + sampled if m > CHAIN}
        keys = sorted(rows - {self.max_width})
        chains = {m for m in sampled if is_chain(m)}
        if RAGGED in greedy + sampled:
            chains |= {self.ragged_key(r) for r in self.ragged.widths}
        return sorted(chains) + keys

    def capture(self, model_runner) -> None:
        """Build a target verify backend and graph set per verify state below the
        startup one, whose backend and graphs serve the largest width. The chain's
        are built as a chain (EAGLE topk 1), so linear-attention layers skip the tree
        links and full attention runs the causal kernel: verifying the same rows as a
        tree cost Qwen3.8 0.2 ms more per step at batch size 8 in the conv update."""
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
        for key in self._captured_keys():
            width = self.member_rows(key)
            override = dict(speculative_num_draft_tokens=width)
            if is_chain(key):
                override["speculative_eagle_topk"] = 1
            with get_spec().override(**override):
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
            self._states[key] = VerifyWidthState(
                attn_backend=attn_backend, graph_runner=graph_runner
            )
        free_after, _ = torch.cuda.mem_get_info(self._device)
        logger.info(
            "DFLASH adaptive tree width: greedy %s, sampled %s, verify states %s; "
            "extra verify backends and graphs took %.2f s and %.2f GB",
            list(self.widths),
            [member_label(m) for m in self.sampled_members],
            [member_label(k) for k in sorted(self._states)],
            time.perf_counter() - tic,
            (free_before - free_after) / (1 << 30),
        )

    def choose_width(self, *, bs: int) -> int:
        """Greedy batches: the width (or the ragged chain) to verify, from earlier
        steps' expected accept lengths (``push_greedy``), before this step's tree is
        built."""
        force = self._config.force_width
        if force is not None and force in self.widths:
            return force
        if self._policy is None:
            return self.max_width
        expected = self._stats.latest()
        if self._accept_len is not None:
            # The pushed values are content factors on the measured levels.
            expected = [
                interpolate_cost(level, bs) * factor
                for level, factor in zip(self._accept_len, expected)
            ]
        return self._policy.choose(bs=bs, expected_accept_lens=expected)

    def push_greedy(self, *, survival, bs: int) -> None:
        """Hand this step's expected accept lengths at every greedy width (with
        measured levels, their content factors), averaged over steps, to later
        steps' choices. ``survival`` is None for a tree without an acceptance
        estimate."""
        if survival is None or self._policy is None:
            return
        mean = survival.prefix_accept_len[:bs, self._width_columns].mean(dim=0)
        if RAGGED in self.widths:
            mean = torch.cat([torch.ones(1, device=mean.device), mean])
        if self._accept_len is not None:
            if self._greedy_raw_average is None:
                self._greedy_raw_average = mean.clone()
            else:
                self._greedy_raw_average.lerp_(mean, _RAW_AVERAGE)
            mean = (mean / self._greedy_raw_average.clamp(min=1e-3)).clamp(min=1e-3)
            mean = mean.pow(self._config.content_weight)
        if self._greedy_average is None:
            self._greedy_average = mean
        else:
            self._greedy_average.lerp_(
                mean,
                _GREEDY_PREDICTION_AVERAGE
                if self._accept_len is None
                else _GREEDY_LEVELS_AVERAGE,
            )
        self._stats.push(self._greedy_average)

    def choose_sampled_member(self, *, bs: int) -> int:
        """Batches with sampling rows: the member to verify, from earlier steps'
        expected accept lengths only. A sampled tree's estimate depends on its draws, so
        reading this step's would make the verified prefix depend on them."""
        force = self._config.force_width
        if force is not None and force in self.sampled_members:
            return force
        if self._sampled_policy is None:
            return self.sampled_members[-1]
        expected = self._sampled_stats.latest()
        if self._static_accept_len is not None:
            # The pushed values are content factors on the measured levels.
            expected = [
                interpolate_cost(level, bs) * factor
                for level, factor in zip(self._static_accept_len, expected)
            ]
        ranked = self._sampled_policy.rank(bs=bs, expected_accept_lens=expected)
        self._sampled_steps += 1
        # Measured member levels need no exploration to stay calibrated.
        if self._static_accept_len is not None or self._sampled_steps % _EXPLORE_EVERY:
            return ranked[0]
        explorations = self._sampled_steps // _EXPLORE_EVERY
        if explorations % _EXPLORE_ROUND_ROBIN:
            return ranked[1]
        return ranked[1:][(explorations // _EXPLORE_ROUND_ROBIN) % (len(ranked) - 1)]

    def predict_sampled(
        self, *, tree_survival, chain_survival, bs: int
    ) -> Optional[torch.Tensor]:
        """Expected accept length of every sampled member, ``[bs, 2 * members]``: the
        per-node model's rescaled by the member's measured ratio, then unscaled. The
        running average of the rescaled batch mean (with measured member levels, of
        the content factors) is pushed for later steps' decisions. None when neither a
        policy nor the dump reads it. The ragged chain has no per-node model of its
        own: it takes the full chain's, or 1 without the chain member."""
        if self._sampled_policy is None and self._dump is None:
            return None
        columns = []
        if self.has_ragged:
            columns.append(
                chain_survival.prefix_accept_len[
                    :bs, self.chain_rows - 1 : self.chain_rows
                ]
                if self.has_chain
                else torch.ones((bs, 1), device=self._device)
            )
        if self.has_chain:
            columns.append(
                chain_survival.prefix_accept_len[:bs, self._sampled_chain_columns]
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
            mean = predicted.mean(dim=0)
            if self._static_accept_len is not None:
                mean = self._content_scaled(raw.mean(dim=0))
            if self._sampled_average is None:
                self._sampled_average = mean
            else:
                self._sampled_average.lerp_(mean, _PREDICTION_AVERAGE)
            self._sampled_stats.push(self._sampled_average)
        return torch.cat([predicted, raw], dim=1)

    def _content_scaled(self, mean_raw: torch.Tensor) -> torch.Tensor:
        """Each member's content factor: its per-node estimate over that estimate's
        long-run average, to the power content_weight."""
        if self._raw_average is None:
            self._raw_average = mean_raw.clone()
        else:
            self._raw_average.lerp_(mean_raw, _RAW_AVERAGE)
        content = (mean_raw / self._raw_average.clamp(min=1e-3)).clamp(min=1e-3)
        return content.pow(self._config.content_weight)

    def activate(self, key: int, model_runner) -> None:
        """Switch the target to the verify state of ``key``: a tree width, or a chain
        member."""
        if key == self._active:
            return
        state = self._states[key]
        model_runner.attn_backend = state.attn_backend
        model_runner.decode_cuda_graph_runner = state.graph_runner
        self._active = key

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
        """Update the sampled calibration from the verified member's build (survival and
        edges in ``proposal``) and that member's accept-length scale."""
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
            stats=self.sampled_stats,
            survival=proposal.survival,
            top_scores_index=proposal.top_scores_index,
            accept_index=accept_index,
            accept_lens=accept_lens,
            num_nodes=self.member_rows(member),
            topk=topk,
        )

    def after_ragged_verify(
        self, *, accept_lens: torch.Tensor, verify_lens: torch.Tensor, greedy: bool
    ) -> None:
        """A ragged chain verify updates no calibration (it verifies no tree build);
        the dump gets each request's accept length and verified rows."""
        if self._dump is not None:
            self._queue_dump(
                {
                    "kind": "greedy" if greedy else "sampled",
                    "member": member_label(RAGGED),
                },
                torch.stack([accept_lens.float(), verify_lens.float()], dim=1),
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
