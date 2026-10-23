# SPDX-License-Identifier: Apache-2.0
"""Pure host-side contract for the native M16/M24 FullK2 schedules.

Keep this module free of torch/FlyDSL imports.  It is both the single source
of truth for wide ownership and a cheap falsification oracle that can run on
build hosts without a GPU software stack.
"""

from __future__ import annotations

from dataclasses import dataclass

TP = 8
TOPK = 8
BLOCKS = 512
WIDE_TOKENS = (16, 24)
MAX_TOKENS = max(WIDE_TOKENS)
ROUTER_BASE = 4 * MAX_TOKENS
ROUTER_TASKS = 16  # logical (expert group, K quarter) partials, not CTA count
ROUTER_ROWS = 8
ROUTER_CTAS = ROUTER_TASKS * (MAX_TOKENS // ROUTER_ROWS)
POOL_BASE = ROUTER_BASE + ROUTER_CTAS
POOL_TASKS = 128
XQ_BASE = MAX_TOKENS
EXPERT_TEAMS = 32


def require_width(tokens: int) -> int:
    if tokens not in WIDE_TOKENS:
        raise ValueError(
            f"native FullK2 width must be one of {WIDE_TOKENS}, got {tokens}"
        )
    return tokens


def front_tiles(tokens: int) -> tuple[tuple[int, int], ...]:
    """Physical M16 tiles; M24 has an explicit live-eight tail."""
    require_width(tokens)
    return ((0, 16),) if tokens == 16 else ((0, 16), (16, 8))


def expert_tiles(tokens: int, fan: int | None = None) -> tuple[tuple[int, int], ...]:
    """Physical expert tiles for one union expert.

    The M24 compact-pick layout makes the first physical M16 tile dense in
    live picks.  The M8 tail therefore exists only when the expert fan exceeds
    sixteen; a zero-fan expert is never scheduled.
    """
    tokens = require_width(tokens)
    if fan is None:
        fan = tokens
    if not 0 <= fan <= tokens:
        raise ValueError(f"expert fan must be in [0, {tokens}], got {fan}")
    if fan == 0:
        return ()
    if tokens == 16 or fan <= 16:
        return ((0, 16),)
    return ((0, 16), (16, 8))


def tail_tiles(tokens: int) -> tuple[tuple[int, int], ...]:
    """Physical M8 tiles used by unpool and final branch gather."""
    require_width(tokens)
    return tuple((base, 8) for base in range(0, tokens, 8))


def union_capacity(tokens: int) -> int:
    return require_width(tokens) * TOPK


def seam_owner(bid: int, tokens: int) -> tuple[int, int] | None:
    if not 0 <= bid < 4 * require_width(tokens):
        return None
    return bid // 4, bid % 4


def final_owner(bid: int, tokens: int) -> tuple[int, int] | None:
    """Map the last 4*M CTAs to (128-column slice, global token)."""
    tokens = require_width(tokens)
    base = BLOCKS - 4 * tokens
    if not base <= bid < BLOCKS:
        return None
    owner = bid - base
    return owner // tokens, owner % tokens


def gather_owner(bid: int, tokens: int) -> tuple[int, int] | None:
    """64/96 CTAs: a 32-column slice for each physical M8 row tile."""
    tokens = require_width(tokens)
    owners = 4 * tokens
    if not 0 <= bid < owners:
        return None
    tile, column = divmod(bid, 32)
    return 8 * tile, column


def expert_slots(team: int, nu: int) -> tuple[int, ...]:
    if not 0 <= team < EXPERT_TEAMS:
        raise ValueError("expert team must be in [0, 32)")
    if not 0 <= nu <= 24 * TOPK:
        raise ValueError("union size is outside the M24 capacity")
    return tuple(range(team, nu, EXPERT_TEAMS))


def deterministic_union(
    routes: list[list[int]], tokens: int
) -> tuple[list[int], list[int], list[int]]:
    """Reference ascending union and device-layout slot-major picks.

    M16 keeps the direct ``slot*M + token -> route`` table.  M24 compacts each
    live word to ``token << 4 | route`` so experts with fan-in <=16 use one
    physical M16 tile, including the measured M24/U59 route set.  Unused rows
    are -1.  The device publishes only after experts, picks, and fans complete.
    """
    tokens = require_width(tokens)
    if len(routes) != tokens or any(len(row) != TOPK for row in routes):
        raise ValueError(f"routes must have shape [{tokens}, {TOPK}]")
    if any(not 0 <= expert < 256 for row in routes for expert in row):
        raise ValueError("expert ids must be in [0, 256)")
    if any(len(set(row)) != TOPK for row in routes):
        raise ValueError("each token must route to eight distinct experts")
    experts = sorted({expert for row in routes for expert in row})
    if len(experts) > union_capacity(tokens):
        raise AssertionError("union exceeds its mathematical capacity")
    picks = [-1] * (len(experts) * tokens)
    fans = []
    for slot, expert in enumerate(experts):
        fan = 0
        for token, row in enumerate(routes):
            if expert in row:
                at = fan if tokens == 24 else token
                word = (
                    token << 4 | row.index(expert)
                    if tokens == 24
                    else row.index(expert)
                )
                picks[slot * tokens + at] = word
                fan += 1
        fans.append(fan)
    return experts, picks, fans


@dataclass(frozen=True)
class WideSchedule:
    tokens: int

    def __post_init__(self) -> None:
        require_width(self.tokens)

    @property
    def seam_ctas(self) -> range:
        return range(4 * self.tokens)

    @property
    def router_ctas(self) -> range:
        return range(
            ROUTER_BASE, ROUTER_BASE + ROUTER_TASKS * (self.tokens // ROUTER_ROWS)
        )

    @property
    def route_ctas(self) -> range:
        return range(self.tokens)

    @property
    def xq_ctas(self) -> range:
        return range(XQ_BASE, XQ_BASE + self.tokens)

    @property
    def pool_ctas(self) -> range:
        return range(POOL_BASE, POOL_BASE + POOL_TASKS)

    @property
    def final_ctas(self) -> range:
        return range(BLOCKS - 4 * self.tokens, BLOCKS)

    @property
    def gather_ctas(self) -> range:
        return range(4 * self.tokens)

    def validate(self) -> None:
        # Early owners execute in program order.  Any overlap with seam owners
        # would make a CTA wait on work that it has not yet produced.
        assert set(self.seam_ctas).isdisjoint(self.router_ctas)
        assert set(self.seam_ctas).isdisjoint(self.pool_ctas)
        assert set(self.router_ctas).isdisjoint(self.pool_ctas)
        assert set(self.route_ctas).isdisjoint(self.xq_ctas)
        # Both consumers finish their seam work first. Early producers never
        # own seam work, so these waits cannot point backwards on a CTA.
        assert set(self.route_ctas) | set(self.xq_ctas) <= set(self.seam_ctas)
        assert POOL_BASE + POOL_TASKS <= BLOCKS
        assert max(self.pool_ctas) < min(self.final_ctas)
        assert (
            len({seam_owner(b, self.tokens) for b in self.seam_ctas})
            == 4 * self.tokens
        )
        assert (
            len({final_owner(b, self.tokens) for b in self.final_ctas})
            == 4 * self.tokens
        )
        assert (
            len({gather_owner(b, self.tokens) for b in self.gather_ctas})
            == 4 * self.tokens
        )


for _tokens in WIDE_TOKENS:
    WideSchedule(_tokens).validate()

