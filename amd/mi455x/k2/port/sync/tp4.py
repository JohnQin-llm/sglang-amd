# SPDX-License-Identifier: Apache-2.0
"""K2's rank-count constants and owner maps at TP4 / EP1 on MI455X, rescaled
from the MI355X TP8 values (``k2bundle.ops.stages.moe`` / ``k2.py`` /
``oproj.py`` / ``quarter_seam.py``). The stage ports import these instead of
re-deriving them, so the rank count is set in one place.

Every rank keeps all 256 experts (EP1) and 1/TP of every per-rank split. The
model dimensions are unchanged (H 8192, MoE hidden 4096, 256 experts, top-8,
full routed intermediate 4096). ``ROWS`` (16 output rows a task) and the K
splits are the MI355X ones; the derived task counts that no longer fit the
MI355X CTA placement are flagged in ``PLACEMENT_CONFLICTS`` for the stage
porters (the placement is theirs to redo, not this module's).
"""

from __future__ import annotations

from .abi import K2_DONE_RANKS
from .plan import BLOCKS, WAVES

TP = 4  # MI355X: 8
EP = 1  # every rank holds every expert (MI355X K2: also EP1 inside TP8)
MAX_TOKENS = 8

# model dimensions (rank-count independent)
H = 8192  # hidden
MH = 4096  # MoE hidden (pooled)
E = 256
TOPK = 8
INTER = 4096  # an expert's full routed intermediate
ROWS = 16

# per-rank splits
EPR = E // TP  # 64 router rows a rank; MI355X: 32
RI = INTER // TP  # 1024 routed intermediate a rank; MI355X: 512
POOL_N = MH // TP  # 1024 pooled columns a rank; MI355X: 512
UP_N = H // TP  # 2048 branch columns a rank; MI355X: 1024
AR_OWNER_COLS = H // TP  # 2048 AR#1 columns a rank sums; MI355X: 1024
FIN_COLS = 128
FIN_TASKS = MH // FIN_COLS  # 32 (rank independent)
OWNED_FIN_TASKS = FIN_TASKS // TP  # 8 owned 128-column slices; MI355X: 4
OWNER_FIN_CTAS = OWNED_FIN_TASKS * MAX_TOKENS  # 64; MI355X: 32
DOWN_TASKS = MH // ROWS  # 256 W2 output tiles (rank independent)
DOWN_TILES_PER_OWNER = DOWN_TASKS // TP  # 64; MI355X: 32
ROWS_PER_OWNER = MAX_TOKENS // TP  # 2 rows a rank in the EP1 row-owner exchange

# derived counts (wave32: WAVES = 16; MI355X wave64: 8)
R_PARTS = 4  # K quarters of the router GEMV
R_GROUPS = EPR // WAVES  # 4: 4 experts a wave; MI355X: 32 // 8 = 4
R_TASKS = R_GROUPS * R_PARTS  # 16 (unchanged)
P_PARTS = 4
P_GROUPS = POOL_N // ROWS  # 64; MI355X: 32
P_TASKS = P_GROUPS * P_PARTS  # 256; MI355X: 128
UP_TASKS = UP_N // ROWS  # 128; MI355X: 64
INTER_W = RI // 8  # 128 E2M1 words a route; MI355X: 64
UG_GROUPS = RI // 32  # 32 32-column groups a picked expert; MI355X: 16
W2_STEPS = RI // 128  # 8 K steps of the down GEMM; MI355X: 4
W2_SCALE_G = RI // 32  # 32; MI355X: 16

# mailbox / peer-region sizes, pairs per parity at s rows
RLOG_PARTS_PAIRS = TP * R_PARTS * MAX_TOKENS * EPR  # 8192 (unchanged: TP * EPR = E)


def ar_pairs(s: int) -> int:
    """AR region [src][t][column pair]: TP * s * H / 2 (TP4: 131072 at s=8)."""
    return TP * s * H // 2


def moe_peer_pairs(s: int) -> dict:
    """``moe.peer_pairs`` at TP4. yr halves (TP * s * TOPK * MH / 2); rlog keeps
    its size (TP * EPR = E)."""
    return {
        "rlog": TP * s * EPR,
        "pool": s * MH // 2,
        "yr": TP * s * TOPK * MH // 2,
        "br": s * H // 2,
        "owned_fmsq": FIN_TASKS * s * TOPK,
        "owned_fin": s * MH // 2,
    }


# owner maps (Python ints or traced Int32 alike)
def down_owner(rgd):
    """The rank whose finalize owns W2 output tile ``rgd`` (16 columns): the yr
    destination of a down CTA at s = 8 (MI355X ``bid // (DOWN_TASKS // TP)``)."""
    return rgd // DOWN_TILES_PER_OWNER


def two_cta_down_owner(bid):
    """The experimental two-CTA build's member -> output rank: 16 members, now
    four adjacent members a rank (MI355X: ``(bid % 16) // 2``)."""
    return (bid % 16) // (16 // TP)


def ar_owner(col):
    """The rank that sums AR#1 column ``col`` (reduce-scatter destination;
    MI355X ``col // (H // TP)``)."""
    return col // AR_OWNER_COLS


def fin_slice(rank, owner, s):
    """Finalize slice ``j`` of a rank's owner CTA ``owner`` at s = 8:
    ``rank * OWNED_FIN_TASKS + owner // s`` (MI355X form, OWNED_FIN_TASKS 4 -> 8)."""
    return rank * OWNED_FIN_TASKS + owner // s


def fin_slice_owner(j):
    """The rank owning finalize slice ``j``."""
    return j // OWNED_FIN_TASKS


def up_cols(rank):
    """First branch (unpool) column of ``rank``: ``rank * UP_N``."""
    return rank * UP_N


def row_owner(t):
    """EP1 row-owner exchange (S2 decision): row ``t`` of 8 is owned by rank
    t // 2, which receives all 8 routes of it from every rank (128 KB a peer)."""
    return t // ROWS_PER_OWNER


assert K2_DONE_RANKS == TP
assert EPR * TP == E and RI * TP == INTER and POOL_N * TP == MH and UP_N * TP == H
assert FIN_TASKS % TP == 0 and DOWN_TASKS % TP == 0 and MAX_TOKENS % TP == 0

# MI355X CTA placements (k2.py) that the TP4 task counts no longer satisfy,
# name -> (needed, available). The stage porters must re-place these.
PLACEMENT_CONFLICTS = {
    # pooler CTAs P0 = NSL + R_TASKS = 80 .. 80 + P_TASKS must fit the grid
    "pool: P0 + P_TASKS <= BLOCKS": (64 + R_TASKS + P_TASKS, BLOCKS),
    # unpooler CTAs U0 = BLOCKS - UP_TASKS must not overlap gather G0 .. G0 + 64
    "up: G0 + GATHER_TASKS <= U0": (128 + 64, BLOCKS - UP_TASKS),
}
