"""K2's o_proj on gfx1250 at TP4 (port of ``k2bundle/ops/stages/oproj.py``): the core rows (T x 2048 bf16, this
rank's K slice: 32 q heads x 64) against the rank's RowParallel shard of o_proj (bf16 [8192, 2048]) -> bf16
partials -> AR#1.

512 tiles of 16 rows, two a CTA (b, b + 256).  The CTA's 16 waves split them: waves 8 j .. 8 j + 7 own tile j,
each two 128-K chunks of its K = 2048 (256 B of weights a lane, every load issued before the core rows land),
so a row's fp32 sum is its tile's 8 wave sums in wave order.  MI355X's C1 build (``stage_oproj_deep``) kept one
wave a tile with a single K chain to match the production gfx950 kernel bit for bit; on gfx1250 there is no
such reference to match, and the split-K form has a quarter of the per-wave latency chain.

``stage_oproj_deep`` (C1, q8) reduce-scatters: a tile's partials go to its column owner only (``ar_owner``,
owner = row // 2048).  ``stage_oproj`` (the default build) pushes them to every rank (``ar_peer``).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from port.sync.sync import cta_barrier

from . import common as C
from . import gemv
from .common import bf16_round, rsrc, traced

TP = C.TP
H = C.H  # o_proj's rows (hidden)
CORE = H // TP  # 2048: o_proj's K a rank
ROWS = 16
OTASKS = H // ROWS  # 512
O_ROUNDS = OTASKS // C.BLOCKS  # 2 tiles a CTA
CORE_ROW = CORE + gemv.LDS_PAD  # bf16 a padded core row in LDS
SPLIT = C.WAVES // O_ROUNDS  # 8 waves a tile
OWNER_COLS = H // TP  # 2048 columns a rank owns after the reduce-scatter


def ar_pairs(s):
    """One parity's AR region: [src rank][token][column pair]."""
    return TP * s * H // 2


def owner_of_tile(rg):
    return (rg * ROWS) // OWNER_COLS


def o_loads(c, bid):
    """This wave's chunks of its tile (``bid + BLOCKS * (wave // SPLIT)``): issue before ``load_core``."""
    lane, wave = c["lane"], c["wave"]
    j = wave // SPLIT
    rg = bid + C.BLOCKS * j
    return gemv.loads(lane, wave % SPLIT, c["args"]["w_o"], rg * ROWS, CORE, waves=SPLIT, nbytes=H * CORE * 2)


@traced
def load_core(c):
    """The core rows (T x 2048 bf16, rows ``CORE`` apart) -> LDS, rows padded."""
    gemv.rows_to_lds(c["tid"], c["args"]["core"], CORE, c["S"], c["xl"], CORE_ROW)


@traced
def _publish_tiles(c, bid, owner_only):
    """Thread (tile j, token, 4-row group): its tile's wave sums in wave order -> bf16 -> AR#1."""
    s, tid, red = c["S"], c["tid"], c["red"]
    per_tile = s * (ROWS // 4)
    if tid < O_ROUNDS * per_tile:
        j = tid // per_tile
        tt = (tid % per_tile) // (ROWS // 4)
        rq = tid % (ROWS // 4)
        vals = []
        for d in range_constexpr(4):
            r = 4 * rq + d
            tot = fx.Float32(0.0)
            for w in range_constexpr(SPLIT):
                tot = tot + fx.ptr_load(red + C.acc_slot(j * SPLIT + w, r, tt))
            vals.append(bf16_round(tot))
        rg = bid + C.BLOCKS * j
        at = (c["rank"] * s + tt) * H + rg * ROWS + 4 * rq
        if const_expr(owner_only):
            for jj in range_constexpr(O_ROUNDS):
                if j == jj:
                    c["put_bf"](c["ar_owner"][jj], at, vals)
        else:
            for p in range_constexpr(TP):
                c["put_bf"](c["ar_peer"][p], at, vals)
    cta_barrier()


@traced
def _oproj_compute(c, bid):
    s, lane, wave, red = c["S"], c["lane"], c["wave"], c["red"]
    ops = o_loads(c, bid)
    load_core(c)
    acc = gemv.wmmas(lane, wave % SPLIT, c["xl"], CORE_ROW, CORE, s, ops, waves=SPLIT)
    gemv.store_acc(red, wave, lane, acc)
    gpu.barrier()


@traced
def stage_oproj_deep(c, bid):
    """C1 (q8) o_proj: CTA ``bid``'s tiles bid, bid + 256 -> the column owners' AR#1 slots (reduce-scatter)."""
    _oproj_compute(c, bid)
    _publish_tiles(c, bid, True)


@traced
def stage_oproj(c, bid):
    """Default-build o_proj: the same tiles -> every rank's AR#1 slot (push-all)."""
    _oproj_compute(c, bid)
    _publish_tiles(c, bid, False)
