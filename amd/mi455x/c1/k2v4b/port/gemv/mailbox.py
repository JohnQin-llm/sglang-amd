"""A minimal gfx1250 tagged mailbox for the single-GPU stage tests.

Same calling convention as ``k2bundle/ops/common/sync.Mailbox`` (``put`` / ``put_bf`` / ``put_words`` / ``poll``
on (region, pair index) with a launch tag next to every 32-bit value), so the stage code is the code K2 will run.
Differences, all test-harness only: a region is an ``Addr`` (address, peer?) without trace bookkeeping; the
spin is bounded by an iteration count (no ``s_memrealtime`` on gfx12) and a give-up raises word 0 of ``err``
instead of the health/alarm protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from .common import CM_DEV, CM_DEV_ST, CM_SYS, CM_SYS_ST, bf16_pair, rsrc, traced

POLL_MAX = 12
SPIN_MAX = 1 << 22


@dataclass(frozen=True)
class Addr:
    value: object  # Int64 address
    peer: bool = False  # system scope (another GPU's buffer) vs device scope (scratch)


def shift(a, delta):
    return Addr(a.value + delta, a.peer)


def _ld_cm(a):
    return CM_SYS if a.peer else CM_DEV


def _st_cm(a):
    return CM_SYS_ST if a.peer else CM_DEV_ST


@traced
def _spin(load_all, pending, err):
    v = load_all()
    n = fx.Int32(0)
    while pending(v) & (n < SPIN_MAX):
        rocdl.s_sleep(1)
        v = load_all()
        n = n + 1
    if n >= SPIN_MAX:
        bo.buffer_store(fx.Int32(1), rsrc(err), 0, cache_modifier=CM_DEV_ST)
    return v


class Mailbox:
    def __init__(self, tag, err):
        self.tag = fx.Int32(tag)
        self.err = err

    def put(self, a, i, v):
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(
            fx.Vector.from_elements([bits, self.tag], fx.Int32), rsrc(a.value), i * 2, cache_modifier=_st_cm(a)
        )

    def put_bf(self, a, i, vs):
        """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), self.tag]
        bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(a.value), i, cache_modifier=_st_cm(a))

    def put_words(self, a, i, words):
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        bo.buffer_store(fx.Vector.from_elements(vals, fx.Int32), rsrc(a.value), i * 2, cache_modifier=_st_cm(a))

    def poll(self, specs, batch=POLL_MAX):
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        cms = {_ld_cm(a) for a, _, _ in specs}
        assert len(cms) == 1, "one poll batch spans two scopes"
        cm = cms.pop()
        tag = self.tag

        def load_all():
            words = []
            for a, i, n in specs:
                w = fx.Vector(
                    bo.buffer_load(rsrc(a.value), fx.Int32(i) * 2, vec_width=2 * n, dtype=T.i32, cache_modifier=cm)
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        v = _spin(load_all, pending, self.err)
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs
