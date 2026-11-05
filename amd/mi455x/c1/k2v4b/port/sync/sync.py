# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/sync.py
"""The tagged mailbox: hand-offs between the CTAs of one GPU (device scope) and
between GPUs (system scope), every address a named region (``Addr``) recorded
when the kernel is traced (``trace``).

gfx1250 (MI455X) port of ``k2bundle.ops.common.sync``; the public names and
signatures are the MI355X ones (``README.md``). What changed:

- cache policy: GFX12 SCOPE bits (``memmodel.CM_DEV`` = SCOPE_DEV = 16,
  ``CM_SYS`` = SCOPE_SYS = 24) instead of gfx950 SC1 / SC0|SC1. A tagged pair
  is a relaxed scoped 64-bit store / load: ``buffer_store_b64 ... scope:
  SCOPE_SYS`` is what hipcc emits for ``__hip_atomic_store(..., RELAXED,
  SYSTEM)`` (S0 measured that form at 2.02 us a hop, 0.4 us under release /
  acquire, no torn pair in 3e6 epochs);
- ``s_waitcnt vmcnt(0)`` -> ``s_wait_storecnt 0`` (gfx12 counts loads and
  stores apart; every drain here is of stores);
- ``publish`` adds a release fence at the hand-off's scope before the flag: on
  gfx1250 a plain store can sit dirty in its XCD's L2, and LLVM's release
  sequence (``global_wb scope:SCOPE_DEV``) is what writes it back. gfx950's
  ``publish`` relied on the store drain alone;
- the clock is ``llvm.readsteadycounter`` (gfx12 has no ``s_memrealtime``),
  still 100 MHz;
- ``import_alarms`` loads ``tp`` flag words four at a time (TP4: one load);
- the wait budget is a keyword (``ticks``, default WAIT_TICKS) so tests can
  fail fast; the MI355X call forms are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .abi import FAILED, RECORD_WORDS, WAIT_SECONDS
from .memmodel import (
    CM_DEV,
    CM_SYS,
    TICKS_PER_SECOND,
    bf16_pair,
    fence_acquire,
    fence_release,
    ld_i32,
    memrealtime,
    rsrc,
    traced,
    uniform,
    wait_stores,
)
from .plan import Space, note

POLL_MAX = 12  # mailbox specs polled per batch (bounds live registers)

import os as _os
# gfx1250: with s_sleep(1) between re-polls the spinning waves starve the producers (the 4-rank
# probe s3/probe_b2b.py fails within 4-30 launches); s_sleep(32) fixes it (AITER mega_moe_tp: POLL_SLEEP 16). Invalidate: off.
_INV_POLL = _os.environ.get("K2_INV_POLL", "0") == "1"
_POLL_TH = int(_os.environ.get("K2_POLL_TH", "0"))  # GFX12 load temporal hint on polls (1 and 3 do not fix the stale line)
WAIT_TICKS = int(WAIT_SECONDS * TICKS_PER_SECOND)
_POLL_MONITOR = _os.environ.get("K2_POLL_MONITOR", "0") == "1"  # arm global_load_monitor, then s_monitor_sleep


def _monitor_arm(addr):
    """global_load_monitor_b32 on ``addr`` (arms this wave's monitor on its L2 line)."""
    from flydsl._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(T.i32, [fx.Int64(addr).ir_value()],
                     "global_load_monitor_b32 $0, $1, off scope:SCOPE_SE\n s_wait_loadcnt 0x0", "=v,v",
                     has_side_effects=True)


def _mon_wait(load_all):
    """Arm the monitor on the batch's first tag word, then sleep until it changes (plain helper: the
    kernel rewriter does not see attribute calls here)."""
    load_all.arm()
    _monitor_sleep()


def _monitor_sleep():
    """Sleep until the armed line is written or evicted, at most _POLL_SLEEP units (a timeout, as s_sleep)."""
    from flydsl._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(T.i32, [], f"s_monitor_sleep {_POLL_SLEEP}\n s_mov_b32 $0, 0", "=s", has_side_effects=True)

SCOPE_SYS_BITS = 3 << 3
_POLL_SLEEP = int(_os.environ.get("K2_POLL_SLEEP", "32"))  # s_sleep between re-polls (64 clocks a unit)
_POLL_BACKOFF = _os.environ.get("K2_POLL_BACKOFF", "0") == "1"  # sleep 1 (4 polls), 8 (8 polls), then _POLL_SLEEP
_POLL_ATOMIC = _os.environ.get("K2_POLL_ATOMIC", "0") == "1"  # poll with atomic or-0 (performed at the coherence point)
_POLL_GLOBAL = _POLL_ATOMIC or _os.environ.get("K2_POLL_GLOBAL", "0") == "1"  # poll with LLVM monotonic global atomics (mega_moe_tp style)
_PUB_FENCE = _os.environ.get("K2_PUB_FENCE", "1") == "1"  # 0: no L2 write-back in publish (data stored write-through)
_INV_EVERY = int(_os.environ.get("K2_INV_EVERY", "1"))  # invalidate on every Nth re-poll (power of 2)
_INV_SCOPE = _os.environ.get("K2_INV_SCOPE", "SCOPE_DEV")  # SCOPE_DEV passes; SCOPE_SE fails; SYS = LLVM acquire fence
_POLL_CHK = int(_os.environ.get("K2_POLL_CHK", "8"))  # deadline clock + health word read every Nth re-poll (power of 2)
assert _POLL_CHK & (_POLL_CHK - 1) == 0, "K2_POLL_CHK must be a power of 2"
# k2v4: a shorter s_sleep at the hand-offs only a few waves poll (the poll storm needs thousands of waves)
_FAST_SLEEP = int(_os.environ.get("K2_FAST_SLEEP", "32"))
# "site" (K2_FAST_SLEEP) or "site:N" entries, separated by "," or "+"
_FAST_SITES_DEFAULT = "ar:4,xq:4,rlog:4,xqrdy:8,xrdy:8,route:8,fin:8,yr:8,fmsq:8,finld:8"
_FAST_SITES = dict(
    (x.partition(":")[0], int(x.partition(":")[2] or _FAST_SLEEP))
    for x in _os.environ.get("K2_FAST_SITES", _FAST_SITES_DEFAULT).replace("+", ",").split(",") if x
)


# k2v4b (default off): a fast site's wait falls back to _POLL_SLEEP after this many re-polls (0: never), so a long wait (a peer's
# queues unmapped for ms by NUMA balancing) cannot keep hundreds of waves polling at sleep 4 / 8
_FAST_POLLS = int(_os.environ.get("K2_FAST_POLLS", "0"))


def site_sleep(site):
    """The re-poll s_sleep of hand-off ``site`` (``K2_FAST_SITES``), None: ``_POLL_SLEEP``."""
    return _FAST_SITES.get(site)


def _fast_key():
    return f"cs{_POLL_CHK}fp{_FAST_POLLS}" + "+".join(f"{k}{v}" for k, v in sorted(_FAST_SITES.items()))


_PUT_WB = _os.environ.get("K2_PUT_WB", "0") == "1"  # experiment: write the L2 back after every tagged store


def _put_wb(a):
    from flydsl._mlir.dialects import llvm as _llvm

    sc = "SCOPE_SYS" if a.space == Space.PEER else "SCOPE_DEV"
    _llvm.inline_asm(T.i32, [], f"s_wait_storecnt 0x0\n global_wb scope:{sc}\n s_wait_storecnt 0x0\n s_mov_b32 $0, 0", "=s", has_side_effects=True)


def _poll_inv():
    """Invalidate the caches below ``_INV_SCOPE`` before a re-poll, so the next load misses a stale line."""
    if const_expr(_INV_SCOPE == "SYS"):
        fence_acquire(CM_SYS)
    else:
        from flydsl._mlir.dialects import llvm as _llvm

        _llvm.inline_asm(
            T.i32, [], f"global_inv scope:{_INV_SCOPE}\n s_wait_loadcnt 0x0\n s_mov_b32 $0, 0", "=s",
            has_side_effects=True,
        )
  # realtime ticks at 100 MHz


@traced
def _give_up(health, alarm, region_id, index, tag, seen):
    """A CTA's give-up: its first is recorded in its ``health`` record; it then
    raises the output rank's flag for this rank (``alarm``) and drains that
    store before it sets FAILED, so no later store of this rank's can overtake
    the flag."""
    bid = fx.block_idx.x
    rec = (bid + 1) * RECORD_WORDS
    if ld_health(health, rec) == 0:
        at = fx.Int32(memrealtime() & 0x7FFFFFFF)
        record = [1, region_id, index, tag, seen, bid, at, 0]
        for h in range_constexpr(2):  # 16 B a store
            bo.buffer_store(
                fx.Vector.from_elements(
                    [fx.Int32(w) for w in record[4 * h : 4 * h + 4]], fx.Int32
                ),
                rsrc(health),
                rec + 4 * h,
                cache_modifier=CM_SYS,
            )
    bo.buffer_store(fx.Int32(1), rsrc(alarm), 0, cache_modifier=CM_SYS)
    wait_stores()
    bo.buffer_store(fx.Int32(1), rsrc(health), FAILED, cache_modifier=CM_SYS)


@traced
def _spin(load_all, pending, health, alarm, region_id, index, tag, ticks=WAIT_TICKS, sleep=None):
    """Re-issue ``load_all`` until ``pending`` clears, for at most ``ticks``, and
    not at all once this rank has failed (``health`` word FAILED), so a failed
    step runs to its end instead of hanging. A side-effecting op in the loop keeps
    the loads from being hoisted.

    A CTA's first give-up is recorded in its ``health`` record (``_give_up``)."""
    t0 = memrealtime()
    v = load_all()
    why = fx.Int32(0)  # 1: timed out, 2: the rank had failed
    it = fx.Int32(0)
    while pending(v) & (why == 0):
        if const_expr(sleep is not None):
            if const_expr(_FAST_POLLS > 0):
                if it < fx.Int32(_FAST_POLLS):
                    rocdl.s_sleep(sleep)
                else:
                    rocdl.s_sleep(_POLL_SLEEP)
            else:
                rocdl.s_sleep(sleep)
        elif const_expr(_POLL_BACKOFF):
            if it < fx.Int32(4):
                rocdl.s_sleep(1)
            else:
                if it < fx.Int32(12):
                    rocdl.s_sleep(8)
                else:
                    rocdl.s_sleep(_POLL_SLEEP)
        else:
            if const_expr(_POLL_MONITOR):
                _mon_wait(load_all)
            else:
                rocdl.s_sleep(_POLL_SLEEP)
        if const_expr(_INV_POLL):
            if (it == fx.Int32(0)) | ((it & fx.Int32(_INV_EVERY - 1)) == fx.Int32(_INV_EVERY - 1)):
                _poll_inv()
        it = it + 1
        v = load_all()
        if const_expr(_POLL_CHK > 1):
            if (it & fx.Int32(_POLL_CHK - 1)) == fx.Int32(0):
                why = (ld_health(health, FAILED) != 0).select(
                    fx.Int32(2), (memrealtime() - t0 > ticks).select(fx.Int32(1), fx.Int32(0)))
        else:
            failed = ld_health(health, FAILED) != 0
            late = memrealtime() - t0 > ticks
            why = failed.select(fx.Int32(2), late.select(fx.Int32(1), fx.Int32(0)))
    if why == 1:
        _give_up(health, alarm, region_id, index, tag, v[1])
    return v


@traced
def _spin_terminal(
    load_all, pending, health, alarm, region_id, index, tag, ticks=WAIT_TICKS
):
    """Bounded mailbox spin used by K2's terminal failure rendezvous.

    Unlike :func:`_spin`, this must not stop merely because local health is
    already FAILED: a failed rank still has to publish completion and import
    every peer's alarm before any rank is allowed to return from the kernel.
    """
    t0 = memrealtime()
    v = load_all()
    timed_out = fx.Int32(0)
    it = fx.Int32(0)
    while pending(v) & (timed_out == 0):
        if const_expr(_POLL_BACKOFF):
            if it < fx.Int32(4):
                rocdl.s_sleep(1)
            else:
                if it < fx.Int32(12):
                    rocdl.s_sleep(8)
                else:
                    rocdl.s_sleep(_POLL_SLEEP)
        else:
            if const_expr(_POLL_MONITOR):
                _mon_wait(load_all)
            else:
                rocdl.s_sleep(_POLL_SLEEP)
        if const_expr(_INV_POLL):
            if (it == fx.Int32(0)) | ((it & fx.Int32(_INV_EVERY - 1)) == fx.Int32(_INV_EVERY - 1)):
                _poll_inv()
        it = it + 1
        v = load_all()
        if const_expr(_POLL_CHK > 1):
            if (it & fx.Int32(_POLL_CHK - 1)) == fx.Int32(0):
                timed_out = (memrealtime() - t0 > ticks).select(fx.Int32(1), fx.Int32(0))
        else:
            timed_out = (memrealtime() - t0 > ticks).select(fx.Int32(1), fx.Int32(0))
    if timed_out == 1:
        _give_up(health, alarm, region_id, index, tag, v[1])
    return v


@traced
def stall(fault_rank, fault_epoch, fault_layer, ticks, rank, epoch, layer):
    """The fault hook (tests only): CTA 0 of rank ``fault_rank`` stalls ``ticks``
    once, in the launch for slot ``fault_layer`` of the step at epoch
    ``fault_epoch``."""
    if (
        (rank == fault_rank)
        & (uniform(ld_i32(epoch, 0)) == fault_epoch)
        & (layer == fault_layer)
        & (fx.block_idx.x == 0)
    ):
        t0 = memrealtime()
        now = memrealtime()
        while now - t0 < ticks:
            rocdl.s_sleep(127)
            now = memrealtime()


def ld_health(health, i):
    return fx.Int32(
        bo.buffer_load(rsrc(health), i, vec_width=1, dtype=T.i32, cache_modifier=CM_SYS)
    )


@dataclass(frozen=True)
class Addr:
    """A mailbox region's address: the region's name and space, recorded when the
    kernel is traced (``trace``), and the traced address. The space also
    sets the hand-off's scope: device for SCRATCH, system for PEER."""

    region: str
    space: Space
    value: object
    offset: int
    extent: int


def sreg(base, offset, region, extent):
    """Scratch region ``region`` at byte ``offset`` of ``base``."""
    return Addr(region, Space.SCRATCH, base + fx.Int64(offset), offset, extent)


def preg(base, offset, region, extent):
    """Peer-symmetric region ``region`` at byte ``offset`` of a rank's buffer
    ``base``."""
    return Addr(region, Space.PEER, base + fx.Int64(offset), offset, extent)


def shift(a, delta):
    """``a`` moved by ``delta`` bytes (a traced or Python integer), same region."""
    return Addr(a.region, a.space, a.value + delta, a.offset, a.extent)


# the store / load cache policy of a hand-off in each space
_SCOPE_CM = {Space.SCRATCH: CM_DEV, Space.PEER: CM_SYS}


def _noted(a, kind):
    note(a.region, a.space, kind, a.offset, a.extent)
    return a.value


class Mailbox:
    """Tagged-pair hand-off between CTAs (device scope) and GPUs (system scope).

    Every 32-bit value is stored next to this launch's tag; a consumer
    polls the payload until all tags match, so a hand-off costs one round trip
    (no store drain, no separate flag). The tag is ``layer + 1``: K2 passes
    ``step_tag(epoch, slot) - 1``, so it tells the steps (epoch) and the layer
    launches of a step (slot) apart and a pair of an earlier launch never
    carries a live tag. Every address is an ``Addr``: its space picks the scope.
    """

    def __init__(self, layer, health, alarm, region_ids, *, wait_ticks=WAIT_TICKS):
        """``health``: this rank's health words (FAILED, then a wait record a
        CTA); ``alarm``: this rank's flag word on the output rank;
        ``region_ids``: region name -> the id a record names it by;
        ``wait_ticks``: a wait's budget (100 MHz ticks; tests shorten it)."""
        self.tag = layer + 1
        self.health = health
        self.alarm = alarm
        self.region_ids = region_ids
        self.wait_ticks = wait_ticks

    def put(self, a, i, v):
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(
            fx.Vector.from_elements([bits, self.tag], fx.Int32),
            rsrc(_noted(a, "put")),
            i * 2,
            cache_modifier=_SCOPE_CM[a.space],
        )
        if const_expr(_PUT_WB):
            _put_wb(a)

    def put_bf(self, a, i, vs):
        """Elements i .. i+len(vs) (2 or 4, i aligned) as packed bf16 pairs, one
        store."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(_noted(a, "put")),
            i,
            cache_modifier=_SCOPE_CM[a.space],
        )
        if const_expr(_PUT_WB):
            _put_wb(a)

    def put_words(self, a, i, words):
        """Raw 32-bit words (1 or 2) at pairs i.. as (word, tag). Integer payloads
        (fp8 words) must come this way: through an f32 value a NaN bit pattern is
        not kept."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            rsrc(_noted(a, "put")),
            i * 2,
            cache_modifier=_SCOPE_CM[a.space],
        )
        if const_expr(_PUT_WB):
            _put_wb(a)

    def _batch(self, specs):
        """(load_all, pending, region id) of one poll batch, all of one space."""
        spaces = {a.space for a, _, _ in specs}
        assert len(spaces) == 1, f"one poll batch spans {spaces}"
        cm = _SCOPE_CM[spaces.pop()]
        if const_expr(_POLL_TH != 0):
            cm = cm | _POLL_TH
        bases = [_noted(a, "poll") for a, _, _ in specs]
        tag = self.tag

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        if const_expr(_POLL_GLOBAL):
            sc = "one-as" if (cm & SCOPE_SYS_BITS) == SCOPE_SYS_BITS else "agent"

            def load_all():
                words = []
                for b, (_, i, n) in zip(bases, specs):
                    for q in range_constexpr(n):
                        pa = fx.Int64(b) + (fx.Int64(fx.Int32(i)) + q) * 8
                        pp = fx.inttoptr(fx.PointerType.get(T.i64, fx.AddressSpace.Global, 8), pa)
                        if const_expr(_POLL_ATOMIC):
                            w = fx.Int64(fx.atomic_or(pp, fx.Int64(0), syncscope=sc))
                        else:
                            w = fx.Int64(fx.generic_load(pp, dtype=fx.Int64, memory_order=fx.AtomicOrdering.Monotonic,
                                                         syncscope=sc, volatile=True))
                        words += [fx.Int32(w & fx.Int64(0xFFFFFFFF)), fx.Int32(w >> 32)]
                return fx.Vector.from_elements(words, fx.Int32)

            b0, i0 = bases[0], specs[0][1]
            load_all.arm = lambda: _monitor_arm(fx.Int64(b0) + fx.Int64(fx.Int32(i0)) * 8 + 4)
            return load_all, pending, self.region_ids.get(specs[0][0].region, -1)

        def load_all():
            words = []
            for b, (_, i, n) in zip(bases, specs):
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        b0, i0 = bases[0], specs[0][1]
        load_all.arm = lambda: _monitor_arm(fx.Int64(b0) + fx.Int64(fx.Int32(i0)) * 8 + 4)
        return load_all, pending, self.region_ids.get(specs[0][0].region, -1)

    @staticmethod
    def _split(specs, v):
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs

    def poll(self, specs, batch=POLL_MAX, site=None):
        """Batched poll of pairs ``specs`` = [(Addr, pair index, npairs in {1, 2})],
        all of one space.

        All pairs are re-loaded together while any tag is stale, so a batch costs
        one round trip after its last producer lands. Returns the value words.
        ``site``: the hand-off's name for ``site_sleep``."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch, site) + self.poll(specs[batch:], batch, site)
        load_all, pending, rid = self._batch(specs)
        v = _spin(
            load_all, pending, self.health, self.alarm, rid, specs[0][1], self.tag,
            self.wait_ticks, site_sleep(site),
        )  # fmt: skip
        return self._split(specs, v)

    def poll_terminal(self, specs, batch=POLL_MAX):
        """Poll tagged pairs even after local failure.

        This is reserved for the launch-tail rendezvous.  Ordinary dataflow
        waits use :meth:`poll`, which exits promptly once the rank has failed.
        """
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll_terminal(specs[:batch], batch) + self.poll_terminal(
                specs[batch:], batch
            )
        load_all, pending, rid = self._batch(specs)
        v = _spin_terminal(
            load_all, pending, self.health, self.alarm, rid, specs[0][1], self.tag,
            self.wait_ticks,
        )  # fmt: skip
        return self._split(specs, v)


_LDS_BARRIER = _os.environ.get("K2_LDS_BARRIER", "1") == "1"


def cta_barrier():
    """CTA barrier for LDS hand-offs only. ``gpu.barrier`` waits for this wave's outstanding global stores
    too (``s_wait_storecnt 0``): after a peer put that is a whole XGMI write round trip. This one waits for
    LDS ops only, so it orders no global store (use ``publish`` / ``wait_stores`` for that).
    ``K2_LDS_BARRIER=0``: ``gpu.barrier``."""
    if const_expr(_LDS_BARRIER):
        from flydsl._mlir.dialects import llvm as _llvm

        _llvm.inline_asm(
            T.i32, [], "s_wait_dscnt 0x0\n s_barrier_signal -1\n s_barrier_wait -1\n s_mov_b32 $0, 0",
            "=s,~{memory}", has_side_effects=True,
        )
    else:
        gpu.barrier()


@traced
def _gate(poll, specs, who, site=None):
    if who:
        poll(specs, len(specs), site)
    gpu.barrier()


def gate(c, specs, who, site=None):
    """CTA gate: threads ``who`` (one wave) poll ``specs``, then the CTA meets; the polls that follow find
    their pairs at the first load, so one wave a CTA polls instead of all of them."""
    _gate(c["poll"], specs, who, site)


@traced
def publish_wt(put, a, i, v, who):
    """``publish`` for data stored write-through at the pair's scope: the drained stores are past the L2
    already, so no ``global_wb`` (which writes back the XCD's whole L2)."""
    wait_stores()
    gpu.barrier()
    if who:
        put(a, i, v)


def stamp(c, k):
    """Debug timeline word ``k`` of this CTA (K2 with ``K2_DEBUG_TL=1`` only)."""
    f = c.get("tlx")
    if f is not None:
        f(k)


@traced
def import_alarms(health, alarm, rank, tp):
    """Copy the output-rank's shared failure state into this rank's health."""
    head = alarm - fx.Int64(rank) * 4
    words = []
    for q in range_constexpr((tp + 3) // 4):  # buffer loads top out at 16 B
        w = fx.Vector(
            bo.buffer_load(
                rsrc(head), 4 * q, vec_width=4, dtype=T.i32, cache_modifier=CM_SYS
            )
        )
        words += [w[e] for e in range(4)]
    failed = words[0] != 0
    for r in range_constexpr(1, tp):
        failed = failed | (words[r] != 0)
    if failed:
        bo.buffer_store(fx.Int32(1), rsrc(health), FAILED, cache_modifier=CM_SYS)
        wait_stores()


@traced
def publish(put, a, i, v, who):
    """This CTA's plain stores made readable by every CTA (SCRATCH ``a``) or every
    GPU (PEER ``a``), then (thread ``who``) the mailbox pair (``a``, i) := v.

    Every wave drains its stores and the CTA meets, so the stores of all its
    waves have reached the L2; thread ``who``'s release fence at the pair's
    scope then writes the L2 back that far (gfx1250: ``global_wb
    scope:SCOPE_DEV`` / ``SCOPE_SYS``), and only then is the flag stored. A
    workgroup-scope fence would do neither. Consumers read the announced data
    at device scope (``CM_DEV`` loads), which no stale L2 line can satisfy."""
    wait_stores()
    gpu.barrier()
    if who:
        if const_expr(_PUB_FENCE):
            fence_release(_SCOPE_CM[a.space])
        put(a, i, v)
