# SPDX-License-Identifier: Apache-2.0
"""gfx1250 (MI455X) memory-model primitives the mailbox needs: the buffer
cache-policy words, the store / load drains, release fences, the 100 MHz clock,
and the few ``ops`` helpers ``sync`` / ``ranks`` call, under their
``k2bundle.ops.common.ops`` names so the ported ``ops.py`` can re-export them.

Cache policy (LLVM ``CPol`` for GFX12, the buffer intrinsics' ``aux``): bits
0-2 TH (temporal hint), bits 3-4 SCOPE. gfx950's SC0 / SC1 / NT bits do not
exist here; the same numbers mean other things (gfx950 CM_SYS = 17 would be
SCOPE_DEV | TH_NT, gfx950 CM_NT = 2 would be TH_HT). What hipcc emits for the
HIP atomics K2's hand-offs correspond to (``probe/memmodel_probe.s``):

- relaxed system-scope 64-bit store / load: ``global_{store,load}_b64 ...
  scope:SCOPE_SYS``, no fence, no wait: the tagged mailbox pair;
- relaxed agent-scope: the same with ``scope:SCOPE_DEV``;
- release fence, agent: ``s_wait_storecnt 0; global_wb scope:SCOPE_DEV;
  s_wait_storecnt 0`` (the per-XCD L2 is not coherent across the device, so
  plain stores need a write-back before a flag may announce them);
- release fence, system: the same with ``scope:SCOPE_SYS``;
- acquire, agent: ``global_inv scope:SCOPE_DEV``;
- the clock: ``s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`` (100 MHz, as gfx950's
  ``s_memrealtime``, which gfx12 no longer has).
"""

from __future__ import annotations

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import rocdl
from flydsl.expr.typing import T

# GFX12 CPol
TH_RT = 0  # regular temporal (default)
TH_NT = 1  # nontemporal
TH_HT = 2  # high temporal
SCOPE_CU = 0 << 3
SCOPE_SE = 1 << 3
SCOPE_DEV = 2 << 3
SCOPE_SYS = 3 << 3

CM_DEV = SCOPE_DEV  # 16: device scope (past the per-XCD L2); gfx950: SC1 = 16
CM_SYS = SCOPE_SYS  # 24: system scope (peers over XGMI); gfx950: SC0 | SC1 = 17
CM_NT = TH_NT  # 1: stream a once-read weight past the caches; gfx950: NT = 2

TICKS_PER_SECOND = 100_000_000  # the realtime counter (S0: wall clock 100000 kHz)

_SCOPE_NAME = {SCOPE_DEV: "agent", SCOPE_SYS: "one-as"}


def traced(fn):
    """Apply flydsl's AST rewriting to a helper, so its ``while`` / ``if`` over
    traced values lower to scf ops. Kernels rewrite only their own body."""
    return ASTRewriter.transform(fn)


def memrealtime():
    """The 100 MHz device clock: ``llvm.readsteadycounter`` (gfx12's
    ``s_sendmsg_rtn_b64 REALTIME``; the gfx950 name is kept)."""
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))


def wait_stores():
    """Every store this wave issued complete (gfx950: ``s_waitcnt vmcnt(0)``,
    which also covered loads; gfx12 counts them apart)."""
    rocdl.s_wait_storecnt(0)


def wait_loads():
    rocdl.s_wait_loadcnt(0)


def fence_release(scope):
    """Release fence at ``scope`` (CM_DEV or CM_SYS): the LLVM memory model's
    sequence, i.e. drain stores and write the L2 back to that scope."""
    llvm.FenceOp(llvm.AtomicOrdering.release, syncscope=_SCOPE_NAME[scope])


def fence_acquire(scope):
    """Acquire fence at ``scope``: invalidate the caches below it."""
    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope=_SCOPE_NAME[scope])


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def ld_i32(ptr, i):
    """Word ``i`` of a buffer, a plain Int32 load."""
    return fx.Int32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32))


def load_ptr64(table, i):
    """Entry ``i`` of an i64 table in memory, wave-uniform (each half through
    ``uniform``): a peer's buffer base, a kernel argument block's pointer."""
    pw = fx.Vector(bo.buffer_load(rsrc(table), i * 2, vec_width=2, dtype=T.i32))
    return (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(fx.Uint32(uniform(pw[0])))


def bf16_pair(a, b):
    """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Float32)[0]
    )


def bf_lo(w):
    """The f32 of a packed bf16 pair word's low half."""
    return (w << 16).bitcast(fx.Float32)


def bf_hi(w):
    """The f32 of a packed bf16 pair word's high half."""
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def bf2_f32(w):
    """Packed bf16 pair word -> (f32 low, f32 high)."""
    return bf_lo(w), bf_hi(w)
