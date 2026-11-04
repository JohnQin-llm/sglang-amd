"""GPU unit tests of the CTA-level primitives at THREADS = 512 (16 waves of 32):
ops.block_max, ops.block_excl_scan, ops.batched_rounds, ops.rows_to_lds. All
exact (integer / max / copies).

Run: flock /data/muse455/locks/gpu0.lock docker exec muse_mi455x bash -c \
  'cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=0 timeout 300 python3 \
   /data/muse455/k2/ep1/port/common/test_block.py'
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gpu  # noqa: E402

import flydsl.expr as fx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flydsl.expr import gpu as fgpu  # noqa: E402
from flydsl.expr import range_constexpr  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402

import ops  # noqa: E402

R = _gpu.Report(f"CTA primitives (THREADS {ops.THREADS}, WAVES {ops.WAVES} x {ops.WAVE})")
rng = np.random.default_rng(512)
TH, W = ops.THREADS, ops.WAVES
R.check("WAVES = THREADS // 32", W == TH // 32 and ops.WAVE == 32)


def rs(p):
    return ops.rsrc(p)


def ldf(p, i):
    return fx.Float32(ops.bo.buffer_load(rs(p), i, vec_width=1, dtype=T.f32))


def ldi(p, i):
    return fx.Int32(ops.bo.buffer_load(rs(p), i, vec_width=1, dtype=T.i32))


def st(v, p, i):
    ops.bo.buffer_store(ops._bits(v), rs(p), i)


def dev(x):
    return torch.from_numpy(np.ascontiguousarray(x)).cuda()


@fx.struct
class Scratch:
    red: fx.Array[fx.Float32, W, 16]
    buf: fx.Array[fx.Int32, W, 16]
    rows: fx.Array[fx.Int32, 8 * 40, 16]


# -------------------------------------------------- block_max / excl_scan
xf = (rng.standard_normal(TH) * 100).astype(np.float32)
xi = rng.integers(-1000, 1000, TH).astype(np.int32)


def body(tid, a, b, c, d, sm):
    lane, wave = tid % 32, ops.uniform(tid // 32)
    st(ops.block_max(ldf(a, tid), lane, wave, sm.red.ptr), b, tid)
    off, tot = ops.block_excl_scan(ldi(c, tid), lane, wave, sm.buf.ptr)
    st(off, b, TH + tid)
    st(tot, b, 2 * TH + tid)
    # a second call reusing the buffers (the barriers keep them apart)
    st(ops.block_max(-ldf(a, tid), lane, wave, sm.red.ptr), b, 3 * TH + tid)


out = torch.zeros(4 * TH, dtype=torch.int32, device="cuda")
_gpu.run(_gpu.kernel(body, TH, lds=Scratch), dev(xf), out, dev(xi))
o = out.cpu().numpy()
R.check("block_max over 16 waves", np.all(o[:TH].view(np.float32) == xf.max()))
R.check("block_excl_scan: exclusive prefix in thread order", np.array_equal(o[TH:2 * TH], np.cumsum(xi) - xi))
R.check("block_excl_scan: CTA total", np.all(o[2 * TH:3 * TH] == xi.sum()))
R.check("block_max reused right after", np.all(o[3 * TH:].view(np.float32) == (-xf).max()))

# ---------------------------------------------------------- batched_rounds
n = 3000  # 6 rounds of 512, the last partial: two batches of LOAD_BATCH = 4
src = rng.integers(-2**30, 2**30, n).astype(np.int32)


def body(tid, a, b, c, d):
    ops.batched_rounds(tid, n, lambda v: ldi(a, v), lambda v, g: st(g * 3 + v, b, v))


out = torch.full((n + 64,), -7, dtype=torch.int32, device="cuda")
L = _gpu.kernel(body, TH)
_gpu.run(L, dev(src), out)
o = out.cpu().numpy()
R.check(f"batched_rounds: every v < {n} used once with its own load", np.array_equal(o[:n], src * 3 + np.arange(n)))
R.check("batched_rounds: nothing past n", np.all(o[n:] == -7))
asm = _gpu.kernel_text(_gpu.isa(L), L.kernel_name)
first_store = asm.find("buffer_store")
R.check("batched_rounds: a batch's 4 loads issue before its first store",
        asm[:first_store].count("buffer_load_b32") >= ops.LOAD_BATCH)

# ------------------------------------------------------------- rows_to_lds
ROWS, RW, DR = 8, 36, 40
srcw = rng.integers(-2**31, 2**31 - 1, ROWS * RW, dtype=np.int64).astype(np.int32)


def body(tid, a, b, c, d, sm):
    dst = sm.rows.ptr
    for i in range_constexpr(-(-ROWS * DR // TH)):
        e = tid + TH * i
        if e < ROWS * DR:
            fx.ptr_store(fx.Int32(-1), dst + e)
    fgpu.barrier()
    ops.rows_to_lds(tid, a, ROWS, RW, dst, DR)
    fgpu.barrier()
    for i in range_constexpr(-(-ROWS * DR // TH)):
        e = tid + TH * i
        if e < ROWS * DR:
            st(fx.ptr_load(dst + e), b, e)


out = torch.zeros(ROWS * DR, dtype=torch.int32, device="cuda")
L = _gpu.kernel(ops.traced(body), TH, lds=Scratch)
_gpu.run(L, dev(srcw), out)
o = out.cpu().numpy().reshape(ROWS, DR)
R.check(f"rows_to_lds: {ROWS} rows x {RW} words -> LDS rows of {DR}",
        np.array_equal(o[:, :RW], srcw.reshape(ROWS, RW)) and np.all(o[:, RW:] == -1))
asm = _gpu.kernel_text(_gpu.isa(L), L.kernel_name)
R.check("rows_to_lds: b128 loads at device scope", "buffer_load_b128" in asm and "scope:SCOPE_DEV" in asm)

R.done()
