"""GPU unit tests of the wave32 lane primitives of ``ops.py``: xshfl, xred,
butterfly, wave_sum / wave_max (32 lanes, and the logical wave64 (lo, hi) pair
in the gfx950 order), permlane_swap, lane_gather, readlane, uniform, row_shr,
wave_incl_scan, ballot / popcount / lanes_below, fresh.

Every check is exact: float reductions are compared bit for bit against a numpy
float32 emulation of the same lane order (for pairs: the gfx950 64-lane
butterfly), integer results by equality.

Run: flock /data/muse455/locks/gpu0.lock docker exec muse_mi455x bash -c \
  'cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=0 timeout 300 python3 \
   /data/muse455/k2/cand/k2v6/port/common/test_lane.py'
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gpu  # noqa: E402  (sets the dump env before flydsl loads)

import flydsl.expr as fx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402

import ops  # noqa: E402

R = _gpu.Report("ops lane primitives (wave32)")
rng = np.random.default_rng(1250)


def rs(p):
    return ops.rsrc(p)


def ldf(p, i):
    return fx.Float32(ops.bo.buffer_load(rs(p), i, vec_width=1, dtype=T.f32))


def ldi(p, i):
    return fx.Int32(ops.bo.buffer_load(rs(p), i, vec_width=1, dtype=T.i32))


def st(v, p, i):
    ops.bo.buffer_store(ops._bits(v), rs(p), i)


def f32_input(n):
    """Mixed magnitudes, so a different summation order changes the bits."""
    x = rng.standard_normal(n).astype(np.float32) * np.exp2(rng.integers(-12, 12, n)).astype(np.float32)
    return x


def order_sensitive(n=64):
    """An input whose 64-lane sum differs bitwise between the gfx950 order
    (32 first) and the two other natural orders, so passing proves the order."""
    while True:
        x = f32_input(n)
        a = emu_butterfly(x, (32, 16, 8, 4, 2, 1), np.add)[0].view(np.int32)
        b = emu_butterfly(x, (1, 2, 4, 8, 16, 32), np.add)[0].view(np.int32)
        c = emu_butterfly(x, (16, 8, 4, 2, 1, 32), np.add)[0].view(np.int32)
        if a != b and a != c:
            return x


def emu_butterfly(v, offsets, op):
    v = v.copy()
    n = len(v)
    for off in offsets:
        idx = np.arange(n)
        lo = np.minimum(idx, idx ^ off)
        hi = np.maximum(idx, idx ^ off)
        v = op(v[lo], v[hi]).astype(v.dtype)
    return v


def gpu(body, a, out_words, threads=32, dtype=torch.int32):
    out = torch.zeros(out_words, dtype=dtype, device="cuda")
    L = _gpu.kernel(body, threads)
    _gpu.run(L, a, out)
    return out.cpu(), L


def ftens(x):
    return torch.from_numpy(np.ascontiguousarray(x)).cuda()


def bits_eq(got_i32, ref_f32):
    g = got_i32.numpy().view(np.float32)
    return bool(np.array_equal(g.view(np.int32), np.asarray(ref_f32, np.float32).view(np.int32)))


# ------------------------------------------------------------- xshfl / xred
x = f32_input(32)
xi = rng.integers(-2**31, 2**31 - 1, 32, dtype=np.int64).astype(np.int32)
for off in (1, 2, 4, 8, 16):
    def body(tid, a, b, c, d, off=off):
        st(ops.xshfl(ldf(a, tid), off), b, tid)
        st(ops.xshfl(ldi(c, tid), off), b, 32 + tid)
        st(ops.xred(ldf(a, tid), off, lambda p, q: p + q), b, 64 + tid)
    out = torch.zeros(96, dtype=torch.int32, device="cuda")
    L = _gpu.kernel(body, 32, name=f"k2c_xshfl_{off}")
    _gpu.run(L, ftens(x), out, ftens(xi))
    o = out.cpu()
    perm = np.arange(32) ^ off
    R.check(f"xshfl f32 off {off} = lane ^ {off}", bits_eq(o[:32], x[perm]))
    R.check(f"xshfl i32 off {off}", np.array_equal(o[32:64].numpy(), xi[perm]))
    R.check(f"xred add off {off} bit-exact", bits_eq(o[64:], emu_butterfly(x, (off,), np.add)))
    asm = _gpu.kernel_text(_gpu.isa(L), L.kernel_name)
    want = "v_permlanex16_b32" if off == 16 else f"row_xmask:{off}"
    R.check(f"xshfl off {off} lowers to {want} (no LDS)", want in asm and "ds_bpermute" not in asm)

# ---------------------------------------------------- butterflies, 32 lanes
for name, offs, op, nop in (
    ("wave_sum", None, None, np.add),
    ("wave_max", None, None, np.maximum),
    ("butterfly (8,4,2,1)", (8, 4, 2, 1), None, np.add),
    ("butterfly (1,2,4,8,16)", (1, 2, 4, 8, 16), None, np.add),
    ("butterfly (16,8,4,2,1) max", (16, 8, 4, 2, 1), fx.max, np.maximum),
):
    x = f32_input(32)

    def body(tid, a, b, c, d, name=name, offs=offs, op=op):
        v = ldf(a, tid)
        if name == "wave_sum":
            r = ops.wave_sum(v)
        elif name == "wave_max":
            r = ops.wave_max(v)
        elif op is None:
            r = ops.butterfly(v, offs)
        else:
            r = ops.butterfly(v, offs, op)
        st(r, b, tid)

    o, _ = gpu(body, ftens(x), 32)
    ref = emu_butterfly(x, offs or (16, 8, 4, 2, 1), nop)
    R.check(f"{name} 32 lanes bit-exact", bits_eq(o, ref))

# ------------------------------------- logical wave64 pairs (gfx950 order)
# the stage calls: wave_sum, butterfly (1, 2, 32, 16, 8, 4) / (8, 4, 2, 1, 16) /
# (16, 8, 4, 2, 1) / (2, 1) max, xshfl(., 1) + butterfly (16, 8, 4, 2)
for name, offs, op, nop in (
    ("wave_sum pair", (32, 16, 8, 4, 2, 1), "sum", np.add),
    ("wave_max pair", (32, 16, 8, 4, 2, 1), "max", np.maximum),
    ("butterfly pair (1,2,32,16,8,4)", (1, 2, 32, 16, 8, 4), None, np.add),
    ("butterfly pair (8,4,2,1,16) (no 32: stays a pair)", (8, 4, 2, 1, 16), None, np.add),
    ("butterfly pair (2,1) max", (2, 1), fx.max, np.maximum),
):
    x = order_sensitive()

    def body(tid, a, b, c, d, offs=offs, op=op):
        v = (ldf(a, tid), ldf(a, 32 + tid))
        if op == "sum":
            r = ops.wave_sum(v)
        elif op == "max":
            r = ops.wave_max(v)
        elif op is None:
            r = ops.butterfly(v, offs)
        else:
            r = ops.butterfly(v, offs, op)
        if isinstance(r, tuple):
            st(r[0], b, tid)
            st(r[1], b, 32 + tid)
        else:
            st(r, b, tid)
            st(r, b, 32 + tid)

    o, _ = gpu(body, ftens(x), 64)
    ref = emu_butterfly(x, offs, nop)
    R.check(f"{name} = gfx950 64-lane, bit-exact", bits_eq(o, ref))

# a different order really changes the bits (the test can tell orders apart)
x = f32_input(64)
a1 = emu_butterfly(x, (32, 16, 8, 4, 2, 1), np.add)[0]
a2 = emu_butterfly(x, (1, 2, 4, 8, 16, 32), np.add)[0]
R.check("orders distinguishable on this input", a1.view(np.int32) != a2.view(np.int32))

# scalar offset 32 is refused at trace time
try:
    _gpu.run(_gpu.kernel(lambda tid, a, b, c, d: st(ops.butterfly(ldf(a, tid), (32, 16)), b, tid), 32),
             ftens(x), torch.zeros(32, dtype=torch.int32, device="cuda"))
    R.check("butterfly(scalar, 32) refused", False)
except ValueError as e:
    R.check("butterfly(scalar, 32) refused", "pair" in str(e))

# ----------------------------------------------------------- permlane_swap
xa = rng.integers(0, 1 << 30, 32).astype(np.int32)
ya = rng.integers(0, 1 << 30, 32).astype(np.int32)


def body(tid, a, b, c, d):
    p, q = ops.permlane_swap(16, ldi(a, tid), ldi(c, tid))
    st(p, b, tid)
    st(q, b, 32 + tid)
    (p0, p1), (q0, q1) = ops.permlane_swap(32, (ldi(a, tid), ldi(c, tid)), (ldi(c, tid) + 1, ldi(a, tid) + 1))
    for j, v in enumerate((p0, p1, q0, q1)):
        st(v, b, 64 + 32 * j + tid)


out = torch.zeros(192, dtype=torch.int32, device="cuda")
_gpu.run(_gpu.kernel(body, 32), ftens(xa), out, ftens(ya))
o = out.cpu().numpy()
R.check("permlane_swap 16: x' = [x 0-15, y 0-15]", np.array_equal(o[:32], np.r_[xa[:16], ya[:16]]))
R.check("permlane_swap 16: y' = [x 16-31, y 16-31]", np.array_equal(o[32:64], np.r_[xa[16:], ya[16:]]))
R.check("permlane_swap 32 on pairs: (x_lo, y_lo), (x_hi, y_hi)",
        np.array_equal(o[64:96], xa) and np.array_equal(o[96:128], ya + 1)
        and np.array_equal(o[128:160], ya) and np.array_equal(o[160:], xa + 1))

# ------------------------------------- lane_gather / readlane / uniform / fresh
xa = rng.integers(-1000, 1000, 32).astype(np.int32)


def body(tid, a, b, c, d):
    v = ldi(a, tid)
    st(ops.lane_gather(v, (tid * 7 + 3) % 32), b, tid)
    st(ops.lane_gather(v, tid + 32), b, 32 + tid)
    st(ops.readlane(v, 5), b, 64 + tid)
    st(ops.readlane(fx.Float32(v), 9), b, 96 + tid)
    st(ops.uniform(v), b, 128 + tid)
    st(ops.fresh(v) * 3, b, 160 + tid)


o, L = gpu(body, ftens(xa), 192)
o = o.numpy()
lanes = np.arange(32)
R.check("lane_gather(v, src)", np.array_equal(o[:32], xa[(lanes * 7 + 3) % 32]))
R.check("lane_gather source lane taken mod 32", np.array_equal(o[32:64], xa))
R.check("readlane(v, 5)", np.all(o[64:96] == xa[5]))
R.check("readlane(f32, 9)", np.all(o[96:128].view(np.float32) == np.float32(xa[9])))
R.check("uniform(v) = lane 0", np.all(o[128:160] == xa[0]))
R.check("fresh(v) * 3", np.array_equal(o[160:], xa * 3))
R.check("fresh keeps its v_mov", "v_mov_b32" in _gpu.kernel_text(_gpu.isa(L), L.kernel_name))

# ------------------------------------------------------------------ row_shr
x = f32_input(32)
ks = (0, 1, 3, 8, 15)


def body(tid, a, b, c, d):
    v = ldf(a, tid)
    lane = tid % 32
    for j, k in enumerate(ks):
        st(ops.row_shr(v, lane, k), b, 32 * j + tid)
    st(ops.row_shr(v, lane, ldi(c, 0)), b, 32 * len(ks) + tid)


out = torch.zeros(32 * (len(ks) + 1), dtype=torch.int32, device="cuda")
_gpu.run(_gpu.kernel(body, 32), ftens(x), out, torch.tensor([5], dtype=torch.int32, device="cuda"))
o = out.cpu()


def ref_shr(k):
    r = np.zeros(32, np.float32)
    for l in range(32):
        if l % 16 >= k:
            r[l] = x[l - k]
    return r


for j, k in enumerate(ks):
    R.check(f"row_shr k={k} (DPP, bound_ctrl)", bits_eq(o[32 * j:32 * j + 32], ref_shr(k)))
R.check("row_shr traced k=5 (bpermute path)", bits_eq(o[32 * len(ks):], ref_shr(5)))

# ----------------------------------------------------------- wave_incl_scan
xa = rng.integers(-50000, 50000, 64).astype(np.int32)


def body(tid, a, b, c, d):
    lane = tid % 32
    st(ops.wave_incl_scan(ldi(a, tid), lane), b, tid)
    lo, hi = ops.wave_incl_scan((ldi(a, tid), ldi(a, 32 + tid)), lane)
    st(lo, b, 32 + tid)
    st(hi, b, 64 + tid)


o, _ = gpu(body, ftens(xa), 96)
o = o.numpy()
R.check("wave_incl_scan 32 lanes", np.array_equal(o[:32], np.cumsum(xa[:32]).astype(np.int32)))
R.check("wave_incl_scan pair = 64-lane scan", np.array_equal(o[32:96], np.cumsum(xa).astype(np.int32)))

# ------------------------------------------- ballot / popcount / lanes_below
xa = rng.integers(0, 100, 64).astype(np.int32)
TH = 50


def body(tid, a, b, c, d):
    lane = tid % 32
    bal = ops.ballot(ldi(a, tid) >= TH)
    st(fx.Int32(bal), b, tid)
    st(fx.Int32(bal >> 32), b, 32 + tid)
    st(ops.popcount(bal), b, 64 + tid)
    st(ops.lanes_below(bal, lane), b, 96 + tid)
    bp = ops.ballot((ldi(a, tid) >= TH, ldi(a, 32 + tid) >= TH))
    st(fx.Int32(bp), b, 128 + tid)
    st(fx.Int32(bp >> 32), b, 160 + tid)
    st(ops.popcount(bp), b, 192 + tid)
    st(ops.lanes_below(bp, lane + 32), b, 224 + tid)


o, L = gpu(body, ftens(xa), 256)
o = o.numpy()
p = xa >= TH
m32 = sum(1 << i for i in range(32) if p[i])
m64 = sum(1 << i for i in range(64) if p[i])
R.check("ballot: low word = 32-lane mask", np.all(o[:32].view(np.uint32) == np.uint32(m32)))
R.check("ballot: bits 32-63 zero", np.all(o[32:64] == 0))
R.check("popcount(ballot)", np.all(o[64:96] == p[:32].sum()))
R.check("lanes_below(bal, lane)", np.array_equal(o[96:128], np.cumsum(p[:32]) - p[:32]))
R.check("ballot(pair) = 64 logical lanes",
        np.all(o[128:160].view(np.uint32) == np.uint32(m64 & 0xFFFFFFFF))
        and np.all(o[160:192].view(np.uint32) == np.uint32(m64 >> 32)))
R.check("popcount(ballot(pair))", np.all(o[192:224] == p.sum()))
R.check("lanes_below(pair ballot, logical lane 32 + l)", np.array_equal(o[224:], (np.cumsum(p) - p)[32:]))
asm = _gpu.kernel_text(_gpu.isa(L), L.kernel_name)
R.check("ballot is a 32-bit SGPR compare (wave32)", "v_cmp" in asm)

R.done()
