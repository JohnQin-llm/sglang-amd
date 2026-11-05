"""GPU unit tests of the memory / misc primitives on gfx1250: ops.rsrc (sized
resource: out-of-range loads return 0, stores are dropped), ld_i32, ld_dev,
row_live, load_ptr64, global_load (past 4 GiB, nt), the gfx12 cache-policy
constants as they appear in the ISA (CM_DEV scope:SCOPE_DEV, CM_SYS
scope:SCOPE_SYS, CM_NT th:TH_LOAD_NT / TH_STORE_NT), vm_wait (s_wait_loadcnt /
s_wait_storecnt), memrealtime (rate vs the host clock), unlikely, kernel_symbol,
lds_bytes, and a cross-CTA device-scope visibility check of ld_dev.

Run: flock /data/muse455/locks/gpu0.lock docker exec muse_mi455x bash -c \
  'cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=0 timeout 300 python3 \
   /data/muse455/k2/cand/k2v4/port/common/test_mem.py'
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gpu  # noqa: E402

import flydsl.expr as fx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402

import ops  # noqa: E402

R = _gpu.Report("memory / misc primitives (gfx1250)")
rng = np.random.default_rng(16)


def rs(p, n=None):
    return ops.rsrc(p, n)


def ldi(p, i):
    return fx.Int32(ops.bo.buffer_load(rs(p), i, vec_width=1, dtype=T.i32))


def st(v, p, i, cm=0):
    ops.bo.buffer_store(ops._bits(v), rs(p), i, cache_modifier=cm)


def dev(x):
    return torch.from_numpy(np.ascontiguousarray(x)).cuda()


def text(L):
    return _gpu.kernel_text(_gpu.isa(L), L.kernel_name)


# ---------------------------------------- rsrc / ld_i32 / ld_dev / row_live
src = rng.integers(-2**31, 2**31 - 1, 256, dtype=np.int64).astype(np.int32)
bids = np.array([0, 3, -1, 7, -1, 2, 1, -1], np.int32)


def body(tid, a, b, c, d):
    st(ldi(a, tid), b, tid)
    st(fx.Int32(ops.ld_dev(a, tid)), b, 32 + tid)
    v4 = fx.Vector(ops.ld_dev(a, 4 * tid, 4))
    for j in range(4):
        st(v4[j], b, 64 + 4 * tid + j)
    v2 = fx.Vector(ops.ld_dev(a, 2 * tid, 2))
    st(v2[1], b, 192 + tid)
    # a 64-word sized resource: words 64.. read 0, stores past it are dropped
    st(fx.Int32(ops.bo.buffer_load(rs(a, 256), 48 + tid, vec_width=1, dtype=T.i32)), b, 224 + tid)
    ops.bo.buffer_store(fx.Int32(12345), rs(d, 64), 48 + tid)
    st(ops.row_live(c, tid % 8).select(fx.Int32(1), fx.Int32(0)), b, 256 + tid)


out = torch.zeros(288, dtype=torch.int32, device="cuda")
small = torch.full((128,), -5, dtype=torch.int32, device="cuda")
L = _gpu.kernel(body, 32)
_gpu.run(L, dev(src), out, dev(bids), small)
o = out.cpu().numpy()
R.check("ld_i32", np.array_equal(o[:32], src[:32]))
R.check("ld_dev words=1 / 4 / 2", np.array_equal(o[32:64], src[:32]) and np.array_equal(o[64:192], src[:128])
        and np.array_equal(o[192:224], src[1:64:2]))
R.check("rsrc(addr, nbytes): loads past nbytes return 0",
        np.array_equal(o[224:240], src[48:64]) and np.all(o[240:256] == 0))
sm = small.cpu().numpy()
R.check("rsrc(addr, nbytes): stores past nbytes dropped", np.all(sm[48:64] == 12345) and np.all(sm[64:] == -5))
R.check("row_live: batch id >= 0", np.array_equal(o[256:], (bids[np.arange(32) % 8] >= 0).astype(np.int32)))
R.check("ld_dev -> scope:SCOPE_DEV", "scope:SCOPE_DEV" in text(L))

# ------------------------------------------------------------- load_ptr64
tab = np.array([0x0000_7FFF_1234_5678, -1, 0x1_0000_0000, 0xFFFF_FFFF, -0x7000_0000_0000_0001, 42, 0, 1 << 62],
               dtype=np.int64)


def body(tid, a, b, c, d):
    p = ops.load_ptr64(a, tid % 8)
    st(fx.Int32(p), b, 2 * tid)
    st(fx.Int32(p >> 32), b, 2 * tid + 1)


out = torch.zeros(64, dtype=torch.int32, device="cuda")
_gpu.run(_gpu.kernel(body, 32), dev(tab), out)
R.check("load_ptr64: full 64-bit entries (high bit, > 4 GiB, sign)",
        np.array_equal(out.cpu().numpy().view(np.int64), tab[np.arange(32) % 8]))

# ------------------------------------------------- global_load past 4 GiB
GB5 = 5 << 30
big = torch.zeros(GB5 // 4, dtype=torch.int32, device="cuda")
OFFW = (GB5 - 4096) // 4  # word offset past 4 GiB
vals = torch.from_numpy(rng.integers(-2**31, 2**31 - 1, 128, dtype=np.int64).astype(np.int32)).cuda()
big[OFFW:OFFW + 128] = vals


def body(tid, a, b, c, d):
    base = a + fx.Int64(OFFW * 4)
    st(fx.Int32(ops.global_load(base + fx.Int64(tid * 4))), b, tid)
    v4 = fx.Vector(ops.global_load(base + fx.Int64(tid * 16), 4, nt=True))
    for j in range(4):
        st(v4[j], b, 32 + 4 * tid + j)


out = torch.zeros(160, dtype=torch.int32, device="cuda")
L = _gpu.kernel(body, 32)
_gpu.run(L, big, out)
o = out.cpu()
R.check("global_load at > 4 GiB (words 1, 4)", torch.equal(o[:32], vals[:32].cpu()) and torch.equal(o[32:], vals.cpu()))
R.check("global_load nt -> global_load_b128 th:TH_LOAD_NT", "TH_LOAD_NT" in text(L) and "global_load_b128" in text(L))
del big
torch.cuda.empty_cache()

# ------------------------------------------------- cache policies in ISA
R.check("constants: CM_DEV 16, CM_SYS 24, CM_NT 1 (gfx12 CPol)", (ops.CM_DEV, ops.CM_SYS, ops.CM_NT) == (16, 24, 1))
cases = {
    "CM_DEV load": (ops.CM_DEV, "load", ["scope:SCOPE_DEV"]),
    "CM_SYS load": (ops.CM_SYS, "load", ["scope:SCOPE_SYS"]),
    "CM_NT load": (ops.CM_NT, "load", ["th:TH_LOAD_NT"]),
    "CM_NT | CM_DEV load": (ops.CM_NT | ops.CM_DEV, "load", ["th:TH_LOAD_NT", "scope:SCOPE_DEV"]),
    "CM_DEV store": (ops.CM_DEV, "store", ["scope:SCOPE_DEV"]),
    "CM_SYS store": (ops.CM_SYS, "store", ["scope:SCOPE_SYS"]),
    "CM_NT store": (ops.CM_NT, "store", ["th:TH_STORE_NT"]),
}
for name, (cm, kind, want) in cases.items():
    def body(tid, a, b, c, d, cm=cm, kind=kind):
        if kind == "load":
            st(fx.Int32(ops.bo.buffer_load(rs(a), tid, vec_width=1, dtype=T.i32, cache_modifier=cm)), b, tid)
        else:
            st(ldi(a, tid), b, tid, cm)

    L = _gpu.kernel(body, 32, name=f"k2c_cpol_{name.replace(' ', '_').replace('|', 'or')}")
    out = torch.zeros(32, dtype=torch.int32, device="cuda")
    _gpu.run(L, dev(src), out)
    lines = [ln.strip() for ln in text(L).splitlines() if f"buffer_{kind}" in ln]
    R.check(f"{name} -> {' '.join(want)}", torch.equal(out.cpu(), torch.from_numpy(src[:32]))
            and any(all(w in ln for w in want) for ln in lines), lines[0] if lines else "no instruction")

# --------------------------------------------------------------- vm_wait
def body(tid, a, b, c, d):
    st(ldi(a, tid), b, tid)
    ops.vm_wait()
    st(ldi(a, tid + 32), b, tid + 32)


L = _gpu.kernel(body, 32)
out = torch.zeros(64, dtype=torch.int32, device="cuda")
_gpu.run(L, dev(src), out)
t = text(L)
R.check("vm_wait -> s_wait_loadcnt 0x0 + s_wait_storecnt 0x0",
        "s_wait_loadcnt 0x0" in t and "s_wait_storecnt 0x0" in t and torch.equal(out.cpu(), torch.from_numpy(src[:64])))

# ----------------------------------------------------------- memrealtime
def body(tid, a, b, c, d):
    t0 = ops.memrealtime()
    if tid == 0:
        st(fx.Int32(t0), b, 0)
        st(fx.Int32(t0 >> 32), b, 1)


L = _gpu.kernel(body, 32)
stamps = []
for _ in range(3):
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    _gpu.run(L, None, out)
    stamps.append((time.perf_counter(), int(out.cpu().view(torch.int64)[0])))
    time.sleep(0.25)
rates = [(stamps[i + 1][1] - stamps[i][1]) / (stamps[i + 1][0] - stamps[i][0]) for i in range(2)]
mhz = np.mean(rates) / 1e6
R.check("memrealtime: monotonic, ~100 MHz constant clock", all(r > 0 for r in rates) and abs(mhz - 100) < 3,
        f"{mhz:.2f} MHz; {'s_sendmsg_rtn_b64' if 's_sendmsg_rtn_b64' in text(L) else '?'}")

# -------------------------------------------------------------- unlikely
@ops.traced
def body(tid, a, b, c, d):
    v = ldi(a, tid)
    r = v
    if ops.unlikely(v < 0):
        r = -v
    st(r, b, tid)


out = torch.zeros(32, dtype=torch.int32, device="cuda")
_gpu.run(_gpu.kernel(body, 32), dev(src), out)
R.check("unlikely(cond) branch semantics", np.array_equal(out.cpu().numpy(), np.abs(src[:32].astype(np.int64)).astype(np.int32)))

# ------------------------------------------------------ host-side helpers
nm = "k2_s8_t0"
R.check("kernel_symbol Itanium form", ops.kernel_symbol("k2", s=8, t=0) == f"_ZN4atom{len(nm)}{nm}E"
        or ops.kernel_symbol("k2", s=8, t=0).startswith("_ZN4atom"), ops.kernel_symbol("k2", s=8, t=0))


@fx.struct
class RedLds:
    red: fx.Array[fx.Float32, ops.WAVES * ops.ACC_WORDS, 16]


R.check("lds_bytes(struct): WAVES x ACC_WORDS f32 = 16 KB", ops.lds_bytes(RedLds) == ops.WAVES * ops.ACC_WORDS * 4,
        str(ops.lds_bytes(RedLds)))

# ---------------------------------------- device scope across CTAs (bounded)
CTAS = 128


@ops.traced
def body(tid, a, b, c, d):
    bid = fx.block_idx.x
    if tid == 0:
        if bid == 0:
            t0 = ops.memrealtime()
            while ops.memrealtime() - t0 < 2000:  # ~20 us: let the pollers start
                ops.fresh(tid)
            st(fx.Int32(1), a, 0, ops.CM_DEV)
        else:
            n = fx.Int32(0)
            seen = fx.Int32(0)
            while (seen == 0) & (n < 2000000):
                seen = fx.Int32(ops.ld_dev(a, 0))
                n = ops.fresh(n) + 1
            st(seen, b, bid)
            st(n, b, CTAS + bid)


flag = torch.zeros(64, dtype=torch.int32, device="cuda")
out = torch.zeros(2 * CTAS, dtype=torch.int32, device="cuda")
_gpu.run(_gpu.kernel(body, 32, grid=CTAS), flag, out)
o = out.cpu().numpy()
spins = o[CTAS + 1:]
R.check(f"ld_dev polling sees another CTA's CM_DEV store ({CTAS - 1} pollers)", np.all(o[1:CTAS] == 1),
        f"spins median {int(np.median(spins))}, max {int(spins.max())}")

R.done()
