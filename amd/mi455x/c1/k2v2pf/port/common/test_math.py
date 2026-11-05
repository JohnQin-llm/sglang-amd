"""GPU unit tests of the arithmetic primitives: ops.hw_rsq / hw_rcp / hw_exp2 /
hw_exp (ulp vs float64), div_rn (bit-exact vs IEEE division), bf16_round /
bf16_pair / bf_lo / bf_hi / bf2_f32, fp8_pack4 (RNE vs torch e4m3fn),
fp8x8_bf16_pk (all 256 codes), and mx.clamp_fp8 / code_ceil / pow2 / rcp_pow2
(exact); host checks of mx.scale_index (aiter shuffle_scale) and
mx.scale_index_gfx1250 (layout_gfx1250.shuffle_scale_gfx1250).

Run: flock /data/muse455/locks/gpu0.lock docker exec muse_mi455x bash -c \
  'cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=0 timeout 300 python3 \
   /data/muse455/k2/cand/k2v2_land1/port/common/test_math.py'
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gpu  # noqa: E402

import flydsl.expr as fx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flydsl.expr import range_constexpr  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402

import mx  # noqa: E402
import ops  # noqa: E402

R = _gpu.Report("ops / mx arithmetic primitives (gfx1250)")
rng = np.random.default_rng(455)
TH = 256
N = 1 << 16
GRID = N // TH


def rs(p):
    return ops.rsrc(p)


def ldf(p, i):
    return fx.Float32(ops.bo.buffer_load(rs(p), i, vec_width=1, dtype=T.f32))


def ldi(p, i):
    return fx.Int32(ops.bo.buffer_load(rs(p), i, vec_width=1, dtype=T.i32))


def st(v, p, i):
    ops.bo.buffer_store(ops._bits(v), rs(p), i)


def elementwise(fn, *inputs, outs=1, name=None):
    """fn(i, a, c, d) -> list of ``outs`` values for element i; inputs (<= 3
    int32/float32 arrays of N) at a, c, d; outputs [outs, N] int32 bits."""
    def body(tid, a, b, c, d):
        i = fx.block_idx.x * TH + tid
        for j, v in enumerate(fn(i, a, c, d)):
            st(v, b, j * N + i)

    L = _gpu.kernel(body, TH, name=name, grid=GRID)
    out = torch.zeros(outs * N, dtype=torch.int32, device="cuda")
    t = [torch.from_numpy(np.ascontiguousarray(x)).cuda() for x in inputs] + [None] * (3 - len(inputs))
    _gpu.run(L, t[0], out, t[1], t[2])
    return out.cpu().numpy().reshape(outs, N)


def ulp_err(got, ref64):
    ref32 = ref64.astype(np.float32)
    ulp = np.spacing(np.abs(ref32)).astype(np.float64)
    return np.abs(got.astype(np.float64) - ref64) / ulp


# ---------------------------------------------------------- hardware math
pos = np.exp2(rng.uniform(-100, 100, N)).astype(np.float32)
sgn = np.where(rng.random(N) < 0.5, -1, 1).astype(np.float32) * pos
e2in = rng.uniform(-120, 120, N).astype(np.float32)
ein = rng.uniform(-80, 80, N).astype(np.float32)
o = elementwise(lambda i, a, c, d: [ops.hw_rsq(ldf(a, i)), ops.hw_rcp(ldf(c, i)), ops.hw_exp2(ldf(d, i))],
                pos, sgn, e2in, outs=3)
r = o.view(np.float32)
u_rsq = ulp_err(r[0], 1.0 / np.sqrt(pos.astype(np.float64))).max()
u_rcp = ulp_err(r[1], 1.0 / sgn.astype(np.float64)).max()
u_e2 = ulp_err(r[2], np.exp2(e2in.astype(np.float64))).max()
R.check("hw_rsq <= 1 ulp", u_rsq <= 1.0, f"max {u_rsq:.3f} ulp")
R.check("hw_rcp <= 1 ulp", u_rcp <= 1.0, f"max {u_rcp:.3f} ulp")
R.check("hw_exp2 <= 1 ulp (normal results)", u_e2 <= 1.0, f"max {u_e2:.3f} ulp")
o = elementwise(lambda i, a, c, d: [ops.hw_exp(ldf(a, i))], ein)
x32 = ein * np.float32(1.4426950408889634)  # the f32 product hw_exp feeds exp2
u_e = ulp_err(o.view(np.float32)[0], np.exp2(x32.astype(np.float64))).max()
rel = np.abs(o.view(np.float32)[0] / np.exp(ein.astype(np.float64)) - 1).max()
R.check("hw_exp = exp2(f32(x * log2 e)) <= 1 ulp", u_e <= 1.0, f"max {u_e:.3f} ulp; vs exp(x): rel {rel:.2e}")

# ---------------------------------------------------------------- div_rn
xs = (rng.standard_normal(N) * np.exp2(rng.integers(-30, 30, N))).astype(np.float32)
ds = (rng.standard_normal(N) * np.exp2(rng.integers(-30, 30, N))).astype(np.float32)
ds[ds == 0] = 1
rh = (np.float32(1.0) / ds).astype(np.float32)
o = elementwise(lambda i, a, c, d: [ops.div_rn(ldf(a, i), ldf(c, i), ldf(d, i)),
                                    ops.div_rn(ldf(a, i), ldf(c, i), fx.Float32(1.0) / ldf(c, i)),
                                    ldf(a, i) * ldf(d, i)],
                xs, ds, rh, outs=3)
ref = (xs / ds).astype(np.float32).view(np.int32)
R.check("div_rn(x, d, host 1/d) bit-exact vs IEEE x / d", np.array_equal(o[0], ref), f"{N} pairs")
R.check("div_rn(x, d, device 1/d) bit-exact vs IEEE x / d", np.array_equal(o[1], ref))
R.check("(plain x * (1/d) is not: the test can tell)", not np.array_equal(o[2], ref),
        f"{int((o[2] != ref).sum())} of {N} differ")

# --------------------------------------------------------- bf16 packing
bits = rng.integers(-2**31, 2**31 - 1, N, dtype=np.int64).astype(np.int32)
ties = ((rng.integers(0, 1 << 15, N // 4) << 16) | 0x8000).astype(np.int32)  # exact halfway cases
bits[: N // 4] = ties
bits[N // 4: N // 4 + 4] = np.array([0x7F800000, -0x800000, 0x7F7FFFFF, 0x00000001], np.int64).astype(np.int32)
f = bits.view(np.float32)
fb = f.copy()
fb2 = np.roll(f, 1)
o = elementwise(lambda i, a, c, d: [ops.bf16_round(ldf(a, i)), ops.bf16_pair(ldf(a, i), ldf(c, i))],
                f, fb2, outs=2)
tb = torch.from_numpy(f).to(torch.bfloat16)
ref_round = tb.float().numpy().view(np.int32)
nan = np.isnan(f)
R.check("bf16_round = torch RNE (incl. ties, inf, max, denormal)",
        np.array_equal(o[0][~nan], ref_round[~nan]) and np.isnan(o[0].view(np.float32)[nan]).all(),
        f"{int(nan.sum())} NaN inputs stay NaN")
lo16 = tb.view(torch.int16).numpy().astype(np.int32) & 0xFFFF
hi16 = torch.from_numpy(fb2).to(torch.bfloat16).view(torch.int16).numpy().astype(np.int32) & 0xFFFF
ref_pair = (lo16 | (hi16 << 16)).astype(np.int64).astype(np.uint32).view(np.int32)
okp = ~(np.isnan(f) | np.isnan(fb2))
R.check("bf16_pair(a, b) = (bf16(a) | bf16(b) << 16)", np.array_equal(o[1][okp], ref_pair[okp]))

w = rng.integers(-2**31, 2**31 - 1, N, dtype=np.int64).astype(np.int32)
o = elementwise(lambda i, a, c, d: [ops.bf_lo(ldi(a, i)), ops.bf_hi(ldi(a, i)), *ops.bf2_f32(ldi(a, i))], w, outs=4)
wl = (w.astype(np.int64) << 16).astype(np.uint32).view(np.int32)
wh = (w & np.int32(-65536)).astype(np.int32)
R.check("bf_lo / bf_hi / bf2_f32 exact",
        np.array_equal(o[0], wl) and np.array_equal(o[1], wh) and np.array_equal(o[2], wl) and np.array_equal(o[3], wh))

# ---------------------------------------------------------- fp8 packing
grid8 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
fin = grid8[np.isfinite(grid8)]
fin = np.unique(fin)
mids = ((fin[1:] + fin[:-1]) / 2).astype(np.float32)  # exact halfway points (RNE ties)
vals = np.concatenate([fin, mids, rng.uniform(-448, 448, N).astype(np.float32)])[: 4 * (N // 4)]
vals = np.resize(vals, N).astype(np.float32)
vals = np.clip(vals, -448, 448)
q = vals.reshape(-1, 4)
o = elementwise(lambda i, a, c, d: [ops.fp8_pack4(ldf(a, 4 * (i % (N // 4))), ldf(a, 4 * (i % (N // 4)) + 1),
                                                  ldf(a, 4 * (i % (N // 4)) + 2), ldf(a, 4 * (i % (N // 4)) + 3))],
                vals)
ref8 = torch.from_numpy(vals).to(torch.float8_e4m3fn).view(torch.uint8).numpy().reshape(-1, 4)
refw = (ref8[:, 0].astype(np.uint32) | ref8[:, 1].astype(np.uint32) << 8 | ref8[:, 2].astype(np.uint32) << 16
        | ref8[:, 3].astype(np.uint32) << 24).view(np.int32)
R.check("fp8_pack4 = torch e4m3fn RNE (all codes, all ties, random)", np.array_equal(o[0][: N // 4], refw),
        f"{len(fin)} codes, {len(mids)} ties")

codes = np.arange(256, dtype=np.uint8)
words = np.resize(codes.view(np.int32), N).astype(np.int32)  # 64 dwords repeated


def pk(i, a, c, d):
    j = (i % 32) * 2
    v = ops.fp8x8_bf16_pk([ldi(a, j), ldi(a, j + 1)]).bitcast(fx.Int32)
    return [v[k] for k in range(4)]


o = elementwise(pk, words, outs=4)
got = np.stack([o[k][:32] for k in range(4)], 1).reshape(-1).view(np.uint16)  # 32 x 8 bf16
ref = torch.from_numpy(codes).view(torch.float8_e4m3fn).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
fin8 = np.isfinite(grid8)
gotf = torch.from_numpy(got.view(np.int16).copy()).view(torch.bfloat16).float().numpy()
R.check("fp8x8_bf16_pk exact on all 254 finite codes, NaN -> NaN",
        np.array_equal(got[fin8], ref[fin8]) and np.isnan(gotf[~fin8]).all())

# ------------------------------------------------------------- mx codes
v = np.concatenate([np.exp2(rng.uniform(-140, 130, N - 8)), [0, 1, 2, 0.75, 448, 1e-45, 3e38, 2.0**-127]])
v = v.astype(np.float32)
codes = rng.integers(1, 254, N).astype(np.int32)
cv = (rng.standard_normal(N) * 1000).astype(np.float32)
o = elementwise(lambda i, a, c, d: [mx.code_ceil(ldf(a, i)), mx.pow2(ldi(c, i)), mx.rcp_pow2(ldi(c, i)),
                                    mx.clamp_fp8(ldf(d, i)), ldf(d, i) * mx.rcp_pow2(ldi(c, i))],
                v, codes, cv, outs=5)
m, e = np.frexp(v.astype(np.float64))
want = np.where(v == 0, 0, np.where(m == 0.5, e - 1, e) + 127)
R.check("code_ceil = biased exponent of the smallest pow2 >= v, clamped [1, 254]",
        np.array_equal(o[0], np.clip(want, 1, 254)))
R.check("pow2(code) = 2^(code - 127)", np.array_equal(o[1].view(np.float32), np.exp2(codes - 127.0).astype(np.float32)))
R.check("rcp_pow2(code) = 2^(127 - code)", np.array_equal(o[2].view(np.float32), np.exp2(127.0 - codes).astype(np.float32)))
R.check("clamp_fp8 = clip(v, -448, 448)", np.array_equal(o[3].view(np.float32), np.clip(cv, -448, 448)))
dv = (cv / np.exp2(codes - 127.0).astype(np.float32)).astype(np.float32)
normal = np.abs(dv) > np.float32(2.0 ** -126)
R.check("x * rcp_pow2(code) == x / pow2(code) bit for bit (normal results)",
        np.array_equal(o[4][normal], dv.view(np.int32)[normal]))

# ------------------------------------------------- scale layouts (host)
rows, cols = 64, 32
ids = torch.arange(rows * cols, dtype=torch.int64).view(rows, cols)
sh = ids.view(rows // 32, 2, 16, cols // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).contiguous().view(-1)
pos_of = torch.empty_like(sh)
pos_of[sh] = torch.arange(sh.numel())
ok = all(mx.scale_index(r_, c_, cols) == pos_of[r_ * cols + c_].item() for r_ in range(rows) for c_ in range(cols))
R.check("scale_index = aiter shuffle_scale byte (unchanged layout)", ok)
sys.path.insert(0, "/data/muse455/k2/port")
from layout_gfx1250 import s_dword, shuffle_scale_gfx1250  # noqa: E402

E, n, k = 2, 64, 1024
ids = torch.arange(E * n * (k // 32), dtype=torch.int64).view(E, n, k // 32)
sh = shuffle_scale_gfx1250(ids).reshape(-1)
pos_of = torch.empty_like(sh)
pos_of[sh] = torch.arange(sh.numel())
ok = all(mx.scale_index_gfx1250(e_ * n + r_, c_, k // 32) == pos_of[(e_ * n + r_) * (k // 32) + c_].item()
         for e_ in range(E) for r_ in range(n) for c_ in range(k // 32))
R.check("scale_index_gfx1250 = layout_gfx1250.shuffle_scale_gfx1250 byte", ok)
ok = all(mx.scale_index_gfx1250(rg * 16 + lane % 16, 4 * (st_ - st_ % 2 + lane // 16), k // 32) // 4
         == s_dword(0, rg, st_, lane, n=n, k=k)
         for rg in range(n // 16) for st_ in range(k // 128) for lane in range(32))
R.check("its dwords = the WMMA scale dword a lane loads (layout_gfx1250.s_dword)", ok)

R.done()
