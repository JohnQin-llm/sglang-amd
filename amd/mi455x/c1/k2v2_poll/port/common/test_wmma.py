"""GPU unit tests of the matrix primitives on gfx1250 (one wave = one 16x16 tile):

* ops.mfma_bf16 (v_wmma_f32_16x16x32_bf16) and ops.mfma_fp8 (16x16x64 fp8, and
  the 4-dword 16x16x32 form): the operand layout (lane l: row / column l % 16,
  16-byte chunk j = K run 2 j + l // 16) and the D layout (column l % 16, rows
  8 (l // 16) + v); exact on integer data, ~1e-6 on random data.
* ops.acc_store + ops.row_sum: 16 waves' accumulators through LDS, summed in
  wave order, bit-exact against the per-wave D read back.
* mx.mfma_scaled FP4 x FP4 for every opsel pair (exact on narrow scales, 1e-6
  on wide ones), FP4 A x FP8 B (the A8W4 activations) and FP8 x FP8 (1e-4:
  the hardware's FP8 scaled dot is not exact).
* mx.fp4_tile_load + mx.scale_index_gfx1250 end to end: a (16,16)-preshuffled
  FP4 weight [64 x 1024] with gfx1250-ordered scales times 16 MXFP4 tokens, one
  wave a 16-row group over 8 K steps (opsel = step parity), vs float64 dequant.

Run: flock /data/muse455/locks/gpu0.lock docker exec muse_mi455x bash -c \
  'cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=0 timeout 300 python3 \
   /data/muse455/k2/cand/k2v2_land1/port/common/test_wmma.py'
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gpu  # noqa: E402

import flydsl.expr as fx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flydsl.expr import gpu as fgpu  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402

import mx  # noqa: E402
import ops  # noqa: E402

sys.path.insert(0, "/data/muse455/k2/port")
from layout_gfx1250 import shuffle_scale_gfx1250, shuffle_weight  # noqa: E402

R = _gpu.Report("WMMA primitives (gfx1250, wave32)")
torch.manual_seed(1250)
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)


def rs(p):
    return ops.rsrc(p)


def ld_dwords(p, base, n):
    """n (4, 8 or 16) dwords from dword ``base`` as one vector (b128 loads)."""
    parts = [fx.Vector(ops.bo.buffer_load(rs(p), base + 4 * q, vec_width=4, dtype=T.i32)) for q in range(n // 4)]
    while len(parts) > 1:
        m = len(parts[0])
        parts = [parts[i].shuffle(parts[i + 1], list(range(2 * m))) for i in range(0, len(parts), 2)]
    return parts[0]


def st_acc(acc, p, lane):
    for v in range(8):
        ops.bo.buffer_store(acc[v], rs(p), lane * 8 + v)


# ------------------------------------------------------------ host packing
def pack_lanes(rowbytes):
    """[16, B] bytes (a row / column's K elements in memory order) -> [32 lanes,
    B / 2 bytes]: lane l's 16-byte chunk j = bytes 16 (2 j + l // 16) .. of row
    l % 16 (the hypothesis every operand below is packed by)."""
    rowbytes = np.ascontiguousarray(rowbytes, dtype=np.uint8)
    nb = rowbytes.shape[1]
    out = np.zeros((32, nb // 2), np.uint8)
    for lane in range(32):
        for j in range(nb // 32):
            c = 2 * j + lane // 16
            out[lane, 16 * j:16 * j + 16] = rowbytes[lane % 16, 16 * c:16 * c + 16]
    return out


def fp4_bytes(codes):
    """[16, K] e2m1 codes -> [16, K / 2] bytes, low nibble the even k."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)


def unpack_d(o):
    """[32 lanes x 8] -> D [16 rows, 16 cols]: lane l column l % 16, rows 8 (l // 16) + v."""
    o = o.reshape(32, 8).double()
    d = torch.empty(16, 16, dtype=torch.float64)
    for lane in range(32):
        for v in range(8):
            d[8 * (lane // 16) + v, lane % 16] = o[lane, v]
    return d


def dev(x):
    return torch.from_numpy(np.ascontiguousarray(x)).cuda()


# ---------------------------------------------------------------- bf16
def run_bf16(A, B):
    """A [16, 32], B [32, 16] (bf16-exact float tensors) -> D."""
    ab = A.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().reshape(16, 64)
    bb = B.T.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().reshape(16, 64)
    pa, pb = pack_lanes(ab).view(np.int32), pack_lanes(bb).view(np.int32)

    def body(tid, a, b, c, d):
        acc = fx.Vector.filled(8, 0.0, fx.Float32)
        av = ld_dwords(a, tid * 8, 8).bitcast(fx.BFloat16)
        bv = ld_dwords(c, tid * 8, 8).bitcast(fx.BFloat16)
        st_acc(ops.mfma_bf16(av, bv, acc), b, tid)

    out = torch.zeros(256, dtype=torch.float32, device="cuda")
    L = _gpu.kernel(body, 32)
    _gpu.run(L, dev(pa), out, dev(pb))
    return unpack_d(out.cpu()), L


A = torch.randint(-8, 9, (16, 32)).double()
B = torch.randint(-8, 9, (32, 16)).double()
D, L = run_bf16(A, B)
R.check("mfma_bf16: layout + exact on integer data", torch.equal(D, A @ B))
R.check("mfma_bf16 lowers to v_wmma_f32_16x16x32_bf16", "v_wmma_f32_16x16x32_bf16" in _gpu.isa(L))
A = torch.randn(16, 32).to(torch.bfloat16).double()
B = torch.randn(32, 16).to(torch.bfloat16).double()
D, _ = run_bf16(A, B)
err = ((D - A @ B).abs().max() / (A @ B).abs().max()).item()
R.check("mfma_bf16 random: rel err <= 1e-6", err <= 1e-6, f"{err:.2e}")

# ----------------------------------------------------------- fp8 unscaled
def run_fp8(A, B, k):
    a8 = A.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    b8 = B.T.contiguous().to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    if k == 32:  # the 4-dword form: K run 2 j + l // 16 for j = 0 only
        a8 = np.concatenate([a8, np.zeros((16, 32), np.uint8)], 1)
        b8 = np.concatenate([b8, np.zeros((16, 32), np.uint8)], 1)
    pa, pb = pack_lanes(a8).view(np.int32), pack_lanes(b8).view(np.int32)
    nd = 4 if k == 32 else 8

    def body(tid, a, b, c, d):
        acc = fx.Vector.filled(8, 0.0, fx.Float32)
        av = ld_dwords(a, tid * 8, nd)
        bv = ld_dwords(c, tid * 8, nd)
        st_acc(ops.mfma_fp8(av, bv, acc), b, tid)

    out = torch.zeros(256, dtype=torch.float32, device="cuda")
    _gpu.run(_gpu.kernel(body, 32), dev(pa), out, dev(pb))
    return unpack_d(out.cpu())


for k in (64, 32):
    A = (torch.randint(-8, 9, (16, k)) / 2).double()
    B = (torch.randint(-8, 9, (k, 16)) / 2).double()
    R.check(f"mfma_fp8 16x16x{k}: layout + exact on small values", torch.equal(run_fp8(A, B, k), A @ B))
A = torch.randn(16, 64).to(torch.float8_e4m3fn).double()
B = torch.randn(64, 16).to(torch.float8_e4m3fn).double()
D = run_fp8(A, B, 64)
err = ((D - A @ B).abs().max() / (A @ B).abs().max()).item()
R.check("mfma_fp8 random: rel err <= 1e-6", err <= 1e-6, f"{err:.2e}")

# ------------------------------------------- acc_store + row_sum, 16 waves
W = ops.WAVES


@fx.struct
class Red:
    red: fx.Array[fx.Float32, W * ops.ACC_WORDS, 16]


A = torch.randn(W, 16, 32).to(torch.bfloat16)
B = torch.randn(W, 32, 16).to(torch.bfloat16)
pa = np.stack([pack_lanes(A[w].contiguous().view(torch.uint8).numpy().reshape(16, 64)) for w in range(W)]).view(np.int32)
pb = np.stack([pack_lanes(B[w].T.contiguous().view(torch.uint8).numpy().reshape(16, 64)) for w in range(W)]).view(np.int32)


def body(tid, a, b, c, d, sm):
    lane, wave = tid % 32, tid // 32
    acc = fx.Vector.filled(8, 0.0, fx.Float32)
    acc = ops.mfma_bf16(ld_dwords(a, tid * 8, 8).bitcast(fx.BFloat16), ld_dwords(c, tid * 8, 8).bitcast(fx.BFloat16), acc)
    for v in range(8):
        ops.bo.buffer_store(acc[v], rs(d), tid * 8 + v)
    red = sm.red.ptr
    ops.acc_store(red, acc, wave, lane)
    fgpu.barrier()
    if tid < 256:
        r, t = tid // 16, tid % 16
        ops.bo.buffer_store(ops.row_sum(red, r, t), rs(b), tid)
        ops.bo.buffer_store(ops.row_sum(red, r, t, waves=(3,)), rs(b), 256 + tid)
        ops.bo.buffer_store(ops.row_sum(red, r, t, waves=(5, 1, 2)), rs(b), 512 + tid)


out = torch.zeros(768, dtype=torch.float32, device="cuda")
per = torch.zeros(W * 256, dtype=torch.float32, device="cuda")
_gpu.run(_gpu.kernel(body, 32 * W, lds=Red), dev(pa), out, dev(pb), per)
Dw = [unpack_d(per.cpu()[256 * w:256 * w + 256]).float().numpy() for w in range(W)]
ok_w = all(np.allclose(Dw[w], (A[w].double() @ B[w].double()).numpy(), rtol=1e-5, atol=1e-5) for w in range(W))
R.check(f"{W} waves' WMMAs correct (per-wave D)", ok_w)


def seq(ws):
    t = np.zeros((16, 16), np.float32)
    for w in ws:
        t = (t + Dw[w]).astype(np.float32)
    return t


o = out.cpu().numpy().reshape(3, 16, 16)
R.check(f"row_sum(red, r, t): all {W} waves in wave order, bit-exact", np.array_equal(o[0].view(np.int32), seq(range(W)).view(np.int32)))
R.check("row_sum waves=(3,)", np.array_equal(o[1], Dw[3]))
R.check("row_sum waves=(5, 1, 2) in that order, bit-exact", np.array_equal(o[2].view(np.int32), seq((5, 1, 2)).view(np.int32)))


# ------------------------------------------------------------ scaled WMMA
def pack_scales(sc, sel):
    """sc [16, 4] e8m0 -> [32] dwords: the lanes opsel ``sel`` reads hold them, the others junk."""
    out = np.full((32, 4), 0x7F, np.uint8)
    out[(1 - sel) * 16:(2 - sel) * 16] = np.random.randint(100, 150, (16, 4))
    out[sel * 16:sel * 16 + 16] = sc
    return out.view(np.int32).reshape(32)


def run_scaled(a_rows, b_rows, sa, sb, a_fmt, b_fmt, sel_a, sel_b):
    """a_rows / b_rows: [16, bytes] operand rows (FP4 packed or FP8 bytes)."""
    pa, pb = pack_lanes(a_rows).view(np.int32), pack_lanes(b_rows).view(np.int32)
    na, nb = pa.shape[1], pb.shape[1]
    sbuf = np.concatenate([pack_scales(sa, sel_a), pack_scales(sb, sel_b)])

    def body(tid, a, b, c, d):
        acc = fx.Vector.filled(8, 0.0, fx.Float32)
        av = ld_dwords(a, tid * na, na)
        bv = ld_dwords(c, tid * nb, nb)
        s_a = fx.Int32(ops.bo.buffer_load(rs(d), tid, vec_width=1, dtype=T.i32))
        s_b = fx.Int32(ops.bo.buffer_load(rs(d), 32 + tid, vec_width=1, dtype=T.i32))
        st_acc(mx.mfma_scaled(av, bv, acc, s_a, s_b, a_fmt, b_fmt, sel_a, sel_b), b, tid)

    out = torch.zeros(256, dtype=torch.float32, device="cuda")
    L = _gpu.kernel(body, 32)
    _gpu.run(L, dev(pa), out, dev(pb), dev(sbuf))
    return unpack_d(out.cpu()), L


def deq_fp4(codes, sc):
    return E2M1[torch.from_numpy(codes).long()] * torch.exp2(torch.from_numpy(sc).double() - 127).repeat_interleave(32, 1)


def deq_fp8(byts, sc):
    v = torch.from_numpy(byts).view(torch.float8_e4m3fn).double()
    return v * torch.exp2(torch.from_numpy(sc).double() - 127).repeat_interleave(32, 1)


for sel_a in (0, 1):
    for sel_b in (0, 1):
        for narrow in (True, False):
            ca = np.random.randint(0, 16, (16, 128)).astype(np.uint8)
            cb = np.random.randint(0, 16, (16, 128)).astype(np.uint8)
            lo, hi = (126, 129) if narrow else (118, 137)
            sa = np.random.randint(lo, hi, (16, 4)).astype(np.uint8)
            sb = np.random.randint(lo, hi, (16, 4)).astype(np.uint8)
            D, L = run_scaled(fp4_bytes(ca), fp4_bytes(cb), sa, sb, mx.FP4, mx.FP4, sel_a, sel_b)
            ref = deq_fp4(ca, sa) @ deq_fp4(cb, sb).T
            if narrow:
                R.check(f"mfma_scaled FP4xFP4 opsel ({sel_a},{sel_b}): exact (narrow scales)", torch.equal(D, ref))
            else:
                err = ((D - ref).abs().max() / ref.abs().max()).item()
                R.check(f"mfma_scaled FP4xFP4 opsel ({sel_a},{sel_b}): rel <= 1e-6 (wide scales)", err <= 1e-6, f"{err:.2e}")
R.check("mfma_scaled lowers to v_wmma_scale_f32_16x16x128_f8f6f4", "v_wmma_scale_f32_16x16x128_f8f6f4" in _gpu.isa(L))

for fa, fb, name in ((mx.FP4, mx.FP8, "FP4 A x FP8 B (A8W4)"), (mx.FP8, mx.FP8, "FP8 x FP8")):
    errs = []
    for trial in range(3):
        sa = np.random.randint(122, 133, (16, 4)).astype(np.uint8)
        sb = np.random.randint(122, 133, (16, 4)).astype(np.uint8)
        if fa == mx.FP4:
            ca = np.random.randint(0, 16, (16, 128)).astype(np.uint8)
            arows, adq = fp4_bytes(ca), deq_fp4(ca, sa)
        else:
            arows = torch.randn(16, 128).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
            adq = deq_fp8(arows, sa)
        brows = torch.randn(16, 128).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        D, _ = run_scaled(arows, brows, sa, sb, fa, fb, trial % 2, (trial + 1) % 2)
        ref = adq @ deq_fp8(brows, sb).T
        errs.append(((D - ref).abs().max() / ref.abs().max()).item())
    R.check(f"mfma_scaled {name}: layout, rel <= 1e-4", max(errs) <= 1e-4, f"max {max(errs):.2e} (hw fp8 dot is inexact)")

# --------------------------------- fp4_tile_load + gfx1250 scales, end to end
N_ROWS, K = 64, 1024
STEPS = K // 128
wc = np.random.randint(0, 16, (N_ROWS, K)).astype(np.uint8)
ws = np.random.randint(126, 129, (N_ROWS, K // 32)).astype(np.uint8)
xc = np.random.randint(0, 16, (16, K)).astype(np.uint8)
xs = np.random.randint(126, 129, (16, K // 32)).astype(np.uint8)
w_sh = shuffle_weight(torch.from_numpy(fp4_bytes(wc)).view(1, N_ROWS, K // 2)).numpy().reshape(-1)
s_sh = shuffle_scale_gfx1250(torch.from_numpy(ws).view(1, N_ROWS, K // 32)).numpy().reshape(-1)
xq = np.stack([pack_lanes(fp4_bytes(xc[:, 128 * s:128 * s + 128])) for s in range(STEPS)]).view(np.int32)  # [st, 32, 8]
xsc = np.zeros((STEPS // 2, 32, 4), np.uint8)
for p in range(STEPS // 2):
    for lane in range(32):
        stp = 2 * p + lane // 16
        xsc[p, lane] = xs[lane % 16, 4 * stp:4 * stp + 4]
xsc = xsc.view(np.int32).reshape(-1)
xbuf = np.concatenate([xq.reshape(-1), xsc])
XSC0 = xq.size


def body(tid, a, b, c, d):
    lane, rg = tid % 32, tid // 32
    r_w, r_s = rs(a), rs(c)
    acc = fx.Vector.filled(8, 0.0, fx.Float32)
    for st_ in range(STEPS):
        av = mx.fp4_tile_load(r_w, 0, rg, K, st_, lane * 4)
        bv = ld_dwords(d, (st_ * 32 + lane) * 8, 8)
        sa = fx.Int32(ops.bo.buffer_load(
            r_s, mx.scale_index_gfx1250(rg * 16 + lane % 16, 4 * (st_ - st_ % 2 + lane // 16), K // 32) // 4,
            vec_width=1, dtype=T.i32))
        sb = fx.Int32(ops.bo.buffer_load(rs(d), XSC0 + (st_ // 2) * 32 + lane, vec_width=1, dtype=T.i32))
        acc = mx.mfma_scaled(av, bv, acc, sa, sb, mx.FP4, mx.FP4, st_ % 2, st_ % 2)
    for v in range(8):
        ops.bo.buffer_store(acc[v], rs(b), tid * 8 + v)


out = torch.zeros(N_ROWS // 16 * 256, dtype=torch.float32, device="cuda")
L = _gpu.kernel(body, 32 * (N_ROWS // 16))
_gpu.run(L, dev(w_sh.view(np.int32)), out, dev(s_sh.view(np.int32)), dev(xbuf))
got = torch.cat([unpack_d(out.cpu()[256 * g:256 * g + 256]) for g in range(N_ROWS // 16)])
ref = deq_fp4(wc, ws) @ deq_fp4(xc, xs).T
R.check(f"fp4_tile_load + scale_index_gfx1250 + mfma_scaled: [{N_ROWS} x {K}] x 16 tokens exact", torch.equal(got, ref),
        f"max |err| {(got - ref).abs().max().item():.3e}")
asm = _gpu.kernel_text(_gpu.isa(L), L.kernel_name)
R.check("fp4_tile_load: two b128 loads a step, th:TH_LOAD_NT",
        asm.count("buffer_load_b128") >= 2 * STEPS and "TH_LOAD_NT" in asm)

R.done()
