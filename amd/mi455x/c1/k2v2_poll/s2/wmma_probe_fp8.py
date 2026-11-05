"""Probe the FP8 (e4m3) B-operand layout of V_WMMA_SCALE_F32_16X16X128_F8F6F4 on gfx1250 (A = FP4 weights,
fmtA=4; B = FP8 activations, fmtB=0; 16 dwords a lane).

Method: A is random e2m1 (known), all scales 1.0. For each B byte position (lane, byte) one run sets that byte to
e4m3 1.0 (0x38) for 16 lanes at once (lanes 0..15 or 16..31) and everything else to 0, then D[:, n] = A[:, k] tells
(n, k) for each lane by matching D's columns against A's columns. Prints the layout map and checks hypotheses.
Then a random full test (random A, random fp8 B, random scales) against torch with the found layout.

Run: cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=3 python3 /data/muse455/k2/cand/k2v2_land1/s2/wmma_probe_fp8.py
"""

import json
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int64, T

sys.path.insert(0, "/data/muse455/k2/port")
from layout_gfx1250 import pack_operand  # noqa: E402

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def rsrc(addr):
    return bo.create_buffer_resource_from_addr(addr)


def make(sel_a, sel_b):
    @flyc.kernel(name=f"wmma_probe_fp8_{sel_a}{sel_b}", known_block_size=[32, 1, 1])
    def probe(a: Int64, b: Int64, sa: Int64, sb: Int64, out: Int64):
        lane = fx.thread_idx.x
        ra, rb, rsa, rsb, ro = rsrc(a), rsrc(b), rsrc(sa), rsrc(sb), rsrc(out)
        av = fx.Vector(bo.buffer_load(ra, lane * 8, vec_width=4, dtype=T.i32)).shuffle(
            fx.Vector(bo.buffer_load(ra, lane * 8 + 4, vec_width=4, dtype=T.i32)), list(range(8)))
        bq = [fx.Vector(bo.buffer_load(rb, lane * 16 + 4 * j, vec_width=4, dtype=T.i32)) for j in range(4)]
        bv = bq[0].shuffle(bq[1], list(range(8))).shuffle(bq[2].shuffle(bq[3], list(range(8))), list(range(16)))
        s_a = bo.buffer_load(rsa, lane, vec_width=1, dtype=T.i32)
        s_b = bo.buffer_load(rsb, lane, vec_width=1, dtype=T.i32)
        acc = fx.Vector.filled(8, 0.0, fx.Float32)
        d = fx.Vector(rocdl.wmma_scale_f32_16x16x128_f8f6f4(
            T.vec(8, T.f32), av, bv, acc, s_a, s_b, fmtA=4, fmtB=0, scaleAType=sel_a, scaleBType=sel_b))
        for i in range_constexpr(8):
            bo.buffer_store(d[i], ro, lane * 8 + i)

    @flyc.jit
    def launch(a: Int64, b: Int64, sa: Int64, sb: Int64, out: Int64, stream: fx.Stream = fx.Stream(None)):
        probe(a, b, sa, sb, out).launch(grid=(1,), block=(32,), stream=stream)

    return launch


def run(launch, a_lanes, b_lanes, sa, sb):
    out = torch.zeros(32 * 8, dtype=torch.float32, device="cuda")
    launch(a_lanes.data_ptr(), b_lanes.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(),
           stream=fx.Stream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    o = out.view(32, 8).cpu()
    D = torch.zeros(16, 16)
    for lane in range(32):
        for v in range(8):
            D[8 * (lane // 16) + v, lane % 16] = o[lane, v]
    return D


def hyp_k(lane, byte, h):
    g = lane // 16
    if h == "H1_8elem":  # dword pair j -> K 16 j + 8 g + (byte % 8)
        j = byte // 8
        return 16 * j + 8 * g + byte % 8
    if h == "H2_16elem":  # 16-byte chunk c -> K 32 c + 16 g + (byte % 16)
        c = byte // 16
        return 32 * c + 16 * g + byte % 16
    if h == "H3_32elem":  # 32-byte half c -> K 64 c + 32 g + (byte % 32)
        c = byte // 32
        return 64 * c + 32 * g + byte % 32
    raise ValueError(h)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    launch = make(0, 0)
    codes = torch.randint(0, 16, (16, 128))
    codes[codes == 8] = 0  # no -0
    A = E2M1[codes]
    a_lanes = pack_operand(codes).reshape(-1).to(dev)
    ones = torch.full((32,), 0x7F7F7F7F, dtype=torch.int64).to(torch.int32).to(dev)
    kmap = {}
    for half in (0, 1):
        for byte in range(64):
            b = torch.zeros(32, 64, dtype=torch.uint8)
            b[16 * half:16 * half + 16, byte] = 0x38
            D = run(launch, a_lanes, b.view(torch.int32).reshape(-1).to(dev), ones, ones)
            for n in range(16):
                col = D[:, n]
                if col.abs().sum() == 0:
                    kmap[(16 * half + n, byte)] = None
                    continue
                match = [k for k in range(128) if torch.equal(col, A[:, k])]
                kmap[(16 * half + n, byte)] = match[0] if len(match) == 1 else ("ambig", match[:4])
    lanes_n_ok = all(v is not None for v in kmap.values())
    print("every (lane, byte) maps to one column n = lane % 16:", lanes_n_ok)
    for h in ("H1_8elem", "H2_16elem", "H3_32elem"):
        ok = all(kmap[(l, b)] == hyp_k(l, b, h) for l in range(32) for b in range(64))
        print(f"  {h}: {'MATCH' if ok else 'no'}")
    print("lane 0 bytes 0..63 ->", [kmap[(0, b)] for b in range(64)])
    print("lane 16 bytes 0..63 ->", [kmap[(16, b)] for b in range(64)])
    json.dump({f"{l},{b}": kmap[(l, b)] for l in (0, 1, 16, 17) for b in range(64)},
              open("/data/muse455/k2/cand/k2v2_land1/s2/wmma_probe_fp8_map.json", "w"))


if __name__ == "__main__" and "--full" not in sys.argv:
    main()


def full_random_test(n_trials=20):
    """Random A (fp4), random fp8 B, random e8m0 scales on both sides vs torch with the H2 layout."""
    dev = "cuda"
    torch.manual_seed(1)
    res = {}
    for sel in (0, 1):
        launch = make(sel, sel)
        worst = {"all": 0.0, "nosub": 0.0}
        for t in range(n_trials):
            codes = torch.randint(0, 16, (16, 128)); codes[codes == 8] = 0
            A = E2M1[codes]
            Bv = (torch.randn(128, 16) * 4).to(torch.float8_e4m3fn)
            if t % 2 == 0:  # no subnormals / tiny values
                Bf = Bv.float(); Bf[Bf.abs() < 2 ** -6] = 0.25; Bv = Bf.to(torch.float8_e4m3fn)
            Bf = Bv.float()
            sa = torch.randint(120, 134, (16, 4)); sb = torch.randint(120, 134, (16, 4))
            ref = (A * torch.exp2(sa.float() - 127).repeat_interleave(32, 1)) @ \
                  (Bf * torch.exp2(sb.float() - 127).repeat_interleave(32, 1).T)
            b_lanes = torch.zeros(32, 64, dtype=torch.uint8)
            bb = Bv.view(torch.uint8)  # [128 k, 16 n]
            for lane in range(32):
                for byte in range(64):
                    b_lanes[lane, byte] = bb[hyp_k(lane, byte, "H2_16elem"), lane % 16]
            def pack_sc(sc):
                d = torch.zeros(32, 4, dtype=torch.uint8)
                d[16 * sel:16 * sel + 16] = sc.to(torch.uint8)
                return d.view(torch.int32).reshape(-1).to(dev)
            D = run(launch, pack_operand(codes).reshape(-1).to(dev), b_lanes.view(torch.int32).reshape(-1).to(dev),
                    pack_sc(sa), pack_sc(sb))
            rel = float((D - ref).norm() / ref.norm())
            worst["all" if t % 2 else "nosub"] = max(worst["all" if t % 2 else "nosub"], rel)
        res[f"opsel{sel}"] = worst
        print(f"random fp4 x fp8 with scales, opsel {sel}: max rel err (with subnormals) {worst['all']:.2e}, "
              f"(no subnormals) {worst['nosub']:.2e}")
    return res


if __name__ == "__main__" and "--full" in sys.argv:
    full_random_test()
