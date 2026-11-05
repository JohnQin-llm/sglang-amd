"""S1: TP4/EP1 loader parity on the real checkpoint (layers 1, 31, 60 by default).

Checks, per layer:
 A. (CPU, all 256 experts, all 4 ranks) byte parity: the 4 ranks' gfx1250-layout slices, un-shuffled and
    un-sliced, reproduce every byte of the native gate/up/down weights and scales.
 B. (CPU) the gfx1250 layout of every rank slice round-trips (inverse(forward) == native slice) and the
    kernel addressing (``w_dword`` / ``s_dword`` with EP1 shapes) gathers the right WMMA lane operands.
 C. (GPU, HIP_VISIBLE_DEVICES selects; 8 experts per layer) numeric parity: the full expert FFN
    y = W2 (silu(W1 x) * W3 x) with full dequantised weights == sum over ranks of the partial FFNs with each
    rank's slice, in float64 (equal to ~1e-12 relative), and the reduction of fp32 partials vs the full fp32
    FFN (reports max rel error).
Writes s1/test_ep1_loader.json.

Run: docker exec muse_mi455x bash -c 'cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=3 python3 /data/muse455/k2/cand/k2v4/s1/test_ep1_loader.py'
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/data/muse455/k2/cand/k2v4")
sys.path.insert(0, "/data/muse455/k2/port")
import ep1_layout as P  # noqa: E402
import layout_gfx1250 as L  # noqa: E402


def check(res, name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)
    res[name] = bool(cond)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,31,60")
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--gpu-experts", default="0,3,17,64,128,191,200,255")
    ap.add_argument("--json", default="/data/muse455/k2/cand/k2v4/s1/test_ep1_loader.json")
    args = ap.parse_args()
    out = {"ep1_bytes_per_gpu_GB": P.ep1_bytes_per_gpu() / 1e9}
    print(f"EP1 MXFP4 expert copy per GPU (layers 1-60): {out['ep1_bytes_per_gpu_GB']:.2f} GB", flush=True)
    allok = True
    g = torch.Generator().manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for layer in [int(x) for x in args.layers.split(",")]:
        res = {}
        t0 = time.time()
        rd = P._Reader()
        print(f"layer {layer}", flush=True)
        ok_bytes = ok_rt = ok_addr = True
        CH = 32
        for e0 in range(0, args.experts, CH):
            ex = range(e0, min(args.experts, e0 + CH))
            nat = [P.load_native_expert(rd, layer, e) for e in ex]
            native = [torch.stack([n[i] for n in nat]) for i in range(6)]
            slices = []
            for r in range(P.TP):
                ip = P.I // P.TP
                w13 = torch.cat([native[0][:, r * ip:(r + 1) * ip], native[1][:, r * ip:(r + 1) * ip]], 1)
                s13 = torch.cat([native[3][:, r * ip:(r + 1) * ip], native[4][:, r * ip:(r + 1) * ip]], 1)
                w2 = native[2][:, :, r * ip // 2:(r + 1) * ip // 2].contiguous()
                s2 = native[5][:, :, r * ip // 32:(r + 1) * ip // 32].contiguous()
                # same result as the per-expert loader function
                if e0 == 0 and r == 1:
                    one = P.slice_expert(*[n[3] for n in native], r)
                    ok_bytes &= all(torch.equal(a, b[3]) for a, b in zip(one, (w13, s13, w2, s2)))
                slices.append((w13.contiguous(), s13.contiguous(), w2, s2))
            gfx = [L.gfx1250_weights(*s) for s in slices]
            # B. round trip of the gfx1250 layout
            for s, gg in zip(slices, gfx):
                ok_rt &= torch.equal(L.unshuffle_weight(gg[0]), s[0]) and torch.equal(L.unshuffle_weight(gg[2]), s[2])
                ok_rt &= torch.equal(L.unshuffle_scale_gfx1250(gg[1]), s[1]) and \
                    torch.equal(L.unshuffle_scale_gfx1250(gg[3]), s[3])
            # A. un-shuffle + un-slice == native
            back = P.unslice([(L.unshuffle_weight(gg[0]), L.unshuffle_scale_gfx1250(gg[1]),
                               L.unshuffle_weight(gg[2]), L.unshuffle_scale_gfx1250(gg[3])) for gg in gfx])
            ok_bytes &= all(torch.equal(a, b) for a, b in zip(back, native))
            # B'. kernel addressing with EP1 shapes on a few random tiles of rank 2
            if e0 == 0:
                r = 2
                gw13 = gfx[r][0].reshape(-1).view(torch.int32)
                gs13 = gfx[r][1].reshape(-1).view(torch.int32)
                gw2 = gfx[r][2].reshape(-1).view(torch.int32)
                gs2 = gfx[r][3].reshape(-1).view(torch.int32)
                for (gw, gs, nat_w, nat_s, n, k) in ((gw13, gs13, slices[r][0], slices[r][1], 2 * P.I // P.TP, P.H),
                                                     (gw2, gs2, slices[r][2], slices[r][3], P.H, P.I // P.TP)):
                    for _ in range(100):
                        ee = int(torch.randint(0, len(ex), (1,), generator=g))
                        rg = int(torch.randint(0, n // 16, (1,), generator=g))
                        st = int(torch.randint(0, k // 128, (1,), generator=g))
                        tile = nat_w[ee, rg * 16:rg * 16 + 16, st * 64:st * 64 + 64]
                        codes = torch.stack([tile & 0xF, tile >> 4], -1).reshape(16, 128)
                        want = L.pack_operand(codes)
                        got = torch.stack([torch.cat([gw[L.w_dword(ee, rg, st, ln, h, n, k):][:4] for h in (0, 1)])
                                           for ln in range(32)])
                        ok_addr &= torch.equal(got, want)
                        for ln in range(32):
                            if ln // 16 != st % 2:
                                continue
                            sd = L.s_dword(ee, rg, st, ln, n, k)
                            ok_addr &= torch.equal(gs[sd:sd + 1].view(torch.uint8),
                                                   nat_s[ee, rg * 16 + ln % 16, 4 * st:4 * st + 4])
        allok &= check(res, f"L{layer} A: 4 rank slices reassemble every byte of {args.experts} experts", ok_bytes)
        allok &= check(res, f"L{layer} B: gfx1250 layout round trip of every rank slice", ok_rt)
        allok &= check(res, f"L{layer} B: EP1-shape kernel addressing gathers WMMA operands + scales", ok_addr)
        # C. numeric parity on GPU
        maxrel64 = maxrel32 = 0.0
        for e in [int(x) for x in args.gpu_experts.split(",")]:
            nat = [t.to(dev) for t in P.load_native_expert(rd, layer, e)]
            x = torch.randn(8, P.H, generator=g).to(dev, torch.float64)
            Wg, Wu, Wd = P.dequant(nat[0], nat[3]), P.dequant(nat[1], nat[4]), P.dequant(nat[2], nat[5])
            full64 = (torch.nn.functional.silu(x @ Wg.double().T) * (x @ Wu.double().T)) @ Wd.double().T
            full32 = (torch.nn.functional.silu(x.float() @ Wg.T) * (x.float() @ Wu.T)) @ Wd.T
            acc64 = torch.zeros_like(full64)
            acc32 = torch.zeros_like(full32)
            for r in range(P.TP):
                w13, s13, w2, s2 = P.slice_expert(*nat, r)
                W13, W2 = P.dequant(w13, s13), P.dequant(w2, s2)
                ip = P.I // P.TP
                h64 = x @ W13.double().T
                acc64 += (torch.nn.functional.silu(h64[:, :ip]) * h64[:, ip:]) @ W2.double().T
                h32 = x.float() @ W13.T
                acc32 += (torch.nn.functional.silu(h32[:, :ip]) * h32[:, ip:]) @ W2.T
            maxrel64 = max(maxrel64, ((acc64 - full64).norm() / full64.norm()).item())
            maxrel32 = max(maxrel32, ((acc32 - full32).norm() / full32.norm()).item())
        res[f"L{layer} C: rel-L2 sum-of-rank-partials vs full FFN, float64"] = maxrel64
        res[f"L{layer} C: rel-L2 sum-of-rank-partials vs full FFN, fp32"] = maxrel32
        allok &= check(res, f"L{layer} C: float64 partial-sum == full expert FFN (rel {maxrel64:.2e} <= 1e-12, "
                            f"fp32 rel {maxrel32:.2e}) on {dev}", maxrel64 <= 1e-12)
        res["seconds"] = time.time() - t0
        out[f"layer{layer}"] = res
        json.dump(out, open(args.json, "w"), indent=1)
    out["all_pass"] = bool(allok)
    json.dump(out, open(args.json, "w"), indent=1)
    print("ALL EP1 LOADER CHECKS PASSED" if allok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
