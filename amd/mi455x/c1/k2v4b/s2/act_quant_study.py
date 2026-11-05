"""S2 numerics study: which activation format for the EP1 expert stages (torch emulation, exact weights).

For the same calibrated inputs as ep1_moe.py, per-route expert outputs are computed in fp32 from dequantised
MXFP4 weights with the two expert inputs (xe -> gate/up, inter -> down) quantised as
  a16 (bf16), a8 (MXFP8 e4m3 + e8m0 per 32, OCP rule), a4 (MXFP4, aiter rule = the FlyDSL kernel's),
then the finalize / unpool / FFN boundary as ep1_moe.py. Reports residual-stream / next-attn cosine vs the stock arm.

Run: cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=3 python3 /data/muse455/k2/cand/k2v4/s2/act_quant_study.py
"""

import argparse
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/muse455/k2/cand/k2v4/s2")
import ep1_moe as S  # noqa: E402

P = S.P


def q_a4(x):
    q, s = S.quant_a4(x)
    return S.dequant_a4(q, s)


def q_a8(x):
    """OCP MXFP8 e4m3: shared exp = floor(log2(amax)) - 8, elements RNE to e4m3, saturating at 448."""
    r, k = x.shape
    xf = x.float().view(r, k // 32, 32)
    amax = xf.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(torch.where(amax > 0, amax, torch.ones_like(amax)))) - 8
    scale = torch.exp2(torch.clamp(e, -127, 127))
    v = torch.clamp(xf / scale, -448, 448).to(torch.float8_e4m3fn).float()
    return (v * scale).view(r, k)


QS = {"a16": lambda x: x.to(torch.bfloat16).float(), "a8": q_a8, "a4": q_a4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,31,60")
    ap.add_argument("--sets", type=int, default=4)
    ap.add_argument("--json", default="/data/muse455/k2/cand/k2v4/s2/act_quant_study.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    base = S.mb.load_module("mmt_base", "/data/muse455/k2/quickwin/muse_moe_triton_base.py")
    rd = P._Reader()
    out = {}
    combos = [("a4", "a4"), ("a8", "a4"), ("a4", "a8"), ("a8", "a8"), ("a16", "a8"), ("a16", "a16")]
    for layer in [int(x) for x in args.layers.split(",")]:
        lay = S.Layer(layer, dev, rd)
        d = lay.d
        corr = json.load(open("/data/muse455/k2/cand/k2v4/s2/ep1_moe_a4.json"))[f"layer{layer}"]["corr"]
        sets = []
        for k in range(args.sets):
            stream, fi = S.make_inputs(d, corr, 1000 * layer + k, dev)
            _, ids, sc, _ = S.route(d, fi)
            sets.append((stream, fi, ids, sc, S.expert_input(d, fi)))
        experts = sorted(set(int(e) for s in sets for e in s[2].view(-1).tolist()))
        for e in experts:
            lay.native[e] = [t.to(dev) for t in P.load_native_expert(rd, layer, e)]
        W = {e: (P.dequant(n[0], n[3]), P.dequant(n[1], n[4]), P.dequant(n[2], n[5])) for e, n in lay.native.items()}
        ep4 = [lay.ep4_stack(r) for r in range(S.TP)]
        res = {}
        cos = {c: ([], []) for c in combos}
        for stream, fi, ids, sc, xe in sets:
            acc = torch.zeros(S.ROWS, S.MH, dtype=torch.float32, device=dev)
            for r in range(S.TP):
                w13, s13, w2, s2 = ep4[r]
                acc += base.muse_moe_triton(xe, sc, ids, w13, w2, 64 * r, d["post_w"], 1e-8, post_norm_gain=1.0,
                                            has_norm=True, w13_scale=s13, w2_scale=s2).float()
            st_s, nx_s = S.boundary(d, stream, F.linear(acc.to(torch.bfloat16), d["unpool"]))
            for c in combos:
                xq = QS[c[0]](xe.float())
                y = torch.zeros(64, S.MH, device=dev)
                for s in range(64):
                    Wg, Wu, Wd = W[int(ids.view(-1)[s])]
                    h = F.silu(xq[s // 8] @ Wg.T) * (xq[s // 8] @ Wu.T)
                    hq = QS[c[1]](h.to(torch.bfloat16).float()[None])[0]
                    y[s] = hq @ Wd.T
                moe = S.finalize(y.to(torch.bfloat16), sc, d["post_w"])
                st, nx = S.boundary(d, stream, F.linear(moe, d["unpool"]))
                cos[c][0].append(S.cos_rows(st, st_s))
                cos[c][1].append(S.cos_rows(nx, nx_s))
        for c in combos:
            a, b = torch.cat(cos[c][0]), torch.cat(cos[c][1])
            res[f"{c[0]}_{c[1]}"] = dict(stream_min=float(a.min()), stream_mean=float(a.mean()),
                                         next_min=float(b.min()), next_mean=float(b.mean()))
            print(f"layer {layer} xe {c[0]:3s} inter {c[1]:3s}: stream cos min {a.min():.6f} mean {a.mean():.6f} | "
                  f"next-attn min {b.min():.6f}", flush=True)
        out[f"layer{layer}"] = res
        json.dump(out, open(args.json, "w"), indent=1)
        del lay, ep4, W
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
