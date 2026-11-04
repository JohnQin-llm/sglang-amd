"""The whole served TP4 QKV op (projection + QK-norm/RoPE/FP8 KV-cache epilogue) at 8 verify rows: served tile vs
the tuned Triton tile vs AITER's gfx1250 TDM split-K GEMM, same `_qkv_epilogue`. Outputs compared bit for bit
(Q, K, V, gate and the cache writes); timing as in qkv_bench.py (62 cold weights in one graph).
"""
import argparse
import json

import sglang
import torch
from sglang.kernels.ops.attention import qk_norm_rope_cache as Q

import qkv_bench as TB
import qkv_bench_gluon as GB

M, N, K, LAYERS = TB.M, TB.N, TB.K, TB.LAYERS
HQ, HK, D = 32, 4, 64
VARIANTS = {"served": None,
            "triton_sk16": ("triton", (16, 64, 128, 4, 3, ".cg")),
            "gluon_tdm_sk8": ("gluon", (32, 128, 8, 3, 2, ".cg"))}


def op(kind, cfg, x, w, pos, cs, loc, kc, vc, ks, vs, outs):
    if kind is None:
        return Q.qkv_proj_qk_norm_rope_cache(x, w, pos, cs, kc, vc, loc, None, 1e-6, 0.125, ks, vs,
                                             num_q_heads=HQ, num_kv_heads=HK)
    sk = cfg[0] if kind == "triton" else cfg[2]
    p = outs["p"][kind]
    if kind == "triton":
        TB.launch(x, w, p, cfg)
    else:
        outs["g"](x, w, p.view(sk, 1, M, N))
    qo, ko, vo, go = outs["o"]
    Q._qkv_epilogue[(M, HQ)](p, pos, cs, loc, None, kc, vc, qo, ko, vo, go, ks, vs, M, N, sk, HQ, HK, D, 1e-6, 0.125,
                             True, False, cs.stride(0), num_warps=1)
    return qo, ko, vo, go


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print("sglang from", sglang.__file__)
    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(LAYERS)]
    pos = torch.arange(130000, 130000 + M, device=dev, dtype=torch.int64)
    ang = torch.arange(262144, device=dev, dtype=torch.float32)[:, None] / (
        10000 ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))[None, :]
    cs = torch.cat([ang.cos(), ang.sin()], 1).contiguous()
    loc = torch.arange(5000, 5000 + M, device=dev, dtype=torch.int64)
    ks = torch.tensor([0.05], device=dev)
    vs = torch.tensor([0.05], device=dev)
    caches = {v: (torch.zeros(8192, HK, D, device=dev, dtype=torch.float8_e4m3fn),
                  torch.zeros(8192, HK, D, device=dev, dtype=torch.float8_e4m3fn)) for v in VARIANTS}
    outs = {"p": {"triton": torch.empty(16, M, N, device=dev, dtype=torch.float32),
                  "gluon": torch.empty(8, M, N, device=dev, dtype=torch.float32)},
            "g": GB.make_launch(VARIANTS["gluon_tdm_sk8"][1]),
            "o": (torch.empty(M, HQ * D, device=dev, dtype=torch.bfloat16), torch.empty(M, HK * D, device=dev, dtype=torch.bfloat16),
                  torch.empty(M, HK * D, device=dev, dtype=torch.bfloat16), torch.empty(M, HQ * D, device=dev, dtype=torch.bfloat16))}
    res, ref = {}, None
    for name, v in VARIANTS.items():
        kind, cfg = v if v else (None, None)
        kc, vc = caches[name]
        y = [t.clone() for t in op(kind, cfg, x, ws[0], pos, cs, loc, kc, vc, ks, vs, outs)]
        torch.cuda.synchronize()
        got = y + [kc[5000:5000 + M].view(torch.uint8).clone(), vc[5000:5000 + M].view(torch.uint8).clone()]
        if ref is None:
            ref = got
        diff = [int((g.view(torch.uint8) != r.view(torch.uint8)).sum()) for g, r in zip(got, ref)]
        us = TB.time_graph(lambda: [op(kind, cfg, x, w, pos, cs, loc, kc, vc, ks, vs, outs) for w in ws])
        res[name] = {"us": us, "bytes_differing_vs_served": diff}
        print(f"{name:14s} {us:6.2f} us per layer  differing bytes (q,k,v,gate,kcache,vcache) vs served: {diff}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
