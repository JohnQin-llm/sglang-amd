"""Microbenchmark of the TP4 fused QKV + output-gate projection (M=8 verify rows, N=4608, K=8192, BF16) on gfx1250.

Each timed graph runs the projection over 62 distinct weight copies (one per layer, ~4.7 GB, so every call reads its
weight from HBM as in serving) and is replayed; the time per call is the graph time / 62. Tiles: a coarse sweep of
split-K x block N x block K, then warps x stages x cache modifier around the best. Numerics: the summed partials
against an FP32 reference. Writes JSON to the path given by --out.

    python3 qkv_bench.py --out /data/muse455/decode_stack/qkv_bench.json
"""
import argparse
import itertools
import json
import time

import torch
import triton
import triton.language as tl

M, N, K, LAYERS = 8, 4608, 8192, 62
HBM_TBS = 16.5   # measured practical streaming rate on this GPU (K2 lane D), TB/s


@triton.jit
def _dense_partial(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, SK: tl.constexpr, CACHE: tl.constexpr):
    pn, pm, pk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = pk * (K // SK) + tl.arange(0, BK)
    ap = X + m[:, None] * K + k[None, :]
    bp = W + n[None, :] * K + k[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(K // SK // BK):
        a = tl.load(ap, m[:, None] < M, other=0)
        b = tl.load(bp, cache_modifier=CACHE)
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK
    tl.store(P + pk * M * N + m[:, None] * N + n[None, :], acc, m[:, None] < M)


def launch(x, w, p, cfg):
    sk, bn, bk, warps, stages, cache = cfg
    _dense_partial[(N // bn, 1, sk)](x, w, p, M, N, K, 16, bn, bk, sk, cache, num_warps=warps, num_stages=stages,
                                     matrix_instr_nonkdim=16)


def time_graph(fn, reps=20):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / LAYERS)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    print("device", torch.cuda.get_device_name(dev), "CUs", torch.cuda.get_device_properties(dev).multi_processor_count)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(LAYERS)]
    ref = x.float() @ ws[0].float().T
    floor = N * K * 2 / (HBM_TBS * 1e12) * 1e6
    res = {"shape": [M, N, K], "layers": LAYERS, "floor_us": floor, "hbm_tbs": HBM_TBS, "runs": []}

    def bench(cfg):
        sk = cfg[0]
        p = torch.empty(sk, M, N, device=dev, dtype=torch.float32)
        try:
            launch(x, ws[0], p, cfg)
            torch.cuda.synchronize()
            err = ((p.sum(0) - ref).abs().max() / ref.abs().max()).item()
            us = time_graph(lambda: [launch(x, w, p, cfg) for w in ws])
        except Exception as ex:  # a tile the compiler or LDS cannot take
            print("skip", cfg, str(ex)[:80])
            return None
        r = {"cfg": list(cfg), "us": us, "rel_err": err, "tbs": N * K * 2 / us / 1e6}
        res["runs"].append(r)
        print(f"  sk={cfg[0]:2d} bn={cfg[1]:3d} bk={cfg[2]:3d} w={cfg[3]} st={cfg[4]} {cfg[5] or '--':3s} "
              f"{us:6.2f} us  {r['tbs']:5.1f} TB/s  err {err:.1e}", flush=True)
        return r

    base = (8, 64, 128, 4, 2, ".cg")
    print("served tile"); res["served"] = bench(base)
    # hipBLASLt for reference
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    res["hipblaslt_us"] = time_graph(lambda: [torch.mm(x, w.T, out=out) for w in ws])
    print(f"  hipBLASLt torch.mm {res['hipblaslt_us']:.2f} us")
    print("coarse sweep")
    t0 = time.time()
    for sk, bn, bk in itertools.product((2, 4, 8, 16, 32), (16, 32, 64, 128), (64, 128, 256)):
        if K // sk // bk >= 1 and K % (sk * bk) == 0:
            bench((sk, bn, bk, 4, 2, ".cg"))
    top = sorted((r for r in res["runs"] if r["rel_err"] < 1e-2), key=lambda r: r["us"])[:5]
    print("fine sweep around", [r["cfg"][:3] for r in top])
    seen = {tuple(r["cfg"]) for r in res["runs"]}
    for r in top:
        sk, bn, bk = r["cfg"][:3]
        for w, st, c in itertools.product((2, 4, 8), (1, 2, 3), (".cg", "")):
            if (sk, bn, bk, w, st, c) not in seen:
                bench((sk, bn, bk, w, st, c))
    ok = sorted((r for r in res["runs"] if r["rel_err"] < 1e-2), key=lambda r: r["us"])
    res["best"] = ok[0]
    print(f"best {ok[0]['cfg']} {ok[0]['us']:.2f} us vs served {res['served']['us']:.2f} us, floor {floor:.2f} us "
          f"({time.time() - t0:.0f} s)")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
