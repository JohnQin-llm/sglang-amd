"""The fused QKV + gate projection (M=8, N=4608, K=8192) on AITER's gfx1250 TDM batched GEMM, as FP32 split-K partials.

AITER's Gluon `_batched_gemm_bf16_bandwidth_bound_kernel` with batch 1 and NUM_KSPLIT > 1 writes FP32 partials
[NUM_KSPLIT, 1, M, N]: the layout the served QKV epilogue (`_qkv_epilogue`) sums. Launched directly (no reduce),
timed like qkv_bench.py (62 cold weight copies in one graph, time per call). Writes JSON to --out.
"""
import argparse
import itertools
import json
import time

import torch
import triton
from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import create_shared_layouts, create_wmma_layouts
from aiter.ops.triton._gluon_kernels.gfx1250.gemm.batched.batched_gemm_bf16 import _KERNEL_MAP

from qkv_bench import HBM_TBS, LAYERS, K, M, N, time_graph

KERN = _KERNEL_MAP["bandwidth_bound"]


def make_launch(cfg):
    bn, bk, sk, nb, warps, cache = cfg
    split = K // sk
    assert split % bk == 0 and split // bk >= 1
    nb = min(nb, split // bk)
    wmma, opa, opb = create_wmma_layouts(warps)
    sha, shb = create_shared_layouts(16, bn, bk, "TN", 16)

    def launch(x, w, p):
        xq, wq = x.view(1, M, K), w.view(1, N, K).transpose(1, 2)
        KERN[(1, sk * triton.cdiv(M, 16) * (N // bn))](
            xq, wq, p, None, M, N, K,
            xq.stride(0), xq.stride(1), xq.stride(2), wq.stride(0), wq.stride(1), wq.stride(2),
            p.stride(1), p.stride(2), p.stride(3), p.stride(0), 0,
            BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, NUM_BUFFERS=nb, LAYOUT="TN", SHARED_LAYOUT_A=sha,
            SHARED_LAYOUT_B=shb, WMMA_LAYOUT=wmma, OPERAND_LAYOUT_A=opa, OPERAND_LAYOUT_B=opb, ADD_BIAS=False,
            NUM_KSPLIT=sk, SPLITK_BLOCK_SIZE=split, num_warps=warps, waves_per_eu=0, cache_modifier=cache)
    return launch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(LAYERS)]
    ref = x.float() @ ws[0].float().T
    floor = N * K * 2 / (HBM_TBS * 1e12) * 1e6
    res = {"shape": [M, N, K], "floor_us": floor, "runs": []}

    def bench(cfg):
        p = torch.empty(cfg[2], 1, M, N, device=dev, dtype=torch.float32)
        try:
            fn = make_launch(cfg)
            fn(x, ws[0], p)
            torch.cuda.synchronize()
            err = ((p.sum(0)[0] - ref).abs().max() / ref.abs().max()).item()
            us = time_graph(lambda: [fn(x, w, p) for w in ws])
        except Exception as ex:
            print("skip", cfg, type(ex).__name__, str(ex)[:100], flush=True)
            return None
        r = {"cfg": list(cfg), "us": us, "rel_err": err, "tbs": N * K * 2 / us / 1e6}
        res["runs"].append(r)
        print(f"  bn={cfg[0]:3d} bk={cfg[1]:3d} sk={cfg[2]:2d} nb={cfg[3]} w={cfg[4]} {cfg[5] or '--':3s} "
              f"{us:6.2f} us {r['tbs']:5.1f} TB/s err {err:.1e}", flush=True)
        return r

    t0 = time.time()
    print("coarse sweep")
    for bn, bk, sk in itertools.product((32, 64, 128, 256), (64, 128, 256, 512), (1, 2, 4, 8, 16, 32)):
        if (K // sk) % bk == 0:
            bench((bn, bk, sk, 3, 4, None))
    ok = lambda: sorted((r for r in res["runs"] if r["rel_err"] < 1e-2), key=lambda r: r["us"])
    top = ok()[:5]
    print("fine sweep around", [r["cfg"][:3] for r in top], flush=True)
    seen = {tuple(r["cfg"]) for r in res["runs"]}
    for r in top:
        bn, bk, sk = r["cfg"][:3]
        for nb, w, c in itertools.product((2, 3, 4), (1, 2, 4, 8), (None, ".cg")):
            if (bn, bk, sk, nb, w, c) not in seen:
                bench((bn, bk, sk, nb, w, c))
    res["best"] = ok()[0]
    print(f"best {res['best']['cfg']} {res['best']['us']:.2f} us ({res['best']['tbs']:.1f} TB/s), floor {floor:.2f} us "
          f"({time.time() - t0:.0f} s)")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
