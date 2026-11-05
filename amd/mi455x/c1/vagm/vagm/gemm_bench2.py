"""Round 2: more QKV partial tiles + split epilogue, an empty-kernel graph floor, and tile sweeps for the DFlash draft
decode GEMMs of the served gfx1250 table (FlyDSL split-K / gluon a16w16) at M = 8, cold weights. Same timing as
gemm_bench.py."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gemm_bench as G  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

G.QKV_VARIANTS.update({
    "g_sk8_bn32_w4_epi2": ("gluon", (32, 128, 8, 3, 4, ".cg"), 8, "split"),
    "g_sk8_bn32_bk256_epi2": ("gluon", (32, 256, 8, 3, 2, ".cg"), 8, "split"),
    "g_sk8_bn16_epi2": ("gluon", (16, 128, 8, 3, 1, ".cg"), 8, "split"),
    "g_sk8_bn32_nb2_epi2": ("gluon", (32, 128, 8, 2, 2, ".cg"), 8, "split"),
    "g_sk8_bn32_nocg_epi2": ("gluon", (32, 128, 8, 3, 2, None), 8, "split"),
    "t_sk8_st3_bk256_epi2": ("triton", (8, 64, 16, 256, 4, 3, ".cg"), 8, "split"),
    "t_sk8_st4_epi2": ("triton", (8, 64, 16, 128, 4, 4, ".cg"), 8, "split"),
    "t_sk8_st3_w2_epi2": ("triton", (8, 64, 16, 128, 2, 3, ".cg"), 8, "split"),
    "t_sk8_bn32_st4_epi2": ("triton", (8, 32, 16, 128, 4, 4, ".cg"), 8, "split"),
})


@triton.jit
def _empty(X):
    pass


def fly(tn, tk, nw, nb, sk):
    return ("flydsl", dict(tile_m=16, tile_n=tn, tile_k=tk, m_warp=1, n_warp=nw, num_buffers=nb, split_k=sk))


def glu(bn, bk, w, nb=4):
    return ("gluon", dict(BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, NUM_BUFFERS=nb, kernel_type="bandwidth_bound",
                          num_warps=w))


DRAFT = {
    "draft_qkv": ((2560, 8192), [fly(32, 128, 2, 3, 16), fly(32, 128, 2, 3, 8), fly(32, 256, 2, 3, 16),
                                 fly(16, 128, 1, 3, 16), fly(32, 128, 2, 4, 32), fly(64, 128, 4, 3, 16),
                                 fly(16, 256, 1, 3, 32), fly(32, 64, 2, 4, 32), glu(32, 512, 4), glu(16, 512, 2)]),
    "draft_o": ((8192, 2048), [glu(64, 256, 2), glu(32, 256, 2), glu(32, 512, 2), glu(16, 256, 1), glu(64, 128, 2),
                               fly(32, 128, 2, 3, 4), fly(32, 128, 2, 3, 2), fly(64, 128, 4, 3, 4),
                               fly(32, 128, 2, 3, 8), fly(16, 128, 1, 3, 4)]),
    "draft_down": ((8192, 7168), [glu(32, 512, 2), glu(32, 256, 2), glu(64, 256, 2), glu(16, 512, 1),
                                  fly(32, 128, 2, 3, 8), fly(32, 128, 2, 3, 4), fly(64, 128, 4, 3, 4),
                                  fly(32, 256, 2, 3, 7)]),
    "draft_gate_up": ((14336, 8192), [glu(64, 256, 2), glu(32, 512, 2), glu(64, 512, 4), glu(128, 256, 4),
                                      fly(64, 128, 4, 3, 2), fly(32, 128, 2, 3, 4), fly(64, 128, 4, 4, 4)]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--tag", default="g2")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    res = {}
    x = torch.zeros(16, device=dev)
    res["empty_kernel_us"] = G.graph_time(lambda: [_empty[(1,)](x) for _ in range(62)], a.trials) / 62
    res["empty_kernel_256_us"] = G.graph_time(lambda: [_empty[(256,)](x) for _ in range(62)], a.trials) / 62
    print("empty kernel in graph", res["empty_kernel_us"], res["empty_kernel_256_us"], flush=True)
    flydsl_mm, gluon_mm = G.DG._kernels()
    M = 8
    res["draft"] = {}
    for name, ((n, k), cands) in DRAFT.items():
        copies = max(2, -(-2_500_000_000 // (n * k * 2)))
        wl = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        xx = torch.randn(M, k, device=dev, dtype=torch.bfloat16)
        out = torch.empty(M, n, device=dev, dtype=torch.bfloat16)
        ref = xx.float() @ wl[0].float().T
        served = G.DG._TABLE[(n, k, torch.bfloat16)]
        rows = []
        for be, cfg in [served] + cands:
            def mm(w, be=be, cfg=cfg):
                if be == "flydsl":
                    return flydsl_mm(xx, w, None, torch.bfloat16, out, **cfg)
                return gluon_mm(xx, w, None, torch.bfloat16, out, config=dict(cfg), backend="gluon")
            try:
                y = mm(wl[0])
                torch.cuda.synchronize()
                err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
                us = G.graph_time(lambda: [mm(w) for w in wl], a.trials) / copies
            except Exception as ex:
                print(name, be, cfg, "FAILED", repr(ex)[:150], flush=True)
                continue
            r = dict(backend=be, cfg=cfg, cold_us=us, rel_err=err, tbs=n * k * 2 / us / 1e6,
                     floor_us=n * k * 2 / G.HBM * 1e6)
            rows.append(r)
            print(name, json.dumps(r), flush=True)
        res["draft"][name] = rows
        del wl
        torch.cuda.empty_cache()
    sys.argv = [sys.argv[0], "--other", "0", "--trials", str(a.trials), "--tag", a.tag + "_qkv", "--variants",
                "served,g_sk8_bn32_epi2,t_sk8_st3_epi2," + ",".join(k for k in G.QKV_VARIANTS if k.endswith("epi2")
                                                                    and k not in ("g_sk8_bn32_epi2", "t_sk8_st3_epi2",
                                                                                  "served_epi2"))]
    os.makedirs(os.path.join(G.ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(G.ROOT, "results", f"gemm_{a.tag}.json"), "w"), indent=1)
    G.main()


if __name__ == "__main__":
    main()
