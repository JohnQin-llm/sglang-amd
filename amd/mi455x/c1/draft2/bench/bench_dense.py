"""Dense-path microbenchmarks at the served C1 verify shapes (8 rows, TP4 shard, gfx1250), graph replay.

  attn  : verify attention (AITER unified_attention + the served overrides: attn v2 full, SWA split sliding)
          followed by the served norm-gate kernel, vs the reduce kernel doing norm + gate (SGLANG_MUSE_REDUCE_GATE_NORM)
  norm  : FFN boundary (_fused_norm_residual_triton_kernel) served vs the hoisted-load variant
  add   : the per-layer ``seq_lens + max_q_len`` add (62 per step served, 1 with SGLANG_MUSE_VERIFY_SEQK_ONCE)

Cold = rotating input sets larger than the device cache (every call reads data last touched hundreds of MB ago).
Times: us per call, median of replays; trials alternate the arms.
"""
import sys

sys.path.insert(0, "/data/muse455/draft2/sglang/python")
import argparse
import json
import os

import torch


def graph_time(fn_list, reps=30):
    """fn_list: callables, one per call in the graph. Returns us per call (median over replays)."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for f in fn_list[:4]:
            f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for f in fn_list:
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / len(fn_list))
    return sorted(ts)[len(ts) // 2]


def bytes_diff(a, b):
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())


# ---------------------------------------------------------------------------------------------- attention
def bench_attn(args, res):
    os.environ["SGLANG_MUSE_ATTN_V2"] = "1"
    os.environ["SGLANG_MUSE_SWA_CFG"] = "1"
    os.environ["SGLANG_MUSE_REDUCE_GATE_NORM"] = "1"
    from sglang.srt.rocm_muse_spark import verify_attention as va

    va.install()
    from aiter.ops.triton.attention.unified_attention import unified_attention
    from sglang.srt.rocm_muse_spark.fused_norm_gate import FusedRMSNormGate
    from sglang.srt.rocm_muse_spark.layers import MuseSpark1xRMSNorm

    dev = torch.device("cuda:0")
    T, HQ, HK, D, PAGE = 8, 32, 4, 64, 16
    eps = 1e-5
    norm_mod = MuseSpark1xRMSNorm(D, eps=eps, has_weight=False)
    ng = FusedRMSNormGate(norm_mod, True, D)
    fp8 = torch.float8_e4m3fn

    for ctx in args.ctx:
        for kind in ("sliding", "full"):
            L = ctx + T
            nblk = (L + PAGE - 1) // PAGE
            sets = []
            nsets = args.cold_sets if args.cold else 1
            for i in range(nsets):
                torch.manual_seed(100 + i)
                kc = (torch.randn(nblk + 1, PAGE, HK, D, device=dev) * 0.5).to(fp8)
                vc = (torch.randn(nblk + 1, PAGE, HK, D, device=dev) * 0.5).to(fp8)
                bt = (torch.randperm(nblk, device=dev, dtype=torch.int32) + 1).view(1, nblk)
                q = torch.randn(T, HQ, D, device=dev, dtype=torch.bfloat16)
                gate = torch.randn(T, HQ * D, device=dev, dtype=torch.bfloat16) * 2
                sets.append(dict(kc=kc, vc=vc, bt=bt, q=q, gate=gate,
                                 o_ref=torch.empty(T, HQ, D, device=dev, dtype=torch.bfloat16),
                                 o_fus=torch.empty(T, HQ, D, device=dev, dtype=torch.bfloat16),
                                 y_ref=None))
            cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
            seqk = torch.tensor([L], device=dev, dtype=torch.int32)
            kd = torch.ones(1, device=dev, dtype=torch.float32)
            vd = torch.ones(1, device=dev, dtype=torch.float32)
            win = (2047, 0) if kind == "sliding" else (-1, -1)

            def attn(st, out):
                unified_attention(q=st["q"], k=st["kc"], v=st["vc"], out=out, cu_seqlens_q=cu, seqused_k=seqk,
                                  max_seqlen_q=T, max_seqlen_k=nblk * PAGE, softmax_scale=D ** -0.5, causal=True,
                                  window_size=win, block_table=st["bt"], softcap=0, q_descale=None,
                                  k_descale=kd, v_descale=vd, sinks=None)

            def served(st):
                def f():
                    attn(st, st["o_ref"])
                    st["y_ref"] = ng.forward_triton(st["o_ref"].view(T, HQ * D), st["gate"])
                return f

            def fused(st):
                def f():
                    va.arm_gate_norm(st["gate"].unsqueeze(1), eps)
                    attn(st, st["o_fus"])
                    assert va.take_gate_norm(), "reduce did not take the norm-gate"
                return f

            def plain(st):
                return lambda: attn(st, st["o_ref"])

            # numerics, eager
            diffs = []
            for st in sets[:3]:
                served(st)()
                fused(st)()
                torch.cuda.synchronize()
                diffs.append(bytes_diff(st["y_ref"].view(T, HQ, D), st["o_fus"]))
            ncall = max(len(sets), 8)
            seq = [sets[i % len(sets)] for i in range(ncall)]
            arms = {"served": [served(s) for s in seq], "fused": [fused(s) for s in seq],
                    "attn_only": [plain(s) for s in seq]}
            out = {k: [] for k in arms}
            for trial in range(args.trials):
                order = list(arms) if trial % 2 == 0 else list(arms)[::-1]
                for k in order:
                    out[k].append(graph_time(arms[k]))
            key = f"attn_{kind}_{ctx // 1024}K_{'cold' if args.cold else 'warm'}"
            res[key] = {k: sorted(v)[len(v) // 2] for k, v in out.items()}
            res[key]["bytes_differing"] = diffs
            res[key]["live_overrides"] = sorted(va._live)
            va._live.clear()
            print(key, json.dumps(res[key]), flush=True)


# ---------------------------------------------------------------------------------------------- norm
def bench_norm(args, res):
    from sglang.srt.rocm_muse_spark.fused_norm_residual import (
        _fused_norm_residual_hoist_kernel as HOIST,
        _fused_norm_residual_triton_kernel as SERVED,
    )

    dev = torch.device("cuda:0")
    M, H = 8, 8192
    nsets = args.cold_sets if args.cold else 1
    torch.manual_seed(0)
    sets = []
    for i in range(nsets):
        sets.append(dict(
            br=torch.randn(M, H, device=dev, dtype=torch.bfloat16),
            st=torch.randn(M, H, device=dev, dtype=torch.float32),
            pw=torch.rand(H, device=dev, dtype=torch.bfloat16) + 0.5,
            qw=torch.rand(H, device=dev, dtype=torch.bfloat16) + 0.5,
            al=torch.rand(H, device=dev, dtype=torch.float32),
            be=torch.rand(H, device=dev, dtype=torch.float32),
            out=torch.empty(M, H, device=dev, dtype=torch.bfloat16),
            ns=torch.empty(M, H, device=dev, dtype=torch.float32),
        ))
    e32 = torch.empty(0, device=dev, dtype=torch.float32)

    for input_scale in (False, True):
        def mk(ker, w, s):
            def f():
                ker[(M,)](s["br"], s["st"], s["out"], s["ns"], e32, s["pw"], s["qw"], s["al"], s["be"], 1e-8, 1e-5,
                          H, H, input_scale, True, True, True, True, False, False, 0, num_warps=w, num_stages=1)
            return f

        ref = None
        arms = {"served_w8": (SERVED, 8)}
        for w in args.warps:
            arms[f"hoist_w{w}"] = (HOIST, w)
        diffs = {}
        s0 = sets[0]
        for name, (ker, w) in arms.items():
            mk(ker, w, s0)()
            torch.cuda.synchronize()
            y = (s0["out"].clone(), s0["ns"].clone())
            if ref is None:
                ref = y
            diffs[name] = [bytes_diff(y[0], ref[0]), bytes_diff(y[1], ref[1])]
        ncall = max(nsets, 64)
        out = {k: [] for k in arms}
        for trial in range(args.trials):
            order = list(arms) if trial % 2 == 0 else list(arms)[::-1]
            for name in order:
                ker, w = arms[name]
                out[name].append(graph_time([mk(ker, w, sets[i % nsets]) for i in range(ncall)]))
        key = f"norm_inscale{int(input_scale)}_{'cold' if args.cold else 'warm'}"
        res[key] = {k: sorted(v)[len(v) // 2] for k, v in out.items()}
        res[key]["bytes_differing"] = diffs
        print(key, json.dumps(res[key]), flush=True)


# ---------------------------------------------------------------------------------------------- add
def bench_add(args, res):
    dev = torch.device("cuda:0")
    seq = torch.tensor([131072], device=dev, dtype=torch.int64)
    keep = []

    def f():
        keep.append(seq + 8)

    res["seqk_add_us_per_call"] = graph_time([f] * 62)
    print("seqk add", res["seqk_add_us_per_call"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["attn", "norm", "add", "all"])
    ap.add_argument("--ctx", type=int, nargs="+", default=[131072])
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--cold-sets", type=int, default=8)
    ap.add_argument("--warps", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    if a.what in ("add", "all"):
        bench_add(a, res)
    if a.what in ("norm", "all"):
        bench_norm(a, res)
    if a.what in ("attn", "all"):
        bench_attn(a, res)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
