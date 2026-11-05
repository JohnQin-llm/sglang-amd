"""Full-attention verify microbenchmark: served AITER split + reduce vs the attn_v2 split kernel (gfx1250, TP4 rank).

Fixtures, the served call and the fp64 reference come from k1att/common.py and vagm/attn_sweep.py (imported, not
modified): 8 verify rows, 32 q / 4 kv heads, D 64, FP8 E4M3 KV [pages, 16, 4, 64] with scattered pages.
Variants:
  served                AITER as served (gfx1250 DEFAULT.json: S128 T128 BM64 w4 st2, reduce SEGMENT_BLOCK 64)
  best                  vagm proposed DEFAULT.json (T64 st3, reduce SEGMENT_BLOCK 128)
  abl<bits>[_t64]       served kernel with timing ablations (1 no masks, 2 no softmax, 4 no PV, 8 no QK)
  v2_T<t>_S<s>_Q<q>_w<w>_st<n>   attn_v2_split + AITER reduce_segments (single-tile path)
Timing: one GPU, HIP graph replay; warm = R back-to-back calls / R; cold = R x [512 MiB flush + call] minus
R x [flush], / R; split = warm without the reduce. Trials alternate variant order; medians.
usage: bench.py --ctx 32768,131072,340000 --variants served,best,v2_T64_S128_Q2_w4_st2 --trials 15
"""
import argparse
import importlib.util
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["TRITON_CACHE_DIR"] = os.path.join(HERE, "tcache")
sys.path.insert(0, "/data/muse455/k2/cand/vagm")
sys.path.insert(0, HERE)

import torch  # noqa: E402

from vagm import attn_sweep as AS  # noqa: E402
from kern.attn_v2 import attn_v2_split  # noqa: E402
from kern.attn_v2t import attn_v2t_split  # noqa: E402
from kern.attn_v2p import attn_v2p_split  # noqa: E402

C, uaw = AS.C, AS.uaw
PROPOSED = "/data/muse455/k2/cand/vagm/results/DEFAULT.proposed.json"
_spec = importlib.util.spec_from_file_location("ua_abl", os.path.join(HERE, "kern", "ua_abl.py"))
ABLM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ABLM)
ORIG_K = uaw.kernel_unified_attention_3d


class AblWrap:
    def __init__(self, bits):
        self.bits = bits

    def __getitem__(self, grid):
        f = ABLM.kernel_unified_attention_3d[grid]
        return lambda *a, **kw: f(*a, ABL=self.bits, **kw)


def served_like(fx, out, seqused, json_path=None, abl=None, ovr=None):
    def call():
        AS.OVR.clear()
        AS.OVR.update(ovr or {"kv_split": {}, "attn_3d": {}, "reduce": {}})
        AS.set_json(json_path)
        uaw.kernel_unified_attention_3d = AblWrap(abl) if abl is not None else ORIG_K
        try:
            C.native_call(fx, out, seqused)
        finally:
            uaw.kernel_unified_attention_3d = ORIG_K
            AS.set_json(None)

    def split():
        orig = uaw._reduce_segments_triton
        uaw._reduce_segments_triton = lambda *x, **k: None
        try:
            call()
        finally:
            uaw._reduce_segments_triton = orig
    return call, split


def v2_like(fx, out, seqused, T, S, Q, w, nst, rsb=None, fold=False, dual=False, trans=False, pfd=0, pfs=0, wpe=0):
    dev = fx["q"].device
    so = torch.empty(C.ROWS, C.NQ, S, C.D, dtype=torch.float32, device=dev)
    sm = torch.empty(C.ROWS, C.NQ, S, dtype=torch.float32, device=dev)
    ss = torch.empty(C.ROWS, C.NQ, S, dtype=torch.float32, device=dev)
    q, k, v, tab = fx["q"], fx["k"], fx["v"], fx["table"][0]
    grid = (S, C.NKV, 1)

    def split():
        args = (so, sm, ss, q, k, v, tab, seqused, fx["cu_q"], fx["k_descale"], fx["v_descale"], C.SCALE,
                q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1),
                v.stride(2))
        if trans:
            kern = attn_v2p_split if trans == "p" else attn_v2t_split
            kern[grid](*args, NQ=C.NQ, G=C.NQ // C.NKV, D=C.D, PAGE=C.PAGE, TILE=T, NSEG=S, BLOCK_M=64,
                       num_warps=w, num_stages=nst)
        else:
            attn_v2_split[grid](*args, NQ=C.NQ, G=C.NQ // C.NKV, D=C.D, PAGE=C.PAGE, TILE=T, NSEG=S, BLOCK_M=64,
                                QSPLIT=Q, FOLD=fold, DUAL=dual, PFD=pfd, PF_SCOPE=pfs, num_warps=w, num_stages=nst,
                                waves_per_eu=wpe)

    def reduce():
        uaw.reduce_segments[(C.ROWS, C.NQ, 1)](
            output_ptr=out, segm_output_ptr=so, segm_max_ptr=sm, segm_expsum_ptr=ss, seq_lens_ptr=seqused,
            num_seqs=1, num_query_heads=C.NQ, out_scale_ptr=None, output_stride_0=out.stride(0),
            output_stride_1=out.stride(1), block_table_stride=fx["table"].stride(0), TILE_SIZE=T, HEAD_SIZE=C.D,
            HEAD_SIZE_PADDED=C.D, V_HEAD_SIZE=C.D, V_HEAD_SIZE_PADDED=C.D, query_start_len_ptr=fx["cu_q"],
            BLOCK_Q=None, NUM_SEGMENTS_PER_SEQ=S, SEGMENT_BLOCK=rsb or S, BLOCK_D=C.D, SLIDING_WINDOW=0,
            WINDOW_SPLIT=False, num_warps=1, num_stages=1, waves_per_eu=2, enable_fp_fusion=True)

    def call():
        split()
        reduce()
    return call, split


def make(name, fx, out, seqused):
    if name == "served":
        return served_like(fx, out, seqused)
    if name == "best":
        return served_like(fx, out, seqused, json_path=PROPOSED)
    if name.startswith("abl"):
        p = name[3:].split("_")
        jp = PROPOSED if "t64" in p[1:] else None
        return served_like(fx, out, seqused, json_path=jp, abl=int(p[0]))
    if name.startswith(("v2_", "v2t_", "v2p_")):
        kv = {s[0]: s[1:] for s in name.split("_")[1:]}
        T, S, Q = int(kv["T"]), int(kv["S"]), int(kv.get("Q", 0))
        if name.startswith(("v2t_", "v2p_")):
            return v2_like(fx, out, seqused, T, S, 0, int(kv.get("w", 4)),
                           int(kv["s"][1:]) if "s" in kv else 2, int(kv["r"]) if "r" in kv else None,
                           trans="p" if name.startswith("v2p_") else "t")
        w = int(kv.get("w", 4))
        nst = int(kv["s"][1:]) if "s" in kv else 2
        rsb = int(kv["r"]) if "r" in kv else None
        return v2_like(fx, out, seqused, T, S, Q, w, nst, rsb, fold=kv.get("F") == "1", dual=kv.get("D") == "1",
                       pfd=int(kv.get("P", 0)), pfs=int(kv.get("C", 0)), wpe=int(kv.get("e", 0)))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="131072")
    ap.add_argument("--variants", default="served,best")
    ap.add_argument("--trials", type=int, default=11)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seeds", default="1,101")
    ap.add_argument("--tag", default="b")
    ap.add_argument("--no-cold", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    names = a.variants.split(",")
    flush = torch.empty(AS.FLUSH_MB * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    results = {}
    for ctx in [int(c) for c in a.ctx.split(",")]:
        seeds = [int(s) for s in a.seeds.split(",")]
        fxs = {s: C.make_fixture("full", ctx, s, dev=dev) for s in seeds}
        refs = {s: AS.reference(fx) for s, fx in fxs.items()}
        fx = fxs[seeds[0]]
        seqused = fx["seq_lens"] + C.ROWS
        out = torch.empty(C.ROWS, C.NQ, C.D, dtype=torch.bfloat16, device=dev)
        res, graphs, served_out, keep = {}, {}, {}, []
        for s, f in fxs.items():
            o = torch.empty_like(out)
            served_like(f, o, f["seq_lens"] + C.ROWS)[0]()
            torch.cuda.synchronize()
            served_out[s] = o.clone()
        for n in names:
            r = {}
            try:
                errs = []
                for s, f in fxs.items():
                    o = torch.full_like(out, float("nan"))
                    make(n, f, o, f["seq_lens"] + C.ROWS)[0]()
                    torch.cuda.synchronize()
                    d = (o.double() - refs[s]).abs()
                    rb = refs[s].to(torch.bfloat16)
                    errs.append(dict(seed=s, max_abs=d.max().item(), mean_abs=d.mean().item(),
                                     nan=int(torch.isnan(o).sum()),
                                     ne_bf16ref=int((o.view(torch.int16) != rb.view(torch.int16)).sum()),
                                     ne_served=int((o.view(torch.int16) != served_out[s].view(torch.int16)).sum())))
                r["err"] = errs
                call, split = make(n, fx, out, seqused)
                keep.append((call, split))  # the v2 partial buffers live outside the graph pool
                R = a.reps
                g = {"warm": AS.graph_of(call, R), "split": AS.graph_of(split, R)}
                if not a.no_cold:
                    g["cold"] = AS.graph_of(lambda c=call: (flush.fill_(1.0), c()), R)
                graphs[n] = g
            except Exception as ex:  # noqa: BLE001
                r["error"] = repr(ex)[:400]
                print(f"[{ctx}] {n} FAILED {r['error']}", flush=True)
            res[n] = r
        gflush = AS.graph_of(lambda: flush.fill_(1.0), a.reps)
        t = {n: {k: [] for k in g} for n, g in graphs.items()}
        tf = []
        for g in graphs.values():
            for gg in g.values():
                AS.time_graph(gg)
        for i in range(a.trials):
            order = list(graphs) if i % 2 == 0 else list(reversed(graphs))
            tf.append(AS.time_graph(gflush))
            for n in order:
                for k, gg in graphs[n].items():
                    t[n][k].append(AS.time_graph(gg))
        fl = st.median(tf) / a.reps
        for n in graphs:
            r = res[n]
            r["warm_us"] = st.median(t[n]["warm"]) / a.reps
            r["split_us"] = st.median(t[n]["split"]) / a.reps
            r["cold_us"] = st.median(t[n]["cold"]) / a.reps - fl if "cold" in t[n] else float("nan")
            e = max(x["max_abs"] for x in r["err"])
            me = max(x["mean_abs"] for x in r["err"])
            nb = sum(x["ne_bf16ref"] for x in r["err"])
            ns = sum(x["ne_served"] for x in r["err"])
            nn = sum(x["nan"] for x in r["err"])
            print(f"[{ctx:6d}] {n:28s} warm {r['warm_us']:6.2f} cold {r['cold_us']:6.2f} split {r['split_us']:6.2f} "
                  f"maxerr {e:.3e} meanerr {me:.4e} ne_bf16ref {nb:5d} ne_served {ns:5d} nan {nn}", flush=True)
        results[str(ctx)] = res
        torch.cuda.synchronize()
        del graphs, keep
        torch.cuda.empty_cache()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"bench_{a.tag}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
