"""Verify-attention microbenchmark (gfx1250, TP4 rank): served AITER / attn_v2 (Triton) vs attn3 (Gluon, TDM).

Fixtures, the served call and the fp64 reference come from k1att/common.py and vagm/attn_sweep.py (imported, not
modified): 8 verify rows, 32 q / 4 kv heads, D 64, FP8 E4M3 KV [pages, 16, 4, 64] with scattered pages.
Variants (layer kind from --kind full | swa):
  served                full: AITER as in the container DEFAULT.json; swa: + SGLANG_MUSE_SWA_CFG split (64 x 32)
  v2_T<t>_S<s>          full: attn_v2_split (served Arbor iter 14 at T128 S128) + reduce_segments single tile
  g3_T<t>_S<s>_B<n>     attn3_split (Gluon) + reduce_segments single tile; swa adds the window range
Timing: one GPU, HIP graph replay; warm = R back-to-back calls / R; cold = R x [512 MiB flush + call] minus
R x [flush], / R; split = warm without the reduce. Trials alternate variant order; medians.
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
import triton  # noqa: E402

from vagm import attn_sweep as AS  # noqa: E402
import kern.attn3 as A3  # noqa: E402

C, uaw = AS.C, AS.uaw
_spec = importlib.util.spec_from_file_location("attn_v2_ref", "/data/muse455/attn2/kern/attn_v2.py")
_V2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_V2)
SWA_OVR = {"kv_split": {"NUM_SEGMENTS": 64, "TILE_SIZE": 32}, "attn_3d": {}, "reduce": {}}


def served_like(fx, out, seqused, ovr=None):
    def call():
        AS.OVR.clear()
        AS.OVR.update(ovr or {"kv_split": {}, "attn_3d": {}, "reduce": {}})
        C.native_call(fx, out, seqused)

    def split():
        orig = uaw._reduce_segments_triton
        uaw._reduce_segments_triton = lambda *x, **k: None
        try:
            call()
        finally:
            uaw._reduce_segments_triton = orig
    return call, split


def reduce_fn(fx, out, seqused, so, sm, ss, T, S, sw=0, window_split=False):
    def reduce():
        uaw.reduce_segments[(C.ROWS, C.NQ, 1)](
            output_ptr=out, segm_output_ptr=so, segm_max_ptr=sm, segm_expsum_ptr=ss, seq_lens_ptr=seqused,
            num_seqs=1, num_query_heads=C.NQ, out_scale_ptr=None, output_stride_0=out.stride(0),
            output_stride_1=out.stride(1), block_table_stride=fx["table"].stride(0), TILE_SIZE=T, HEAD_SIZE=C.D,
            HEAD_SIZE_PADDED=C.D, V_HEAD_SIZE=C.D, V_HEAD_SIZE_PADDED=C.D, query_start_len_ptr=fx["cu_q"],
            BLOCK_Q=None, NUM_SEGMENTS_PER_SEQ=S, SEGMENT_BLOCK=S, BLOCK_D=C.D, SLIDING_WINDOW=sw,
            WINDOW_SPLIT=window_split, num_warps=1, num_stages=1, waves_per_eu=2, enable_fp_fusion=True)
    return reduce


def v2_like(fx, out, seqused, T, S):
    dev = fx["q"].device
    so = torch.empty(C.ROWS, C.NQ, S, C.D, dtype=torch.float32, device=dev)
    sm = torch.empty(C.ROWS, C.NQ, S, dtype=torch.float32, device=dev)
    ss = torch.empty(C.ROWS, C.NQ, S, dtype=torch.float32, device=dev)
    q, k, v, tab = fx["q"], fx["k"], fx["v"], fx["table"][0]

    def split():
        _V2.attn_v2_split[(S, C.NKV, 1)](
            so, sm, ss, q, k, v, tab, seqused, fx["cu_q"], fx["k_descale"], fx["v_descale"], C.SCALE,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
            NQ=C.NQ, G=C.NQ // C.NKV, D=C.D, PAGE=C.PAGE, TILE=T, NSEG=S, BLOCK_M=64, QSPLIT=0, FOLD=True,
            DUAL=False, PFD=0, PF_SCOPE=0, num_warps=4, num_stages=2, waves_per_eu=0)
    red = reduce_fn(fx, out, seqused, so, sm, ss, T, S)

    def call():
        split()
        red()
    return call, split, (so, sm, ss)


def g3_like(fx, out, seqused, T, S, B, kw):
    dev = fx["q"].device
    so = torch.empty(C.ROWS, C.NQ, S, C.D, dtype=torch.float32, device=dev)
    sm = torch.empty(C.ROWS, C.NQ, S, dtype=torch.float32, device=dev)
    ss = torch.empty(C.ROWS, C.NQ, S, dtype=torch.float32, device=dev)
    q, k, v, tab = fx["q"], fx["k"], fx["v"], fx["table"][0]
    kern = getattr(A3, kw.pop("kern", "attn3_split"))
    sw = (1 + fx["window"][0]) if fx["kind"] == "swa" else 0

    def split():
        kern[(S, C.NKV, 1)](
            so, sm, ss, q, k, v, tab, seqused, fx["cu_q"], fx["k_descale"], fx["v_descale"], C.SCALE,
            q.stride(0), q.stride(1), k.shape[0],
            NQ=C.NQ, NKV=C.NKV, G=C.NQ // C.NKV, D=C.D, PAGE=C.PAGE, TILE=T, NSEG=S, NBUF=B, BLOCK_M=64,
            SW=sw, num_warps=4, **kw)
    red = reduce_fn(fx, out, seqused, so, sm, ss, T, S, sw=sw, window_split=sw > 0)

    def call():
        split()
        red()
    return call, split, (so, sm, ss)


def make(name, fx, out, seqused):
    if name == "served":
        return served_like(fx, out, seqused, SWA_OVR if fx["kind"] == "swa" else None)
    kv = {s[0]: s[1:] for s in name.split("_")[1:]}
    if name.startswith("v2_"):
        return v2_like(fx, out, seqused, int(kv["T"]), int(kv["S"]))[:2]
    if name.startswith("g3_"):
        extra = {}
        if "W" in kv:
            extra["waves_per_eu"] = int(kv["W"])
        if "P" in kv:
            extra["PAD"] = int(kv["P"])
        if "A" in kv:
            extra["ABL"] = int(kv["A"])
        if "V" in kv:
            extra["AV"] = int(kv["V"])
        if "F" in kv or "X" in kv:
            raise ValueError("non-fused / wide TDM variants are removed (17:28 UTC GPU hang)")
        return g3_like(fx, out, seqused, int(kv["T"]), int(kv["S"]), int(kv.get("B", 4)), extra)[:2]
    raise ValueError(name)


def require_dry_ok(names, kind):
    """Refuse g3 variants that have not passed dryrun.sh (1K, 4K, 340K) for the current kernel source."""
    import hashlib
    sha = hashlib.sha1(open(os.path.join(HERE, "kern", "attn3.py"), "rb").read()).hexdigest()[:12]
    ok = set()
    path = os.path.join(HERE, "logs", "dry_ok.txt")
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 4 and p[3] == sha:
                ok.add((p[0], p[1], int(p[2])))
    missing = [n for n in names if n.startswith("g3_")
               and not all((n, kind, c) in ok for c in (1024, 4096, 340000))]
    if missing:
        raise SystemExit(f"refusing variants without a passing dry run for kernel {sha}: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="full")
    ap.add_argument("--ctx", default="131072")
    ap.add_argument("--variants", default="served,v2_T128_S128,g3_T64_S128_B4")
    ap.add_argument("--trials", type=int, default=11)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seeds", default="1,101")
    ap.add_argument("--tag", default="b")
    ap.add_argument("--no-cold", action="store_true")
    ap.add_argument("--numerics-only", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    names = a.variants.split(",")
    require_dry_ok(names, a.kind)
    flush = torch.empty(AS.FLUSH_MB * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    results = {}
    for ctx in [int(c) for c in a.ctx.split(",")]:
        seeds = [int(s) for s in a.seeds.split(",")]
        fxs = {s: C.make_fixture(a.kind, ctx, s, dev=dev) for s in seeds}
        refs = {s: AS.reference(fx) for s, fx in fxs.items()}
        fx = fxs[seeds[0]]
        seqused = fx["seq_lens"] + C.ROWS
        out = torch.empty(C.ROWS, C.NQ, C.D, dtype=torch.bfloat16, device=dev)
        res, graphs, served_out, keep = {}, {}, {}, []
        for s, f in fxs.items():
            o = torch.empty_like(out)
            make("served", f, o, f["seq_lens"] + C.ROWS)[0]()
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
                if not a.numerics_only:
                    call, split = make(n, fx, out, seqused)
                    keep.append((call, split))
                    R = a.reps
                    g = {"warm": AS.graph_of(call, R), "split": AS.graph_of(split, R)}
                    if not a.no_cold:
                        g["cold"] = AS.graph_of(lambda c=call: (flush.fill_(1.0), c()), R)
                    graphs[n] = g
            except Exception as ex:  # noqa: BLE001
                r["error"] = repr(ex)[:600]
                print(f"[{ctx}] {n} FAILED {r['error']}", flush=True)
            res[n] = r
        gflush = AS.graph_of(lambda: flush.fill_(1.0), a.reps)
        t = {n: {k: [] for k in g} for n, g in graphs.items()}
        tf = []
        for g in graphs.values():
            for gg in g.values():
                AS.time_graph(gg)
        for i in range(a.trials if graphs else 0):
            order = list(graphs) if i % 2 == 0 else list(reversed(graphs))
            tf.append(AS.time_graph(gflush))
            for n in order:
                for k, gg in graphs[n].items():
                    t[n][k].append(AS.time_graph(gg))
        fl = st.median(tf) / a.reps if tf else 0.0
        for n in names:
            r = res[n]
            if "err" not in r:
                continue
            nan_ = float("nan")
            if n in graphs:
                r["warm_us"] = st.median(t[n]["warm"]) / a.reps
                r["split_us"] = st.median(t[n]["split"]) / a.reps
                r["cold_us"] = st.median(t[n]["cold"]) / a.reps - fl if "cold" in t[n] else nan_
            e = max(x["max_abs"] for x in r["err"])
            me = max(x["mean_abs"] for x in r["err"])
            nb = sum(x["ne_bf16ref"] for x in r["err"])
            ns = sum(x["ne_served"] for x in r["err"])
            nn = sum(x["nan"] for x in r["err"])
            emax = " ".join(f"{x['max_abs']:.3e}" for x in r["err"])
            print(f"[{a.kind} {ctx:6d}] {n:22s} warm {r.get('warm_us', nan_):6.2f} cold {r.get('cold_us', nan_):6.2f} "
                  f"split {r.get('split_us', nan_):6.2f} maxerr {e:.3e} ({emax}) meanerr {me:.4e} "
                  f"ne_bf16ref {nb:5d} ne_served {ns:5d} nan {nn}", flush=True)
        results[str(ctx)] = res
        torch.cuda.synchronize()
        del graphs, keep
        torch.cuda.empty_cache()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"bench_{a.kind}_{a.tag}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
