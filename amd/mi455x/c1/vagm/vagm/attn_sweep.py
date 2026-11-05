"""Verify-attention config sweep (served AITER Triton kernel_unified_attention_3d + reduce_segments, gfx1250).

The served kernels are imported unchanged from the container's AITER; only the configs the wrapper reads are
overridden (get_unified_attention_config patched in the wrapper module): kv_split (NUM_SEGMENTS, TILE_SIZE,
SEGMENT_GRID_FIRST), attn_3d (BLOCK_M, num_warps, num_stages, waves_per_eu), reduce (num_warps, SEGMENT_BLOCK,
BLOCK_D). Fixtures and the served call are k1att/common.py (served layout, 8 verify rows, TP4 rank shapes).

Timing: one GPU, HIP graph replay. warm = graph of R back-to-back calls / R; cold = graph of R x [512 MiB flush,
call] minus graph of R x [flush], / R; split = warm with reduce_segments removed. Trials alternate variants;
median per variant. Numerics: bf16 output vs an fp64 reference (dequantized FP8 KV, causal + window mask).
usage: attn_sweep.py --kind swa|full --ctx 131072 --variants name,name|all --trials 9 --tag t
"""
import argparse
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(ROOT, "tcache"))

import torch  # noqa: E402

sys.path.insert(0, "/data/muse455/k2/cand/k1att")
from k1att import common as C  # noqa: E402

uaw = C.uaw
OVR = {}
_orig_cfg = uaw.get_unified_attention_config


def _patched_cfg(op, params, backend="triton", arch=None):
    cfg = _orig_cfg(op, params, backend=backend, arch=arch)
    cfg.update(OVR.get(op, {}))
    return cfg


uaw.get_unified_attention_config = _patched_cfg
FLUSH_MB = 512


def _load_kern(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"vagm_{name}", os.path.join(HERE, "kern", f"ua_{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel_unified_attention_3d, mod.reduce_segments


KERNS = {"orig": (uaw.kernel_unified_attention_3d, uaw.reduce_segments)}


def set_kern(name):
    if name not in KERNS:
        KERNS[name] = _load_kern(name)
    uaw.kernel_unified_attention_3d, uaw.reduce_segments = KERNS[name]


import aiter.ops.triton.utils.unified_attention_utils as UAU  # noqa: E402

_orig_load_json = UAU.load_config_json
JSON = {"path": None}


def _load_json(path, *a, **k):
    if JSON["path"] and path.endswith("gfx1250/triton/attention/unified_attention/DEFAULT.json"):
        with open(JSON["path"]) as f:
            return json.load(f)
    return _orig_load_json(path, *a, **k)


UAU.load_config_json = _load_json


def set_json(path):
    if JSON["path"] != path:
        JSON["path"] = path
        UAU._get_unified_attention_config_cached.cache_clear()


def V(S=None, T=None, BM=None, w=None, st_=None, wpe=None, sgf=None, rw=None, rsb=None, rbd=None, kern="orig",
      json_=None, qfp8=False):
    o = {"kv_split": {}, "attn_3d": {}, "reduce": {}, "kern": kern, "json": json_, "qfp8": qfp8}
    if S is not None:
        o["kv_split"]["NUM_SEGMENTS"] = S
    if T is not None:
        o["kv_split"]["TILE_SIZE"] = T
    if sgf is not None:
        o["kv_split"]["SEGMENT_GRID_FIRST"] = sgf
    for k, v in (("BLOCK_M", BM), ("num_warps", w), ("num_stages", st_), ("waves_per_eu", wpe)):
        if v is not None:
            o["attn_3d"][k] = v
    for k, v in (("num_warps", rw), ("SEGMENT_BLOCK", rsb), ("BLOCK_D", rbd)):
        if v is not None:
            o["reduce"][k] = v
    return o


VARIANTS = {
    "swa": {
        "served": V(),
        "s32t64": V(S=32), "s64t64": V(S=64), "s32t32": V(S=32, T=32), "s64t32": V(S=64, T=32),
        "s128t32": V(S=128, T=32), "s128t16": V(S=128, T=16), "s64t16": V(S=64, T=16),
        "sgf": V(sgf=1), "s64t32sgf": V(S=64, T=32, sgf=1),
        "bm16": V(BM=16), "bm32": V(BM=32), "s32t64bm16": V(S=32, BM=16), "s64t32bm16": V(S=64, T=32, BM=16),
        "s32t64bm32": V(S=32, BM=32),
        "w2": V(w=2), "w8": V(w=8), "st1": V(st_=1), "st3": V(st_=3),
        "s64t32w2": V(S=64, T=32, w=2), "s64t32st1": V(S=64, T=32, st_=1),
        "rbd32": V(rbd=32), "rbd16": V(rbd=16), "rw2": V(rw=2),
        "s64t32rbd32": V(S=64, T=32, rbd=32), "s64t32rbd16": V(S=64, T=32, rbd=16),
        "s32t32st3": V(S=32, T=32, st_=3), "s64t32st3": V(S=64, T=32, st_=3), "s32t32bm16": V(S=32, T=32, BM=16),
        "s32t32rbd32": V(S=32, T=32, rbd=32), "s64t32st3rbd32": V(S=64, T=32, st_=3, rbd=32),
        "s16t32": V(S=16, T=32), "s16t32st3": V(S=16, T=32, st_=3), "st3bm16": V(st_=3, BM=16),
        "s32t32st3bm16": V(S=32, T=32, st_=3, BM=16), "s32t64st3": V(S=32, st_=3),
        "s32t32st1": V(S=32, T=32, st_=1),
        "pf": V(kern="pf"), "pf_st1": V(st_=1, kern="pf"), "pf_s32t32": V(S=32, T=32, kern="pf"),
        "pf_s32t32st1": V(S=32, T=32, st_=1, kern="pf"), "pf_s16t32": V(S=16, T=32, kern="pf"),
        "pf_s16t32st1": V(S=16, T=32, st_=1, kern="pf"), "pf_s64t32": V(S=64, T=32, kern="pf"),
        "pf2_s16t32st1": V(S=16, T=32, st_=1, kern="pf2"), "pf2_s16t64": V(S=16, kern="pf2"),
        "pf_s64t32rbd32": V(S=64, T=32, rbd=32, kern="pf"), "pf_s32t32rbd32": V(S=32, T=32, rbd=32, kern="pf"),
        "ss": V(kern="ss"), "ss_s64t32rbd32": V(S=64, T=32, rbd=32, kern="ss"), "ss_s32t32st3": V(S=32, T=32, st_=3, kern="ss"),
        "ss_s64t32": V(S=64, T=32, kern="ss"), "pfss_s64t32rbd32": V(S=64, T=32, rbd=32, kern="pfss"),
        "pfss_s32t32": V(S=32, T=32, kern="pfss"), "pfss": V(kern="pfss"), "ss_st3": V(st_=3, kern="ss"),
        "ss_s32t64st3": V(S=32, st_=3, kern="ss"),
        "json_proposed": V(json_="/data/muse455/k2/cand/vagm/results/DEFAULT.proposed.json"), "json_proposed_bd32": V(json_="/data/muse455/k2/cand/vagm/results/DEFAULT.proposed_bd32.json"),
        "s64t32st3rsb128": V(S=64, T=32, st_=3, rsb=128), "s64t32st3rsb128rbd32": V(S=64, T=32, st_=3, rsb=128, rbd=32),
    },
    "full": {
        "served": V(),
        "s64": V(S=64), "s256": V(S=256), "s512": V(S=512),
        "t256s64": V(S=64, T=256), "t256s128": V(S=128, T=256), "t256s256": V(S=256, T=256),
        "t64s256": V(S=256, T=64), "t64s512": V(S=512, T=64),
        "bm32": V(BM=32), "bm16": V(BM=16), "bm16s64": V(S=64, BM=16), "bm32s64": V(S=64, BM=32),
        "st1": V(st_=1), "st3": V(st_=3), "w8": V(w=8), "w2": V(w=2),
        "wpe1": V(wpe=1), "wpe2": V(wpe=2),
        "s256st3": V(S=256, st_=3), "s64st3": V(S=64, st_=3),
        "rsb32": V(rsb=32), "rsb128": V(rsb=128), "rbd32": V(rbd=32), "rbd16": V(rbd=16), "rw2": V(rw=2),
        "s64rbd16": V(S=64, rbd=16), "s256rbd16": V(S=256, rbd=16),
        "st3rsb128": V(st_=3, rsb=128), "st3rsb128rbd32": V(st_=3, rsb=128, rbd=32), "rsb128rbd32": V(rsb=128, rbd=32),
        "st3rsb128rw2": V(st_=3, rsb=128, rw=2), "st4rsb128": V(st_=4, rsb=128),
        "pf": V(kern="pf"), "pf_st1": V(st_=1, kern="pf"), "pf_st3": V(st_=3, kern="pf"),
        "pf_rsb128": V(rsb=128, kern="pf"), "pf_st1_rsb128": V(st_=1, rsb=128, kern="pf"),
        "pf_s64": V(S=64, kern="pf"), "pf_s256": V(S=256, kern="pf"), "pf_t64s256": V(S=256, T=64, kern="pf"),
        "pf_t256s64": V(S=64, T=256, kern="pf"), "pf_st1_s256": V(S=256, st_=1, kern="pf"),
        "pf2": V(kern="pf2"), "pf2_st1": V(st_=1, kern="pf2"), "pf2_rsb128": V(rsb=128, kern="pf2"),
        "pf2_st1_rsb128": V(st_=1, rsb=128, kern="pf2"), "pf2_s64": V(S=64, kern="pf2"),
        "pf2_t64s256": V(S=256, T=64, kern="pf2"), "pf2_t64": V(T=64, kern="pf2"), "pf_t64": V(T=64, kern="pf"),
        "pf2_s256": V(S=256, kern="pf2"),
        "ss": V(kern="ss"), "pfss": V(kern="pfss"), "pf2ss": V(kern="pf2ss"), "pfss_rsb128": V(rsb=128, kern="pfss"),
        "pf2ss_rsb128": V(rsb=128, kern="pf2ss"), "pfss_st1_rsb128": V(st_=1, rsb=128, kern="pfss"),
        "pf2ss_st1_rsb128": V(st_=1, rsb=128, kern="pf2ss"),
        "t64": V(T=64), "t64st3": V(T=64, st_=3), "t64rsb128": V(T=64, rsb=128), "t64st3rsb128rbd32": V(T=64, st_=3, rsb=128, rbd=32),
        "t64st3rsb128": V(T=64, st_=3, rsb=128), "t32": V(T=32), "t32st3": V(T=32, st_=3), "t32rsb128": V(T=32, rsb=128),
        "t64s64": V(T=64, S=64), "t64s64st3": V(T=64, S=64, st_=3), "t32s256": V(T=32, S=256), "t64w8": V(T=64, w=8),
        "t64wpe2": V(T=64, wpe=2), "t32s64": V(T=32, S=64),
        "w1t32s512": V(w=1, T=32, S=512, rbd=16), "w1t64s512": V(w=1, T=64, S=512, rbd=16),
        "w1t32s256": V(w=1, T=32, S=256, rbd=16), "w1t64s256": V(w=1, T=64, S=256, rbd=16),
        "w1t64s512st3": V(w=1, T=64, S=512, st_=3, rbd=16), "w2t64s256": V(w=2, T=64, S=256, rbd=16),
        "w1t128s512": V(w=1, T=128, S=512, rbd=16), "w1t64s1024": V(w=1, T=64, S=1024, rbd=16),
        "qfp8": V(qfp8=True), "qfp8_t64st3rsb128": V(T=64, st_=3, rsb=128, qfp8=True),
        "qfp8_t256s64": V(T=256, S=64, qfp8=True),
        "json_proposed": V(json_="/data/muse455/k2/cand/vagm/results/DEFAULT.proposed.json"), "json_proposed_bd32": V(json_="/data/muse455/k2/cand/vagm/results/DEFAULT.proposed_bd32.json"),
        "s64t32st3rsb128": V(S=64, T=32, st_=3, rsb=128), "s64t32st3rsb128rbd32": V(S=64, T=32, st_=3, rsb=128, rbd=32),
    },
}


def graph_of(fn, reps):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    return g


def time_graph(g):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000.0


def reference(fx):
    """fp64 attention over the live keys (served masks: causal, window SLIDING_WINDOW = 1 + window[0])."""
    ctx, kind = fx["ctx"], fx["kind"]
    seq = ctx + C.ROWS
    n_log = math.ceil(seq / C.PAGE)
    pages = fx["table"][0, :n_log].long()
    k = fx["k"].view(torch.uint8)[pages].view(C.FP8).reshape(-1, C.NKV, C.D)[:seq].double()
    v = fx["v"].view(torch.uint8)[pages].view(C.FP8).reshape(-1, C.NKV, C.D)[:seq].double()
    q = fx["q"].double().view(C.ROWS, C.NKV, C.NQ // C.NKV, C.D)
    pos_q = ctx + torch.arange(C.ROWS, device=q.device)
    pos_k = torch.arange(seq, device=q.device)
    mask = pos_k[None, :] <= pos_q[:, None]
    if kind == "swa":
        sw = 1 + fx["window"][0]
        mask &= (pos_q[:, None] - pos_k[None, :]) < sw
    lo = 0 if kind == "full" else max(ctx - (1 + fx["window"][0]), 0)
    k, v, mask = k[lo:], v[lo:], mask[:, lo:]
    out = torch.empty(C.ROWS, C.NKV, C.NQ // C.NKV, C.D, dtype=torch.float64, device=q.device)
    for h in range(C.NKV):
        s = torch.einsum("rgd,kd->rgk", q[:, h], k[:, h]) * C.SCALE
        s = s.masked_fill(~mask[:, None, :], float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[:, h] = torch.einsum("rgk,kd->rgd", p, v[:, h])
    return out.view(C.ROWS, C.NQ, C.D)


def qcall(fx, out, seqused, ov):
    if not ov.get("qfp8"):
        return C.native_call(fx, out, seqused)
    if "q8" not in fx:
        fx["q8"] = fx["q"].to(C.FP8)
        fx["q_descale"] = torch.ones(1, dtype=torch.float32, device=fx["q"].device)
    uaw.unified_attention(
        q=fx["q8"], k=fx["k"], v=fx["v"], out=out, cu_seqlens_q=fx["cu_q"],
        seqused_k=seqused, max_seqlen_q=C.ROWS, max_seqlen_k=fx["table"].shape[1] * C.PAGE,
        softmax_scale=C.SCALE, causal=True, window_size=fx["window"], block_table=fx["table"],
        softcap=0.0, q_descale=fx["q_descale"], k_descale=fx["k_descale"], v_descale=fx["v_descale"], sinks=None)


def launch_info(fx, ov):
    set_json(ov["json"])
    lp = C.native_launch_params(fx)
    set_json(None)
    S = ov["kv_split"].get("NUM_SEGMENTS", lp["NUM_SEGMENTS"])
    T = ov["kv_split"].get("TILE_SIZE", lp["TILE_SIZE"])
    BM = max(ov["attn_3d"].get("BLOCK_M", lp["attn3d"]["BLOCK_M"]), 8)
    BQ = BM // (C.NQ // C.NKV)
    qb = C.ROWS // BQ + 1
    seq = fx["ctx"] + C.ROWS
    ntiles = -(-seq // T)
    first = max(seq - C.ROWS - C.SW + 1, 0) // T if fx["kind"] == "swa" else 0
    span = ntiles - first
    tps = max(-(-span // S), 1)
    live = -(-span // tps)
    return dict(S=S, T=T, BM=BM, grid=S * C.NKV * qb, live_segments=live, tiles_per_segment=tps,
                live_programs=live * C.NKV * (C.ROWS // BQ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="swa")
    ap.add_argument("--ctx", default="131072")
    ap.add_argument("--variants", default="all")
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seeds", default="1,101")
    ap.add_argument("--tag", default="sweep")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    table = VARIANTS[a.kind]
    names = list(table) if a.variants == "all" else a.variants.split(",")
    flush = torch.empty(FLUSH_MB * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    results = {}
    for ctx in [int(c) for c in a.ctx.split(",")]:
        seeds = [int(s) for s in a.seeds.split(",")]
        fxs = {s: C.make_fixture(a.kind, ctx, s, dev=dev) for s in seeds}
        refs = {s: reference(fx) for s, fx in fxs.items()}
        fx = fxs[seeds[0]]
        seqused = fx["seq_lens"] + C.ROWS
        out = torch.empty(C.ROWS, C.NQ, C.D, dtype=torch.bfloat16, device=dev)
        live_kv = (min(ctx + C.ROWS, C.SW + C.ROWS) if a.kind == "swa" else ctx + C.ROWS) * C.NKV * C.D * 2
        res = {}
        graphs = {}
        served_out = {}
        for n in names:
            ov = table[n]
            OVR.clear()
            OVR.update(ov)
            r = dict(launch_info(fx, ov))
            try:
                errs = []
                for s, f in fxs.items():
                    su = f["seq_lens"] + C.ROWS
                    o = torch.empty_like(out)
                    OVR.clear()
                    OVR.update(ov)
                    set_kern(ov["kern"])
                    set_json(ov["json"])
                    qcall(f, o, su, ov)
                    torch.cuda.synchronize()
                    d = (o.double() - refs[s]).abs()
                    rb = refs[s].to(torch.bfloat16)
                    if n == names[0]:
                        served_out[s] = o.clone()
                    errs.append(dict(seed=s, max_abs=d.max().item(), mean_abs=d.mean().item(),
                                     rel_max=(d.max() / refs[s].abs().max()).item(),
                                     ne_bf16ref=int((o.view(torch.int16) != rb.view(torch.int16)).sum()),
                                     ne_first=int((o.view(torch.int16) != served_out[s].view(torch.int16)).sum())))
                r["err"] = errs

                def call(ov=ov):
                    OVR.clear()
                    OVR.update(ov)
                    set_kern(ov["kern"])
                    set_json(ov["json"])
                    qcall(fx, out, seqused, ov)

                def split_only(ov=ov):
                    orig = uaw._reduce_segments_triton
                    uaw._reduce_segments_triton = lambda *x, **k: None
                    try:
                        call(ov)
                    finally:
                        uaw._reduce_segments_triton = orig

                R = a.reps
                graphs[n] = {"warm": graph_of(call, R), "cold": graph_of(lambda c=call: (flush.fill_(1.0), c()), R),
                             "split": graph_of(split_only, R)}
            except Exception as ex:
                r["error"] = repr(ex)[:300]
                print(f"[{a.kind} {ctx}] {n} FAILED {r['error']}", flush=True)
            res[n] = r
        OVR.clear()
        set_kern("orig")
        set_json(None)
        gflush = graph_of(lambda: flush.fill_(1.0), a.reps)
        t = {n: {k: [] for k in g} for n, g in graphs.items()}
        tf = []
        for g in graphs.values():
            for gg in g.values():
                time_graph(gg)
        for i in range(a.trials):
            order = list(graphs) if i % 2 == 0 else list(reversed(graphs))
            tf.append(time_graph(gflush))
            for n in order:
                for k, gg in graphs[n].items():
                    t[n][k].append(time_graph(gg))
        fl = st.median(tf) / a.reps
        for n in graphs:
            r = res[n]
            r["warm_us"] = st.median(t[n]["warm"]) / a.reps
            r["cold_us"] = st.median(t[n]["cold"]) / a.reps - fl
            r["split_warm_us"] = st.median(t[n]["split"]) / a.reps
            r["reduce_plus_gap_us"] = r["warm_us"] - r["split_warm_us"]
            r["kv_bytes"] = live_kv
            r["tbs_cold"] = live_kv / r["cold_us"] / 1e6
            e = max(x["max_abs"] for x in r["err"])
            me = max(x["mean_abs"] for x in r["err"])
            nb = sum(x["ne_bf16ref"] for x in r["err"])
            print(f"[{a.kind} {ctx:6d}] {n:14s} S{r['S']:4d} T{r['T']:4d} BM{r['BM']:3d} grid {r['grid']:5d} "
                  f"live {r['live_programs']:5d} (seg {r['live_segments']}x{r['tiles_per_segment']}t) "
                  f"warm {r['warm_us']:6.2f} cold {r['cold_us']:6.2f} split {r['split_warm_us']:6.2f} "
                  f"red+gap {r['reduce_plus_gap_us']:5.2f} {r['tbs_cold']:5.2f} TB/s maxerr {e:.3e} meanerr {me:.3e} ne_bf16ref {nb}", flush=True)
        results[str(ctx)] = res
        del graphs
        torch.cuda.empty_cache()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"attn_{a.kind}_{a.tag}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
