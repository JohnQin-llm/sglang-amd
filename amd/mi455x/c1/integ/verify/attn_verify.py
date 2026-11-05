"""Changes 3 + 4 inside the integrated tree: AITER unified_attention with the SGLang verify-attention overrides.

Imports sglang from the worktree (PYTHONPATH) and the fixtures / fp64 reference / graph timing of
/data/muse455/attn2/bench.py and vagm/attn_sweep.py (8 verify rows, 32 q / 4 kv heads, D 64, FP8 KV, page 16).
- full layers: served (flags off) vs SGLANG_MUSE_ATTN_V2 (attn_v2_split + served reduce, single-tile path);
  also attn2's own v2_T128_S128_w4_st2_F1 launch, which the routed path must match bit for bit.
- sliding layers: served vs SGLANG_MUSE_SWA_CFG; also vagm's s64t32 override, which it must match bit for bit.
- flags off after install: byte-identical to AITER before install.
Timing: one GPU, graph of R calls; warm = R calls / R, cold = R x [512 MiB flush + call] - R x flush; alternating
trials, medians.
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, "/data/muse455/attn2")
sys.path.insert(0, "/data/muse455/k2/cand/vagm")
import bench as B  # noqa: E402
import torch  # noqa: E402

os.environ["TRITON_CACHE_DIR"] = "/data/muse455/integ/verify/tcache"

import sglang  # noqa: E402

assert sglang.__file__.startswith("/data/muse455/integ/sglang/"), sglang.__file__
from sglang.srt.rocm_muse_spark import verify_attention as VA  # noqa: E402

AS, C, uaw = B.AS, B.C, B.uaw


def bits(o):
    return o.view(torch.int16)


def run(fx, out_like, flags=None, ovr=None):
    o = torch.full_like(out_like, float("nan"))
    with_flags(flags, lambda: served_call(fx, o, ovr))
    torch.cuda.synchronize()
    return o


def served_call(fx, o, ovr=None):
    AS.OVR.clear()
    AS.OVR.update(ovr or {})
    try:
        C.native_call(fx, o, fx["seq_lens"] + C.ROWS)
    finally:
        AS.OVR.clear()


def with_flags(flags, fn):
    old = (VA.ATTN_V2, VA.SWA_CFG)
    VA.ATTN_V2, VA.SWA_CFG = flags or (False, False)
    try:
        return fn()
    finally:
        VA.ATTN_V2, VA.SWA_CFG = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="32768,131072,340000")
    ap.add_argument("--seeds", default="1,2,101,102")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="/data/muse455/integ/verify/results/attn_verify.json")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    seeds = [int(s) for s in a.seeds.split(",")]
    ctxs = [int(c) for c in a.ctx.split(",")]
    kinds = ("full", "swa")
    fxs = {(k, c, s): C.make_fixture(k, c, s, dev=dev) for k in kinds for c in ctxs for s in seeds}
    out0 = torch.empty(C.ROWS, C.NQ, C.D, dtype=torch.bfloat16, device=dev)

    # AITER as served, before any override is installed
    pre = {key: run(fx, out0) for key, fx in fxs.items()}
    VA.ATTN_V2, VA.SWA_CFG = True, True
    VA.install()
    VA.ATTN_V2, VA.SWA_CFG = False, False
    assert uaw._unified_attention_3d_triton.__module__ == VA.__name__

    variants = {"full": {"served": (False, False), "attn_v2": (True, False), "all_flags": (True, True)},
                "swa": {"served": (False, False), "swa_cfg": (False, True), "all_flags": (True, True)}}
    res = {"flags_off_identical": {}, "numerics": {}, "timing": {}, "fallback": {}}
    for key, fx in fxs.items():
        kind, ctx, s = key
        ref = AS.reference(fx)
        off = run(fx, out0)
        res["flags_off_identical"]["%s_%d_%d" % key] = int((bits(off) != bits(pre[key])).sum())
        rows = {}
        for n, fl in variants[kind].items():
            o = run(fx, out0, fl)
            d = (o.double() - ref).abs()
            rows[n] = dict(max_abs=d.max().item(), mean_abs=d.mean().item(), nan=int(torch.isnan(o).sum()),
                           ne_served=int((bits(o) != bits(pre[key])).sum()))
            if kind == "full" and n == "attn_v2":
                o2 = torch.full_like(out0, float("nan"))
                B.v2_like(fx, o2, fx["seq_lens"] + C.ROWS, 128, 128, 0, 4, 2, fold=True)[0]()
                torch.cuda.synchronize()
                rows[n]["ne_attn2_v2_T128"] = int((bits(o) != bits(o2)).sum())
            if kind == "swa" and n == "swa_cfg":
                o2 = run(fx, out0, None, {"kv_split": {"NUM_SEGMENTS": 64, "TILE_SIZE": 32}})
                rows[n]["ne_vagm_s64t32"] = int((bits(o) != bits(o2)).sum())
        res["numerics"]["%s_%d_%d" % key] = rows
        print("numerics", key, json.dumps(rows), "flags_off_ne", res["flags_off_identical"]["%s_%d_%d" % key],
              flush=True)

    # routing predicate on the served params: each precondition alone must send the call back to AITER
    seen = []
    inner = uaw._unified_attention_3d_triton

    def spy(params, *x, **kw):
        seen.append((params, x, kw))
        return inner(params, *x, **kw)

    uaw._unified_attention_3d_triton = spy
    run(fxs[("full", ctxs[0], seeds[0])], out0, (True, False))
    uaw._unified_attention_3d_triton = inner
    p, x, kw = seen[0]
    nseg, tile = x[3], x[4]
    VA.ATTN_V2 = True
    cases = {"served C1 verify": p, "num_seqs 2": p._replace(num_seqs=2), "9 rows": p._replace(num_tokens=9),
             "sinks": p._replace(sinks=torch.zeros(1)), "softcap": p._replace(softcap=30.0),
             "alibi": p._replace(use_alibi_slopes=True), "output_scale": p._replace(output_scale=torch.ones(1)),
             "shuffled": p._replace(shuffled_kv_cache=True), "head 128": p._replace(head_size=128),
             "fp8 q": p._replace(q_dtype=torch.float8_e4m3fn), "sliding": p._replace(sliding_window=2048)}
    res["fallback"] = {n: VA._v2_applies(c, nseg, tile, False) for n, c in cases.items()}
    res["fallback"]["tile 64"] = VA._v2_applies(p, nseg, 64, False)
    res["fallback"]["segments/tile"] = [nseg, tile]
    VA.ATTN_V2 = False
    print("v2 applies:", res["fallback"], flush=True)

    flush = torch.empty(AS.FLUSH_MB * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    gflush = AS.graph_of(lambda: flush.fill_(1.0), a.reps)
    for kind in kinds:
        for ctx in ctxs:
            fx = fxs[(kind, ctx, seeds[0])]
            out = torch.empty_like(out0)
            graphs = {}
            for n, fl in variants[kind].items():
                if n == "all_flags":
                    continue
                call = (lambda fx=fx, out=out: served_call(fx, out))
                graphs[n] = with_flags(fl, lambda: {
                    "warm": AS.graph_of(call, a.reps),
                    "cold": AS.graph_of(lambda: (flush.fill_(1.0), call()), a.reps)})
            for g in graphs.values():
                for gg in g.values():
                    AS.time_graph(gg)
            t = {n: {"warm": [], "cold": []} for n in graphs}
            tf = []
            for i in range(a.trials):
                tf.append(AS.time_graph(gflush))
                for n in (list(graphs) if i % 2 == 0 else list(reversed(graphs))):
                    for m, gg in graphs[n].items():
                        t[n][m].append(AS.time_graph(gg))
            fl_us = st.median(tf) / a.reps
            row = {n: {"warm_us": st.median(v["warm"]) / a.reps, "cold_us": st.median(v["cold"]) / a.reps - fl_us}
                   for n, v in t.items()}
            res["timing"][f"{kind}_{ctx}"] = row
            print("timing", kind, ctx, json.dumps(row), flush=True)
            del graphs
            torch.cuda.synchronize()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
