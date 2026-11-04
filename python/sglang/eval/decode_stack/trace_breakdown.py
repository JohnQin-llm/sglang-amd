"""Kernel breakdown of the draft+verify decode cycle from one torch-profiler trace (rank 0).

Kernels are assigned to the step they ran in by the GPU-side step annotations (step[DRAFT bs=1], step[VERIFY bs=1]),
so prefill chunks in the same trace are left out. Verify-step kernels are split by category; the draft step is one
row. Output: ms per cycle and share of the cycle, plus GPU idle inside the cycle.

    python3 trace_breakdown.py TRACE.json.gz OUT.json [label]
"""
import bisect
import collections
import gzip
import json
import re
import sys

CATS = [
    ("MoE mono kernel (K2)", r"muse_mono|mono_k2|k2g_s"),
    ("MoE expert GEMMs", r"fused_moe_gemm|moe_gemm"),
    ("MoE routing / align / finalize", r"_align_small|finalize_norm|routing|topk|_joint_epilogue|router_pooler"),
    ("attention", r"unified_attention|reduce_segments"),
    ("dense GEMMs (QKV, o_proj, router, lm_head)", r"_dense_partial|gemm_a16w16|^Cijk_|decode_gemm|gfx1250_gemm|bf16_tiled|splitk|_gemm_"),
    ("all-reduce / all-gather", r"ar_ll|all_reduce|allreduce|ag_gfx|nccl|all_gather"),
    ("norms, RoPE, QKV epilogue, KV-cache write", r"norm|rotary|reshape_and_cache|qk_norm|rope|kv_write|qkv_epilogue"),
    ("copies and small elementwise", r"copyBuffer|elementwise|triton_poi|Convert|fill|softcap|argmax|memcpy|memset"),
]


CONFIG = "/data/Muse-Spark-Open-816B-A42B-Quark-MXFP4/config.json"
ATTN_MAIN = r"unified_attention_(2d|3d)"


def main(path, out, label=""):
    cfg = json.load(open(CONFIG))
    types = cfg.get("text_config", cfg)["layer_types"]
    nfull = sum(t == "full_attention" for t in types)
    att = {"full_attention": f"attention: {nfull} full-attention layers",
           "sliding_attention": f"attention: {len(types) - nfull} sliding-window layers"}
    ev = json.load(gzip.open(path))["traceEvents"]
    steps = sorted((e["ts"], e["ts"] + e["dur"], "draft" if "DRAFT" in e["name"] else "verify")
                   for e in ev if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X"
                   and re.match(r"step\[(DRAFT|VERIFY) bs=1\]", e.get("name", "")))
    starts = [s[0] for s in steps]
    k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
    draft, by, other = 0.0, collections.defaultdict(float), collections.defaultdict(float)
    layer, cur, last = collections.Counter(), None, None
    for e in sorted(k, key=lambda e: e["ts"]):
        i = bisect.bisect_right(starts, e["ts"]) - 1
        if i < 0 or e["ts"] >= steps[i][1]:
            continue
        if steps[i][2] == "draft":
            draft += e["dur"]
            continue
        for name, pat in CATS:
            if re.search(pat, e["name"]):
                if name == "attention":
                    if re.search(ATTN_MAIN, e["name"]):
                        last = att[types[layer[i] % len(types)]]
                        layer[i] += 1
                    name = last or name
                by[name] += e["dur"]
                break
        else:
            other[e["name"][:60]] += e["dur"]
    n = sum(1 for s in steps if s[2] == "verify")
    span = sum(s[1] - s[0] for s in steps)
    busy = draft + sum(by.values()) + sum(other.values())
    names = [x for name, _ in CATS for x in ([att["full_attention"], att["sliding_attention"]] if name == "attention" else [name])]
    rows = [{"name": name, "ms": by[name] / n / 1000} for name in names if by[name] > 0]
    if any(c % len(types) for c in layer.values()):
        print(f"warning: attention calls per verify step {sorted(set(layer.values()))} are not a multiple of {len(types)}")
    rows.append({"name": "draft model (whole step)", "ms": draft / n / 1000})
    o = sum(other.values())
    if o:
        rows.append({"name": "other verify kernels", "ms": o / n / 1000,
                     "top": [(a, round(b / n / 1000, 3)) for a, b in sorted(other.items(), key=lambda x: -x[1])[:5]]})
    rows.append({"name": "GPU idle inside the steps", "ms": max(span - busy, 0) / n / 1000})
    cyc = span / n / 1000
    for r in rows:
        r["pct"] = 100 * r["ms"] / cyc
    res = {"label": label, "trace": path, "cycles": n, "cycle_ms": cyc, "rows": rows}
    json.dump(res, open(out, "w"), indent=1)
    print(f"{label}: {n} cycles, {cyc:.2f} ms per draft+verify cycle")
    for r in rows:
        print(f"  {r['ms']:7.3f} ms  {r['pct']:5.1f}%  {r['name']}" + (f"  {r.get('top')}" if r.get("top") else ""))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
