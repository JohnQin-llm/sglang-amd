"""Served time per layer of the span K2 replaces (o_proj .. unpooler, FFN boundary excluded), VERIFY steps.
Each layer = the kernels between two attention kernels; the span = from the first GEMM after attention up to and
including the GEMM after the second all-reduce (the unpooler)."""
import gzip, json, statistics, sys

res = {}
for path in sys.argv[1:]:
    d = json.load(gzip.open(path))
    ev = d["traceEvents"]
    steps = [e for e in ev if e.get("cat") == "gpu_user_annotation" and "VERIFY" in e.get("name", "")]
    kall = sorted([e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"], key=lambda e: e["ts"])
    spans, parts = [], {}
    for s in steps:
        k = [e for e in kall if s["ts"] <= e["ts"] < s["ts"] + s["dur"]]
        att = [i for i, e in enumerate(k) if "unified_attention" in e["name"]]
        for a, b in zip(att, att[1:]):
            seg = k[a + 1:b]
            ars = [i for i, e in enumerate(seg) if "ar_ll" in e["name"]]
            if len(ars) != 2 or "rms_norm_gate" not in seg[0]["name"]:
                continue
            end = ars[1] + 1  # the unpooler GEMM follows the second AR
            span = seg[1:end + 1]
            spans.append(sum(e["dur"] for e in span))
            wall = span[-1]["ts"] + span[-1]["dur"] - span[0]["ts"]
            parts.setdefault("wall", []).append(wall)
            for e in span:
                n = e["name"][:40]
                parts.setdefault(n, []).append(e["dur"])
    rank = path.split("TP-")[1][0]
    res[rank] = {"layers": len(spans), "span_kernel_us_mean": statistics.mean(spans),
                 "span_kernel_us_median": statistics.median(spans),
                 "span_wall_us_mean": statistics.mean(parts["wall"]),
                 "parts_us_mean": {n: statistics.mean(v) * len(v) / len(spans) for n, v in parts.items() if n != "wall"}}
    print(f"rank {rank}: {len(spans)} layers, span kernel time mean {res[rank]['span_kernel_us_mean']:.1f} us "
          f"(median {res[rank]['span_kernel_us_median']:.1f}), span wall mean {res[rank]['span_wall_us_mean']:.1f} us")
    for n, v in sorted(res[rank]["parts_us_mean"].items(), key=lambda x: -x[1]):
        print(f"    {v:7.1f}  {n}")
json.dump(res, open("/data/muse455/k2/cand/k2v6/s3/served_span_layers.json", "w"), indent=1)
