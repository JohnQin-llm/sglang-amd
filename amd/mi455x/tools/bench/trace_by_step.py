"""GPU kernel time inside each step annotation (gpu_user_annotation step[...]) and its top kernels."""
import collections, gzip, json, re, sys
ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
steps = [e for e in ev if e.get("cat") == "gpu_user_annotation" and str(e.get("name", "")).startswith("step[")]
k.sort(key=lambda e: e["ts"])
agg = collections.defaultdict(lambda: {"n": 0, "span": 0.0, "busy": 0.0, "top": collections.defaultdict(float)})
for s in steps:
    kind = re.sub(r" toks=\d+", "", s["name"])
    a, b = s["ts"], s["ts"] + s["dur"]
    inside = [e for e in k if a <= e["ts"] < b]
    g = agg[s["name"] if "EXTEND" in s["name"] else kind]
    g["n"] += 1; g["span"] += s["dur"]; g["busy"] += sum(e["dur"] for e in inside)
    for e in inside:
        g["top"][e["name"][:70]] += e["dur"]
for name, g in agg.items():
    n = g["n"]
    print(f"{name}: {n} x, {g['span']/n/1000:.2f} ms each, GPU busy {g['busy']/n/1000:.2f} ms each")
    for kn, us in sorted(g["top"].items(), key=lambda x: -x[1])[:8]:
        print(f"    {us/n/1000:7.2f} ms  {kn}")
