"""Served per-layer time of the span K2 replaces, from a torch-profiler trace (VERIFY steps only, GPU kernels).
Usage: python3 served_span.py trace.json.gz [...]"""
import collections, gzip, json, re, sys

out = {}
for path in sys.argv[1:]:
    d = json.load(gzip.open(path))
    ev = d["traceEvents"]
    steps = [e for e in ev if e.get("cat") in ("gpu_user_annotation",) and "VERIFY" in e.get("name", "")]
    if not steps:
        steps = [e for e in ev if e.get("cat") == "user_annotation" and "VERIFY" in e.get("name", "")]
    k = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
    inwin = lambda e: any(s["ts"] <= e["ts"] < s["ts"] + s["dur"] for s in steps)
    kv = [e for e in k if inwin(e)]
    by = collections.defaultdict(lambda: [0.0, 0])
    for e in kv:
        by[e["name"][:90]][0] += e["dur"]
        by[e["name"][:90]][1] += 1
    tot = sum(v[0] for v in by.values())
    step_us = [s["dur"] for s in steps]
    rows = sorted(by.items(), key=lambda x: -x[1][0])
    out[path] = {"verify_steps": len(steps), "step_us": step_us, "kernel_us_per_step": tot / max(1, len(steps)),
                 "kernels": [(n, v[0] / len(steps), v[1] / len(steps)) for n, v in rows[:30]]}
    print(path.split("/")[-1], "verify steps", len(steps), "step wall us", [round(x) for x in step_us])
    print(f"  kernel time per verify step {tot / len(steps):.0f} us")
    for n, us, cnt in out[path]["kernels"]:
        print(f"  {us:8.1f} us/step {cnt:6.1f} calls/step  {us / max(cnt,1):7.2f} us/call  {n}")
json.dump(out, open("/data/muse455/k2/cand/k2v2_land1/s3/served_span.json", "w"), indent=1)
