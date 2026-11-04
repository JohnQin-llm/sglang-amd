"""Kernel time by name and category in one torch-profiler trace (GPU events only)."""
import collections, gzip, json, re, sys

d = json.load(gzip.open(sys.argv[1]))
ev = d["traceEvents"]
cats = collections.Counter(e.get("cat") for e in ev)
print("categories:", dict(cats.most_common(10)))
k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
if not k:
    sys.exit("no GPU kernel events")
t0, t1 = min(e["ts"] for e in k), max(e["ts"] + e["dur"] for e in k)
busy = sum(e["dur"] for e in k)
print(f"{len(k)} GPU events over {(t1 - t0) / 1000:.2f} ms, busy {busy / 1000:.2f} ms ({100 * busy / (t1 - t0):.0f}%)")
steps = [e for e in ev if e.get("cat") == "user_annotation" and re.search(r"(?i)step|forward|decode|verify", e.get("name", ""))]
print("step annotations:", collections.Counter(e["name"][:40] for e in steps).most_common(6))
by = collections.defaultdict(float)
for e in k:
    by[e["name"]] += e["dur"]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
for name, us in sorted(by.items(), key=lambda x: -x[1])[:top]:
    n = sum(1 for e in k if e["name"] == name)
    print(f"{100 * us / busy:5.1f}%  {us / 1000:8.2f} ms  {n:5d} calls  {name[:110]}")
