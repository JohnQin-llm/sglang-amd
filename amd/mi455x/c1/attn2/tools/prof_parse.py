"""Summarize rocprofv3 counter CSVs (results/prof/p*/) per split kernel: median per dispatch over the n calls.

Dispatch order in prof.py: per variant, 2 warm-up + n timed calls of (split, reduce). Kernels are matched by
name; the variants run in the order given on the command line, so dispatch groups map to variants in order.
"""
import csv
import glob
import statistics as st
import sys
from collections import defaultdict

ROOT = "/data/muse455/attn2/results/prof"
variants = sys.argv[1].split(",")
per = defaultdict(lambda: defaultdict(list))  # (variant) -> counter -> values
for g in sorted(glob.glob(f"{ROOT}/p*")):
    files = glob.glob(f"{g}/**/*counter_collection.csv", recursive=True)
    if not files:
        continue
    rows = list(csv.DictReader(open(files[0])))
    disp = defaultdict(dict)
    names = {}
    for r in rows:
        k = r["Kernel_Name"]
        if "reduce_segments" in k or ("unified_attention_3d" not in k and "attn_v2" not in k):
            continue
        d = int(r["Dispatch_Id"])
        disp[d][r["Counter_Name"]] = disp[d].get(r["Counter_Name"], 0.0) + float(r["Counter_Value"])
        names[d] = k
    ids = sorted(disp)
    n = len(ids) // len(variants)
    for i, v in enumerate(variants):
        for d in ids[i * n:(i + 1) * n][2:]:
            for c, x in disp[d].items():
                per[v][c].append(x)
counters = sorted({c for v in per.values() for c in v})
print(f"{'counter':28s} " + " ".join(f"{v[:22]:>22s}" for v in variants))
for c in counters:
    print(f"{c:28s} " + " ".join(f"{st.median(per[v][c]) if per[v][c] else float('nan'):22.4g}" for v in variants))
