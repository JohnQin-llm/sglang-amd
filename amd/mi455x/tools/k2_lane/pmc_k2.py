"""K2 hardware counters (KernelForge's rocprofv3 PMC method, gfx1250 counter names) from the lane's PMC passes.

Each pass is one `k2safe.sh -w "rocprofv3 --pmc <counters> -d <dir>/<pass> -o pmc --output-format csv --"` run of
`s3/probe_b2b.py --layers 31 --eager 10` (eager back-to-back launches; counters are not collected in graph replays).
Reads every <dir>/<pass>/pmc_counter_collection.csv, keeps the K2 dispatches (kernel name `muse_mono_k2*`), and
prints per GPU the median over launches of each counter, the dispatch time and the kernel's resources.

    python3 pmc_k2.py /data/muse455/k2/lane/pmc [--json out.json]
"""
import collections
import csv
import glob
import json
import os
import statistics as st
import sys


def load(root):
    per = collections.defaultdict(lambda: collections.defaultdict(dict))   # agent -> dispatch -> counter -> value
    meta = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "*counter_collection.csv"))):
        p = os.path.basename(os.path.dirname(f))
        for r in csv.DictReader(open(f)):
            if "muse_mono_k2" not in r["Kernel_Name"]:
                continue
            key = (p, r["Dispatch_Id"])
            d = per[r["Agent_Id"]][key]
            d[r["Counter_Name"]] = d.get(r["Counter_Name"], 0.0) + float(r["Counter_Value"])
            d["_dur_us"] = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1000
            meta = {k: r[k] for k in ("Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Block_Size", "VGPR_Count",
                                      "SGPR_Count", "Scratch_Size")}
    out = {}
    for agent, disp in sorted(per.items()):
        cols = collections.defaultdict(list)
        for d in disp.values():
            for c, v in d.items():
                cols[c].append(v)
        out[agent] = {c: st.median(v) for c, v in cols.items()}
        out[agent]["_launches"] = len(disp)
    return out, meta


def main():
    root = sys.argv[1]
    out, meta = load(root)
    print("kernel", meta)
    names = sorted({c for a in out.values() for c in a})
    print(f"{'counter':34s}" + "".join(f"{a:>16s}" for a in out))
    for c in names:
        print(f"{c:34s}" + "".join(f"{out[a].get(c, float('nan')):16.4g}" for a in out))
    if "--json" in sys.argv:
        json.dump({"meta": meta, "per_agent": out}, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
