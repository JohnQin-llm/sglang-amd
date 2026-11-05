"""Span A/B of three trees (ab_spike.py sync loop: graph of 20 chained calls, slowest rank's p50 per call, us):
k2v2_land1 (land1), k2v4_land1 (v4land1), k2v4b (v4b); golden digests (stream_out + branch_out + route ids)."""
import glob
import json
import os
import statistics as st

SP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "s3", "spike")
TREES = ("land1", "v4land1", "v4b")


def p50s(path):
    d = json.load(open(path))
    per = [[r[0] / d["reps"] for r in rr] for rr in d["sync"]]
    return max(st.median(p) for p in per), d["golden_digest"], all(d["replay_matches_golden"])


print("| layer | mode | trial | k2v2_land1 | k2v4_land1 | k2v4b | v4b - v4land1 | v4b vs land1 | digests equal | replay == golden |")
print("|---|---|---|---|---|---|---|---|---|---|")
means = {}
for L in (1, 31, 60):
    for mode in ("warm", "cold"):
        trials = sorted({int(os.path.basename(p).split("_t")[1].split("_")[0])
                         for p in glob.glob(f"{SP}/ab_*_t*_L{L}_{mode}.json")})
        for t in trials:
            a = {k: p50s(f"{SP}/ab_{k}_t{t}_L{L}_{mode}.json") for k in TREES
                 if os.path.isfile(f"{SP}/ab_{k}_t{t}_L{L}_{mode}.json")}
            if len(a) < 3:
                continue
            dg = {x for k in TREES for x in a[k][1]}
            for k in TREES:
                means.setdefault((L, mode, k), []).append(a[k][0])
            print(f"| {L} | {mode} | {t} | {a['land1'][0]:.2f} | {a['v4land1'][0]:.2f} | {a['v4b'][0]:.2f} | "
                  f"{a['v4b'][0] - a['v4land1'][0]:+.2f} | {a['land1'][0] / a['v4b'][0]:.3f}x | "
                  f"{'yes ' + next(iter(dg)) if len(dg) == 1 else 'NO ' + str(dg)} | {all(a[k][2] for k in TREES)} |")
for L in (1, 31, 60):
    for mode in ("warm", "cold"):
        v = {k: means.get((L, mode, k)) for k in TREES}
        if all(v.values()):
            m = {k: st.mean(x) for k, x in v.items()}
            print(f"mean L{L} {mode} ({len(v['v4b'])} trials): land1 {m['land1']:.2f}  v4land1 {m['v4land1']:.2f}  "
                  f"v4b {m['v4b']:.2f}  (v4b - v4land1 {m['v4b'] - m['v4land1']:+.2f}, v4b vs land1 "
                  f"{m['land1'] / m['v4b']:.3f}x)")
