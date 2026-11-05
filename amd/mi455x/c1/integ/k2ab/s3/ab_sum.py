"""Summarise ab_run.sh trials: per layer, mode, trial and tree the slowest rank's p50 per-call us (chain of 20 calls
a graph replay, us / call), and the golden digests (stream_out + branch_out + route ids) of both trees."""
import glob
import json
import os
import statistics as st

SP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "s3", "spike")


def p50s(path):
    d = json.load(open(path))
    per = [[r[0] / d["reps"] for r in rr] for rr in d["sync"]]
    return [st.median(p) for p in per], d["golden_digest"], d["replay_matches_golden"]


rows = []
for L in (1, 31, 60):
    for mode in ("warm", "cold"):
        for t in sorted({int(os.path.basename(p).split("_t")[1].split("_")[0]) for p in glob.glob(f"{SP}/ab_*_t*_L{L}_{mode}.json")}):
            a = {tree: p50s(f"{SP}/ab_{tree}_t{t}_L{L}_{mode}.json") for tree in ("land1", "k2v4")
                 if os.path.isfile(f"{SP}/ab_{tree}_t{t}_L{L}_{mode}.json")}
            if len(a) < 2:
                continue
            s, c = max(a["land1"][0]), max(a["k2v4"][0])
            same = a["land1"][1] == a["k2v4"][1] and len(set(a["k2v4"][1])) == 1
            rep = all(a["land1"][2]) and all(a["k2v4"][2])
            rows.append((L, mode, t, s, c, same, rep, a["k2v4"][1][0]))
print("| layer | mode | trial | k2v2_land1 us | k2v4 us | delta | speedup | outputs bit-identical | replay == golden |")
print("|---|---|---|---|---|---|---|---|---|")
for L, mode, t, s, c, same, rep, dg in rows:
    print(f"| {L} | {mode} | {t} | {s:.2f} | {c:.2f} | {c - s:+.2f} | {s / c:.3f}x | {'yes ' + dg if same else 'NO'} | {rep} |")
for L in (1, 31, 60):
    for mode in ("warm", "cold"):
        v = [(s, c) for LL, m, t, s, c, *_ in rows if LL == L and m == mode]
        if v:
            s = st.mean(x[0] for x in v)
            c = st.mean(x[1] for x in v)
            print(f"mean L{L} {mode}: land1 {s:.2f} k2v4 {c:.2f} delta {c - s:+.2f} ({s / c:.3f}x, {len(v)} trials)")
