"""Summarise ab_run6.sh: per layer / mode / trial / arm the slowest rank's p50 per-call us, deltas vs k2v4b, digests."""
import json
import os
import statistics as st

SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spike")
ARMS = ("v4b", "v6", "v6repm", "v6vmm", "v6fl")


def one(arm, t, L, mode):
    p = f"{SP}/ab_{arm}_t{t}_L{L}_{mode}.json"
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    per = [st.median([r[0] / d["reps"] for r in rr]) for rr in d["sync"]]
    return max(per), d["golden_digest"][0], len(set(d["golden_digest"])) == 1 and all(d["replay_matches_golden"])


print("| layer | mode | trial | " + " | ".join(ARMS) + " | digests equal, replay == golden |")
print("|---|---|---|" + "---|" * len(ARMS) + "---|")
means = {}
for L in (1, 31, 60):
    for mode in ("warm", "cold"):
        for t in (1, 2):
            r = {a: one(a, t, L, mode) for a in ARMS}
            if any(v is None for v in r.values()):
                continue
            ok = len({v[1] for v in r.values()}) == 1 and all(v[2] for v in r.values())
            base = r["v4b"][0]
            cells = [f"{r['v4b'][0]:.2f}"] + [f"{r[a][0]:.2f} ({r[a][0] - base:+.2f})" for a in ARMS[1:]]
            print(f"| {L} | {mode} | {t} | " + " | ".join(cells) + f" | {'yes ' + r['v4b'][1] if ok else 'NO'} |")
            for a in ARMS:
                means.setdefault((L, mode, a), []).append(r[a][0])
print()
for L in (1, 31, 60):
    for mode in ("warm", "cold"):
        b = st.mean(means[(L, mode, "v4b")])
        print(f"mean L{L} {mode}: " + ", ".join(f"{a} {st.mean(means[(L, mode, a)]):.2f} ({st.mean(means[(L, mode, a)]) - b:+.2f})" for a in ARMS))
