"""Correlate KFD quiesce -> resume windows (kfd_evict3.bt) with spike.py slow iterations.  python3 trace_an.py <trace> <tag>"""
import collections, glob, json, sys
sys.path.insert(0, "s3")
from spike_an import queued_times

tr, tag = sys.argv[1], sys.argv[2]
ev = []
for l in open(tr):
    p = l.split()
    if len(p) > 2 and p[0].isdigit():
        d = {"t": int(p[0]), "k": p[1]}
        d.update(x.split("=", 1) for x in p[2:] if "=" in x)
        ev.append(d)
files = sorted(glob.glob(f"s3/spike/{tag}_L*.json"))
meta = json.load(open(files[0]))["meta"]
rank_of = {m["pid"]: i for i, m in enumerate(meta)}
win, open_ = [], {}
for e in ev:
    if e["k"] == "quiesce":
        open_.setdefault(e["nspid"], e["t"])
    elif e["k"] == "resume" and e["nspid"] in open_:
        win.append((open_.pop(e["nspid"]), e["t"], rank_of.get(int(e["nspid"]), "?")))
nw = [(e["t"], int(e["us"]), rank_of.get(int(e["nspid"]), "?")) for e in ev if e["k"] == "numa_work"]
loops = []
for f in files:
    r = json.load(open(f))
    for lp in ("sync", "queued", "sync1"):
        if f"{lp}_t" in r:
            loops.append((r[f"{lp}_t"][0], r[f"{lp}_t"][1], f.split("/")[-1], lp, r))
t_in = lambda t: any(a <= t <= b for a, b, *_ in loops)
wt = [w for w in win if t_in(w[0])]
d = sorted((b - a) / 1000 for a, b, _ in wt)
print(f"quiesce->resume windows: {len(win)} total, {len(wt)} inside timed loops; by rank {dict(collections.Counter(w[2] for w in wt))}")
if d:
    print(f"  window us: min {d[0]:.0f} p50 {d[len(d)//2]:.0f} p90 {d[int(.9*len(d))]:.0f} max {d[-1]:.0f}")
print(f"numa_work >50us inside loops: {sum(t_in(t) for t, _, _ in nw)}")
pre = [min([(w[0] - t) / 1000 for t, _, rk in nw if rk == w[2] and 0 <= w[0] - t < 50e6] or [None]) if any(rk == w[2] and 0 <= w[0] - t < 50e6 for t, _, rk in nw) else None for w in wt]
print(f"windows starting <=50 ms after the same rank's task_numa_work: {sum(p is not None for p in pre)} of {len(wt)}")
spk_hit = spk_all = 0
used = set()
rows = []
for a, b, f, lp, r in loops:
    if lp == "queued":
        Q = r["queued"]; qt = [queued_times(Q[k]) for k in range(4)]
        for ci in range(len(qt[0])):
            for j in range(1, len(qt[0][ci])):
                us = [qt[k][ci][j][0] for k in range(4)]
                if max(us) < 1000:
                    continue
                s = min(qt[k][ci][j][1] - int(qt[k][ci][j][0] * 1000) for k in range(4))
                e_ = max(qt[k][ci][j][1] for k in range(4))
                orig = max(range(4), key=lambda k: qt[k][ci][j][1])
                ov = [i for i, w in enumerate(win) if w[0] < e_ and w[1] > s - 200_000]
                spk_all += 1; spk_hit += bool(ov); used.update(ov)
                rows.append((f, lp, round(max(us)), [(win[i][2], round((win[i][1] - win[i][0]) / 1000)) for i in ov]))
    else:
        S = r[lp]; reps = r["reps"] if lp == "sync" else 1
        for i in range(len(S[0])):
            us = [S[k][i][0] / reps for k in range(4)]
            if max(us) < 150 or max(S[k][i][0] for k in range(4)) < 1000:
                continue
            s = min(S[k][i][3] for k in range(4)); e_ = max(S[k][i][5] for k in range(4))
            ov = [j for j, w in enumerate(win) if w[0] < e_ and w[1] > s - 200_000]
            spk_all += 1; spk_hit += bool(ov); used.update(ov)
            rows.append((f, lp, round(max(S[k][i][0] for k in range(4))), [(win[j][2], round((win[j][1] - win[j][0]) / 1000)) for j in ov]))
print(f"slow replays (>1 ms extra) overlapping a quiesce->resume window: {spk_hit} of {spk_all}")
inwin = [i for i, w in enumerate(win) if t_in(w[0])]
print(f"windows inside loops that coincide with a slow replay: {len(set(inwin) & used)} of {len(inwin)}")
for x in rows[:15]:
    print("  ", x)
