"""Analysis of spike.py output: distribution table, cross-rank structure of slow iterations, host stamps.

  python3 spike_an.py <tag> [--detail]
"""

import glob
import json
import os
import sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spike")


def pct(q, p):
    return q[min(len(q) - 1, int(p * len(q)))]


def stats(x):
    q = sorted(x)
    return (len(q), pct(q, 0.5), pct(q, 0.99), pct(q, 0.999), q[-1], sum(v > 150 for v in q), sum(v > 1000 for v in q))


def per_iter(res, loop):
    if loop == "sync":
        return [[r[0] / res["reps"] for r in rr] for rr in res["sync"]]
    if loop == "sync1":
        return [[r[0] for r in rr] for rr in res["sync1"]]
    return [[x for c in rr["us"] for x in c[1:]] for rr in res["queued"]]


def queued_times(rr):
    """(us, est. end time ns) of every queued call: chunk end = sync return, walked backwards."""
    out = []
    for c, (t0, tq, ts) in zip(rr["us"], rr["chunks"]):
        end = ts
        rows = []
        for us in reversed(c):
            rows.append((us, end))
            end -= int(us * 1000)
        out.append(list(reversed(rows)))
    return out


def main():
    tag = sys.argv[1]
    detail = "--detail" in sys.argv
    files = sorted(glob.glob(os.path.join(D, f"{tag}_L*_*.json")), key=lambda f: (int(f.split("_L")[-1].split("_")[0]), f))
    print(f"| case | loop | rank | iters | p50 | p99 | p99.9 | max | >150 us | >1 ms |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    summary = {}
    for f in files:
        res = json.load(open(f))
        case = f"L{res['layer']} {res['mode']}"
        for loop in ("sync", "queued", "sync1"):
            if loop not in res:
                continue
            per = per_iter(res, loop)
            for r, x in enumerate(per):
                n, p50, p99, p999, mx, c150, c1m = stats(x)
                print(f"| {case} | {loop} | {r} | {n} | {p50:.1f} | {p99:.1f} | {p999:.1f} | {mx:.0f} | {c150} | {c1m} |")
            summary[(case, loop)] = per
    if not detail:
        return
    for f in files:
        res = json.load(open(f))
        case = f"L{res['layer']} {res['mode']}"
        if "sync" in res:
            S = res["sync"]
            n = len(S[0])
            print(f"\n== {case} sync: slow iterations (per-layer us > 150 on any rank)")
            for i in range(n):
                us = [S[r][i][0] / res["reps"] for r in range(4)]
                if max(us) <= 150:
                    continue
                tb1 = [S[r][i][2] for r in range(4)]
                te = [S[r][i][3] for r in range(4)]
                tr = [S[r][i][4] for r in range(4)]
                m = min(tb1)
                print(f"  it {i}: us/layer {[round(u) for u in us]} replay_ms {[round(S[r][i][0] / 1000, 2) for r in range(4)]} "
                      f"bar_exit+us {[round((t - m) / 1000) for t in tb1]} e0rec+us {[round((t - m) / 1000) for t in te]} "
                      f"replay_ret+us {[round((t - m) / 1000) for t in tr]} launch_us {[round((tr[r] - te[r]) / 1000) for r in range(4)]} "
                      f"minflt {[S[r][i][6] for r in range(4)]} nivcsw {[S[r][i][9] for r in range(4)]} nvcsw {[S[r][i][8] for r in range(4)]}")
        if "queued" in res:
            Q = res["queued"]
            print(f"\n== {case} queued: chunks (enqueue ms vs GPU ms) and slow calls")
            for r in range(4):
                enq = [(c[1] - c[0]) / 1e6 for c in Q[r]["chunks"]]
                gpu = [sum(u) / 1000 for u in Q[r]["us"]]
                print(f"  rank {r}: enqueue ms {[round(e, 1) for e in enq]}  gpu ms {[round(g, 1) for g in gpu]}")
            qt = [queued_times(Q[r]) for r in range(4)]
            for ci in range(len(qt[0])):
                for j in range(len(qt[0][ci])):
                    us = [qt[r][ci][j][0] for r in range(4)]
                    if max(us) > 150:
                        ends = [qt[r][ci][j][1] for r in range(4)]
                        m = min(ends)
                        print(f"  chunk {ci} call {j}: us {[round(u) for u in us]} end+us {[round((e - m) / 1000) for e in ends]} "
                              f"t_end {m}")


if __name__ == "__main__":
    main()
