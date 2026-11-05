"""Serving arm analysis: per-step chunk gaps (serve_client.py) vs KFD quiesce -> resume windows of the schedulers.

  python3 serve_an.py <arm> [<arm> ...]
"""

import json
import re
import sys

D = "/data/muse455/k2/cand/k2v2_poll"


def pct(q, p):
    return q[min(len(q) - 1, int(p * len(q)))]


for arm in sys.argv[1:]:
    reqs = json.load(open(f"{D}/s3/serve/client_{arm}.json"))
    sched = {}
    for line in open(f"{D}/logs/server_{arm}.log", errors="replace"):
        m = re.search(r"TP(\d) EP\d\] Process (\d+) gpu_id", line)
        if m:
            sched[m.group(2)] = int(m.group(1))
    win, open_ = [], {}
    for line in open(f"{D}/s3/serve/trace_{arm}.txt", errors="replace"):
        p = line.split()
        if len(p) < 3 or not p[0].isdigit():
            continue
        kv = dict(x.split("=", 1) for x in p[2:] if "=" in x)
        if p[1] == "quiesce":
            open_.setdefault(kv["nspid"], int(p[0]))
        elif p[1] == "resume" and kv["nspid"] in open_:
            win.append((open_.pop(kv["nspid"]), int(p[0]), sched.get(kv["nspid"], kv["nspid"])))
    gaps = []
    tok = []
    t_lo = t_hi = None
    for r in reqs:
        if r["warmup"]:
            continue
        ch = r["chunks"]
        t_lo = ch[0][0] if t_lo is None else t_lo
        t_hi = ch[-1][0]
        for (a, ta), (b, tb) in zip(ch[1:], ch[2:]):
            gaps.append((b - a, a, b))
            if ta is not None and tb is not None:
                tok.append(tb - ta)
    q = sorted(g[0] / 1e6 for g in gaps)
    T = (t_hi - t_lo) / 1e9
    wi = [w for w in win if t_lo <= w[0] <= t_hi]
    slow = [g for g in gaps if g[0] > 15e6]
    hit = sum(any(w[0] < g[2] and w[1] > g[1] for w in wi) for g in slow)
    gw = sum(any(w[0] < g[2] and w[1] > g[1] for w in wi) for g in gaps)
    d = sorted((w[1] - w[0]) / 1e6 for w in wi)
    print(f"== {arm}: {len(gaps)} steps in {T:.0f} s, tokens/step mean {sum(tok) / max(len(tok), 1):.2f}")
    print(f"   step ms p50 {pct(q, .5):.2f} p90 {pct(q, .9):.2f} p99 {pct(q, .99):.2f} p99.9 {pct(q, .999):.2f} "
          f"max {q[-1]:.1f}; >12 ms {sum(x > 12 for x in q)}, >15 ms {sum(x > 15 for x in q)}, "
          f">20 ms {sum(x > 20 for x in q)}, >50 ms {sum(x > 50 for x in q)}")
    print(f"   quiesce windows during decode: {len(wi)} ({len(wi) / T:.2f}/s), by TP rank "
          f"{ {k: sum(w[2] == k for w in wi) for k in sorted({w[2] for w in wi}, key=str)} }"
          + (f", ms p50 {pct(d, .5):.1f} mean {sum(d) / len(d):.1f} max {d[-1]:.1f}" if d else ""))
    print(f"   steps overlapping a window: {gw} ({100 * gw / len(gaps):.2f}%); steps > 15 ms overlapping a window: "
          f"{hit} of {len(slow)}")
    print(f"   mean step {sum(q) / len(q):.3f} ms; mean of steps not overlapping a window "
          f"{sum(g[0] for g in gaps if not any(w[0] < g[2] and w[1] > g[1] for w in wi)) / max(len(gaps) - gw, 1) / 1e6:.3f} ms")
