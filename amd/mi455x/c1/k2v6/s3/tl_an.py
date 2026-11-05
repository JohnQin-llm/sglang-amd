"""Per-stage K2 timeline from probe_b2b_tl.py raw dumps (K2_TL_RAW): every CTA's stamps aligned to its XCD's
launch start (XCD = bid % 8; the XCD clocks are offset), per rank. Cross-rank offsets: each rank's clock is
aligned so that its FIN all-gather and AR all-gather waits are consistent (see ``offsets``).

    python3 tl_an.py base_L31.json [more.json ...]
"""
import json
import statistics as st
import sys

M = 1 << 31

EV = [  # (label, stamp, CTA set)
    ("start", 0, "all"),
    ("oproj done (pool start)", 4, "all"),
    ("seam: AR seen", 22, "seam"),
    ("seam done", 44, "seam"),
    ("pool: x seen", 27, "all"),
    ("pool done", 35, "all"),
    ("router done", 42, "all"),
    ("route: RLOG seen", 31, "route"),
    ("route done", 45, "route"),
    ("xq: all POOL seen", 46, "xq"),
    ("xq done", 43, "xq"),
    ("route seen (all)", 50, "all"),
    ("UG w issued", 49, "all"),
    ("XQ ready seen", 25, "all"),
    ("UG start", 7, "all"),
    ("UG end (down start)", 8, "all"),
    ("down acquired", 24, "all"),
    ("down1 UGRDY seen", 14, "all"),
    ("down done (YR out)", 28, "nonxq"),
    ("fin start", 9, "fin"),
    ("fin: YR seen", 36, "fin"),
    ("fin: FMSQ seen", 37, "fin"),
    ("fin done", 38, "fin"),
    ("up start", 10, "all"),
    ("up: FIN seen", 29, "all"),
    ("up: FIN in LDS", 39, "all"),
    ("up done", 40, "all"),
    ("end", 13, "all"),
]
SETS = {"all": range(256), "seam": range(0, 8), "route": range(96, 104), "xq": range(8, 16),
        "fin": range(0, 256) if __import__("os").environ.get("TL_FIN") == "all" else range(192, 256), "nonxq": [b for b in range(256) if not 8 <= b < 16]}


def aligned(T):
    base = {x: min(T[b][0] for b in range(x, 256, 8)) for x in range(8)}
    return [[((v - base[b % 8]) % M) / 100 if v else None for v in T[b]] for b in range(256)]


def events(A):
    out = {}
    for lab, k, s in EV:
        v = sorted(A[b][k] for b in SETS[s] if A[b][k] is not None)
        if v:
            out[lab] = (v[0], st.median(v), v[-1])
    return out


def offsets(evs):
    """Rank clock offsets o_r (rank r local = global + o_r, o_0 = 0): the FIN all-gather is symmetric (every rank
    publishes its 2 rows to every rank and gates on all 8), so every rank sees the last FIN at the same global
    time + the same hop: o_r = seen_r - seen_0 for 'up: FIN seen' (min over CTAs)."""
    lab = "seam: AR seen" if __import__("os").environ.get("TL_ALIGN") == "ar" else "up: FIN seen"
    s0 = evs[0][lab][0]
    return [evs[r][lab][0] - s0 for r in range(4)]


def main():
    for path in sys.argv[1:]:
        d = json.load(open(path))
        As = [aligned(T) for T in d["raw"]]
        evs = [events(A) for A in As]
        off = offsets(evs)
        print(f"== {path} layer {d['layer']}: rank clock offsets (us, from the FIN all-gather) "
              f"{[round(o, 1) for o in off]}")
        print(f"{'event':26s} " + " ".join(f"{'r%d min/med/max' % r:>20s}" for r in range(4)) + "   global max")
        for lab, _, _ in EV:
            row, gm = [], -1e9
            for r in range(4):
                e = evs[r].get(lab)
                if e:
                    row.append(f"{e[0] - off[r]:6.1f}/{e[1] - off[r]:6.1f}/{e[2] - off[r]:6.1f}")
                    gm = max(gm, e[2] - off[r])
                else:
                    row.append(" " * 20)
            print(f"{lab:26s} " + " ".join(f"{x:>20s}" for x in row) + f"   {gm:6.1f}")


if __name__ == "__main__":
    main()
