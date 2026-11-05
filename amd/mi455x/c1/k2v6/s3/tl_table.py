"""K2 stage table from probe_b2b_tl.py raw dumps (K2_TL_RAW, optionally K2_TL_PAIR=1).

Per stage: start / end in the global frame (each rank's XCD-aligned clock shifted by its FIN all-gather offset,
see tl_an.offsets; ~±1 us between ranks), the stage's own time (median per-CTA duration, same-CTA stamps: no
offset error), the wait for its input (median per-CTA poll time), and the floor (bytes / 16.5 TB/s or the hop).

    python3 tl_table.py base_L1.json base_L31.json base_L60.json
"""
import json
import statistics as st
import sys

from tl_an import aligned, events, offsets

M = 1 << 31
BW = 16.5e6  # bytes / us
HOP = 2.05

# name, (own-time stamps a->b), (wait stamps a->b), CTA set, floor us, floor note
STAGES = [
    ("o_proj (33.5 MB)", (0, 4), None, range(16, 256), 33.5e6 / BW, "HBM"),
    ("AR#1 hop -> seam (256 KB/peer)", None, (2, 22), range(0, 8), HOP, "1 XGMI hop"),
    ("seam (2 norms)", (22, 44), None, range(0, 8), 0.5, "latency"),
    ("x -> pool (device hop)", None, (4, 27), range(16, 256), 1.0, "device hop"),
    ("pool GEMV (16.8 MB, prefetched)", (27, 35), None, range(16, 256), 16.8e6 / BW, "HBM"),
    ("router (8.4 MB) -> RLOGL", (35, 42), None, range(16, 256), 8.4e6 / BW, "HBM"),
    ("route: RLOG wait", None, (3, 31), range(96, 104), 1.0, "device hop"),
    ("route compute", (31, 45), None, range(96, 104), 0.5, "latency"),
    ("POOLP all-gather -> xq (256 KB/peer)", None, (5, 46), range(8, 16), HOP, "1 XGMI hop"),
    ("xq (norm + MXFP8)", (46, 43), None, range(8, 16), 0.3, "latency"),
    ("route table + UG w issue", (50, 49), None, range(16, 256), 0.3, "latency"),
    ("XQRDY wait (+acquire)", None, (49, 7), range(16, 256), 1.0, "device hop"),
    ("UG (183 MB)", (7, 8), None, range(256), 183e6 / BW, "HBM"),
    ("UG->down acquire", (8, 24), None, range(256), 0.3, "latency"),
    ("down (91 MB)", (24, 28), None, [b for b in range(256) if not 8 <= b < 16], 91e6 / BW, "HBM"),
    ("YR hop -> finalize (256 KB/peer)", None, (9, 36), range(192, 256), HOP, "1 XGMI hop"),
    ("FMSQ (row's 32 slices)", None, (36, 37), range(192, 256), 1.0, "device hop"),
    ("finalize out", (37, 38), None, range(192, 256), 0.3, "latency"),
    ("FIN all-gather -> unpool gate", None, (10, 29), range(256), HOP, "1 XGMI hop"),
    ("FIN -> LDS (128 KB pairs a CTA)", (29, 39), None, range(256), 0.5, "latency"),
    ("unpool GEMV (64 MB, prefetched)", (39, 40), None, range(256), 0.5, "compute"),
]


def dur(T, cs, a, b):
    v = [((T[c][b] - T[c][a]) % M) / 100 for c in cs if T[c][a] and T[c][b]]
    return (st.median(v), max(v)) if v else None


def main():
    for path in sys.argv[1:]:
        d = json.load(open(path))
        raws = d["raw"]
        evs = [events(aligned(T)) for T in raws]
        off = offsets(evs)
        print(f"== layer {d['layer']} ({path}); rank offsets {[round(o, 1) for o in off]}")
        print(f"{'stage':40s} {'own med/max':>12s} {'wait med/max':>13s} {'floor':>6s}  per rank own|wait medians")
        for name, own, wait, cs, fl, note in STAGES:
            o = [dur(T, cs, *own) for T in raws] if own else None
            w = [dur(T, cs, *wait) for T in raws] if wait else None
            om = f"{max(x[0] for x in o):5.1f}/{max(x[1] for x in o):5.1f}" if o else ""
            wm = f"{max(x[0] for x in w):5.1f}/{max(x[1] for x in w):5.1f}" if w else ""
            pr = " ".join(f"{(o or w)[r][0]:4.1f}" for r in range(4))
            print(f"{name:40s} {om:>12s} {wm:>13s} {fl:6.1f}  {pr}  ({note})")
        ends = [(evs[r]["end"][2] - off[r]) for r in range(4)]
        starts = [(evs[r]["start"][0] - off[r]) for r in range(4)]
        print(f"  launch start (global, first CTA) per rank {[round(x, 1) for x in starts]}; end (last CTA) "
              f"{[round(x, 1) for x in ends]}; span max {max(ends) - min(starts):.1f}")
        if d.get("raw_prev") and any(any(r) for r in d["raw_prev"][0]):
            gaps = []
            for r in range(4):
                P, T = d["raw_prev"][r], raws[r]
                g = []
                for x in range(8):
                    pe = max(P[b][13] for b in range(x, 256, 8))
                    ns = min(T[b][0] for b in range(x, 256, 8))
                    g.append(((ns - pe) % M) / 100)
                gaps.append(round(st.median(g), 1))
            print(f"  launch boundary (prev launch last CTA end -> next launch first CTA start, same XCD clock), "
                  f"per rank: {gaps} us")


if __name__ == "__main__":
    main()
