#!/usr/bin/env python3
"""Split SGLang prefill chunk time into fixed, per-new-token and prefix-attention parts.

Usage: python3 -I eval_prefill_cost_split.py --log server.log --out OUT.json

Timing source: SGLang's 'input throughput' on a Prefill line is the previous prefill's new tokens
divided by the wall time since the previous prefill line (perf_counter, sub-second). When two
consecutive scheduler lines are both Prefill lines with no running decode, that interval is the
previous chunk's forward alone. Chunk offset (tokens already in the KV before the chunk) is the
logged #cached-token plus, for a chunked continuation (previous chunk was full and no new request
was admitted), the tokens of the earlier chunks of the same request.
Model (least squares): t = a + b*new + c*new*(offset + new/2)/1e6   [s]
  a: fixed per chunk (launch/host/eager), b: per-token GEMM/MoE, c: causal attention over prefix.
Two estimators bracket each other: the fit, and the direct medians of full chunks by offset.
"""
import argparse, json, re
import numpy as np

P = re.compile(r"^\[(\S+ \S+) TP0[^\]]*\] (Prefill|Decode) batch, (.*)$")


def parse(path):
    ev = []
    for line in open(path, errors="replace"):
        m = P.match(line)
        if not m:
            continue
        d = {}
        for p in m.group(3).split(", "):
            if ":" in p:
                k, v = p.split(":", 1)
                try:
                    d[k.strip()] = float(v)
                except ValueError:
                    d[k.strip()] = v.strip()
        ev.append((m.group(1), m.group(2), d))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ev = parse(a.log)
    rows = []
    off = 0.0
    prev = None
    for t, kind, d in ev:
        if kind != "Prefill":
            prev = None
            continue
        cached = d["#cached-token"]
        if prev is not None and prev[2]["#new-token"] >= a.chunk - 1 and d["#new-seq"] == 1 and cached == 0 \
                and d["#queue-req"] <= prev[2]["#queue-req"]:
            off = off + prev[2]["#new-token"]
        else:
            off = cached
        d["_off"] = off
        if prev is not None and prev[2]["#running-req"] == 0 and d["#running-req"] == 0 \
                and d.get("input throughput (token/s)", 0) > 0:
            pn = prev[2]["#new-token"]
            rows.append({"time": prev[0], "new": pn, "off": prev[2]["_off"],
                         "seqs": prev[2]["#new-seq"], "sec": pn / d["input throughput (token/s)"]})
        prev = (t, kind, d)
    rows = [r for r in rows if r["sec"] < 30]
    X = np.array([[1.0, r["new"], r["new"] * (r["off"] + r["new"] / 2) / 1e6] for r in rows])
    y = np.array([r["sec"] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    full = [r for r in rows if r["new"] >= a.chunk - 1]
    bins = {}
    for lo, hi in [(0, 1), (1, 50e3), (50e3, 100e3), (100e3, 150e3), (150e3, 250e3), (250e3, 1e9)]:
        s = [r["sec"] for r in full if lo <= r["off"] < hi]
        if s:
            bins[f"{int(lo)}-{int(hi)}"] = {"n": len(s), "median_s": round(float(np.median(s)), 3),
                                            "tok_per_s": round(a.chunk / float(np.median(s)), 0)}
    small = [r for r in rows if r["new"] < 4096]
    out = {"clean_chunks": len(rows), "fit": {"fixed_s": coef[0], "per_token_us": coef[1] * 1e6,
                                               "attn_s_per_Mtok2": coef[2], "r2": r2},
           "full_chunks_by_offset": bins,
           "small_chunks": [{k: r[k] for k in ("new", "off", "sec")} for r in small],
           "rows": rows}

    def split(new, off):
        parts = [coef[0], coef[1] * new, coef[2] * new * (off + new / 2) / 1e6]
        return {"fixed": round(parts[0], 3), "per_token": round(parts[1], 3), "attention": round(parts[2], 3),
                "total": round(sum(parts), 3)}
    out["examples"] = {f"new{n}_off{o}": split(n, o) for n, o in
                       [(16384, 0), (16384, 100000), (16384, 200000), (2000, 100000), (500, 150000)]}
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("clean_chunks", "fit", "full_chunks_by_offset", "examples")}, indent=1, default=float))
    print("small", out["small_chunks"])


if __name__ == "__main__":
    main()
