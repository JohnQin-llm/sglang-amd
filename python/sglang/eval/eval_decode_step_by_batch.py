"""Decode step time by batch size from an SGLang server log (bench window only).

step_ms = running_requests * accept_len / gen_throughput * 1000, taken only from
decode log lines whose interval since the previous TP0 log line holds no prefill
batch (so prefill time does not leak into the decode step).

usage: python3 eval_decode_step_by_batch.py SERVER_LOG [WINDOW_S=600] [OUT_JSON]
"""
import datetime as dt
import json
import re
import statistics
import sys


def ts(line):
    return dt.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp()


def num(line, key):
    m = re.search(re.escape(key) + r": ([\d.]+)", line)
    return float(m.group(1)) if m else None


def main():
    path = sys.argv[1]
    window = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    lines = [l for l in open(path, errors="ignore") if "TP0" in l and ("Decode batch" in l or "Prefill batch" in l)]
    end = ts(lines[-1])
    lines = [l for l in lines if ts(l) >= end - window]
    by_bs, prefill_seen = {}, False
    for l in lines:
        if "Prefill batch" in l:
            prefill_seen = True
            continue
        clean = not prefill_seen
        prefill_seen = False
        bs, acc, gt = num(l, "#running-req"), num(l, "accept len"), num(l, "gen throughput (token/s)")
        ctx = num(l, "#full token")
        if not clean or not gt or not bs:
            continue
        by_bs.setdefault(int(bs), []).append((bs * acc / gt * 1000.0, ctx / bs))
    out = {}
    for bs in sorted(by_bs):
        v = by_bs[bs]
        step = statistics.median(x[0] for x in v)
        out[bs] = {
            "lines": len(v),
            "step_ms_median": round(step, 2),
            "per_user_tok_s": round(4.05 / step * 1000.0, 1),
            "mean_ctx_per_req": int(statistics.mean(x[1] for x in v)),
        }
        print(bs, out[bs])
    if len(sys.argv) > 3:
        json.dump(out, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
