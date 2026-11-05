#!/usr/bin/env python3
"""Split a served bench window into prefill, decode and idle time from the SGLang server log.

Usage: python3 -I eval_server_time_split.py --log server.log --start HH:MM:SS --end HH:MM:SS
       [--date YYYY-MM-DD] [--idle-gap 5] --out OUT.json

Each rank-0 scheduler line is an event. The gap from an event to the next is charged to that
event's kind: a 'Prefill batch' line is logged when its chunk is scheduled, so the gap after
it is that chunk's forward; a 'Decode batch' line closes a run of decode steps, so the gap
after it is decode. Gaps longer than --idle-gap are idle (a prefill keeps up to --idle-gap).
Timestamps have 1 s resolution: single events are rounded, totals over ~1,000 events are
good to a few seconds. Also reports prefill tokens computed and cached, and rate by chunk.
"""
import argparse
import datetime as dt
import json
import re

PAT = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0[^\]]*\] (Prefill|Decode) batch, (.*)$")


def kv(s):
    out = {}
    for part in s.split(", "):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--date")
    ap.add_argument("--idle-gap", type=float, default=5.0)
    ap.add_argument("--decode-log-interval", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ev = []
    for line in open(a.log, errors="replace"):
        m = PAT.match(line)
        if m:
            t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            ev.append((t, m.group(2), kv(m.group(3))))
    date = a.date or ev[-1][0].strftime("%Y-%m-%d")
    t0 = dt.datetime.strptime(f"{date} {a.start}", "%Y-%m-%d %H:%M:%S")
    t1 = dt.datetime.strptime(f"{date} {a.end}", "%Y-%m-%d %H:%M:%S")
    win = [e for e in ev if t0 <= e[0] <= t1]
    span = (t1 - t0).total_seconds()
    charge = {"prefill": 0.0, "decode": 0.0, "idle": (win[0][0] - t0).total_seconds() if win else span}
    pre_new = pre_cached = pre_lines = dec_lines = pre_with_running = 0
    by_chunk = {}
    for i, (t, kind, f) in enumerate(win):
        nxt = win[i + 1][0] if i + 1 < len(win) else t1
        gap = (nxt - t).total_seconds()
        keep = a.idle_gap if kind == "Prefill" else 0.0
        if gap > a.idle_gap:
            charge["idle"] += gap - keep
            gap = keep
        if kind == "Prefill":
            pre_lines += 1
            charge["prefill"] += gap
            n = int(f.get("#new-token", 0))
            pre_new += n
            pre_cached += int(f.get("#cached-token", 0))
            pre_with_running += f.get("#running-req", 0) > 0
            b = "16384" if n >= 16384 else ("4096-16383" if n >= 4096 else ("512-4095" if n >= 512 else "<512"))
            d = by_chunk.setdefault(b, {"lines": 0, "tokens": 0, "seconds": 0.0})
            d["lines"] += 1
            d["tokens"] += n
            d["seconds"] += gap
        else:
            dec_lines += 1
            charge["decode"] += gap
    # Step-based decode time: each decode line closes --decode-log-interval verify steps. The
    # step at a batch size is the median of running * accept_len / gen_throughput over lines
    # whose previous event is also a decode line about 1 s earlier (no prefill in between).
    steps = {}
    for i in range(1, len(win)):
        t, kind, f = win[i]
        pt, pk, _ = win[i - 1]
        if kind == "Decode" and pk == "Decode" and (t - pt).total_seconds() <= 2 and f.get("gen throughput (token/s)", 0) > 0:
            bs = int(f["#running-req"])
            steps.setdefault(bs, []).append(bs * f["accept len"] / f["gen throughput (token/s)"])
    med = {bs: sorted(v)[len(v) // 2] for bs, v in steps.items()}
    dec_model = 0.0
    dec_by_bs = {}
    for t, kind, f in win:
        if kind != "Decode":
            continue
        bs = int(f["#running-req"])
        st = med.get(bs) or med[min(med, key=lambda b: abs(b - bs))]
        dec_model += a.decode_log_interval * st
        dec_by_bs[bs] = dec_by_bs.get(bs, 0.0) + a.decode_log_interval * st
    for d in by_chunk.values():
        d["tok_per_s"] = round(d["tokens"] / d["seconds"], 1) if d["seconds"] else None
    res = {
        "window_s": span,
        "seconds": {k: round(v, 1) for k, v in charge.items()},
        "share": {k: round(v / span, 3) for k, v in charge.items()},
        "prefill_lines": pre_lines,
        "prefill_lines_with_running_decode": int(pre_with_running),
        "decode_lines": dec_lines,
        "prefill_tokens_computed": pre_new,
        "prefill_tokens_cached": pre_cached,
        "prefill_rate_tok_s": round(pre_new / charge["prefill"], 1) if charge["prefill"] else None,
        "prefill_by_chunk": by_chunk,
        "step_ms_by_batch": {bs: round(1000 * v, 1) for bs, v in sorted(med.items())},
        "decode_seconds_step_model": round(dec_model, 1),
        "decode_seconds_by_batch": {bs: round(v, 1) for bs, v in sorted(dec_by_bs.items())},
        "prefill_seconds_residual": round(span - charge["idle"] - dec_model, 1),
        "prefill_rate_residual_tok_s": round(pre_new / max(1e-9, span - charge["idle"] - dec_model), 1),
    }
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
