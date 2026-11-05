#!/usr/bin/env python3
"""Price a whole AgentX bench window from two independent sides, to bracket where the time goes.

Usage: python3 -I eval_whole_run_lanes.py --export profile_export.jsonl --log server.log
       [--lanes 16] [--chips 4] [--duration 600] [--interval 20] --out OUT.json

Client side (per-request export, profiling phase only): lane-seconds of the window spent
waiting for the first token (queue + prefill), decoding, and idle at the client (think time,
no request in flight). Little's law turns these into mean requests in each state. Also
recomputes throughput per chip from the export, to check the metric's accounting.

Server side (rank-0 scheduler log over the same window): mean running requests and mean
queue weighted by log-interval time, and decode time spent while requests sat in the queue
(the time the prefill-decode interval makes decode run ahead of admission).

The two sides must agree on mean decoding requests and mean queue; if they do not, the
model of the run is wrong. CPU only.
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
    ap.add_argument("--export", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--lanes", type=int, default=16)
    ap.add_argument("--chips", type=int, default=4)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.export) if l.strip()]
    prof = [r for r in recs if r["metadata"].get("benchmark_phase") != "warmup"]
    t0 = min(r["metadata"]["credit_issued_ns"] for r in prof) / 1e9
    t1 = t0 + a.duration
    lane_ttft = lane_dec = 0.0
    tok_in = tok_out = 0
    n_done = n_cancel = n_open = 0
    lat_ttft, lat_dec = [], []
    for r in prof:
        m = r["metadata"]
        s, e = m["request_start_ns"] / 1e9, m["request_end_ns"] / 1e9
        if s >= t1:
            continue
        mt = r["metrics"]
        ttft = mt.get("time_to_first_token", {}).get("value")
        first = s + ttft / 1e3 if ttft is not None else e
        lane_ttft += max(0.0, min(first, t1) - max(s, t0))
        lane_dec += max(0.0, min(e, t1) - max(first, t0))
        if m.get("was_cancelled"):
            n_cancel += 1
            continue
        if e <= t1:
            n_done += 1
            tok_in += mt.get("input_sequence_length", {}).get("value", 0)
            tok_out += mt.get("output_sequence_length", {}).get("value", 0)
            lat_ttft.append(first - s)
            lat_dec.append(e - first)
        else:
            n_open += 1
    lane_total = a.lanes * a.duration
    med = lambda x: sorted(x)[len(x) // 2] if x else None
    client = {
        "window_start_epoch": t0,
        "completed_in_window": n_done, "unfinished_at_end": n_open, "cancelled": n_cancel,
        "lane_seconds_total": lane_total,
        "lane_seconds_waiting_first_token": round(lane_ttft, 1),
        "lane_seconds_decoding": round(lane_dec, 1),
        "lane_seconds_client_idle": round(lane_total - lane_ttft - lane_dec, 1),
        "mean_waiting_first_token": round(lane_ttft / a.duration, 2),
        "mean_decoding": round(lane_dec / a.duration, 2),
        "mean_in_flight": round((lane_ttft + lane_dec) / a.duration, 2),
        "median_ttft_s": round(med(lat_ttft), 2), "median_decode_s": round(med(lat_dec), 2),
        "tput_per_chip_from_export": round((tok_in + tok_out) / a.duration / a.chips, 1),
        "output_share_of_tokens": round(tok_out / max(1, tok_in + tok_out), 4),
        "mean_input_tokens_per_completed": round(tok_in / max(1, n_done)),
        "mean_output_tokens_per_completed": round(tok_out / max(1, n_done)),
    }

    ev = []
    for line in open(a.log, errors="replace"):
        mm = PAT.match(line)
        if mm:
            t = dt.datetime.strptime(mm.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            ev.append((t, mm.group(2), kv(mm.group(3))))
    # the log is wall-clock local time; shift by whole hours onto the export's epoch
    pre = [e[0] for e in ev if e[1] == "Prefill"]
    best = None
    for h in range(-14, 15):
        o = h * 3600.0
        n = sum(1 for t in pre if t0 <= t + o <= t1)
        if best is None or n > best[0]:
            best = (n, o)
    off = best[1]
    ev = [(t + off, k, f) for t, k, f in ev]
    win = [e for e in ev if t0 <= e[0] <= t1]
    run_w = q_w = dec_q = dec_noq = pre_t = 0.0
    pre_batches = pre_with_running = new_seq = pre_q_left = 0
    for i, (t, kind, f) in enumerate(win):
        nxt = win[i + 1][0] if i + 1 < len(win) else t1
        gap = min(nxt - t, 5.0)
        run, q = f.get("#running-req", 0.0), f.get("#queue-req", 0.0)
        run_w += run * gap
        q_w += q * gap
        if kind == "Decode":
            if q > 0:
                dec_q += gap
            else:
                dec_noq += gap
        else:
            pre_t += gap
            pre_batches += 1
            new_seq += f.get("#new-seq", 0)
            pre_with_running += run > 0
            pre_q_left += q > 0
    server = {
        "log_offset_s": off, "events": len(win),
        "mean_running_time_weighted": round(run_w / a.duration, 2),
        "mean_queue_time_weighted": round(q_w / a.duration, 2),
        "decode_seconds_with_queue_nonempty": round(dec_q, 1),
        "decode_seconds_with_queue_empty": round(dec_noq, 1),
        "prefill_seconds": round(pre_t, 1),
        "prefill_batches": pre_batches, "new_seqs_admitted": int(new_seq),
        "prefill_batches_with_running_decode": pre_with_running,
        "prefill_batches_with_queue_left_after": pre_q_left,
        "admitted_per_second": round(new_seq / a.duration, 3),
        "prefill_decode_interval": a.interval,
    }
    bracket = {
        "client_mean_decoding": client["mean_decoding"],
        "server_mean_running": server["mean_running_time_weighted"],
        "client_mean_waiting_first_token": client["mean_waiting_first_token"],
        "server_mean_queue": server["mean_queue_time_weighted"],
    }
    out = {"client": client, "server": server, "bracket": bracket}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
