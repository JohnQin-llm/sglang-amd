"""Single-stream decode load (C1 shape: one request at a time) with per-chunk arrival times.

Streams /generate (stream_interval 1: one chunk per decode step), ignore_eos, back to back for --seconds.
Records CLOCK_MONOTONIC ns of every chunk and its cumulative completion_tokens. Output JSON: list of requests.
  python3 serve_client.py --out x.json --seconds 180
"""

import argparse
import json
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--url", default="http://127.0.0.1:30000/generate")
    a = ap.parse_args()
    prompt = ("Write a detailed, step by step technical explanation of how a persistent GPU kernel can hand data "
              "between four GPUs through mailboxes with epoch tags, and what can go wrong. ") * 8
    sess = requests.Session()
    sess.trust_env = False
    out = []
    end = time.monotonic() + a.seconds
    first = True
    while time.monotonic() < end:
        body = {"text": prompt, "stream": True,
                "sampling_params": {"max_new_tokens": 64 if first else a.max_new, "temperature": 0.0,
                                    "ignore_eos": True}}
        t0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        chunks = []
        with sess.post(a.url, json=body, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                t = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                m = json.loads(data).get("meta_info", {})
                chunks.append((t, m.get("completion_tokens")))
        out.append({"t0": t0, "warmup": first, "chunks": chunks})
        first = False
    json.dump(out, open(a.out, "w"))
    n = sum(len(x["chunks"]) for x in out if not x["warmup"])
    print(f"{len(out)} requests, {n} chunks -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
