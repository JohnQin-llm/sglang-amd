"""Decode speed of one request vs context length on a running server (batch 1, streaming /generate)."""
import json, random, sys, time, urllib.request

URL = "http://127.0.0.1:30000/generate"
random.seed(0)

def run(ctx, new=256):
    ids = [random.randrange(1000, 150000) for _ in range(ctx)]
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"max_new_tokens": new, "temperature": 0, "ignore_eos": True}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            n = d["meta_info"]["completion_tokens"]
            if t_first is None:
                t_first = time.time()
    t1 = time.time()
    decode_s = t1 - t_first
    print(json.dumps({"context": ctx, "ttft_s": round(t_first - t0, 2), "new_tokens": n,
                      "decode_tok_per_s": round((n - 1) / decode_s, 1), "ms_per_token": round(1000 * decode_s / (n - 1), 2)}),
          flush=True)

for ctx in map(int, sys.argv[1:]):
    run(ctx)
