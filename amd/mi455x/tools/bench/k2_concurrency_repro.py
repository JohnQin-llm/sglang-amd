"""Reproduce K2's c>1 failure: request A decodes alone (single-request K2 steps); request B arrives mid-decode
(its prefill lands next to A's K2 verify steps). Prints per-request timings; a K2 give-up shows as a stall."""
import json, sys, threading, time, urllib.request, random

URL = "http://127.0.0.1:30000/generate"


def gen(ctx, new, out, key, seed):
    rnd = random.Random(seed)
    ids = [rnd.randrange(1000, 150000) for _ in range(ctx)]
    body = json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": new, "temperature": 0,
                                                             "ignore_eos": True}}).encode()
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=600)
    json.loads(r.read())
    out[key] = round(time.time() - t0, 2)


for trial, delay in enumerate([0.5, 1.0, 1.5, 2.0, 0.25]):
    out = {}
    a = threading.Thread(target=gen, args=(1000, 2048, out, "A_1k_2048tok", trial))
    b = threading.Thread(target=gen, args=(8000, 128, out, "B_8k_128tok", 100 + trial))
    a.start(); time.sleep(delay); b.start(); a.join(); b.join()
    print(json.dumps({"trial": trial, "B_delay_s": delay, **out}), flush=True)
