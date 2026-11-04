#!/usr/bin/env bash
# Correctness: 16 arithmetic questions with exact answers, greedy, through the chat endpoint the bench
# uses. Passes at 15 of 16 on a real-acceptance server. Under InferenceX's golden simulated acceptance
# accepted drafts are not verified, so answers degrade by design (the recipe scores 6-8/16); the
# campaign then sets GSM8K_MIN (a fraction, read from the campaign config only) to catch a broken server. Numerics changes must
# also pass 15/16 on a real-acceptance probe.
#
# Arbor injects PORT and MODEL_NAME; writes CORRECTNESS_OUTPUT when set.
set -euo pipefail
: "${PORT:?}" "${MODEL_NAME:?}"
python3 - "$PORT" "$MODEL_NAME" "${CORRECTNESS_OUTPUT:-/dev/null}" <<'PY'
import json, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

port, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
cases = [(a, b) for a, b in ((37, 43), (128, 77), (901, 19), (66, 66), (452, 31), (17, 289), (1234, 9),
                             (73, 88), (5050, 2), (999, 101), (64, 125), (321, 47), (12, 3456), (707, 13),
                             (88, 99), (250, 44))]

def ask(case):
    a, b = case
    body = {"model": model, "temperature": 0, "max_tokens": 4096,
            "messages": [{"role": "user", "content": f"What is {a} * {b}? Reply with the number only."}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        text = json.loads(r.read())["choices"][0]["message"].get("content") or ""
    nums = re.findall(r"-?\d[\d,]*", text)
    got = int(nums[-1].replace(",", "")) if nums else None
    return {"q": f"{a}*{b}", "want": a * b, "got": got, "ok": got == a * b}

with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(ask, cases))
correct = sum(r["ok"] for r in results)
need = round(float(os.environ["GSM8K_MIN"]) * len(cases)) if os.environ.get("GSM8K_MIN") else 15
passed = correct >= need
json.dump({"passed": passed, "correct": correct, "need": need, "total": len(cases), "results": results}, open(out, "w"), indent=1)
print(f"[correctness] {correct}/{len(cases)} (need {need}) {'PASS' if passed else 'FAIL'}")
sys.exit(0 if passed else 1)
PY
