"""Kernel-lane harness for K2 on MI455X C1: the kept K2 tree (stock) vs a candidate tree, one layer, 8 rows.

Each arm is one `s3/run_k2.py` run of its tree through k2safe.sh (all four GPUs, real layer weights, HIP-graph
timing per rank, numerics against the stock served path). The arms alternate by trial (odd: stock first, even:
candidate first) so drift cancels over trials. Writes the kernel lane's harness format (arbor/measure/kernel_metric.py).

    python3 k2_ab.py --candidate k2tree:NAME --candidate-path /data/muse455/k2/cand/NAME --layer 31 --trial 1 --out X.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

K2SAFE = "/data/muse455/k2/k2safe.sh"
STOCK = "/data/muse455/k2/ep1"
ROWS = 8


def run_arm(tree, layer, reps, iters, timeout):
    res = os.path.join(tree, "s3", "run_k2_a8.json")
    t0 = time.time()
    cmd = [K2SAFE, "-t", str(timeout), tree, "s3/run_k2.py", "--layers", str(layer), "--reps", str(reps),
           "--iters", str(iters)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0 or not os.path.isfile(res) or os.path.getmtime(res) < t0:
        sys.exit(f"[k2_ab] {tree} layer {layer}: run_k2 failed (rc {p.returncode}):\n{log[-1500:]}")
    row = json.load(open(res))[str(layer)]
    ranks = [float(x) for x in row["us_per_layer_median_ranks"]]
    chk = row.get("check") or {}
    routes = chk.get("routes") or {"ok": True}
    checks = {"stream_cos_min": chk.get("stream2_cos_min"), "next_attn_cos_min": chk.get("stream_out_cos_min"),
              "branch_cos_min": chk.get("branch_cos_min"), "routes_equal": bool(routes.get("ok")),
              "pass": bool(chk.get("pass_target")), "replay_matches_first": row.get("replay_matches_first")}
    arm = {"median_us": sorted(ranks)[len(ranks) // 2], "rank_median_us": ranks, "min_us": row.get("us_per_layer_min")}
    return arm, checks, log[-600:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--candidate-path", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stock", default=STOCK)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    arms = {"stock": a.stock, "candidate": a.candidate_path}
    order = ["stock", "candidate"] if a.trial % 2 else ["candidate", "stock"]
    cold, checks, logs = {}, {}, {}
    for name in order:
        cold[name], checks[name], logs[name] = run_arm(arms[name], a.layer, a.reps, a.iters, a.timeout)
        print(f"[k2_ab] L{a.layer} t{a.trial} {name}: {cold[name]['median_us']:.1f} us/layer "
              f"(ranks {[round(x, 1) for x in cold[name]['rank_median_us']]}), checks {checks[name]}", flush=True)
    doc = {"args": {"rows": [ROWS], "layer": a.layer, "trial": a.trial, "candidate": a.candidate,
                    "candidate_path": a.candidate_path, "stock_tree": a.stock, "order": order},
           "rows": {str(ROWS): {"cold": cold, "checks": checks}},
           "checks_per_rank": {"0": {str(ROWS): checks}}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(doc, open(a.out, "w"), indent=1)
    print(f"[k2_ab] wrote {a.out}: speedup {cold['stock']['median_us'] / cold['candidate']['median_us']:.3f}x")


if __name__ == "__main__":
    main()
