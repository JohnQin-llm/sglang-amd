"""Summaries of the serve_stress runs in out/: give-ups per run, cross-tree bitwise checks, the sticky window.

    compare.py [reference tag, default v4b_nompol]
Every run uses seed 0, so step s of every run picks the same case on every layer except at injected steps.
"""
import json
import os
import sys

import torch

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
REF = sys.argv[1] if len(sys.argv) > 1 else "v4b_nb0"


def load(tag):
    j = os.path.join(D, tag + ".json")
    if not os.path.exists(j):
        return None
    r = json.load(open(j))
    p = os.path.join(D, tag + "_dig.pt")
    r["dig"] = torch.load(p) if os.path.exists(p) else None
    return r


tags = sorted(f[:-5] for f in os.listdir(D) if f.endswith(".json"))
runs = {t: load(t) for t in tags}
print("| run | sweep launches | sweep give-ups (nu) | serve calls | serve give-up steps | injected nu63 steps | "
      "step p50 / p99.9 / max us | steps > +1 ms | digest |")
print("|---|---|---|---|---|---|---|---|---|")
for t, r in runs.items():
    sw = r.get("sweep", [])
    swg = [f"L{x['layer']}:{x['nu']}" for x in sw if x["gave_up"]]
    sv = (r.get("serve") or [None])[0]
    if sv:
        line = (f"{sv['calls']} | {len(sv['failed_steps'])} | {len(sv['injected'])} | "
                f"{sv['p50_us']} / {sv['p999_us']} / {sv['max_us']} | {sv['steps_over_+1ms']} | "
                f"{sv['digest_ok_steps']}")
    else:
        line = "- | - | - | - | - | -"
    print(f"| {t} | {len(sw)} | {len(swg)} ({', '.join(swg)}) | {line} |")

# give-up records
for t, r in runs.items():
    for ev in r.get("giveups", []):
        regs = sorted({x for a in ev["ranks"] for x in a["regions"]})
        lays = sorted({x for a in ev["ranks"] for x in a["layers"]})
        n = [a["n_ctas"] for a in ev["ranks"]]
        print(f"{t}: {ev['where']}: flags {ev['flags']}, CTAs a rank {n}, regions {regs}, layers {lays}; "
              f"rank0 first: {ev['ranks'][0]['first'][:2]}")


def mask(r):
    sv = r["serve"][0]
    m = torch.ones(sv["steps"], dtype=torch.bool)
    for s in sv["failed_steps"]:
        m[s] = False
    for s in sv["injected"]:
        m[int(s)] = False
    if sv.get("sticky_steps"):
        a, b = sv["sticky_steps"]
        m[a:b + 1] = False
    return m


ref = runs.get(REF)
if ref and ref.get("dig") is not None:
    print(f"\nbitwise vs {REF} (steps valid in both runs; (step, layer) outputs, stream_out + branch digests):")
    for t, r in runs.items():
        if t == REF or r.get("dig") is None:
            continue
        n = min(r["dig"].shape[0], ref["dig"].shape[0])
        m = mask(r)[:n] & mask(ref)[:n]
        a, b = r["dig"][:n][m], ref["dig"][:n][m]
        bad = int((a != b).any(-1).sum())
        print(f"  {t}: {int(m.sum())} steps x {a.shape[1]} layers compared, {bad} differ")
        sv = r["serve"][0]
        if sv.get("sticky_steps"):
            lo, hi = sv["sticky_steps"]
            hi = min(hi, n - 1)
            mm = mask(ref)[lo:hi + 1]
            a, b = r["dig"][lo:hi + 1][mm], ref["dig"][lo:hi + 1][mm]
            bad = (a != b).any(-1)
            print(f"  {t} sticky window (FAILED left set, steps {lo}..{hi}): {int(bad.sum())} of {bad.numel()} "
                  f"(step, layer) outputs differ from {REF} ({100.0 * float(bad.float().mean()):.1f} %); "
                  f"steps with any wrong layer {int(bad.any(-1).sum())} of {bad.shape[0]}; "
                  f"sticky step p50 {sv.get('sticky_p50_us')} us vs normal {sv['p50_us']} us")

# sweep digests per (layer, case) vs the reference's
if ref and ref.get("sweep"):
    rd = {(x["layer"], x["case"]): x for x in ref["sweep"]}
    print(f"\nsweep digests vs {REF} (eager, one launch per case):")
    for t, r in runs.items():
        if t == REF or not r.get("sweep"):
            continue
        diff = [(x["layer"], x["nu"]) for x in r["sweep"]
                if (x["layer"], x["case"]) in rd and x["digest"] != rd[(x["layer"], x["case"])]["digest"]]
        n = sum((x["layer"], x["case"]) in rd for x in r["sweep"])
        print(f"  {t}: {n} cases compared, {len(diff)} differ {diff}")
