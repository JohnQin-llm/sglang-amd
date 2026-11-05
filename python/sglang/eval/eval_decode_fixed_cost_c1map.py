"""Map the MI455X C1 batch-size-1 kernel breakdown onto the C16 fixed decode step (no GPU needed).

Inputs (read-only):
  --c1-stock  handoff knowledge/profile/stock_128k.json  (C1 stock tree, gfx1250, rank 0, draft+verify cycle, batch 1, 128K)
  --c1-kept   handoff knowledge/profile/now_128k.json    (C1 kept tree, same shape)
  --c16-steps campaign/c16/probes/decode_steps/iter11.json (served step by batch size from the bench log)
Attention at batch size 1 on the C16 tree comes from ledger c16-verify-attn-graph-floor / c16-verify-swa-split
(graph replay): full attention 8 us (short) .. 74 us (batch 3, 90K); sliding window 7.5 us. The C1 stock attention
row is replaced by that range because the C16 tree has the 3D split-KV verify kernels and C1 stock did not.

Caveat: the C1 trace was taken on the MI450 day-0 image (ROCm 10.2), this campaign runs the MI45x image (ROCm 10.1);
the rows are an estimate of the C16 batch-1 step, not a measurement of it.

    python3 eval_decode_fixed_cost_c1map.py --c1-stock S --c1-kept K --c16-steps J --out OUT.json
"""
import argparse
import json


def rows(path):
    d = json.load(open(path))
    return d["cycle_ms"], {r["name"]: r["ms"] for r in d["rows"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-stock", required=True)
    ap.add_argument("--c1-kept", required=True)
    ap.add_argument("--c16-steps", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cyc_s, stock = rows(a.c1_stock)
    cyc_k, kept = rows(a.c1_kept)
    step1 = json.load(open(a.c16_steps))["1"]["step_ms_median"]

    attn = [k for k in stock if k.startswith("attention")]
    non_attn = {k: v for k, v in stock.items() if k not in attn}
    gpu_non_attn = sum(non_attn.values())
    # C16 tree attention at batch size 1 (graph replay), ms per verify step
    full_lo, full_hi, swa = 16 * 0.008, 16 * 0.074, 46 * 0.0075
    attn_lo, attn_hi = full_lo + swa, full_hi + swa
    est_lo, est_hi = gpu_non_attn + attn_lo, gpu_non_attn + attn_hi

    cats = {}
    for k, v in non_attn.items():
        k_kept = kept.get(k, 0.0)
        if k == "MoE expert GEMMs":
            k_kept += kept.get("MoE mono kernel (K2)", 0.0)
        cats[k] = {
            "c1_stock_ms": round(v, 3),
            "share_of_c16_step1_pct": round(100 * v / step1, 1),
            "c1_kept_ms": round(k_kept, 3),
            "c1_saving_ms": round(v - k_kept, 3),
            "c1_saving_pct_of_c16_step1": round(100 * (v - k_kept) / step1, 1),
        }
    out = {
        "c16_step_bs1_ms": step1,
        "c1_stock_cycle_ms": round(cyc_s, 2),
        "c1_kept_cycle_ms": round(cyc_k, 2),
        "c1_stock_gpu_non_attention_ms": round(gpu_non_attn, 2),
        "c16_attention_bs1_ms_range": [round(attn_lo, 2), round(attn_hi, 2)],
        "c16_gpu_estimate_ms_range": [round(est_lo, 2), round(est_hi, 2)],
        "unexplained_host_or_image_ms_range": [round(step1 - est_hi, 2), round(step1 - est_lo, 2)],
        "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1]["c1_stock_ms"])),
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
