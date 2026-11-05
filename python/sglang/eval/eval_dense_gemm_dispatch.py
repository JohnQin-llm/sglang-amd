"""CPU-only map of which kernel serves each dense BF16 decode GEMM on gfx1250.

No GPU, no torch, no aiter import (safe while the GPUs are wedged): it reads the
dispatch tables the served code consults and reports, per (N, K, M), the path.

Paths, as read from the source (sglang rocm_muse_spark/model.py _dense_linear,
srt/layers/quantization/unquant.py apply, aiter tuned_gemm.get_GEMM_A16W16_config):
  1. tiled : pre/post-expert projections, 17..64 rows, (N, K, bm) in the MI355X LAUNCH table
  2. csv   : aiter bf16_tuned_gemm.csv row for (gfx1250, cu_num, M, N, K)
  3. gluon : gfx1250 fallback = aiter gluon gemm_a16w16; config from
             GEMM-A16W16-N=..-K=...json if present, else DEFAULT.json (M_LEQ_* buckets)

Usage (inside the container): python3 -I eval_dense_gemm_dispatch.py --out OUT.json
"""

import argparse
import ast
import csv
import json
import os
import re

SGL = "/sgl-workspace/sglang/python/sglang"
AITER = "/sgl-workspace/aiter/aiter"
TILED = f"{SGL}/kernels/ops/gemm/bf16_tiled_small_m_gemm.py"
CSV = f"{AITER}/configs/bf16_tuned_gemm.csv"
GLUON_DIR = f"{AITER}/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16"

# Tensor parallel 4 dense decode shapes (N, K), as named in the C1 MI455X trace.
SHAPES = {
    "qkv_gate": (4608, 8192),
    "o_proj": (8192, 2048),
    "pre_expert_proj": (4096, 8192),
    "post_expert_proj": (8192, 4096),
}
TILED_USERS = {"pre_expert_proj", "post_expert_proj"}
ROWS = [1, 8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128]


def _tiled_table():
    vals = {}
    for node in ast.parse(open(TILED).read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in ("MIN_M", "MAX_M", "LAUNCH"):
                vals[name] = ast.literal_eval(node.value)
    return vals


def _bm(m):
    return max(16, 1 << (m - 1).bit_length())


def _csv_rows():
    rows = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["gfx"] == "gfx1250":
                rows[(int(r["M"]), int(r["N"]), int(r["K"]))] = (
                    r["libtype"], float(r["us"]), int(r["cu_num"]))
    return rows


def _gluon_cfg(n, k):
    p = os.path.join(GLUON_DIR, f"GEMM-A16W16-N={n}-K={k}.json")
    used = p if os.path.exists(p) else os.path.join(GLUON_DIR, "DEFAULT.json")
    return used, json.load(open(used))


def _bucket(cfg, m):
    best = None
    for key in cfg:
        mm = re.match(r"M_LEQ_(\d+)$", key)
        if mm and m <= int(mm.group(1)) and (best is None or int(mm.group(1)) < best[0]):
            best = (int(mm.group(1)), key)
    return best[1] if best else "any"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tiled, csvr = _tiled_table(), _csv_rows()
    out = {"csv_gfx1250_shapes": sorted({(n, k) for (_, n, k) in csvr}), "shapes": {}}
    for name, (n, k) in SHAPES.items():
        used, cfg = _gluon_cfg(n, k)
        per = {}
        for m in ROWS:
            if (name in TILED_USERS and tiled["MIN_M"] <= m <= tiled["MAX_M"]
                    and (n, k, _bm(m)) in tiled["LAUNCH"]):
                per[m] = {"path": "tiled_mi355x_tp8_tuning", "launch": tiled["LAUNCH"][(n, k, _bm(m))]}
            elif (m, n, k) in csvr:
                per[m] = {"path": "aiter_csv", "row": csvr[(m, n, k)]}
            else:
                b = _bucket(cfg, m)
                per[m] = {"path": "gluon_gemm_a16w16_untuned", "config_file": os.path.basename(used),
                          "bucket": b, "config": cfg[b]}
        out["shapes"][name] = {"N": n, "K": k, "rows": per}
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    for name, s in out["shapes"].items():
        paths = {}
        for m, r in s["rows"].items():
            paths.setdefault(r["path"] + "/" + r.get("config_file", ""), []).append(m)
        print(name, s["N"], s["K"], paths)


if __name__ == "__main__":
    main()
