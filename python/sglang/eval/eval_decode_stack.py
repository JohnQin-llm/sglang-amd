"""What is left of the decode stack (m-decode-stack) on the kept tree, measured at the served shapes.

Run every microbenchmark through the GPU lock wrapper, e.g.
    /data/muse455/k2/k2safe.sh /sgl-workspace/sglang/python/sglang/eval/decode_stack qkv_fullop_bench.py --out X.json

Components and results (the JSON files beside the scripts):
- decode_stack/trace_breakdown.py: the kept tree's draft + verify cycle at 128K context, by kernel category
  (breakdown_iter9_128k.json): 9.05 ms; MoE mono kernel 4.60 ms, attention 1.07, draft 0.85, norms / RoPE / QKV
  epilogue 0.74, dense GEMMs 0.72, GPU idle 0.61.
- decode_stack/qkv_bench.py, qkv_bench_gluon.py: the fused QKV + gate projection (8 x 4608 x 8192 BF16, 62 cold
  weights in one HIP graph): served Triton tile 9.55 us, best Triton tile 6.68 us, AITER gfx1250 TDM split-K
  6.44 us (11.7 TB/s); both kernel families saturate near 11.5 TB/s at a 75.5 MB launch.
- decode_stack/qkv_fullop_bench.py: the whole served op (projection + QK-norm / RoPE / FP8 KV-cache epilogue):
  13.12 -> 10.09 us per layer with the TDM GEMM, bit-identical outputs (qkv_fullop.json).
- decode_stack/norm_bench.py: the fused post-norm + residual-gate + pre-norm launch sweep at 8 rows: 2.76 us at
  the served 8 warps, 2.60 us at 16, bit-identical (norm_bench.json).
"""
import json
import os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decode_stack")


def summary():
    q = json.load(open(os.path.join(HERE, "qkv_fullop.json")))
    n = json.load(open(os.path.join(HERE, "norm_bench.json")))
    b = json.load(open(os.path.join(HERE, "breakdown_iter9_128k.json")))
    saved_qkv = (q["served"]["us"] - q["gluon_tdm_sk8"]["us"]) * 62 / 1000
    saved_norm = (n["w8_s1"]["us"] - n["w16_s1"]["us"]) * 64 / 1000
    return {"cycle_ms": b["cycle_ms"], "qkv_saved_ms": saved_qkv, "norm_saved_ms": saved_norm,
            "increment_ceiling_pct": 100 * (saved_qkv + saved_norm) / b["cycle_ms"]}


if __name__ == "__main__":
    print(json.dumps(summary(), indent=1))
