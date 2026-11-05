"""CPU-only: which kernel each dense decode GEMM of the served Muse tree reaches, and the
weight bytes each rank reads per decode step (tensor parallel 4). No torch import, no GPU.

usage: python3 -I eval_dense_weight_bytes.py --sglang PKG_DIR --config MODEL_CONFIG --out OUT
"""
import argparse, json, re, os

ap = argparse.ArgumentParser()
ap.add_argument("--sglang", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--tp", type=int, default=4)
ap.add_argument("--hbm_tbps", type=float, default=13.3)  # ledger c16-moe-verify-floor
ap.add_argument("--out", required=True)
a = ap.parse_args()

src = open(os.path.join(a.sglang, "srt/rocm_muse_spark/model.py")).read()
rp = open(os.path.join(a.sglang, "srt/rocm_muse_spark/router_pooler.py")).read()
tiled = open(os.path.join(a.sglang, "kernels/ops/gemm/bf16_tiled_small_m_gemm.py")).read()
cfg = json.load(open(a.config)); t = cfg.get("text_config", cfg)
H, Hm, L = t["hidden_size"], t["moe_hidden_size"], t["num_hidden_layers"]
nq, nkv, hd, V = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"], t["vocab_size"]
moe_layers = sum(1 for x in t.get("mlp_layer_types", ["sparse"] * L) if x == "sparse")

checks = {
    "pre_post_are_nn_linear": bool(re.search(r"self\.pre_expert_proj = \(\s*nn\.Linear", src))
    and bool(re.search(r"self\.post_expert_proj = \(\s*nn\.Linear", src)),
    "aiter_dense_default_off": 'os.environ.get("MUSE_AITER_DENSE", "0")' in src,
    "unpool_fusion_needs_tp8_ep8": "moe.tp_size == 8" in src and "moe.ep_size == 8" in src,
    "shared_router_pooler_needs_no_local_expert_mapping": "op.local_expert_mapping is None" in rp,
}
head = tiled.split("LAUNCH = {")[0]
layout = sorted(set(re.findall(r"\((\d+), (\d+)\): \(\d+, \d+\)", head)))
launch = re.findall(r"\((\d+), (\d+), (\d+)\): \(\d+, \d+\)", tiled)
checks["tiled_layouts"] = layout
checks["tiled_pre_rows"] = sorted(int(m) for n, k, m in launch if (n, k) == ("4096", "8192"))
checks["tiled_post_has_layout"] = ("8192", "4096") in layout

qkv_n = (2 * nq + 2 * nkv) * hd // a.tp  # q, k, v plus output gate, per rank
gb = lambda n, k, layers, b=2: n * k * b * layers / 1e9
rows = {
    "qkv_gate (column parallel)": dict(n=qkv_n, k=H, gb=gb(qkv_n, H, L),
        path="unquant.apply -> tgemm.mm -> gluon gemm_a16w16 DEFAULT.json"),
    "o_proj (row parallel)": dict(n=H, k=nq * hd // a.tp, gb=gb(H, nq * hd // a.tp, L),
        path="unquant.apply -> tgemm.mm -> gluon gemm_a16w16 DEFAULT.json"),
    "pre_expert_proj (replicated on every rank)": dict(n=Hm, k=H, gb=gb(Hm, H, moe_layers),
        path="nn.Linear -> hipBLASLt; Triton tiled (MI355X launch) at 17-64 rows"),
    "post_expert_proj (replicated on every rank)": dict(n=H, k=Hm, gb=gb(H, Hm, moe_layers),
        path="nn.Linear -> hipBLASLt at every row count"),
    "lm_head fp32 (vocab parallel)": dict(n=V // a.tp, k=H, gb=gb(V // a.tp, H, 1, 4), path="not traced"),
}
tot = sum(r["gb"] for r in rows.values())
for r in rows.values():
    r["floor_ms"] = r["gb"] / a.hbm_tbps
rep = sum(r["gb"] for k, r in rows.items() if "replicated" in k)
out = dict(checks=checks, moe_layers=moe_layers, rows=rows, total_gb=tot,
           total_floor_ms=tot / a.hbm_tbps, replicated_gb=rep, replicated_share=rep / tot,
           tp_split_saving_floor_ms=rep * (1 - 1 / a.tp) / a.hbm_tbps,
           c1_measured_dense_ms_bs1=4.32)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out, indent=1))
