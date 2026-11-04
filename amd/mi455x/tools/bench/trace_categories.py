"""GPU time per decode step by category, from one torch-profiler trace of N draft+verify steps."""
import collections, gzip, json, re, sys
path, steps = sys.argv[1], int(sys.argv[2])
ev = json.load(gzip.open(path))["traceEvents"]
k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
t0, t1 = min(e["ts"] for e in k), max(e["ts"] + e["dur"] for e in k)
CATS = [
    ("attention, full layers (split-KV)", r"unified_attention_3d|kernel_unified_attention_3d|reduce_segments"),
    ("attention, sliding-window layers (2D)", r"unified_attention_2d"),
    ("draft attention", r"^_fwd_kernel|dflash|draft"),
    ("MoE experts (Triton GEMMs)", r"fused_moe_gemm"),
    ("MoE align / routing / finalize", r"_align_small|finalize_norm|routing|topk|_dense_partial|_joint_epilogue"),
    ("dense BF16 GEMM (aiter a16w16)", r"gemm_a16w16"),
    ("dense GEMM (hipBLASLt)", r"^Cijk_"),
    ("all-reduce / all-gather", r"ar_ll|all_reduce|allreduce|ag_gfx|nccl"),
    ("norms, RoPE, KV write, elementwise", r"norm|rotary|reshape_and_cache|elementwise|qk_norm|triton_poi|Convert|copyBuffer|softcap|argmax"),
]
by = collections.defaultdict(float); other = collections.defaultdict(float)
for e in k:
    for name, pat in CATS:
        if re.search(pat, e["name"]):
            by[name] += e["dur"]; break
    else:
        other[e["name"]] += e["dur"]
busy = sum(e["dur"] for e in k)
span = t1 - t0
print(f"{steps} steps: span {span/1000:.1f} ms ({span/steps/1000:.2f} ms/step), GPU busy {busy/1000:.1f} ms ({100*busy/span:.0f}%), idle {(span-busy)/steps/1000:.2f} ms/step")
for name, _ in CATS:
    print(f"  {by[name]/steps/1000:6.2f} ms/step  {100*by[name]/span:5.1f}%  {name}")
o = sum(other.values())
print(f"  {o/steps/1000:6.2f} ms/step  {100*o/span:5.1f}%  other:", ", ".join(f"{n[:40]} {v/steps/1000:.2f}" for n, v in sorted(other.items(), key=lambda x: -x[1])[:5]))
