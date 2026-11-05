"""Resource footprint and cold/warm time of the served TP4 M=8 QKV kernels (_dense_partial split-K 8 + _qkv_epilogue)."""
import json
import re
import sys

sys.path.insert(0, "/data/muse455/sglang/python")
import torch
import triton

from sglang.kernels.ops.gemm.bf16_splitk_gemm import _dense_partial

M, N, K = 8, 4608, 8192
SK, BN, BM, W = 8, 64, 16, 4
dev = torch.device("cuda:0")
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
LAY = 62
ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(LAY)]
p = torch.empty(SK, M, N, device=dev, dtype=torch.float32)


def launch(w):
    return _dense_partial[(N // BN, 1, SK)](x, w, p, M, N, K, BM, BN, 128, SK, ".cg", num_warps=W, num_stages=2,
                                            matrix_instr_nonkdim=16)


k = launch(ws[0])
torch.cuda.synchronize()
asm = k.asm.get("amdgcn", "")
meta = {}
for key in ("vgpr_count", "sgpr_count", "group_segment_fixed_size", "private_segment_fixed_size", "agpr_count",
            "max_flat_workgroup_size", "wavefront_size"):
    m = re.search(r"\." + key + r":\s+(\d+)", asm)
    meta[key] = int(m.group(1)) if m else None
meta["n_regs"] = getattr(k, "n_regs", None)
meta["n_spills"] = getattr(k, "n_spills", None)
meta["shared"] = k.metadata.shared if hasattr(k.metadata, "shared") else None
meta["grid"] = [N // BN, 1, SK]
meta["threads"] = W * 32
open("/data/muse455/pdltest/bench/dense_partial.amdgcn", "w").write(asm)


def gtime(fn, reps=20):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


meta["cold_us_per_call"] = gtime(lambda: [launch(w) for w in ws]) / LAY
meta["warm_us_per_call"] = gtime(lambda: [launch(ws[0]) for _ in range(LAY)]) / LAY
print("TRITON", json.dumps(meta))
json.dump(meta, open("/data/muse455/pdltest/bench/tri_meta.json", "w"), indent=1)
