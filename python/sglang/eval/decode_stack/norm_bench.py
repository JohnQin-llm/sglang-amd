"""The served fused post-norm + residual-gate + pre-norm kernel at 8 verify rows (H=8192): launch config sweep.

Flags as served for the cross-layer call (post_feedforward_layernorm with weight, channel-wise residual gate,
input_layernorm with weight, FP32 stream). Timing: 64 launches per graph on rotating inputs (time per call);
outputs compared bit for bit against the served config (num_warps=8, num_stages=1).
"""
import argparse
import json

import torch
from sglang.srt.rocm_muse_spark.fused_norm_residual import _fused_norm_residual_triton_kernel as KER

M, H, CALLS = 8, 8192, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    br = [torch.randn(M, H, device=dev, dtype=torch.bfloat16) for _ in range(CALLS)]
    st = [torch.randn(M, H, device=dev, dtype=torch.float32) for _ in range(CALLS)]
    pw = torch.rand(H, device=dev, dtype=torch.bfloat16) + 0.5
    qw = torch.rand(H, device=dev, dtype=torch.bfloat16) + 0.5
    al = torch.rand(H, device=dev, dtype=torch.float32)
    be = torch.rand(H, device=dev, dtype=torch.float32)
    e32 = torch.empty(0, device=dev, dtype=torch.float32)
    out = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    ns = torch.empty(M, H, device=dev, dtype=torch.float32)

    def launch(i, w, s):
        KER[(M,)](br[i], st[i], out, ns, e32, pw, qw, al, be, 1e-6, 1e-6, H, H, False, True, True, True, True, False,
                  False, 0, num_warps=w, num_stages=s)

    def time_cfg(w, s):
        launch(0, w, s)
        torch.cuda.synchronize()
        y = (out.clone(), ns.clone())
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for i in range(CALLS):
                launch(i, w, s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            for i in range(CALLS):
                launch(i, w, s)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(20):
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000 / CALLS)
        return sorted(ts)[10], y

    res = {}
    base_us, ref = time_cfg(8, 1)
    for w in (2, 4, 8, 16, 32):
        for s in (1, 2):
            try:
                us, y = time_cfg(w, s)
            except Exception as ex:
                print("skip", w, s, str(ex)[:80]); continue
            diff = [int((p.view(torch.uint8) != q.view(torch.uint8)).sum()) for p, q in zip(y, ref)]
            res[f"w{w}_s{s}"] = {"us": us, "bytes_differing": diff}
            print(f"num_warps={w:2d} num_stages={s}: {us:5.2f} us per call, differing bytes (out, new_stream) {diff}",
                  flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
