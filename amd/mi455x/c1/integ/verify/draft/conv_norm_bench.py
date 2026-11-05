"""Microbenchmark + numerics: DFlash output-side grouped conv + residual add + RMSNorm, served (inductor conv kernel
+ aiter Triton add-RMSNorm, two launches) vs dflash_conv_norm.conv_out_add_rmsnorm (one launch).

Served shape: 8 rows (draft block 8, bs 1), hidden 8192, conv taps 2, group 16. 10 instances per draft step
(attention and MLP side of 5 layers; the last feeds the final norm). Timing: CUDA graph, alternating trials.
"""
import statistics
import sys

sys.path.insert(0, "/data/muse455/integ/sglang/python")

import torch
from aiter.ops.triton.normalization.rmsnorm import rmsnorm2d_fwd_with_add

from sglang.kernels.ops.speculative.dflash_conv_norm import conv_out_add_rmsnorm
from sglang.srt.models.dflash import DFlashGroupedConv

T, H, BLOCK, TAPS, GS, EPS, NINST = 8, 8192, 8, 2, 16, 1e-5, 10
dev = torch.device("cuda", 0)
torch.manual_seed(0)


def make():
    conv = DFlashGroupedConv(H, BLOCK, TAPS, GS).to(dev).to(torch.bfloat16)
    with torch.no_grad():
        conv.base_kernel.add_(torch.randn_like(conv.base_kernel) * 0.05)
    insts = []
    for _ in range(NINST):
        coef = (torch.randn(T, 2 * TAPS * (H // GS), device=dev) * 0.3).to(torch.bfloat16)
        insts.append(dict(
            y=(torch.randn(T, H, device=dev) * 0.7).to(torch.bfloat16),
            delta=coef.reshape(T, 2, TAPS, H // GS)[..., 1, :, :],
            res=(torch.randn(T, H, device=dev) * 4).to(torch.bfloat16),
            w=(torch.rand(H, device=dev) + 0.5).to(torch.bfloat16),
        ))
    return conv, insts


def served(conv, insts):
    outs = []
    for d in insts:
        x = conv.finish(d["y"], d["delta"])
        out, res_out = torch.empty_like(x), torch.empty_like(x)
        rmsnorm2d_fwd_with_add(out, x, d["res"], res_out, d["w"], EPS)  # RMSNorm.forward_aiter's call
        outs.append((out, res_out))
    return outs


def fused(conv, insts):
    return [conv_out_add_rmsnorm(d["y"], d["delta"], conv.base_kernel[1], d["res"], d["w"], EPS, BLOCK, GS)
            for d in insts]


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def time_graph(g, iters=200):
    for _ in range(10):
        g.replay()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / iters


def main():
    conv, insts = make()
    with torch.no_grad():
        a, b = served(conv, insts), fused(conv, insts)
        bit = all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(a, b))
        dn = max(float((x[0].float() - y[0].float()).abs().max()) for x, y in zip(a, b))
        dr = max(float((x[1].float() - y[1].float()).abs().max()) for x, y in zip(a, b))
        nd = sum(int((x[0] != y[0]).sum()) for x, y in zip(a, b))
        rel = max(float(((x[0].float() - y[0].float()).abs() / y[0].float().abs().clamp_min(1e-3)).max())
                  for x, y in zip(a, b))
        print(f"[numerics] {NINST} x [8, 8192]: normed + residual bitwise equal to served: {bit}; "
              f"normed max abs {dn:.3g} max rel {rel:.3g} ({nd} of {NINST * T * H} elements differ), "
              f"residual max abs {dr:.3g}")
        ga, gb = graph_of(lambda: served(conv, insts)), graph_of(lambda: fused(conv, insts))
        ta, tb = [], []
        for i in range(9):
            for g, t in ((ga, ta), (gb, tb)) if i % 2 == 0 else ((gb, tb), (ga, ta)):
                t.append(time_graph(g))
        ma, mb = statistics.median(ta), statistics.median(tb)
        print(f"[time] 10 conv-finish + add-RMSNorm per draft step: served {ma:.2f} us, fused {mb:.2f} us, "
              f"saved {ma - mb:.2f} us (trials served {min(ta):.2f}-{max(ta):.2f}, fused {min(tb):.2f}-{max(tb):.2f})")


if __name__ == "__main__":
    main()
