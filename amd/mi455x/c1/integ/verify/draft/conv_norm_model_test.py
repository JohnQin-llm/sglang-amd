"""Integration check of SGLANG_MUSE_DFLASH_CONV_NORM_FUSE through DFlashDraftModel.forward / DFlashDecoderLayer.forward.

5 layers with the real DFlashGroupedConv and RMSNorm modules (aiter dispatch) and stand-in attention / MLP
(fixed BF16 matmuls), hidden 8192, 8 rows. The final hidden states with the flag off and on are compared
bitwise, and the draft forward is timed in a CUDA graph.
"""
import statistics
import sys

sys.path.insert(0, "/data/muse455/integ/sglang/python")

import torch
from torch import nn

import sglang.srt.models.dflash as dflash
from sglang.srt.layers.layernorm import RMSNorm

T, H, L = 8, 8192, 5
dev = torch.device("cuda", 0)
torch.manual_seed(0)


class StubAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter((torch.randn(H, H) * H ** -0.5).to(torch.bfloat16), requires_grad=False)

    def forward(self, positions, hidden_states, forward_batch):
        return hidden_states @ self.w


class StubMLP(StubAttn):
    def forward(self, x):
        return x @ self.w


def make_model():
    layers = []
    for _ in range(L):
        ly = dflash.DFlashDecoderLayer.__new__(dflash.DFlashDecoderLayer)
        nn.Module.__init__(ly)
        ly.input_layernorm = RMSNorm(H, eps=1e-5)
        ly.post_attention_layernorm = RMSNorm(H, eps=1e-5)
        ly.self_attn = StubAttn()
        ly.mlp = StubMLP()
        ly.attention_conv = dflash.DFlashGroupedConv(H, 8, 2, 16)
        ly.mlp_conv = dflash.DFlashGroupedConv(H, 8, 2, 16)
        layers.append(ly)
    m = dflash.DFlashDraftModel.__new__(dflash.DFlashDraftModel)
    nn.Module.__init__(m)
    m.layers = nn.ModuleList(layers)
    m.norm = RMSNorm(H, eps=1e-5)
    m.embed_tokens = None
    m = m.to(dev).to(torch.bfloat16)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 1:
                p.copy_(torch.rand_like(p.float()).add_(0.5).to(p.dtype))
        for ly in layers:
            for c in (ly.attention_conv, ly.mlp_conv):
                c.base_kernel.add_(torch.randn_like(c.base_kernel) * 0.05)
                c.kernel_projection.weight.copy_(
                    (torch.randn_like(c.kernel_projection.weight.float()) * H ** -0.5 * 0.3).to(torch.bfloat16))
    return m


def forward(m, x, fuse):
    dflash._DFLASH_CONV_NORM_FUSE = fuse
    return m.forward(None, None, None, input_embeds=x).hidden_states


def main():
    m = make_model()
    print("RMSNorm dispatch:", m.norm._forward_method.__name__, "aiter triton:", dflash._aiter_triton_rmsnorm())
    x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
    with torch.no_grad():
        a = forward(m, x, False)
        b = forward(m, x, True)
        print(f"[numerics] 5-layer draft forward, final hidden bitwise equal flag off vs on: {torch.equal(a, b)} "
              f"(max abs {float((a.float() - b.float()).abs().max()):.3g})")
        res = {}
        for fuse in (False, True):
            dflash._DFLASH_CONV_NORM_FUSE = fuse
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    forward(m, x, fuse)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                forward(m, x, fuse)
            res[fuse] = g
        ts = {False: [], True: []}
        for i in range(7):
            for fuse in ((False, True) if i % 2 == 0 else (True, False)):
                g = res[fuse]
                for _ in range(5):
                    g.replay()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(50):
                    g.replay()
                e1.record()
                torch.cuda.synchronize()
                ts[fuse].append(e0.elapsed_time(e1) * 1000 / 50)
        a, b = statistics.median(ts[False]), statistics.median(ts[True])
        print(f"[time] stub draft forward (5 layers incl. stand-in matmuls): flag off {a:.1f} us, on {b:.1f} us, "
              f"saved {a - b:.1f} us")


def probe_topk_ties():
    """Order torch.topk (radix path, [8, 50512] BF16, k 16) gives to equal values."""
    g = torch.Generator(device="cuda").manual_seed(7)
    lg = (torch.randn(8, 50512, device=dev, generator=g) * 2).to(torch.bfloat16)
    lg[0, :40] = 9.0
    lg[1, [5, 900, 17000, 33000, 50000]] = 9.5
    lg[1, [7, 1000, 20000, 40000]] = 9.25
    v, i = torch.topk(lg, 16, dim=-1)
    for r in range(3):
        groups = {}
        for val, idx in zip(v[r].float().tolist(), i[r].tolist()):
            groups.setdefault(val, []).append(idx)
        print(f"[topk ties] row {r}: " + "; ".join(f"{k:g}: {g}" for k, g in groups.items() if len(g) > 1))


if __name__ == "__main__":
    main()
    probe_topk_ties()
