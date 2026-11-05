"""Check attn3_split partials against torch per segment (full layer, one fixture)."""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["TRITON_CACHE_DIR"] = os.path.join(HERE, "tcache")
sys.path.insert(0, "/data/muse455/k2/cand/vagm")
sys.path.insert(0, HERE)
import torch  # noqa: E402

from vagm import attn_sweep as AS  # noqa: E402
import kern.attn3 as A3  # noqa: E402

C = AS.C
ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
T, S, B = 64, 128, 4
kw = dict(a.split("=") for a in sys.argv[2:])
kw = {k: int(v) for k, v in kw.items()}
dev = torch.device("cuda", 0)
fx = C.make_fixture("full", ctx, 1, dev=dev)
seq = ctx + C.ROWS
seqused = fx["seq_lens"] + C.ROWS
so = torch.full((C.ROWS, C.NQ, S, C.D), 12345.0, device=dev)
sm = torch.full((C.ROWS, C.NQ, S), 12345.0, device=dev)
ss = torch.full((C.ROWS, C.NQ, S), 12345.0, device=dev)
q, k, v, tab = fx["q"], fx["k"], fx["v"], fx["table"][0]
A3.attn3_split[(S, C.NKV, 1)](
    so, sm, ss, q, k, v, tab, seqused, fx["cu_q"], fx["k_descale"], fx["v_descale"], C.SCALE,
    q.stride(0), q.stride(1), k.shape[0],
    NQ=C.NQ, NKV=C.NKV, G=8, D=C.D, PAGE=C.PAGE, TILE=T, NSEG=S, NBUF=B, BLOCK_M=64, num_warps=4, **kw)
torch.cuda.synchronize()
n_tiles = math.ceil(seq / T)
tps = max(math.ceil(n_tiles / S), 1)
live = math.ceil(n_tiles / tps)
print("seq", seq, "tiles", n_tiles, "tps", tps, "live", live)
print("written max", int((sm != 12345.0).sum()), "of", C.ROWS * C.NQ * live, "nan max", int(torch.isnan(sm).sum()),
      "nan sum", int(torch.isnan(ss).sum()), "nan out", int(torch.isnan(so).sum()))
n_log = math.ceil(seq / C.PAGE)
pages = tab[:n_log].long()
kk = k.view(torch.uint8)[pages].view(C.FP8).reshape(-1, C.NKV, C.D)[:seq].float()
vv = v.view(torch.uint8)[pages].view(C.FP8).reshape(-1, C.NKV, C.D)[:seq].float()
c = C.SCALE * 1.4426950408889634
for seg in [0, 1, live - 1]:
    lo, hi = seg * tps * T, min((seg + 1) * tps * T, seq)
    for h in [0, 1]:
        qh = q.float()[:, h * 8:(h + 1) * 8]  # [8 rows, 8 heads, D]
        s_ = torch.einsum("rgd,kd->rgk", qh, kk[lo:hi, h])
        pos_q = ctx + torch.arange(C.ROWS, device=dev)
        mask = torch.arange(lo, hi, device=dev)[None, :] <= pos_q[:, None]
        s_ = s_.masked_fill(~mask[:, None, :], float("-inf"))
        m = (s_.amax(-1) * c)
        p = torch.exp2(s_ * c - m[..., None])
        l_ = p.sum(-1)
        o = torch.einsum("rgk,kd->rgd", p, vv[lo:hi, h])
        gm, gl_, go = sm[:, h * 8:(h + 1) * 8, seg], ss[:, h * 8:(h + 1) * 8, seg], so[:, h * 8:(h + 1) * 8, seg]
        print(f"seg {seg} kvh {h}: max err {(gm - m).abs().max().item():.3e} sum rel {((gl_ - l_) / l_).abs().max().item():.3e}"
              f" out err {(go - o).abs().max().item():.3e} | kern m[0,:3] {gm[0, :3].tolist()} ref {m[0, :3].tolist()}"
              f" | kern l[0,:2] {gl_[0, :2].tolist()} ref {l_[0, :2].tolist()} | o[0,0,:3] {go[0, 0, :3].tolist()} ref {o[0, 0, :3].tolist()}")
