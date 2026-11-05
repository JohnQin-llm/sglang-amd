"""K2 TP4/EP1 numerics harness (lead): references, inputs and the bars.

Two torch arms over the same inputs (one layer, T verify rows, the attention core output and the residual stream):

  k2ref : the TP4/EP1 K2 dataflow with the kernel's rounding points (INTERFACE.md §1 "dataflow"): per-rank o_proj
          partials bf16 -> rank-order fp32 sum -> bf16; post_attn_norm, gated residual (stream_out fp32),
          pre_ffn_norm -> bf16 x; fp32 router; bf16 pooler; pre_expert_norm -> bf16 xe -> MXFP8; per rank the EP1
          slice: gate/up (fp32) -> bf16 silu(g) u -> MXFP8 -> down -> bf16 partial per route; rank-order fp32 sum
          -> bf16 y; finalize (v = bf16(y bf16(w+1)), w_k / sqrt(mean v^2 + eps), bf16 sum); bf16 unpooler.
  stock : the served math (A16 experts, bf16 inter, the same rounding at the layer seams), the bar's reference.

Both close the FFN boundary the served way (native kernel, not in K2): stream2 = a stream_out + b
post_ffn_norm(branch); the bar is the cosine of stream2 rows (floor 0.997, target 0.999), and routes equal except
near-ties (8th vs 9th biased score within two bf16 ulps of the 8th).

``check(outs, case, ref)`` grades a kernel's outputs (``stream_out`` fp32 [T, H], ``branch_out`` bf16 [T, H],
optionally ``ids`` int32 [T, 8]) against ``stock`` and reports the distance to ``k2ref`` too (kernel bugs show up
there first: k2ref and a correct kernel differ only by accumulation order and FP8 WMMA precision).

Run (reference vs stock only, no kernel):
  flock /data/muse455/locks/gpu1.lock docker exec muse_mi455x bash -c \
    'cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=1 python3 /data/muse455/k2/cand/k2v6/port/validation.py'
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/muse455/k2/cand/k2v6")
sys.path.insert(0, "/data/muse455/k2/cand/k2v6/s2")
import ep1_layout as P  # noqa: E402
from port import consts as K  # noqa: E402
from port.gemv import weights as W  # noqa: E402

H, MH, E, TOPK, TP, RI = K.H, K.MH, K.E, K.TOPK, K.TP, K.RI
CORE_FULL = K.CORE * TP  # 8192: o_proj K (all heads)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FLOOR, TARGET = 0.997, 0.999


# ---------------------------------------------------------------- small math
def bf(x):
    return x.to(torch.bfloat16).float()


def rstd(x, eps):
    return torch.rsqrt((x.float() * x.float()).mean(-1, keepdim=True) + eps)


def quant_a8(x):
    """OCP MXFP8 e4m3 per 32 (INTERFACE.md §1 Activations): -> (dequantized fp32, e4m3 bytes, e8m0 codes)."""
    r, k = x.shape
    xf = x.float().view(r, k // 32, 32)
    amax = xf.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(torch.where(amax > 0, amax, torch.ones_like(amax)))) - 8
    e = torch.where(amax > 0, torch.clamp(e, -127, 127), torch.full_like(e, -127))
    q = torch.clamp(xf / torch.exp2(e), -448, 448).to(torch.float8_e4m3fn)
    deq = (q.float() * torch.exp2(e)).view(r, k)
    return deq, q.view(torch.uint8).view(r, k), (e + 127).to(torch.uint8).view(r, k // 32)


def route(x_bf, w_router, bias, mult=K.ROUTER_MULT):
    """Served FusedMuseSparkRouting: sigmoid(logits mult) + bias -> top-8 (ties: lower id); weights renormalized."""
    logits = x_bf.float() @ w_router.float().T
    sc = torch.sigmoid(logits * mult)
    sel = sc + bias.float()
    # stable top-k: equal keys -> lower expert id
    order = torch.sort(sel, dim=-1, descending=True, stable=True)
    ids = order.indices[:, :TOPK].to(torch.int32)
    s = sc.gather(1, ids.long())
    wts = s / (s.sum(-1, keepdim=True) + K.ROUTE_EPS) * K.ROUTE_SCALE
    gap = order.values[:, TOPK - 1] - order.values[:, TOPK]
    return logits, ids, wts, gap, order.values[:, TOPK - 1]


def bf16_ulp(v):
    v = v.abs().clamp_min(1e-30)
    return torch.exp2(torch.floor(torch.log2(v)) - 7)


def gated(stream, branch, alpha, beta):
    return alpha.float() * stream.float() + beta.float() * branch.float()


# ---------------------------------------------------------------- one layer's weights
class LayerData:
    """Dense weights (bf16 / fp32, as the checkpoint) and the routed experts (native MXFP4 + EP1 slices)."""

    def __init__(self, layer, dev="cuda"):
        if layer in (0, 61):
            raise ValueError("layers 0 and 61 have BF16 experts: not K2 layers")
        self.layer, self.dev = layer, dev
        self.w = W.load(layer, dev)
        self.rd = P._Reader()
        p = f"model.language_model.layers.{layer}.mlp.experts.post_expert_norm.weight"
        self.w["w_post"] = self.rd.get(p).to(dev)
        self.native = {}  # e -> (gate, up, down, gate_s, up_s, down_s) on dev
        self._deq = {}

    def expert(self, e):
        if e not in self.native:
            self.native[e] = [t.to(self.dev) for t in P.load_native_expert(self.rd, self.layer, e)]
        return self.native[e]

    def deq(self, e, rank=None):
        """Dequantized fp32 (Wg, Wu, Wd) of expert e: whole, or rank's EP1 slice (gate/up rows, down K)."""
        key = (e, rank)
        if key not in self._deq:
            g, u, d, gs, us, ds = self.expert(e)
            if rank is None:
                self._deq[key] = (P.dequant(g, gs).float(), P.dequant(u, us).float(), P.dequant(d, ds).float())
            else:
                w13, s13, w2, s2 = P.slice_expert(g, u, d, gs, us, ds, rank)
                w13 = P.dequant(w13, s13).float()
                self._deq[key] = (w13[:RI], w13[RI:], P.dequant(w2, s2).float())
        return self._deq[key]


# ---------------------------------------------------------------- inputs
class Case:
    def __init__(self, core, stream_a):
        self.core = core  # [T, 8192] bf16, all heads (rank r reads columns r*2048 ..)
        self.stream_a = stream_a  # [T, H] fp32

    def core_rank(self, r):
        return self.core[:, r * K.CORE:(r + 1) * K.CORE].contiguous()


def make_case(lay, corr, seed=0, t=K.MAX_TOKENS, dev="cuda"):
    """Synthetic verify rows like S2: T rows share a base direction (corr) in both the core and the stream.
    The core is scaled so that the o_proj output has the stream's rms."""
    g = torch.Generator().manual_seed(seed)

    def mix(n):
        base = torch.randn(1, n, generator=g).expand(t, n)
        noise = torch.randn(t, n, generator=g)
        return corr * base + math.sqrt(1 - corr * corr) * noise

    stream = mix(H).float().to(dev)
    core = mix(CORE_FULL).to(dev)
    o = core @ lay.w["w_o"].float().T
    core = (core * (stream.norm() / o.norm())).to(torch.bfloat16)
    return Case(core, stream)


def calibrate(lay, target=43, seed=0, t=K.MAX_TOKENS):
    """corr giving ~target distinct experts over the T rows (bisection on the k2ref route)."""
    lo, hi = 0.0, 0.999
    best = None
    for _ in range(14):
        mid = (lo + hi) / 2
        case = make_case(lay, mid, seed, t)
        n = int(torch.unique(seam_and_route(lay, case)["ids"]).numel())
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (mid, n)
        if n > target:
            lo = mid
        else:
            hi = mid
        if n == target:
            break
    return best


# ---------------------------------------------------------------- the arms
def seam_and_route(lay, case, oproj_ranked=True):
    w = lay.w
    if oproj_ranked:  # K2: per-rank bf16 partials, rank-order fp32 sum, bf16
        acc = torch.zeros(case.core.shape[0], H, device=case.core.device)
        for r in range(TP):
            wo = w["w_o"][:, r * K.CORE:(r + 1) * K.CORE]
            acc = acc + bf(case.core_rank(r).float() @ wo.float().T)
        xo = bf(acc)
    else:  # one fp32 GEMM, bf16 out
        xo = bf(case.core.float() @ w["w_o"].float().T)
    n = gated(case.stream_a, xo * rstd(xo, K.EPS_POST_ATTN), w["alpha_a"], w["beta_a"])
    x = bf(n * rstd(n, K.EPS_FFN) * (w["w_fn"].float() + 1.0))
    logits, ids, wts, gap, s8 = route(x, w["w_router"], w["bias"])
    return dict(xo=xo, stream_out=n, x=x, logits=logits, ids=ids, wts=wts, gap=gap, s8=s8)


def pre_expert(lay, pooled):
    w = lay.w["w_pre"].float()
    return bf((pooled * rstd(pooled, K.EPS_PRE_EXPERT)) * bf(w + 1.0))


def finalize(lay, y, wts):
    """y [T, 8, MH] (bf16 values) -> [T, MH] bf16 (P3's / the served finalize)."""
    v = bf(y * bf(lay.w["w_post"].float() + 1.0))
    sc = wts[..., None] * torch.rsqrt((v * v).mean(-1, keepdim=True) + K.EPS_POST_EXPERT)
    return bf((sc * v).sum(1))


def k2ref(lay, case, act="a8"):
    """The K2 TP4/EP1 dataflow (kernel rounding points). Returns every intermediate the kernel exposes."""
    o = seam_and_route(lay, case)
    x, ids, wts = o["x"], o["ids"], o["wts"]
    t = x.shape[0]
    pooled = torch.zeros(t, MH, device=x.device)
    for r in range(TP):  # each rank its 1024 columns (an all-gather, no sum)
        cols = slice(r * K.POOL_N, (r + 1) * K.POOL_N)
        pooled[:, cols] = bf(x @ lay.w["w_pool"][cols].float().T)
    xe = pre_expert(lay, pooled)
    xq = quant_a8(xe)[0] if act == "a8" else xe
    ysum = torch.zeros(t, TOPK, MH, device=x.device)
    parts = []
    for r in range(TP):
        yr = torch.zeros(t, TOPK, MH, device=x.device)
        for e in torch.unique(ids).tolist():
            wg, wu, wd = lay.deq(e, r)
            sel = (ids == e).nonzero()
            xs = xq[sel[:, 0]]
            inter = bf(F.silu(xs @ wg.T) * (xs @ wu.T))
            iq = quant_a8(inter)[0] if act == "a8" else inter
            yr[sel[:, 0], sel[:, 1]] = bf(iq @ wd.T)
        parts.append(yr)
        ysum = ysum + yr  # rank order
    y = bf(ysum)
    fin = finalize(lay, y, wts)
    branch = torch.zeros(t, H, device=x.device)
    for r in range(TP):
        cols = slice(r * K.UP_N, (r + 1) * K.UP_N)
        branch[:, cols] = bf(fin @ lay.w["w_unpool"][cols].float().T)
    o.update(pooled=pooled, xe=xe, y_parts=parts, y=y, fin=fin, branch=branch)
    return o


def stock(lay, case, ids=None):
    """Served math: one o_proj GEMM, A16 experts (whole intermediate, bf16 inter), same seams. ``ids``: force
    these routes (near-tie rows graded on the kernel's choice; weights from stock's own scores)."""
    o = seam_and_route(lay, case, oproj_ranked=False)
    if ids is not None:
        sc = torch.sigmoid(o["logits"] * K.ROUTER_MULT).gather(1, ids.long())
        o["ids"], o["wts"] = ids, sc / (sc.sum(-1, keepdim=True) + K.ROUTE_EPS) * K.ROUTE_SCALE
    x, ids, wts = o["x"], o["ids"], o["wts"]
    t = x.shape[0]
    pooled = bf(x @ lay.w["w_pool"].float().T)
    xe = pre_expert(lay, pooled)
    y = torch.zeros(t, TOPK, MH, device=x.device)
    for e in torch.unique(ids).tolist():
        wg, wu, wd = lay.deq(e)
        sel = (ids == e).nonzero()
        xs = xe[sel[:, 0]]
        y[sel[:, 0], sel[:, 1]] = bf(bf(F.silu(xs @ wg.T) * (xs @ wu.T)) @ wd.T)
    fin = finalize(lay, y, wts)
    branch = bf(fin @ lay.w["w_unpool"].float().T)
    o.update(pooled=pooled, xe=xe, y=y, fin=fin, branch=branch)
    return o


def boundary(lay, stream_out, branch):
    """The native FFN boundary: stream2 = a stream_out + b rms_norm((w_pf + 1) branch, 1e-8)."""
    w = lay.w
    b = branch.float() * (w["w_pf"].float() + 1.0)
    pn = bf(b * rstd(b, 1e-8))
    return gated(stream_out, pn, w["alpha_f"], w["beta_f"])


# ---------------------------------------------------------------- grading
def cos_rows(a, b):
    return F.cosine_similarity(a.double(), b.double(), dim=-1)


def routes_ok(ids, ref):
    """Routes equal as sets except near-tie rows of ``ref`` (gap within two bf16 ulps of the 8th score)."""
    bad, ties = [], []
    for t in range(ids.shape[0]):
        if set(ids[t].tolist()) != set(ref["ids"][t].tolist()):
            near = float(ref["gap"][t]) <= 2 * float(bf16_ulp(ref["s8"][t]))
            (ties if near else bad).append(t)
    return {"ok": not bad, "mismatch_rows": bad, "near_tie_rows": ties}


def check(outs, case, lay, ref=None, st=None):
    st = st or stock(lay, case)
    res = {}
    if "ids" in outs:
        res["routes"] = routes_ok(outs["ids"], st)
        if res["routes"]["near_tie_rows"]:  # grade on the kernel's routes (allowed near-tie flips)
            ids = st["ids"].clone()
            for t in res["routes"]["near_tie_rows"]:
                ids[t] = outs["ids"][t]
            st = stock(lay, case, ids)
    s2_k = boundary(lay, outs["stream_out"], outs["branch_out"])
    s2_s = boundary(lay, st["stream_out"], st["branch"])
    c = cos_rows(s2_k, s2_s)
    res.update({
        "stream2_cos_min": float(c.min()),
        "stream2_cos_mean": float(c.mean()),
        "stream_out_cos_min": float(cos_rows(outs["stream_out"], st["stream_out"]).min()),
        "branch_cos_min": float(cos_rows(outs["branch_out"].float(), st["branch"]).min()),
        "floor": FLOOR,
        "target": TARGET,
    })
    res["pass_floor"] = res["stream2_cos_min"] >= FLOOR and res.get("routes", {"ok": True})["ok"]
    res["pass_target"] = res["stream2_cos_min"] >= TARGET and res["pass_floor"]
    if ref is not None:
        res["vs_k2ref"] = {
            "stream_out_maxabs": float((outs["stream_out"] - ref["stream_out"]).abs().max()),
            "branch_cos_min": float(cos_rows(outs["branch_out"].float(), ref["branch"]).min()),
            "branch_bf16_mismatch_frac": float((bf(outs["branch_out"].float()) != ref["branch"]).float().mean()),
        }
    return res


# ---------------------------------------------------------------- CLI: k2ref vs stock (the predicted bar)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,31,60")
    ap.add_argument("--act", default="a8", choices=("a8", "a16"))
    ap.add_argument("--target-distinct", type=int, default=43)
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {"act": args.act, "layers": {}}
    for layer in [int(x) for x in args.layers.split(",")]:
        t0 = time.time()
        lay = LayerData(layer)
        rows = []
        for seed in [int(x) for x in args.seeds.split(",")]:
            corr, n = calibrate(lay, args.target_distinct, seed)
            case = make_case(lay, corr, seed)
            ref = k2ref(lay, case, args.act)
            st = stock(lay, case)
            outs = {"stream_out": ref["stream_out"], "branch_out": ref["branch"].to(torch.bfloat16), "ids": ref["ids"]}
            r = check(outs, case, lay, None, st)
            r.update(seed=seed, corr=corr, distinct=n, min_gap=float(ref["gap"].min()))
            rows.append(r)
            print(f"layer {layer} seed {seed}: distinct {n} stream2 cos min {r['stream2_cos_min']:.6f} "
                  f"branch cos min {r['branch_cos_min']:.6f} routes {r['routes']}", flush=True)
        res["layers"][layer] = {"cases": rows, "seconds": time.time() - t0}
        with open(os.path.join(OUT, f"validation_k2ref_{args.act}.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
