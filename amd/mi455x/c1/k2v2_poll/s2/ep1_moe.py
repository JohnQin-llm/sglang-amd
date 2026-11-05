"""S2: the MoE block at TP4/EP1 with the FlyDSL A4W4 expert stages, simulated rank by rank on one GPU.

Arms (same inputs, same routes):
  stock  : the served path at TP4/EP4: fp32 router, fused routing rule, bf16 pooler + pre-expert norm, served
           Triton `muse_moe_triton` per EP4 rank (64 local experts, A16 activations, per-route post-expert
           norm + weighted sum local), sum over ranks (the all-reduce), bf16 unpooler.
  ep1    : the K2 TP4/EP1 dataflow: same router/pooler, MXFP4 activations (aiter rule), per rank the FlyDSL
           gate/up+SiLU over its 1024-wide intermediate slice of all 256 experts -> MXFP4 inter -> FlyDSL down
           over K=1024 -> bf16 partial y_r per route; partials summed over ranks in rank order (fp32) -> bf16 y
           (the reduce to the row owner), then post-expert norm + routed sum (finalize), unpooler.
  ref    : float64 torch, A16 activations (bf16 xe), dequantised weights, no intermediate rounding.
Then both arms close the FFN boundary the served way: stream' = a*stream + b*post_ffn_norm(branch),
next_attn = pre_attn_norm_{L+1}(stream').

Inputs: synthetic like the MI355X harness (eval_moe_layer.py): 8 draft rows share a base vector, stream = corr*base
+ sqrt(1-corr^2)*noise; corr is calibrated so the 8 rows touch ~43 distinct experts (real M8 verify, layer 31).
ffn_input = bf16(pre_feedforward_layernorm(stream)) (the attention branch is not simulated).

Timing (graph replay, back-to-back copies so weights stream from HBM): EP1 per-rank expert stages (rank r's slice,
input set r), and the served EP4 Triton per rank (max over ranks = the EP4 critical path).

Run: cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=3 python3 /data/muse455/k2/cand/k2v2_land1/s2/ep1_moe.py --layers 1,31,60
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/muse455/k2/cand/k2v2_land1")
sys.path.insert(0, "/data/muse455/k2/port")
sys.path.insert(0, "/data/muse455/k2/quickwin")
import ep1_layout as P  # noqa: E402
import layout_gfx1250 as L  # noqa: E402
import fly_expert_gemm_ep1 as feg  # noqa: E402
import moe_bench as mb  # noqa: E402
from bench_fly_gemm import quant_a4, dequant as dequant_a4  # noqa: E402

HID = 8192
MH = P.H
I = P.I
E = P.E
TP = P.TP
TOPK = 8
ROWS = 8
IP = I // TP


def quant_a8(x):
    """OCP MXFP8 e4m3 per 32: shared exp = floor(log2(amax)) - 8 (e8m0 = exp + 127), RNE, saturating at 448.
    x [R, K] -> (e4m3 bytes u8 [R, K], e8m0 u8 [R, K/32])."""
    r, k = x.shape
    xf = x.float().view(r, k // 32, 32)
    amax = xf.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(torch.where(amax > 0, amax, torch.ones_like(amax)))) - 8
    e = torch.where(amax > 0, torch.clamp(e, -127, 127), torch.full_like(e, -127))
    v = torch.clamp(xf / torch.exp2(e), -448, 448).to(torch.float8_e4m3fn)
    return v.view(torch.uint8).view(r, k).contiguous(), (e + 127).to(torch.uint8).view(r, k // 32).contiguous()


def dequant_a8(q, s):
    r, k = q.shape
    return (q.view(torch.float8_e4m3fn).float().view(r, k // 32, 32) * torch.exp2(s.float() - 127)[..., None]).view(r, k)


def load_dense(rd, layer, dev):
    p = f"model.language_model.layers.{layer}"
    g = lambda k: rd.get(k).to(dev)  # noqa: E731
    d = dict(
        router=g(f"{p}.mlp.gate.weight").float(),
        bias=g(f"{p}.mlp.gate.e_score_correction_bias").float(),
        pool=g(f"{p}.mlp.pre_expert_proj.weight"),
        pre_w=g(f"{p}.mlp.pre_expert_norm.weight"),
        post_w=g(f"{p}.mlp.experts.post_expert_norm.weight"),
        unpool=g(f"{p}.mlp.post_expert_proj.weight"),
        preffn_w=g(f"{p}.pre_feedforward_layernorm.weight"),
        postffn_w=g(f"{p}.post_feedforward_layernorm.weight"),
        ffn_gate=g(f"{p}.post_feedforward_residual_gate.gate"),
        next_w=g(f"model.language_model.layers.{layer + 1}.input_layernorm.weight"),
    )
    return d


def rmsnorm_zc(x, w, eps):
    """MuseSpark1xRMSNorm, zero-centred gamma (+1), output in x's dtype."""
    return F.rms_norm(x, (x.shape[-1],), w + 1.0, eps).to(x.dtype)


def post_norm_in_scale(x, w, eps):
    """MuseSpark1xRMSNormWithInputScale: rms_norm((w+1) * x)."""
    return F.rms_norm((w + 1.0) * x, (x.shape[-1],), None, eps).to(x.dtype)


def gated_residual(stream, branch, gate, temp=0.3):
    w = gate.view(1, -1).float() / temp
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return (alpha * stream + beta * branch.to(stream.dtype)).to(stream.dtype)


def route(d, ffn_in, mult=0.17677669529663687):
    logits = ffn_in.float() @ d["router"].T
    scores = torch.sigmoid(logits * mult)
    sel = scores + d["bias"]
    top = torch.topk(sel, TOPK, dim=-1)
    ids = top.indices.to(torch.int32)
    s = scores.gather(1, ids.long())
    s = s / (s.sum(-1, keepdim=True) + 1e-15)
    # near-tie margin: 8th vs 9th biased score
    srt = torch.sort(sel, dim=-1, descending=True).values
    gap = (srt[:, 7] - srt[:, 8])
    return logits, ids, s, gap


def expert_input(d, ffn_in):
    pooled = F.linear(ffn_in, d["pool"])
    return rmsnorm_zc(pooled, d["pre_w"], 1e-5)


def finalize(y, scores, post_w, eps=1e-8):
    """y [64, MH] (bf16 values) -> [8, MH] bf16, the served finalize arithmetic."""
    scale = (post_w.float() + 1.0).to(post_w.dtype).float()
    yn = (y.float() * scale).to(torch.bfloat16).float()
    inv = torch.rsqrt((yn * yn).mean(-1) + eps)
    w = scores.reshape(-1) * inv
    return (w[:, None] * yn).view(ROWS, TOPK, MH).sum(1).to(torch.bfloat16)


def boundary(d, stream, branch):
    s2 = gated_residual(stream, post_norm_in_scale(branch, d["postffn_w"], 1e-8), d["ffn_gate"])
    nxt = rmsnorm_zc(s2, d["next_w"], 1e-5).to(torch.bfloat16)
    return s2, nxt


def make_inputs(d, corr, seed, dev):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, HID, generator=g).expand(ROWS, HID)
    noise = torch.randn(ROWS, HID, generator=g)
    stream = (corr * base + math.sqrt(1 - corr * corr) * noise).float().to(dev)
    ffn_in = rmsnorm_zc(stream, d["preffn_w"].float(), 1e-5).to(torch.bfloat16)
    return stream, ffn_in


def cos_rows(a, b):
    return F.cosine_similarity(a.double(), b.double(), dim=-1)


class Layer:
    def __init__(self, layer, dev, rd):
        self.layer = layer
        self.dev = dev
        self.d = load_dense(rd, layer, dev)
        self.rd = rd
        self.ep1 = None  # per rank (w13, s13, w2, s2) gfx1250, [256, ...]
        self.native = {}  # expert -> native tensors on dev (routed experts only)

    def load_ep1(self, experts):
        """EP1 slices, gfx1250 layout, all 4 ranks; only `experts` are filled (others zero: never read)."""
        ranks = []
        for r in range(TP):
            w13 = torch.zeros(E, 2 * IP, MH // 2, dtype=torch.uint8)
            s13 = torch.zeros(E, 2 * IP, MH // 32, dtype=torch.uint8)
            w2 = torch.zeros(E, MH, IP // 2, dtype=torch.uint8)
            s2 = torch.zeros(E, MH, IP // 32, dtype=torch.uint8)
            ranks.append([w13, s13, w2, s2])
        for e in experts:
            nat = P.load_native_expert(self.rd, self.layer, e)
            self.native[e] = [t.to(self.dev) for t in nat]
            for r in range(TP):
                for i, t in enumerate(P.slice_expert(*nat, r)):
                    ranks[r][i][e] = t
        self.ep1 = [tuple(t.to(self.dev).contiguous() for t in L.gfx1250_weights(*rk)) for rk in ranks]

    def ep4_stack(self, r):
        """Native EP4 rank-r stack [64, ...] with only the loaded experts filled."""
        w13 = torch.zeros(64, 2 * I, MH // 2, dtype=torch.uint8, device=self.dev)
        s13 = torch.zeros(64, 2 * I, MH // 32, dtype=torch.uint8, device=self.dev)
        w2 = torch.zeros(64, MH, I // 2, dtype=torch.uint8, device=self.dev)
        s2 = torch.zeros(64, MH, I // 32, dtype=torch.uint8, device=self.dev)
        for e, nat in self.native.items():
            if 64 * r <= e < 64 * (r + 1):
                j = e - 64 * r
                w13[j, :I], w13[j, I:] = nat[0], nat[1]
                s13[j, :I], s13[j, I:] = nat[3], nat[4]
                w2[j], s2[j] = nat[2], nat[5]
        return w13, s13, w2, s2


def ep1_experts(lay, xe, ids, scores, launches, align_fn, keep=None, act="a4"):
    """EP1 expert stage for all 4 ranks; returns y (bf16 [64, MH]) after the rank-order reduction and the
    per-rank partials."""
    quant = quant_a8 if act == "a8" else quant_a4
    xq, xs = quant(xe)
    sid, eid, npost = align_fn(ids)
    l1, l2 = launches
    parts = []
    for r in range(TP):
        w13, s13, w2, s2 = lay.ep1[r]
        inter = torch.zeros(ROWS * TOPK, IP, dtype=torch.bfloat16, device=xe.device)
        feg.run(l1, xq, xs, w13, s13, sid, eid, npost, inter)
        iq, isc = quant(inter.float())
        y = torch.zeros(ROWS * TOPK, MH, dtype=torch.bfloat16, device=xe.device)
        feg.run(l2, iq, isc, w2, s2, sid, eid, npost, y)
        parts.append(y)
        if keep is not None:
            keep.append((xq, xs, sid, eid, npost, inter, iq, isc, y))
    acc = torch.zeros(ROWS * TOPK, MH, dtype=torch.float32, device=xe.device)
    for y in parts:  # rank order
        acc += y.float()
    return acc.to(torch.bfloat16), parts


def ref_experts(lay, xe, ids):
    """float64 per-route expert outputs (A16, exact dequant), [64, MH]."""
    out = torch.zeros(ROWS * TOPK, MH, dtype=torch.float64, device=xe.device)
    for s in range(ROWS * TOPK):
        e = int(ids.view(-1)[s])
        nat = lay.native[e]
        Wg = P.dequant(nat[0], nat[3]).double()
        Wu = P.dequant(nat[1], nat[4]).double()
        Wd = P.dequant(nat[2], nat[5]).double()
        x = xe[s // TOPK].double()
        out[s] = (F.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,31,60")
    ap.add_argument("--target-distinct", type=int, default=43)
    ap.add_argument("--sets", type=int, default=4, help="input sets per layer (numerics + timing copies)")
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--prefetch", type=int, default=16)
    ap.add_argument("--nt", type=int, default=1)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--act", default="a8", choices=["a4", "a8"], help="expert activation format")
    ap.add_argument("--json", default="/data/muse455/k2/cand/k2v2_land1/s2/ep1_moe.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    new = mb.load_module("mmt_new", "/data/muse455/k2/quickwin/muse_moe_triton_new.py")
    base = mb.load_module("mmt_base", "/data/muse455/k2/quickwin/muse_moe_triton_base.py")
    slots = ROWS * TOPK
    units = slots  # at most 64 16-row units at 8 rows (each route its own expert in the worst case)
    a8 = args.act == "a8"
    l1 = feg.make(IP, MH, slots, units, True, args.waves, args.prefetch, args.nt, n_exp=E, act_fp8=a8)
    l2 = feg.make(MH, IP, slots, units, False, args.waves, args.prefetch, args.nt, n_exp=E, act_fp8=a8)

    def align256(ids):
        return new.moe_align_block_size_small(ids, E, 0, 16)

    if args.act != "a8" and args.json.endswith("ep1_moe.json"):
        args.json = args.json.replace(".json", f"_{args.act}.json")
    out = json.load(open(args.json)) if os.path.exists(args.json) else {}
    out["config"] = vars(args)
    rd = P._Reader()
    for layer in [int(x) for x in args.layers.split(",")]:
        t0 = time.time()
        lay = Layer(layer, dev, rd)
        d = lay.d
        # calibrate corr on set 0's seed
        calib = {}
        for c in [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            _, fi = make_inputs(d, c, 1000 * layer, dev)
            calib[c] = int(route(d, fi)[1].unique().numel())
        corr = min(calib, key=lambda c: (abs(calib[c] - args.target_distinct), c))
        print(f"layer {layer}: distinct experts by corr {calib} -> corr {corr}", flush=True)
        sets = []
        for k in range(args.sets):
            stream, fi = make_inputs(d, corr, 1000 * layer + k, dev)
            logits, ids, sc, gap = route(d, fi)
            xe = expert_input(d, fi)
            sets.append(dict(stream=stream, ffn_in=fi, ids=ids, scores=sc, gap=gap, xe=xe))
        experts = sorted(set(int(e) for s in sets for e in s["ids"].view(-1).tolist()))
        lay.load_ep1(experts)
        ep4 = [lay.ep4_stack(r) for r in range(TP)]
        res = dict(corr=corr, calib=calib, distinct=[int(s["ids"].unique().numel()) for s in sets],
                   ep4_local_per_rank=[[int(((s["ids"].unique() // 64) == r).sum()) for r in range(TP)] for s in sets],
                   min_route_gap=[float(s["gap"].min()) for s in sets])
        cosd = {k: [] for k in ("stream_ep1_vs_stock", "next_ep1_vs_stock", "moe_ep1_vs_stock", "branch_ep1_vs_stock",
                                "stream_ep1_vs_ref", "stream_stock_vs_ref", "moe_ep1_vs_ref", "moe_stock_vs_ref",
                                "y_route_ep1_vs_ref", "y_route_stock_vs_ref")}
        keeps = []
        for k, s in enumerate(sets):
            xe, ids, sc = s["xe"], s["ids"], s["scores"]
            # stock: served Triton EP4 per rank, summed (all-reduce)
            acc = torch.zeros(ROWS, MH, dtype=torch.float32, device=dev)
            for r in range(TP):
                w13, s13, w2, s2 = ep4[r]
                o = base.muse_moe_triton(xe, sc, ids, w13, w2, 64 * r, d["post_w"], 1e-8, post_norm_gain=1.0,
                                         has_norm=True, w13_scale=s13, w2_scale=s2)
                acc += o.float()
            moe_stock = acc.to(torch.bfloat16)
            # ep1
            keep = []
            y_ep1, _ = ep1_experts(lay, xe, ids, sc, (l1, l2), align256, keep, act=args.act)
            # exactness vs a torch model of the same quantised arithmetic (rank 0 stage 1 and stage 2)
            if k == 0:
                deq = dequant_a8 if a8 else dequant_a4
                xq_, xs_, sid_, eid_, np_, inter_, iq_, isc_, y_ = keep[0]
                xd, idq = deq(xq_, xs_), deq(iq_, isc_)
                e1 = e2 = 0.0
                for s_ in range(slots):
                    e_ = int(ids.view(-1)[s_])
                    w13n, s13n, w2n, s2n = P.slice_expert(*lay.native[e_], 0)
                    gu = xd[s_ // TOPK] @ P.dequant(w13n, s13n).T
                    r1 = (gu[:IP] * torch.sigmoid(gu[:IP]) * gu[IP:]).to(torch.bfloat16).float()
                    e1 = max(e1, float((inter_[s_].float() - r1).norm() / r1.norm()))
                    r2 = (idq[s_] @ P.dequant(w2n, s2n).T).to(torch.bfloat16).float()
                    e2 = max(e2, float((y_[s_].float() - r2).norm() / r2.norm()))
                res["kernel_vs_torch_same_quant_rel_l2"] = dict(stage1=e1, stage2=e2)
                print(f"  FlyDSL {args.act} vs torch model of the same arithmetic (rank 0): stage1 rel-L2 {e1:.2e}, "
                      f"stage2 {e2:.2e}", flush=True)
            keeps.append(keep)
            moe_ep1 = finalize(y_ep1, sc, d["post_w"])
            # reference (float64, A16)
            y_ref = ref_experts(lay, xe, ids)
            scale = (d["post_w"].double() + 1.0)
            yn = y_ref * scale
            inv = torch.rsqrt((yn * yn).mean(-1) + 1e-8)
            moe_ref = ((sc.reshape(-1).double() * inv)[:, None] * yn).view(ROWS, TOPK, MH).sum(1)
            # served-style per-route y for the stock arm (A16 Triton stage outputs) for the per-route cosine
            y_st = torch.zeros(slots, MH, dtype=torch.float32, device=dev)
            for r in range(TP):
                w13, s13, w2, s2 = ep4[r]
                sid, eid, npost = base.moe_align_block_size_small(ids, 64, 64 * r, 16)
                cfg = base._pick_config(ROWS, TOPK, 64, True)
                inter = torch.zeros(slots, I, dtype=torch.bfloat16, device=dev)
                base._launch_gemm(xe, w13, s13, inter, sid, eid, npost, I, MH, slots, TOPK, True, True, True, cfg)
                yy = torch.zeros(slots, MH, dtype=torch.bfloat16, device=dev)
                base._launch_gemm(inter, w2, s2, yy, sid, eid, npost, MH, I, slots, TOPK, False, False, True, cfg)
                loc = ((ids.view(-1) // 64) == r)
                y_st[loc] = yy[loc].float()
            branch_st = F.linear(moe_stock, d["unpool"])
            branch_ep1 = F.linear(moe_ep1, d["unpool"])
            branch_ref = F.linear(moe_ref.to(torch.bfloat16), d["unpool"]).double()  # unpool in bf16 like served
            st_s, nx_s = boundary(d, s["stream"], branch_st)
            st_e, nx_e = boundary(d, s["stream"], branch_ep1)
            st_r, _ = boundary(d, s["stream"].double(), branch_ref)
            cosd["stream_ep1_vs_stock"].append(cos_rows(st_e, st_s))
            cosd["next_ep1_vs_stock"].append(cos_rows(nx_e, nx_s))
            cosd["moe_ep1_vs_stock"].append(cos_rows(moe_ep1, moe_stock))
            cosd["branch_ep1_vs_stock"].append(cos_rows(branch_ep1, branch_st))
            cosd["stream_ep1_vs_ref"].append(cos_rows(st_e, st_r))
            cosd["stream_stock_vs_ref"].append(cos_rows(st_s, st_r))
            cosd["moe_ep1_vs_ref"].append(cos_rows(moe_ep1, moe_ref))
            cosd["moe_stock_vs_ref"].append(cos_rows(moe_stock, moe_ref))
            cosd["y_route_ep1_vs_ref"].append(cos_rows(y_ep1, y_ref))
            cosd["y_route_stock_vs_ref"].append(cos_rows(y_st, y_ref))
        res["cos"] = {k: dict(min=float(torch.cat(v).min()), mean=float(torch.cat(v).mean())) for k, v in cosd.items()}
        for k, v in res["cos"].items():
            print(f"  {k:24s} min {v['min']:.6f} mean {v['mean']:.6f}", flush=True)
        print(f"  distinct {res['distinct']} ep4 local/rank {res['ep4_local_per_rank']} min 8th-9th gap "
              f"{min(res['min_route_gap']):.2e}", flush=True)
        # ---- timing
        if not args.no_time:
            def ep1_rank_fn(stage2=True):
                def fn():
                    for r in range(TP):  # rank r's slice with input set r % sets (distinct bytes per call)
                        xq, xs, sid, eid, npost, inter, iq, isc, y = keeps[r % len(keeps)][r]
                        w13, s13, w2, s2 = lay.ep1[r]
                        feg.run(l1, xq, xs, w13, s13, sid, eid, npost, inter)
                        if stage2:
                            feg.run(l2, iq, isc, w2, s2, sid, eid, npost, y)
                return fn
            t1, _ = mb.time_graph(ep1_rank_fn(False))
            t12, _ = mb.time_graph(ep1_rank_fn(True))
            res["time_ep1_rank_us"] = dict(stage1=t1 / TP, stage2=(t12 - t1) / TP, both=t12 / TP)
            # stock per rank (served Triton, align..finalize), worst rank per set
            st = []
            for r in range(TP):
                w13, s13, w2, s2 = ep4[r]

                def fn(r=r, w13=w13, s13=s13, w2=w2, s2=s2):
                    for s in sets:
                        base.muse_moe_triton(s["xe"], s["scores"], s["ids"], w13, w2, 64 * r, d["post_w"], 1e-8,
                                             post_norm_gain=1.0, has_norm=True, w13_scale=s13, w2_scale=s2)
                us, _ = mb.time_graph(fn)
                st.append(us / len(sets))
            res["time_stock_ep4_rank_us"] = st
            print(f"  time/layer: EP1 FlyDSL per rank stage1 {t1 / TP:.1f} + stage2 {(t12 - t1) / TP:.1f} = "
                  f"{t12 / TP:.1f} us | stock EP4 Triton per rank {['%.1f' % x for x in st]} (worst {max(st):.1f})",
                  flush=True)
        res["seconds"] = time.time() - t0
        out[f"layer{layer}"] = res
        json.dump(out, open(args.json, "w"), indent=1)
        del lay, ep4, keeps
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
