"""Single-GPU test of the gfx1250 TP4/EP1 MoE stage (``moe.py``): the 4 ranks run one after another, each
with its own EP1 slice, scratch and launches; the YR exchange (row owner = t // 2) and the FIN all-gather go
through peer buffers on the one GPU (launch boundaries stand in for the mailbox polls).

Checks per layer and input set (real MXFP4 weights; the lead's S2 inputs: real router, synthetic draft rows
calibrated to ~43 distinct experts):
  exact (bit for bit, torch ``ref.py`` doing the kernel's arithmetic):
    xq   kernel XQ / XSC bytes vs ref from POOL
    ug   kernel INTER / ISC bytes (each rank) vs ref from the kernel's XQ          (stage-isolated)
    down kernel YR bf16 (each rank) vs ref from the kernel's INTER                (stage-isolated)
    fin  kernel FIN (each rank's copy) vs ref from the kernel's YR                (stage-isolated)
    e2e  kernel FIN vs the torch-only chain POOL -> ... -> FIN (A4W4 / A8W4 reference of the whole block)
  information: cosine of FIN vs the served A16 path (TP4/EP4 Triton ``muse_moe_triton`` per rank, summed)
  and vs a float64 A16 reference.
Timing (graph replay, the 4 ranks' slices back to back so weights stream from HBM; / 4 = per rank).

Run: flock /data/muse455/locks/gpu1.lock docker exec muse_mi455x bash -c 'cd /tmp && PYTHONSAFEPATH=1
     HIP_VISIBLE_DEVICES=1 timeout 300 python3 /data/muse455/k2/cand/k2v2_land1/port/moe/test_moe.py --layers 31'
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, "/data/muse455/k2/cand/k2v2_land1/s2")
sys.path.insert(1, "/data/muse455/k2/cand/k2v2_land1")
sys.path.insert(1, "/data/muse455/k2/port")
sys.path.insert(1, "/data/muse455/k2/quickwin")
import flydsl.expr as fx  # noqa: E402

import moe as M  # noqa: E402
import ref as R  # noqa: E402
import ep1_layout as P  # noqa: E402
import ep1_moe as S2  # noqa: E402
import moe_bench as mb  # noqa: E402

S = M.MAX_TOKENS
TOPK = M.TOPK
TP = M.TP


def _st():
    return fx.Stream(torch.cuda.current_stream().cuda_stream)


class Rank:
    """One emulated rank: its slice (gfx1250 layout), scratch, and the stage launches."""

    def __init__(self, r, slice_, cfg, dev, peers):
        self.r = r
        self.w13, self.s13, self.w2, self.s2 = slice_
        self.cfg = cfg
        act = cfg["act"]
        sp = M.scratch_plain(S, act)
        self.buf = {k: torch.zeros(n, dtype=torch.uint8, device=dev) for k, n in sp.items()}
        self.peers = peers  # shared dict: "yr" (one buffer, owners' rows disjoint), "fin" [TP] (one per rank)

    def ptr(self, k):
        return self.buf[k].data_ptr()

    def xq(self, pool, w_pre):
        c = self.cfg
        M.make_xq(S, c["act"])(pool.data_ptr(), w_pre.data_ptr(), self.ptr("xqb"), self.ptr("xscb"), stream=_st())

    def ug(self, route):
        c = self.cfg
        M.make_ug(S, c["act"], c["ksplit"], c["prefetch"], c["nt"])(
            route.data_ptr(), self.ptr("xqb"), self.ptr("xscb"), self.w13.data_ptr(), self.s13.data_ptr(),
            self.ptr("interb"), self.ptr("iscb"), stream=_st())

    def down(self, route):
        c = self.cfg
        yr = self.peers["yr"].data_ptr()  # every owner's YR region: one buffer here
        M.make_down(S, c["act"], c["dwaves"], c["dprefetch"], c["nt"])(
            route.data_ptr(), self.ptr("interb"), self.ptr("iscb"), self.w2.data_ptr(), self.s2.data_ptr(), self.r,
            yr, yr, yr, yr, stream=_st())

    def fin(self, route, w_post, upto):
        fins = [f.data_ptr() for f in self.peers["fin"]]
        M.make_fin(S, upto)(route.data_ptr(), w_post.data_ptr(), self.peers["yr"].data_ptr(), self.ptr("fmsq"),
                            self.r, *fins, stream=_st())


def run_stage(ranks, pool, w_pre, w_post, route):
    for rk in ranks:  # each rank: xq, ug, down (YR to the owners)
        rk.xq(pool, w_pre)
        rk.ug(route)
        rk.down(route)
    for upto in (1, 2):  # owners: FMSQ, then FIN to every rank
        for rk in ranks:
            rk.fin(route, w_post, upto)


def mism(a, b):
    a, b = a.contiguous().view(-1), b.contiguous().view(-1)
    return int((a != b).sum()), a.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,31,60")
    ap.add_argument("--act", default="a4", choices=M.Q.ACTS)
    ap.add_argument("--sets", type=int, default=2)
    ap.add_argument("--target-distinct", type=int, default=43)
    ap.add_argument("--ksplit", type=int, default=1)
    ap.add_argument("--prefetch", type=int, default=16)
    ap.add_argument("--dwaves", type=int, default=4)
    ap.add_argument("--dprefetch", type=int, default=8)
    ap.add_argument("--nt", type=int, default=1)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-stock", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cfg = dict(act=args.act, ksplit=args.ksplit, prefetch=args.prefetch, dwaves=args.dwaves,
               dprefetch=args.dprefetch, nt=args.nt)
    jpath = args.json or f"{HERE}/test_moe_{args.act}.json"
    out = json.load(open(jpath)) if os.path.exists(jpath) else {}
    out["config"] = vars(args)
    dev = torch.device("cuda")
    base = None if args.no_stock else mb.load_module("mmt_base", "/data/muse455/k2/quickwin/muse_moe_triton_base.py")
    rd = P._Reader()
    for layer in [int(x) for x in args.layers.split(",")]:
        M.check_layer(layer)
        t0 = time.time()
        lay = S2.Layer(layer, dev, rd)
        d = lay.d
        calib = {}
        for cr in [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            _, fi = S2.make_inputs(d, cr, 1000 * layer, dev)
            calib[cr] = int(S2.route(d, fi)[1].unique().numel())
        corr = min(calib, key=lambda cr: (abs(calib[cr] - args.target_distinct), cr))
        sets = []
        for k in range(args.sets):
            stream, fi = S2.make_inputs(d, corr, 1000 * layer + k, dev)
            _, ids, sc, gap = S2.route(d, fi)
            pool = F.linear(fi, d["pool"]).contiguous()
            route = torch.cat([ids.int(), sc.float().view(torch.int32)], 1).contiguous()
            sets.append(dict(stream=stream, ffn_in=fi, ids=ids, scores=sc, pool=pool, route=route))
        experts = sorted(set(int(e) for s in sets for e in s["ids"].view(-1).tolist()))
        lay.load_ep1(experts)
        peers = dict(yr=torch.zeros(TP * S * TOPK * M.MH, dtype=torch.bfloat16, device=dev),
                     fin=[torch.zeros(S, M.MH, dtype=torch.bfloat16, device=dev) for _ in range(TP)])
        ranks = [Rank(r, lay.ep1[r], cfg, dev, peers) for r in range(TP)]
        slices = [{e: P.slice_expert(*lay.native[e], r) for e in experts} for r in range(TP)]
        w_pre = d["pre_w"].contiguous()
        w_post = d["post_w"].contiguous()
        res = dict(corr=corr, distinct=[int(s["ids"].unique().numel()) for s in sets])
        ex = {k: [0, 0] for k in ("xq", "xsc", "inter", "isc", "yr", "fin", "fin_replicas", "e2e_fin")}
        ex_sr = {k: [0, 0] for k in ("inter", "yr", "e2e_fin")}
        cosd = {k: [] for k in ("fin_vs_stock_a16", "fin_vs_ref64_a16", "stock_vs_ref64", "ref_a4w4_vs_stock")}

        def acc(dct, k, m):
            dct[k][0] += m[0]
            dct[k][1] += m[1]

        for si, s in enumerate(sets):
            for f in peers["fin"]:
                f.zero_()
            peers["yr"].zero_()
            run_stage(ranks, s["pool"], w_pre, w_post, s["route"])
            torch.cuda.synchronize()
            ids, sc = s["ids"], s["scores"]
            xe, xq_r, xs_r = R.xq(s["pool"], w_pre, args.act)
            kq = ranks[0].buf["xqb"].view(S, -1)
            ks = ranks[0].buf["xscb"].view(S, -1)
            acc(ex, "xq", mism(kq, xq_r))
            acc(ex, "xsc", mism(ks, xs_r))
            yr_k = peers["yr"].view(TP, S * TOPK, M.MH)
            yr_e2e = []
            for r, rk in enumerate(ranks):
                kiq = rk.buf["interb"].view(S * TOPK, -1)
                kisc = rk.buf["iscb"].view(S * TOPK, -1)
                rkq = rk.buf["xqb"].view(S, -1)
                rks = rk.buf["xscb"].view(S, -1)
                for sr, dct in ((False, ex), (True, ex_sr)):
                    _, iq_r, isc_r = R.ug_rank(rkq, rks, ids, slices[r], args.act, M.RI, args.ksplit, sr)
                    acc(dct, "inter", mism(kiq, iq_r))
                    if not sr:
                        acc(ex, "isc", mism(kisc, isc_r))
                    y_r = R.down_rank(kiq, kisc, ids, slices[r], args.act, M.RI, sr)
                    acc(dct, "yr", mism(yr_k[r].float(), y_r))
                    # e2e: torch only, from POOL
                    _, iq_e, isc_e, y_e = R.experts_rank(xq_r, xs_r, ids, slices[r], args.act, M.RI, args.ksplit, sr)
                    if sr:
                        yr_e2e_sr = yr_e2e_sr + [y_e] if r else [y_e]
                    else:
                        yr_e2e.append(y_e)
            fin_r = R.fin([yr_k[r] for r in range(TP)], sc, w_post)
            acc(ex, "fin", mism(peers["fin"][0].float(), fin_r))
            for p in range(1, TP):
                acc(ex, "fin_replicas", mism(peers["fin"][p], peers["fin"][0]))
            fin_e = R.fin(yr_e2e, sc, w_post)
            acc(ex, "e2e_fin", mism(peers["fin"][0].float(), fin_e))
            acc(ex_sr, "e2e_fin", mism(peers["fin"][0].float(), R.fin(yr_e2e_sr, sc, w_post)))
            fin_k = peers["fin"][0]
            # served A16 (stock) and float64 A16 reference
            xe16 = S2.expert_input(d, s["ffn_in"])
            y_ref = S2.ref_experts(lay, xe16, ids)
            yn = y_ref * (d["post_w"].double() + 1.0)
            inv = torch.rsqrt((yn * yn).mean(-1) + 1e-8)
            moe_ref = ((sc.reshape(-1).double() * inv)[:, None] * yn).view(S, TOPK, M.MH).sum(1)
            cosd["fin_vs_ref64_a16"].append(S2.cos_rows(fin_k, moe_ref))
            if base is not None:
                accs = torch.zeros(S, M.MH, dtype=torch.float32, device=dev)
                for r in range(TP):
                    w13, s13, w2, s2 = lay.ep4_stack(r)
                    o = base.muse_moe_triton(xe16, sc, ids, w13, w2, 64 * r, d["post_w"], 1e-8, post_norm_gain=1.0,
                                             has_norm=True, w13_scale=s13, w2_scale=s2)
                    accs += o.float()
                    del w13, s13, w2, s2
                stock = accs.to(torch.bfloat16)
                cosd["fin_vs_stock_a16"].append(S2.cos_rows(fin_k, stock))
                cosd["stock_vs_ref64"].append(S2.cos_rows(stock, moe_ref))
                cosd["ref_a4w4_vs_stock"].append(S2.cos_rows(fin_e, stock))
        res["exact_mismatch"] = {k: f"{v[0]}/{v[1]}" for k, v in ex.items()}
        res["exact_mismatch_single_round_ref"] = {k: f"{v[0]}/{v[1]}" for k, v in ex_sr.items()}
        res["cos"] = {k: dict(min=float(torch.cat(v).min()), mean=float(torch.cat(v).mean()))
                      for k, v in cosd.items() if v}
        print(f"layer {layer}: distinct {res['distinct']}", flush=True)
        print(f"  exact (mismatching elements / total): {res['exact_mismatch']}", flush=True)
        print(f"  same vs single-rounding chain ref: {res['exact_mismatch_single_round_ref']}", flush=True)
        for k, v in res["cos"].items():
            print(f"  cos {k:20s} min {v['min']:.6f} mean {v['mean']:.6f}", flush=True)
        if not args.no_time:
            s0 = sets[0]

            def fn_part(parts):
                def fn():
                    for k, rk in enumerate(ranks):
                        s = sets[k % len(sets)]
                        if "xq" in parts:
                            rk.xq(s["pool"], w_pre)
                        if "ug" in parts:
                            rk.ug(s["route"])
                        if "down" in parts:
                            rk.down(s["route"])
                    if "fin" in parts:
                        for upto in (1, 2):
                            for rk in ranks:
                                rk.fin(s0["route"], w_post, upto)
                return fn

            tm = {}
            for name, parts in (("ug", ("ug",)), ("ug+down", ("ug", "down")), ("xq", ("xq",)), ("fin", ("fin",)),
                                ("all", ("xq", "ug", "down", "fin"))):
                us, _ = mb.time_graph(fn_part(parts))
                tm[name] = us / TP
            tm["down"] = tm["ug+down"] - tm["ug"]
            res["time_per_rank_us"] = tm
            print("  time per rank (us): " + ", ".join(f"{k} {v:.1f}" for k, v in tm.items()), flush=True)
        res["seconds"] = time.time() - t0
        out[f"layer{layer}"] = res
        json.dump(out, open(jpath, "w"), indent=1)
        del lay, ranks, slices, peers
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
