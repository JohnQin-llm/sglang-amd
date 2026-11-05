"""Per-stage tests of the gfx1250 K2 small-GEMM stages on one GPU, TP4 ranks simulated rank by rank.

The stages run in C1's K2 order: o_proj (reduce-scatter) -> AR owner sum -> seam per token -> router (q8 pairs)
and pooler -> route -> [experts, not here] -> unpooler -> gather -> (next layer's K1) seam A.  Inputs are seeded
synthetic activations, weights are layer 31's.  Each stage is checked against a float64 torch reference computed
from the stage's own inputs (so errors do not compound), then timed alone at 8 rows (``bench.py``).

usage: test_stages.py [--stages oproj,owner,seam,router,pool,route,up,gather,seama,seamb] [--no-bench]
       (the stages before the last one named run too: each consumes its predecessor's output)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from gemv import common as C
from gemv import gemv
from gemv import harness as Hn
from gemv import moe_dense as M
from gemv import oproj, quarter_seam, seam, seam_b
from gemv.weights import ROUTER_MULT, load

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
S = 8
TP = C.TP
H = C.H
EPS_ATTN, EPS_FFN, POST_EPS, EPS_QN = 1e-8, 1e-5, 1e-8, 1e-5
EPS = {"eps_attn": EPS_ATTN, "eps_ffn": EPS_FFN, "post_eps": POST_EPS, "eps": EPS_QN}


def sync_check(rig, what):
    torch.cuda.synchronize()
    assert int(rig.err[0]) == 0, f"{what}: a mailbox poll timed out"


def bf16_stats(got, ref64):
    """got bf16, ref float64: exact-match fraction vs bf16(ref), max |diff| in bf16 ulps of bf16(ref), cosine,
    and the cosine bf16(ref) itself reaches (the output rounding's floor)."""
    got = got.reshape(-1)
    ref64 = ref64.reshape(-1)
    rb = ref64.to(torch.bfloat16)
    exact = (got.view(torch.int16) == rb.view(torch.int16)).float().mean().item()
    ulp = torch.where(rb.float().abs() > 0, rb.float().abs(), torch.full_like(rb.float(), 1e-30))
    ulp = 2.0 ** (torch.floor(torch.log2(ulp)) - 7)
    maxulp = ((got.double() - rb.double()).abs() / ulp.double()).max().item()
    g, r = got.double(), ref64
    cos = (g @ r / (g.norm() * r.norm())).item()
    rbd = rb.double()
    floor = (rbd @ r / (rbd.norm() * r.norm())).item()
    return {"exact_vs_bf16_ref": round(exact, 6), "max_ulp": round(maxulp, 3), "cos": cos, "cos_bf16_ref_floor": floor}


def f32_stats(got, ref64):
    got = got.reshape(-1).double()
    ref64 = ref64.reshape(-1)
    rel = ((got - ref64).abs() / ref64.abs().clamp_min(1e-30)).max().item()
    cos = (got @ ref64 / (got.norm() * ref64.norm())).item()
    return {"max_rel": rel, "cos": cos, "max_abs": (got - ref64).abs().max().item()}


def rms_rstd64(x, eps):
    return 1.0 / torch.sqrt((x * x).mean(-1, keepdim=True) + eps)


class State:
    def __init__(self):
        torch.manual_seed(1234)
        self.w = load(31)
        self.rig = Hn.Rig(S)
        self.res = {}
        self.launches = {}
        g = torch.Generator(device="cuda").manual_seed(7)
        self.core = [torch.randn(S, oproj.CORE, generator=g, device="cuda").to(torch.bfloat16) for _ in range(TP)]
        self.stream_a = (torch.randn(S, H, generator=g, device="cuda") * 2.0).contiguous()
        self.fin = (torch.randn(S, M.MH, generator=g, device="cuda")).to(torch.bfloat16)
        self.stream_out = torch.zeros(S, H, device="cuda")
        self.branch_out = torch.zeros(S, H, dtype=torch.bfloat16, device="cuda")
        self.stream_a2 = torch.zeros(S, H, device="cuda")
        self.wo = [self.w["w_o"][:, r * oproj.CORE:(r + 1) * oproj.CORE].contiguous() for r in range(TP)]
        w = self.w
        shuf = gemv.shuffle_rows if gemv.SHUFFLED else (lambda t: t)
        self.wo_k = [shuf(x) for x in self.wo]  # the kernel's byte order; references use the plain tensors
        self.wpool_k = shuf(w["w_pool"])
        self.wunpool_k = shuf(w["w_unpool"])
        self.rig.set(
            alpha_a=w["alpha_a"], beta_a=w["beta_a"], w_fn=w["w_fn"], w_router=w["w_router"], expert_bias=w["bias"],
            w_pool=self.wpool_k, w_unpool=self.wunpool_k, stream_a=self.stream_a, stream_out=self.stream_out,
            branch_out=self.branch_out, w_pf=w["w_pf"], alpha_f=w["alpha_f"], beta_f=w["beta_f"], w_qn=w["w_qn"],
        )  # fmt: skip

    def launch(self, name, body, grid, **kw):
        if name not in self.launches:
            t0 = time.time()
            self.launches[name] = Hn.build(name, body, grid, **kw)
            self.launches[name + "_built"] = time.time() - t0
        return self.launches[name]

    def save(self):
        os.makedirs(OUT, exist_ok=True)
        self.res["variant"] = {
            "interleave": gemv.INTERLEAVE, "shuffled": gemv.SHUFFLED, "weight_cm": C.CM_NT, "xrows_cm": C.CM_XROWS,
            "blocks": C.BLOCKS, "threads": C.THREADS,
        }  # fmt: skip
        with open(os.path.join(OUT, f"test_stages{os.environ.get('K2G_TAG', '')}.json"), "w") as f:
            json.dump(self.res, f, indent=1)


# ---------------------------------------------------------------- stages
def l_oproj(st):
    return st.launch("oproj_deep", lambda c, bid: oproj.stage_oproj_deep(c, bid), C.BLOCKS)


def t_oproj(st):
    rig = st.rig
    lo = l_oproj(st)
    for r in range(TP):
        rig.run(lo, r, core=st.core[r], w_o=st.wo_k[r])
    sync_check(rig, "oproj")
    ref = [(st.core[r].double() @ st.wo[r].double().T) for r in range(TP)]  # [S, H] each
    got = torch.empty(TP, S, H, dtype=torch.bfloat16, device="cuda")
    tags_ok = True
    for o in range(TP):
        v, tg = rig.pairs_value(rig.peer_words(o, "ar"))
        bf = Hn.bf16_of_pairs(v).view(TP, S, H)
        cols = slice(o * oproj.OWNER_COLS, (o + 1) * oproj.OWNER_COLS)
        got[:, :, cols] = bf[:, :, cols]
        tags_ok &= bool((tg.view(TP, S, H // 2)[:, :, o * 1024:(o + 1) * 1024] == 1).all())
    st.ar_partials = got
    r = bf16_stats(got, torch.stack(ref))
    r["tags_ok"] = tags_ok
    st.res["oproj"] = r
    print("oproj", r)


def l_owner(st):
    return st.launch("ar_owner_sum", lambda c, bid: quarter_seam.stage_ar_owner_sum(c, bid), 4 * S)


def t_owner(st):
    rig = st.rig
    lw = l_owner(st)
    for r in range(TP):
        rig.run(lw, r)
    sync_check(rig, "owner")
    part = st.ar_partials.float()
    exp = part[0]
    for src in range(1, TP):
        exp = exp + part[src]
    exp_b = exp.to(torch.bfloat16)
    ok_all = True
    for r in range(TP):
        v, _ = rig.pairs_value(rig.peer_words(r, "xg"))
        xg = Hn.bf16_of_pairs(v).view(S, H)
        ok_all &= bool((xg.view(torch.int16) == exp_b.view(torch.int16)).all())
        xs = Hn.f32_of_pairs(rig.pairs_value(rig.peer_words(r, "xsq"))[0]).view(S, 64)
    st.xg = xg
    sq_ref = (exp_b.double().view(S, 64, 128) ** 2).sum(-1)
    full = sum(st.core[r].double() @ st.wo[r].double().T for r in range(TP))
    res = {
        "xg_bit_exact_vs_rank_order_sum": ok_all,
        "xsq": f32_stats(xs, sq_ref),
        "xg_vs_fp64_oproj": bf16_stats(xg, full),
    }
    st.res["owner"] = res
    print("owner", res)


def seam_ref(x_bf, stream, w):
    x = x_bf.double()
    r1 = rms_rstd64(x, EPS_ATTN)
    n = w["alpha_a"].double() * stream.double() + w["beta_a"].double() * (x * r1)
    r2 = rms_rstd64(n, EPS_FFN)
    y = n * r2 * (w["w_fn"].double() + 1.0)
    return n, y


def l_seam(st):
    return st.launch("seam_token", lambda c, bid: quarter_seam.stage_seam_token(c, bid), S, eps=EPS)


def t_seam(st):
    rig = st.rig
    ls = l_seam(st)
    for r in range(TP):
        rig.run(ls, r)
    sync_check(rig, "seam")
    n_ref, y_ref = seam_ref(st.xg, st.stream_a, st.w)
    xrow = [rig.scratch_words(r, "xrow").view(torch.bfloat16).view(S, H) for r in range(TP)]
    same = all(bool((xrow[r].view(torch.int16) == xrow[0].view(torch.int16)).all()) for r in range(TP))
    flags = all(bool((rig.pairs_value(rig.scratch_words(r, "xrdy"))[1][: 4 * S] == 1).all()) for r in range(TP))
    st.x = xrow[0].clone()
    res = {"stream_out": f32_stats(st.stream_out, n_ref), "x": bf16_stats(st.x, y_ref), "ranks_agree": same,
           "xrdy_set": flags}
    st.res["seam"] = res
    print("seam", res)


def router_body(c, bid):
    ri = (bid % M.R_TASKS) // M.R_PARTS
    rq = bid % M.R_PARTS
    M.router_compute_pair(c, ri, rq, bid // M.R_TASKS, M.router_loads(c, ri, rq))


def l_router(st):
    return st.launch("router_pair", router_body, M.R_TASKS_Q8)


def t_router(st):
    rig = st.rig
    lr = l_router(st)
    for r in range(TP):
        rig.run(lr, r)
    sync_check(rig, "router")
    v, _ = rig.pairs_value(rig.peer_words(0, "rlog_parts"))
    parts = Hn.f32_of_pairs(v).view(TP, M.R_PARTS, S, M.EPR)  # [src][q][t][e]
    logits = parts[:, 0] + parts[:, 1]
    logits = logits + parts[:, 2]
    logits = logits + parts[:, 3]
    logits = logits.permute(1, 0, 2).reshape(S, M.E)
    st.logits = logits.contiguous()
    ref = st.x.double() @ st.w["w_router"].double().T
    st.logits_ref = ref
    agree = all(bool((rig.pairs_value(rig.peer_words(r, "rlog_parts"))[0] == v).all()) for r in range(1, TP))
    res = f32_stats(logits, ref)
    res["max_abs_over_rms"] = res["max_abs"] / ref.pow(2).mean().sqrt().item()
    res["ranks_agree"] = agree
    st.res["router"] = res
    print("router", res)


def pool_body(c, bid):
    pi = bid // M.P_PARTS
    pq = bid % M.P_PARTS
    M.pool_compute(c, pi, pq, M.pool_loads(c, pi, pq))


def l_pool(st):
    return st.launch("pool", pool_body, M.P_TASKS)


def t_pool(st):
    rig = st.rig
    lp = l_pool(st)
    for r in range(TP):
        rig.run(lp, r)
    sync_check(rig, "pool")
    v, tg = rig.pairs_value(rig.peer_words(0, "pool"))
    pool = Hn.bf16_of_pairs(v).view(S, M.MH)
    ref = st.x.double() @ st.w["w_pool"].double().T
    res = bf16_stats(pool, ref)
    res["tags_ok"] = bool((tg == 1).all())
    st.res["pool"] = res
    print("pool", res)


def l_route(st):
    return st.launch("route", lambda c, bid: M.stage_route(c, bid), S, mult=ROUTER_MULT)


def served_routing(logits, bias):
    """The served routing op (``FusedMuseSparkRouting.forward_triton``'s kernel) on ``logits``."""
    from sglang.srt.rocm_muse_spark.fused_muse_spark_routing import fused_muse_spark_routing

    w, i = fused_muse_spark_routing(
        logits.float().contiguous(), bias, 8, 1.0, router_output_multiplier=ROUTER_MULT,
        router_score_normalization="sigmoid", router_score_division_eps=1e-15,
    )  # fmt: skip
    return w.clone(), i.clone()


def served_epilogues(logits, bias):
    """The routing programs of the served 8-row front's epilogues (``router_pooler_topk``) on ``logits`` as their
    partials: ``_joint_epilogue`` (the 8 / 16 / 24-row joint path) and ``_reduce_route`` (the shared path)."""
    from sglang.kernels.ops.moe.router_pooler_topk import _joint_epilogue, _route_preproject_epilogue

    m = logits.shape[0]
    dev = logits.device
    norm = torch.ones(M.MH, dtype=torch.bfloat16, device=dev)
    out = {}
    p = torch.zeros(1, m, 4608, device=dev)
    p[0, :, 4096:4352] = logits
    sc = torch.empty(m, 8, device=dev)
    ids = torch.empty(m, 8, dtype=torch.int32, device=dev)
    y = torch.empty(m, M.MH, dtype=torch.bfloat16, device=dev)
    _joint_epilogue[(2 * m,)](p, bias, norm, sc, ids, y, m, 1, ROUTER_MULT, 1.0, 1e-5, num_warps=4)
    out["joint_epilogue"] = (sc.clone(), ids.clone())
    rp = logits.float().contiguous().view(1, m, 256)
    dp = torch.zeros(8, m, M.MH, device=dev)
    _route_preproject_epilogue[(2 * m,)](rp, dp, bias, norm, sc, ids, y, m, M.MH, 8, 1, ROUTER_MULT, 1.0, 1, 1e-5,
                                         num_warps=4)
    out["reduce_route"] = (sc.clone(), ids.clone())
    return out


def run_route(st, logits):
    """ROUTE of rank 0 for ``logits`` [S, 256] placed in RLOG_PARTS as part 0 (parts 1..3 zero)."""
    rig = st.rig
    lr = l_route(st)
    words = rig.peer_words(0, "rlog_parts").view(TP, M.R_PARTS, S, M.EPR, 2)
    lg = logits.float().view(S, TP, M.EPR).permute(1, 0, 2)
    words[:, 0, :, :, 0] = lg.contiguous().view(torch.int32)
    words[:, 1:, :, :, 0] = 0
    words[..., 1] = 1
    rig.scratch_words(0, "route").zero_()
    rig.run(lr, 0)
    sync_check(rig, "route")
    v, tg = rig.pairs_value(rig.scratch_words(0, "route"))
    v = v.view(S, 2, 8)
    assert bool((tg.view(S, 2, 8) == 1).all())
    return v[:, 1].contiguous().view(torch.float32), v[:, 0].contiguous()


def t_route(st, trials=2000):
    rig = st.rig
    bias = st.w["bias"]
    # (a) the K2 pipeline's own logits (the router stage's parts as produced, summed in-kernel)
    lr = l_route(st)
    rig.scratch_words(0, "route").zero_()
    rig.run(lr, 0)
    sync_check(rig, "route")
    v, _ = rig.pairs_value(rig.scratch_words(0, "route"))
    v = v.view(S, 2, 8)
    kw, ki = v[:, 1].contiguous().view(torch.float32), v[:, 0].contiguous()
    st.route_w, st.route_ids = kw.clone(), ki.clone()
    sw, si = served_routing(st.logits, bias)
    res = {
        "pipeline_ids_equal": bool((ki == si).all()),
        "pipeline_weights_bitequal": bool((kw.view(torch.int32) == sw.view(torch.int32)).all()),
    }
    # vs the fp64 router: top-8 sets equal except near-ties
    _, ri = served_routing(st.logits_ref.float(), bias)
    sc = torch.sigmoid(st.logits_ref * ROUTER_MULT) + bias.double()
    srt = sc.sort(-1, descending=True).values
    gap = srt[:, 7] - srt[:, 8]
    set_eq = [set(ki[t].tolist()) == set(ri[t].tolist()) for t in range(S)]
    two_ulp = 2 * 2.0 ** (torch.floor(torch.log2(srt[:, 7].abs())) - 7)
    res["fp64_router_sets_equal"] = sum(set_eq)
    res["fp64_router_mismatch_rows_are_near_ties"] = all(set_eq[t] or bool(gap[t] <= two_ulp[t]) for t in range(S))
    res["min_gap_8v9"] = gap.min().item()
    # (b) random logits around the real distribution: bit-identical to the served op
    g = torch.Generator(device="cuda").manual_seed(11)
    scale = st.logits.std().item()
    nbad_i = nbad_w = 0
    for k in range(trials // S):
        lg = torch.randn(S, M.E, generator=g, device="cuda") * scale + st.logits.mean()
        if k % 4 == 1:  # exact ties in the biased score
            lg[:, 10] = lg[:, 20]
        if k % 4 == 2:
            lg = (lg * 4).round() / 4
        kw, ki = run_route(st, lg)
        sw, si = served_routing(lg, bias)
        nbad_i += int((ki != si).any(-1).sum())
        nbad_w += int((kw.view(torch.int32) != sw.view(torch.int32)).any(-1).sum())
    res["random_rows"] = (trials // S) * S
    res["random_rows_ids_differ"] = nbad_i
    res["random_rows_weights_differ"] = nbad_w
    # (c) the served 8-row front's own routing epilogues, fed the same logits
    try:
        nbad = {"joint_epilogue": [0, 0], "reduce_route": [0, 0]}
        for k in range(trials // S // 4):
            lg = torch.randn(S, M.E, generator=g, device="cuda") * scale + st.logits.mean()
            if k % 2 == 1:
                lg[:, 30] = lg[:, 3]
            kw, ki = run_route(st, lg)
            for name, (sw, si) in served_epilogues(lg, bias).items():
                nbad[name][0] += int((ki != si).any(-1).sum())
                nbad[name][1] += int((kw.view(torch.int32) != sw.view(torch.int32)).any(-1).sum())
        res["served_epilogue_rows"] = (trials // S // 4) * S
        res["served_epilogue_rows_differ_ids_weights"] = nbad
    except Exception as e:  # noqa: BLE001
        res["served_epilogue_error"] = repr(e)
    # (d) the whole served front (joint packed router/pooler GEMM + epilogue) on the pipeline's x
    try:
        from sglang.kernels.ops.moe.router_pooler_topk import joint_router_pooler_topk, pack_joint_router_pooler_weight

        packed = pack_joint_router_pooler_weight(st.w["w_router"].T.contiguous(), st.w["w_pool"])
        fw, fi, _ = joint_router_pooler_topk(st.x, packed, bias, st.w["w_pre"], ROUTER_MULT, 1.0, 1e-5)
        same = [set(st.route_ids[t].tolist()) == set(fi[t].tolist()) for t in range(S)]
        res["served_front_sets_equal_rows"] = sum(same)
        res["served_front_mismatch_rows_are_near_ties"] = all(same[t] or bool(gap[t] <= two_ulp[t]) for t in range(S))
        res["served_front_weight_max_abs_diff"] = (st.route_w - fw).abs().max().item()
    except Exception as e:  # noqa: BLE001
        res["served_front_error"] = repr(e)
    st.res["route"] = res
    print("route", res)


def up_body(c, bid):
    M.stage_up(c, bid, M.up_loads(c, bid))


def l_up(st):
    return st.launch("up", up_body, M.UP_TASKS)


def t_up(st):
    rig = st.rig
    lu = l_up(st)
    for r in range(TP):
        rig.scratch_words(r, "fin").copy_(st.fin.view(-1).view(torch.int32))
        fr = rig.scratch_words(r, "frdy").view(-1, 2)
        fr[:, 0] = 1
        fr[:, 1] = 1
    for r in range(TP):
        rig.run(lu, r)
    sync_check(rig, "up")
    v, tg = rig.pairs_value(rig.peer_words(0, "br"))
    br = Hn.bf16_of_pairs(v).view(S, H)
    ref = st.fin.double() @ st.w["w_unpool"].double().T
    st.br = br.clone()
    res = bf16_stats(br, ref)
    res["tags_ok"] = bool((tg == 1).all())
    st.res["up"] = res
    print("up", res)


def l_gather(st):
    return st.launch("gather", lambda c, bid: M.stage_gather(c, bid), M.GATHER_TASKS)


def t_gather(st):
    rig = st.rig
    lg = l_gather(st)
    ok = True
    for r in range(TP):
        st.branch_out.zero_()
        rig.run(lg, r)
        sync_check(rig, "gather")
        ok &= bool((st.branch_out.view(torch.int16) == st.br.view(torch.int16)).all())
    st.res["gather"] = {"bit_exact": ok}
    print("gather", ok)


def l_seama(st):
    return st.launch("seam_a", lambda c, bid: seam.stage_seam(c, bid), seam.NS, eps=EPS)


def t_seama(st):
    rig = st.rig
    la = l_seama(st)
    branch = st.br
    stream = st.stream_out.clone()
    rig.run(la, 0, branch_in=branch, stream_in=stream, stream_a=st.stream_a2)
    sync_check(rig, "seama")
    w = st.w
    xh = branch.double() * (w["w_pf"].double() + 1.0)
    sa = w["alpha_f"].double() * stream.double() + w["beta_f"].double() * xh * rms_rstd64(xh, POST_EPS)
    y = sa * rms_rstd64(sa, EPS_QN) * (w["w_qn"].double() + 1.0)
    x = rig.scratch_words(0, "xrow").view(torch.bfloat16).view(S, H)
    res = {"stream_a": f32_stats(st.stream_a2, sa), "attn_in": bf16_stats(x, y)}
    st.res["seama"] = res
    print("seama", res)


def l_seamb(st):
    return st.launch("seam_b", lambda c, bid: seam_b.stage_seam(c, bid), seam_b.NSL, eps=EPS, quarter=False)


def l_oproj_all(st):
    return st.launch("oproj_all", lambda c, bid: oproj.stage_oproj(c, bid), C.BLOCKS, quarter=False)


def t_seamb(st):
    """Default-build path: push-all o_proj, then seam B per slice (needs every rank's partials)."""
    rig = st.rig
    rig.zero()
    lo, ls = l_oproj_all(st), l_seamb(st)
    for r in range(TP):
        rig.run(lo, r, core=st.core[r], w_o=st.wo_k[r])
    for r in range(TP):
        rig.run(ls, r)
    sync_check(rig, "seamb")
    part = Hn.bf16_of_pairs(rig.pairs_value(rig.peer_words(0, "ar"))[0]).view(TP, S, H).float()
    x = part[0]
    for src in range(1, TP):
        x = x + part[src]
    n_ref, y_ref = seam_ref(x.to(torch.bfloat16), st.stream_a, st.w)
    xr = rig.scratch_words(0, "xrow").view(torch.bfloat16).view(S, H)
    res = {"stream_out": f32_stats(st.stream_out, n_ref), "x": bf16_stats(xr, y_ref)}
    st.res["seamb"] = res
    print("seamb", res)


ORDER = ["oproj", "owner", "seam", "router", "pool", "route", "up", "gather", "seama", "seamb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(ORDER))
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--trials", type=int, default=2000)
    a = ap.parse_args()
    want = a.stages.split(",")
    st = State()
    chain = [w for w in want if w in ORDER and w != "seamb"]
    last = max((ORDER.index(w) for w in chain), default=-1)
    for name in ORDER[: last + 1]:
        fn = globals()["t_" + name]
        if name == "route":
            fn(st, a.trials)
        else:
            fn(st)
        st.save()
    if not a.no_bench:
        from gemv.bench import bench_all

        bench_all(st, want)
        st.save()
    if "seamb" in want:  # zeroes the rig: after the benches
        t_seamb(st)
        st.save()


if __name__ == "__main__":
    main()
