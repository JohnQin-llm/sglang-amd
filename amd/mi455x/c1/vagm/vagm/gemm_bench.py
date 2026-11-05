"""Decode GEMMs at M = 8 (gfx1250, TP4 rank shapes), served kernels vs variants, cold and warm, single GPU.

QKV: the served fused op `qkv_proj_qk_norm_rope_cache` = Triton `_dense_partial` (split-K 8, BN 64, BM 16, 4 warps,
2 stages, .cg) -> FP32 partials [8, 8, 4608] -> `_qkv_epilogue` (grid (8, 32), 1 warp: split-K sum, QK norm, RoPE,
FP8 cache write). Variants: partial tiles at split-K 8 (same K slices -> same partial values), split-K 16, the AITER
gfx1250 gluon TDM batched GEMM as partials, and an epilogue copy whose K/V heads run in their own programs.
Other GEMMs: `gfx1250_decode_gemm.mm` (served table) at M = 8 for lm_head and the DFlash draft shapes.

cold: one graph over L distinct weight copies (> 2 GB, so every call streams its weight from HBM), time / L;
warm: the same graph over one weight copy L times. Median of trials. Numerics: bytes differing vs the served op.
"""
import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(ROOT, "tcache"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

sys.path.insert(0, "/data/muse455/decode_stack")
from sglang.kernels.ops.attention import qk_norm_rope_cache as Q  # noqa: E402
from sglang.kernels.ops.gemm import gfx1250_decode_gemm as DG  # noqa: E402

M, N, K = 8, 4608, 8192
HQ, HK, D = 32, 4, 64
HBM = 16.5e12


@triton.jit
def _qkv_epilogue_split(P, POS, COS, LOC, MAP, KC, VC, QO, KO, VO, GO, KS, VS, M: tl.constexpr, N: tl.constexpr,
                        SK: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr,
                        SCALE: tl.constexpr, ROPE: tl.constexpr, HAS_MAP: tl.constexpr, COS_STRIDE: tl.constexpr):
    """`_qkv_epilogue` with the same per-element math; programs head < HQ do gate + q, HQ + kh do k, v, cache."""
    token, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    if head < HQ:
        gate = Q._project_slice(P, token, (HQ + 2 * HK) * D + head * D, M, N, SK, D)
        tl.store(GO + token * HQ * D + head * D + d, gate)
        q = Q._project_slice(P, token, head * D, M, N, SK, D)
        q = Q._norm_rope(q, COS, POS, EPS, SCALE, D, ROPE, True, COS_STRIDE)
        tl.store(QO + token * HQ * D + head * D + d, q)
    else:
        kh = head - HQ
        k = Q._project_slice(P, token, HQ * D + kh * D, M, N, SK, D)
        v = Q._project_slice(P, token, (HQ + HK) * D + kh * D, M, N, SK, D)
        k = Q._norm_rope(k, COS, POS, EPS, 1.0, D, ROPE, False, COS_STRIDE)
        tl.store(KO + token * HK * D + kh * D + d, k)
        tl.store(VO + token * HK * D + kh * D + d, v)
        slot = tl.load(LOC + token).to(tl.int64)
        if slot >= 0:
            if HAS_MAP:
                slot = tl.load(MAP + slot).to(tl.int64)
            if slot >= 0:
                offset = slot * HK * D + kh * D + d
                tl.store(KC + offset, k.to(tl.float32) / tl.load(KS))
                tl.store(VC + offset, v.to(tl.float32) / tl.load(VS))


def graph_time(fn, trials):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(trials):
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    return st.median(ts)


def gluon_partials(cfg):
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (create_shared_layouts,
                                                                                create_wmma_layouts)
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.batched.batched_gemm_bf16 import _KERNEL_MAP
    kern = _KERNEL_MAP["bandwidth_bound"]
    bn, bk, sk, nb, warps, cache = cfg
    split = K // sk
    nb = min(nb, split // bk)
    wmma, opa, opb = create_wmma_layouts(warps)
    sha, shb = create_shared_layouts(16, bn, bk, "TN", 16)

    def launch(x, w, p):
        xq, wq = x.view(1, M, K), w.view(1, N, K).transpose(1, 2)
        pv = p.view(sk, 1, M, N)
        kern[(1, sk * triton.cdiv(M, 16) * (N // bn))](
            xq, wq, pv, None, M, N, K,
            xq.stride(0), xq.stride(1), xq.stride(2), wq.stride(0), wq.stride(1), wq.stride(2),
            pv.stride(1), pv.stride(2), pv.stride(3), pv.stride(0), 0,
            BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, NUM_BUFFERS=nb, LAYOUT="TN", SHARED_LAYOUT_A=sha,
            SHARED_LAYOUT_B=shb, WMMA_LAYOUT=wmma, OPERAND_LAYOUT_A=opa, OPERAND_LAYOUT_B=opb, ADD_BIAS=False,
            NUM_KSPLIT=sk, SPLITK_BLOCK_SIZE=split, num_warps=warps, waves_per_eu=0, cache_modifier=cache)
    return launch


def triton_partials(cfg):
    sk, bn, bm, bk, warps, stages, cache = cfg

    def launch(x, w, p):
        Q._dense_partial[(N // bn, triton.cdiv(M, bm), sk)](x, w, p, M, N, K, bm, bn, bk, sk, cache,
                                                            num_warps=warps, num_stages=stages,
                                                            matrix_instr_nonkdim=16)
    return launch


# name -> (partials kind, cfg, split-K, epilogue)
SERVED_T = (8, 64, 16, 128, 4, 2, ".cg")
QKV_VARIANTS = {
    "served": ("triton", SERVED_T, 8, "served"),
    "t_sk8_st3": ("triton", (8, 64, 16, 128, 4, 3, ".cg"), 8, "served"),
    "t_sk8_bn32": ("triton", (8, 32, 16, 128, 4, 2, ".cg"), 8, "served"),
    "t_sk8_bn32_w2": ("triton", (8, 32, 16, 128, 2, 2, ".cg"), 8, "served"),
    "t_sk8_bn32_st3": ("triton", (8, 32, 16, 128, 4, 3, ".cg"), 8, "served"),
    "t_sk8_bk64_st3": ("triton", (8, 64, 16, 64, 4, 3, ".cg"), 8, "served"),
    "t_sk8_bk256": ("triton", (8, 64, 16, 256, 4, 2, ".cg"), 8, "served"),
    "t_sk8_bn128_w8": ("triton", (8, 128, 16, 128, 8, 2, ".cg"), 8, "served"),
    "t_sk16_st3": ("triton", (16, 64, 16, 128, 4, 3, ".cg"), 16, "served"),
    "g_sk8_bn32": ("gluon", (32, 128, 8, 3, 2, ".cg"), 8, "served"),
    "g_sk8_bn64": ("gluon", (64, 128, 8, 3, 2, ".cg"), 8, "served"),
    "g_sk8_bn32_nb4": ("gluon", (32, 128, 8, 4, 2, ".cg"), 8, "served"),
    "served_epi2": ("triton", SERVED_T, 8, "split"),
    "t_sk8_st3_epi2": ("triton", (8, 64, 16, 128, 4, 3, ".cg"), 8, "split"),
    "g_sk8_bn32_epi2": ("gluon", (32, 128, 8, 3, 2, ".cg"), 8, "split"),
}

OTHER = {  # (N, K, out dtype): served table entry, per cycle count
    "lm_head_fp32": (50512, 8192, torch.float32, 1),
    "draft_logits": (50512, 8192, torch.bfloat16, 1),
    "draft_qkv": (2560, 8192, torch.bfloat16, 5),
    "draft_o": (8192, 2048, torch.bfloat16, 5),
    "draft_gate_up": (14336, 8192, torch.bfloat16, 5),
    "draft_down": (8192, 7168, torch.bfloat16, 5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--layers", type=int, default=62)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--other", type=int, default=1)
    ap.add_argument("--tag", default="g1")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    L = a.layers
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(L)]
    pos = torch.arange(130000, 130000 + M, device=dev, dtype=torch.int64)
    ang = torch.arange(262144, device=dev, dtype=torch.float32)[:, None] / (
        10000 ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))[None, :]
    cs = torch.cat([ang.cos(), ang.sin()], 1).contiguous()
    loc = torch.arange(5000, 5000 + M, device=dev, dtype=torch.int64)
    ks = torch.tensor([0.05], device=dev)
    vs = torch.tensor([0.05], device=dev)
    kc = torch.zeros(8192, HK, D, device=dev, dtype=torch.float8_e4m3fn)
    vc = torch.zeros_like(kc)
    outs = (torch.empty(M, HQ * D, device=dev, dtype=torch.bfloat16), torch.empty(M, HK * D, device=dev, dtype=torch.bfloat16),
            torch.empty(M, HK * D, device=dev, dtype=torch.bfloat16), torch.empty(M, HQ * D, device=dev, dtype=torch.bfloat16))
    parts = {8: torch.empty(8, M, N, device=dev, dtype=torch.float32),
             16: torch.empty(16, M, N, device=dev, dtype=torch.float32)}

    def epi(kind, p, sk):
        qo, ko, vo, go = outs
        args = (p, pos, cs, loc, None, kc, vc, qo, ko, vo, go, ks, vs, M, N, sk, HQ, HK, D, 1e-6, 0.125, True, False,
                cs.stride(0))
        if kind == "served":
            Q._qkv_epilogue[(M, HQ)](*args, num_warps=1)
        else:
            _qkv_epilogue_split[(M, HQ + HK)](*args, num_warps=1)

    res = {"qkv": {}, "other": {}, "floor_qkv_us": N * K * 2 / HBM * 1e6}
    names = list(QKV_VARIANTS) if a.variants == "all" else a.variants.split(",")
    ref = None
    for n in names:
        kind, cfg, sk, ek = QKV_VARIANTS[n]
        try:
            part = triton_partials(cfg) if kind == "triton" else gluon_partials(cfg)
            p = parts[sk]
            if n == "served":
                def op(w, part=part, p=p):
                    Q.qkv_proj_qk_norm_rope_cache(x, w, pos, cs, kc, vc, loc, None, 1e-6, 0.125, ks, vs,
                                                  num_q_heads=HQ, num_kv_heads=HK)
            else:
                def op(w, part=part, p=p, sk=sk, ek=ek):
                    part(x, w, p)
                    epi(ek, p, sk)
            kc.zero_()
            vc.zero_()
            if n == "served":
                y = Q.qkv_proj_qk_norm_rope_cache(x, ws[0], pos, cs, kc, vc, loc, None, 1e-6, 0.125, ks, vs,
                                                  num_q_heads=HQ, num_kv_heads=HK)
                y = [t.clone() for t in y]
            else:
                op(ws[0])
                y = [t.clone() for t in outs]
            torch.cuda.synchronize()
            got = y + [kc[5000:5000 + M].view(torch.uint8).clone(), vc[5000:5000 + M].view(torch.uint8).clone()]
            if ref is None:
                ref = got
            diff = [int((g.reshape(-1).view(torch.uint8) != r.reshape(-1).view(torch.uint8)).sum())
                    for g, r in zip(got, ref)]
            r = {"cold_us": graph_time(lambda: [op(w) for w in ws], a.trials) / L,
                 "warm_us": graph_time(lambda: [op(ws[0]) for _ in range(L)], a.trials) / L,
                 "partials_cold_us": graph_time(lambda: [part(x, w, p) for w in ws], a.trials) / L,
                 "partials_warm_us": graph_time(lambda: [part(x, ws[0], p) for _ in range(L)], a.trials) / L,
                 "epilogue_us": graph_time(lambda: [epi(ek, p, sk) for _ in range(L)], a.trials) / L,
                 "bytes_diff_q_k_v_gate_kc_vc": diff, "cfg": [kind, list(cfg), sk, ek]}
            if kind == "triton":
                r["programs"] = (N // cfg[1]) * triton.cdiv(M, cfg[2]) * sk
            else:
                r["programs"] = sk * (N // cfg[0])
            r["partials_tbs_cold"] = N * K * 2 / r["partials_cold_us"] / 1e6
        except Exception as ex:
            r = {"error": repr(ex)[:300]}
        res["qkv"][n] = r
        print("qkv", n, json.dumps(r), flush=True)

    if a.other:
        for n, (nn, kk, odt, cnt) in OTHER.items():
            copies = max(2, -(-2_500_000_000 // (nn * kk * 2)))
            wl = [torch.randn(nn, kk, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
            xx = torch.randn(M, kk, device=dev, dtype=torch.bfloat16)
            outb = torch.empty(M, nn, device=dev, dtype=odt)
            assert DG.applies(xx, wl[0], odt), n
            y = DG.mm(xx, wl[0], odt, outb)
            refm = xx.float() @ wl[0].float().T
            err = ((y.float() - refm).abs().max() / refm.abs().max()).item()
            r = {"cold_us": graph_time(lambda: [DG.mm(xx, w, odt, outb) for w in wl], a.trials) / copies,
                 "warm_us": graph_time(lambda: [DG.mm(xx, wl[0], odt, outb) for _ in wl], a.trials) / copies,
                 "hipblaslt_cold_us": graph_time(lambda: [torch.mm(xx, w.T, out_dtype=odt) if odt != xx.dtype
                                                          else torch.mm(xx, w.T) for w in wl], a.trials) / copies,
                 "bytes": nn * kk * 2, "floor_us": nn * kk * 2 / HBM * 1e6, "rel_err": err,
                 "entry": [DG._TABLE[(nn, kk, odt)][0], DG._TABLE[(nn, kk, odt)][1]], "per_cycle": cnt}
            r["tbs_cold"] = r["bytes"] / r["cold_us"] / 1e6
            res["other"][n] = r
            print("other", n, json.dumps(r), flush=True)
            del wl
            torch.cuda.empty_cache()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", f"gemm_{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
