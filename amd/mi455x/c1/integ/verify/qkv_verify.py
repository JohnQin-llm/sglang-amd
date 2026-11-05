"""Change 2 inside the integrated tree: fused QKV + gate projection (TP4, N 4608, K 8192), flags off / on.

Imports sglang from the worktree (PYTHONPATH) and, for the byte reference, the served tree's
qk_norm_rope_cache.py loaded from /data/muse455/sglang (read only). Fixtures and timing follow
vagm/gemm_bench.py: cold = one graph over 62 distinct weight copies / 62, warm = one copy 62 times; medians of
alternating trials. Numerics: bytes differing vs served in Q, K, V, gate, K cache, V cache, at several row counts,
seeds and with / without the SWA slot mapping.
"""
import argparse
import importlib.util
import json
import os
import statistics as st
import sys

os.environ["TRITON_CACHE_DIR"] = "/data/muse455/integ/verify/tcache"
import torch  # noqa: E402

import sglang  # noqa: E402

assert sglang.__file__.startswith("/data/muse455/integ/sglang/"), sglang.__file__
from sglang.kernels.ops.attention import qk_norm_rope_cache as Q  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "served_qk_norm_rope_cache", "/data/muse455/sglang/python/sglang/kernels/ops/attention/qk_norm_rope_cache.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

N, K, HQ, HK, D = 4608, 8192, 32, 4, 64


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
    return g


def replay_us(g):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


VARIANTS = {  # name -> (QKV_TDM, QKV_SPLIT_EPI, gluon available)
    "flags_off": (False, False, True),
    "split_epi": (False, True, True),
    "tdm": (True, False, True),
    "tdm+split_epi": (True, True, True),
    "triton_st4+split_epi": (True, True, False),
}


def set_variant(name):
    tdm, split, gluon = VARIANTS[name]
    Q.QKV_TDM, Q.QKV_SPLIT_EPI = tdm, split
    Q._tdm_kernel = None if gluon else False


def fixture(m, seed, dev, mapping):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(m, K, generator=g, device=dev).to(torch.bfloat16)
    w = (torch.randn(N, K, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    pos = torch.randint(0, 260000, (m,), generator=g, device=dev, dtype=torch.int64)
    ang = torch.arange(262144, device=dev, dtype=torch.float32)[:, None] / (
        10000 ** (torch.arange(0, D, 2, device=dev, dtype=torch.float32) / D))[None, :]
    cs = torch.cat([ang.cos(), ang.sin()], 1).contiguous()
    loc = torch.randperm(8192, generator=g, device=dev)[:m].to(torch.int64)
    if m > 2:
        loc[1] = -1  # padded row: no cache write
    mp = None
    if mapping:
        mp = torch.randperm(8192, generator=g, device=dev).to(torch.int64)
        mp[loc[0]] = -1  # unmapped SWA slot
    return x, w, pos, cs, loc, mp


def call(mod, fx, kc, vc, ks, vs):
    x, w, pos, cs, loc, mp = fx
    return mod.qkv_proj_qk_norm_rope_cache(x, w, pos, cs, kc, vc, loc, mp, 1e-6, 0.125, ks, vs,
                                           num_q_heads=HQ, num_kv_heads=HK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--layers", type=int, default=62)
    ap.add_argument("--out", default="/data/muse455/integ/verify/results/qkv_verify.json")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    ks = torch.tensor([0.05], device=dev)
    vs = torch.tensor([0.07], device=dev)
    res = {"numerics": {}, "timing": {}}

    # numerics: every variant vs the served module, bytes of all six outputs
    for m in (1, 3, 8, 16):
        for seed in (0, 1):
            for mapping in (False, True):
                fx = fixture(m, seed, dev, mapping)
                kc0 = torch.randn(8192, HK, D, device=dev).to(torch.float8_e4m3fn)
                vc0 = torch.randn(8192, HK, D, device=dev).to(torch.float8_e4m3fn)
                kc, vc = kc0.clone(), vc0.clone()
                ref = [t.clone() for t in call(S, fx, kc, vc, ks, vs)] + [kc.view(torch.uint8), vc.view(torch.uint8)]
                row = {}
                for n in VARIANTS:
                    set_variant(n)
                    kc, vc = kc0.clone(), vc0.clone()
                    got = list(call(Q, fx, kc, vc, ks, vs)) + [kc.view(torch.uint8), vc.view(torch.uint8)]
                    torch.cuda.synchronize()
                    row[n] = [int((g_.reshape(-1).view(torch.uint8) != r.reshape(-1).view(torch.uint8)).sum())
                              for g_, r in zip(got, ref)]
                key = f"m{m}_seed{seed}_map{int(mapping)}"
                res["numerics"][key] = row
                print("bytes differing vs served (q, k, v, gate, kcache, vcache)", key, json.dumps(row), flush=True)
    print("live:", sorted(Q._live), flush=True)

    # timing at the C1 verify shape (8 rows)
    m, L = 8, a.layers
    x, _, pos, cs, loc, _ = fixture(m, 0, dev, False)
    ws = [(torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16) for _ in range(L)]
    kc = torch.zeros(8192, HK, D, device=dev, dtype=torch.float8_e4m3fn)
    vc = torch.zeros_like(kc)
    names = ["flags_off", "tdm+split_epi", "triton_st4+split_epi", "split_epi"]
    graphs = {}
    for n in names:
        set_variant(n)
        op = (lambda w: Q.qkv_proj_qk_norm_rope_cache(x, w, pos, cs, kc, vc, loc, None, 1e-6, 0.125, ks, vs,
                                                      num_q_heads=HQ, num_kv_heads=HK))
        graphs[n] = {"cold": graph_time(lambda: [op(w) for w in ws], 0),
                     "warm": graph_time(lambda: [op(ws[0]) for _ in range(L)], 0)}
    set_variant("flags_off")
    t = {n: {"cold": [], "warm": []} for n in names}
    for i in range(a.trials):
        for n in (names if i % 2 == 0 else names[::-1]):
            for k_, g in graphs[n].items():
                t[n][k_].append(replay_us(g) / L)
    res["timing"] = {n: {k_: st.median(v) for k_, v in d.items()} for n, d in t.items()}
    for n, r in res["timing"].items():
        print(f"timing {n:22s} cold {r['cold']:6.2f} us  warm {r['warm']:6.2f} us", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
