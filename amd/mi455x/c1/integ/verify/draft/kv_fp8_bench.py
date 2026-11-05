"""Microbenchmark + numerics: DFlash draft FP8 KV writes, served ops vs the fused kernels (dflash_kv_fp8.py).

Served shapes, TP4 rank: 5 draft layers, 16 Q heads / 2 KV heads x 128, FP8 e4m3fn KV, draft block 8 rows (bs 1).
  A. draft step, per layer: table_qk_norm_rope_ + set_kv_buffer (k.to(fp8), v.to(fp8), store_cache)
     vs table_qk_norm_rope_kv_ (one launch).
  B. context commit (after verify), per layer: saturate_to_fp8_range x2, .to(fp8) x2, prefix-valid tiled store
     vs CtxKVCommitFP8 (one launch for all 5 layers).
Timing: CUDA graph of all 5 layers, replayed; trials alternate the two arms. Numerics: bytes of every cache row,
and the in-place QKV, compared bitwise.
"""
import os
import statistics
import sys

sys.path.insert(0, "/data/muse455/integ/sglang/python")

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.table_qk_norm_rope import table_qk_norm_rope_
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.quantization.fp8_kernel import saturate_to_fp8_range
from sglang.kernels.ops.speculative.dflash_kv_fp8 import (
    CtxKVCommitFP8,
    _e4m3fn_bits,
    table_qk_norm_rope_kv_,
)
from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_prefix_valid_impl

L, NHQ, NHK, D, T, BLOCK = 5, 16, 2, 128, 8, 8
SLOTS = 300_000
FP8 = torch.float8_e4m3fn
dev = torch.device("cuda", 0)
torch.manual_seed(0)


@triton.jit
def _conv_test_kernel(X, OUT, N, SATURATE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    o = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = o < N
    x = tl.load(X + o, mask=m).to(tl.float32)
    tl.store(OUT + o, _e4m3fn_bits(x, SATURATE).to(tl.uint8), mask=m)


def check_conversion():
    bits = torch.arange(-32768, 32768, dtype=torch.int32, device=dev).to(torch.int16)
    x = bits.view(torch.bfloat16).contiguous()
    res = {}
    for sat in (False, True):
        out = torch.empty(x.numel(), dtype=torch.uint8, device=dev)
        _conv_test_kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), sat, 1024)
        ref = (saturate_to_fp8_range(x, FP8) if sat else x).to(FP8).view(torch.uint8)
        diff = (out != ref).nonzero().flatten()
        nan_in = torch.isnan(x.float())
        res[sat] = (int(diff.numel()), int((~nan_in[diff]).sum()) if diff.numel() else 0)
        if diff.numel():
            i = diff[:5]
            print(f"  saturate={sat}: first diffs x={x[i].float().tolist()} ours={out[i].tolist()} "
                  f"torch={ref[i].tolist()}")
    print(f"[conv] all 65536 bf16 -> e4m3fn: mismatches (all, non-NaN inputs): plain {res[False]}, "
          f"saturating {res[True]}")
    out = torch.empty(x.numel(), dtype=torch.uint8, device=dev)
    _conv_test_kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), True, 1024)
    plain = x.to(FP8).view(torch.uint8)
    d = (out != plain).nonzero().flatten()
    print(f"[conv] saturating kernel vs torch's unclamped .to(float8_e4m3fn): {d.numel()} mismatches"
          + (f", e.g. x={x[d[:5]].float().tolist()} ours={out[d[:5]].tolist()} torch={plain[d[:5]].tolist()}"
             if d.numel() else ""))
    return res


def make_layers():
    layers = []
    cos_sin = torch.randn(1 << 20, D, device=dev, dtype=torch.bfloat16)  # [max_pos, D] table like the rotary cache
    for _ in range(L):
        qkv = (torch.randn(T, (NHQ + 2 * NHK) * D, device=dev) * 3).to(torch.bfloat16)
        qkv[0, -D:] = torch.tensor([500.0, -600.0, 448.0, 470.0] * (D // 4), device=dev)  # V overflow cases
        layers.append(dict(
            qkv0=qkv,
            qkv=qkv.clone(),
            qw=(torch.rand(D, device=dev) + 0.5).to(torch.bfloat16),
            kw=(torch.rand(D, device=dev) + 0.5).to(torch.bfloat16),
            kc=torch.zeros(SLOTS, NHK, D, device=dev, dtype=torch.uint8),
            vc=torch.zeros(SLOTS, NHK, D, device=dev, dtype=torch.uint8),
        ))
    pos = torch.arange(131_072, 131_072 + T, device=dev, dtype=torch.int64)
    loc = torch.arange(123_456, 123_456 + T, device=dev, dtype=torch.int64)
    return layers, cos_sin, pos, loc


def served_step(layers, cos_sin, pos, loc):
    for ly in layers:
        qkv = ly["qkv"]
        table_qk_norm_rope_(qkv, pos, ly["qw"], ly["kw"], cos_sin, NHQ, NHK, D, 1e-5)
        q, k, v = qkv.split([NHQ * D, NHK * D, NHK * D], dim=-1)
        k = k.view(-1, NHK, D)
        v = v.view(-1, NHK, D)
        # MHATokenToKVPool.set_kv_buffer: cast, view as store dtype, _set_kv_buffer_impl -> store_cache
        ck = k.to(FP8).view(torch.uint8)
        cv = v.to(FP8).view(torch.uint8)
        store_cache(ck.view(-1, NHK * D), cv.view(-1, NHK * D), ly["kc"].view(-1, NHK * D),
                    ly["vc"].view(-1, NHK * D), loc, row_bytes=NHK * D, v_row_bytes=NHK * D,
                    size_limit=SLOTS)


def fused_step(layers, cos_sin, pos, loc):
    for ly in layers:
        table_qk_norm_rope_kv_(ly["qkv"], pos, ly["qw"], ly["kw"], cos_sin, loc, ly["kc"], ly["vc"],
                               NHQ, NHK, D, 1e-5)


def make_ctx(commit_len):
    cap = 64
    k = (torch.randn(L, cap, NHK, D, device=dev) * 4).to(torch.bfloat16)
    v = (torch.randn(L, cap, NHK, D, device=dev) * 60).to(torch.bfloat16)  # some beyond +-448
    v[:, 0, 0, :4] = torch.tensor([500.0, -600.0, 448.0, 470.0], device=dev)
    kc = [torch.zeros(SLOTS, NHK, D, device=dev, dtype=torch.uint8) for _ in range(L)]
    vc = [torch.zeros(SLOTS, NHK, D, device=dev, dtype=torch.uint8) for _ in range(L)]
    loc_2d = torch.arange(200_000, 200_000 + BLOCK, device=dev, dtype=torch.int64).view(1, BLOCK)
    lens = torch.tensor([commit_len], device=dev, dtype=torch.int32)
    return dict(k=k[:, :BLOCK], v=v[:, :BLOCK], kc=kc, vc=vc, loc_2d=loc_2d, lens=lens)


def served_commit(c):
    for l in range(L):
        ck = saturate_to_fp8_range(c["k"][l], FP8)
        cv = saturate_to_fp8_range(c["v"][l], FP8)
        ck = ck.to(FP8)
        cv = cv.to(FP8)
        _set_kv_buffer_prefix_valid_impl(ck.contiguous().view(torch.uint8), cv.contiguous().view(torch.uint8),
                                         c["kc"][l], c["vc"][l], c["loc_2d"], c["lens"], row_dim=NHK * D,
                                         store_dtype=torch.uint8)


def graph_of(fn, reps):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    return g


def time_graph(g, reps, iters=50):
    for _ in range(5):
        g.replay()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / iters / reps


def ab(name, fa, fb, reps=20, trials=7):
    ga, gb = graph_of(fa, reps), graph_of(fb, reps)
    ta, tb = [], []
    for i in range(trials):
        for g, t in ((ga, ta), (gb, tb)) if i % 2 == 0 else ((gb, tb), (ga, ta)):
            t.append(time_graph(g, reps))
    ma, mb = statistics.median(ta), statistics.median(tb)
    print(f"[time] {name}: served {ma:6.2f} us  fused {mb:6.2f} us  saved {ma - mb:6.2f} us per draft cycle "
          f"(served trials {min(ta):.2f}-{max(ta):.2f}, fused {min(tb):.2f}-{max(tb):.2f})")
    return ma, mb


def main():
    print(torch.cuda.get_device_properties(0).gcnArchName, torch.__version__, triton.__version__)
    check_conversion()

    # A. draft-step K/V write: numerics
    layers, cos_sin, pos, loc = make_layers()
    served_step(layers, cos_sin, pos, loc)
    ref = [(ly["qkv"].clone(), ly["kc"][loc].clone(), ly["vc"][loc].clone()) for ly in layers]
    for ly in layers:
        ly["qkv"].copy_(ly["qkv0"])
        ly["kc"].zero_()
        ly["vc"].zero_()
    fused_step(layers, cos_sin, pos, loc)
    ok = all(torch.equal(a, ly["qkv"]) and torch.equal(b, ly["kc"][loc]) and torch.equal(c, ly["vc"][loc])
             for (a, b, c), ly in zip(ref, layers))
    nz = sum(int(ly["kc"].count_nonzero() + ly["vc"].count_nonzero()) for ly in layers)
    print(f"[numerics] draft step: qkv + FP8 K/V rows bitwise equal to served: {ok} "
          f"(written bytes outside loc: {nz - sum(int(ly['kc'][loc].count_nonzero() + ly['vc'][loc].count_nonzero()) for ly in layers)})")
    # timing: qkv is updated in place every replay (values drift but stay finite; shapes and work are equal)
    for ly in layers:
        ly["qkv"].copy_(ly["qkv0"])
    ab("draft step, 5 layers: QK-norm/RoPE + FP8 K/V store",
       lambda: served_step(layers, cos_sin, pos, loc), lambda: fused_step(layers, cos_sin, pos, loc))

    # B. context commit: numerics over commit lengths, then timing at the served acceptance (~4 of 8)
    for n in (1, 4, 8):
        a, b = make_ctx(n), make_ctx(n)
        b["k"].copy_(a["k"]); b["v"].copy_(a["v"])
        served_commit(a)
        CtxKVCommitFP8(b["kc"], b["vc"])(b["k"], b["v"], b["loc_2d"], b["lens"])
        eq = all(torch.equal(x, y) for x, y in zip(a["kc"] + a["vc"], b["kc"] + b["vc"]))
        print(f"[numerics] context commit, commit_len {n}: all 5 layers' K/V caches bitwise equal: {eq}")
    c = make_ctx(4)
    commit = CtxKVCommitFP8(c["kc"], c["vc"])
    ab("context commit, 5 layers: FP8 cast + prefix-valid store",
       lambda: served_commit(c), lambda: commit(c["k"], c["v"], c["loc_2d"], c["lens"]))


if __name__ == "__main__":
    main()
