"""DFlash2 draft microbenchmarks at the served shapes (TP4 rank, gfx1250), graph replay.

  fc : the context projection fc [8192, 40960] (replicated, served) vs this rank's [2048, 40960] row shard
       (SGLANG_MUSE_DFLASH_FC_TP_SHARD) on the same tuned kernel; bitwise check of the shard's columns.
       The all-gather that follows the shard is timed separately (4 ranks) by bench_ag.py.
Cold = rotating weight copies larger than the device cache.
"""
import sys

sys.path.insert(0, "/data/muse455/draft2/sglang/python")
sys.path.insert(0, "/data/muse455/draft2/bench")
import argparse
import json

import torch

from bench_dense import bytes_diff, graph_time  # noqa: E402


def bench_fc(args, res):
    from sglang.kernels.ops.gemm import gfx1250_decode_gemm as g

    dev = torch.device("cuda:0")
    N, K, TP = 8192, 40960, 4
    nsets = args.cold_sets if args.cold else 1
    torch.manual_seed(0)
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(nsets)]
    shards = [w[N // TP * 1 : N // TP * 2].clone() for w in ws]  # rank 1's rows
    for m in args.m:
        x = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
        full = g.mm(x, ws[0])
        part = g.mm(x, shards[0])
        assert full is not None and part is not None
        diff = bytes_diff(full[:, N // TP : 2 * N // TP], part)
        ncall = max(nsets, 8)
        arms = {
            "replicated": [lambda i=i: g.mm(x, ws[i % nsets]) for i in range(ncall)],
            "shard": [lambda i=i: g.mm(x, shards[i % nsets]) for i in range(ncall)],
        }
        out = {k: [] for k in arms}
        for t in range(args.trials):
            for k in (list(arms) if t % 2 == 0 else list(arms)[::-1]):
                out[k].append(graph_time(arms[k], reps=20))
        key = f"fc_m{m}_{'cold' if args.cold else 'warm'}"
        res[key] = {k: sorted(v)[len(v) // 2] for k, v in out.items()}
        res[key]["bytes_differing_shard_vs_replicated_cols"] = diff
        print(key, json.dumps(res[key]), flush=True)


def bench_fcsweep(args, res):
    """Faster (split-K, not bitwise) kernels for the [2048, 40960] shard, cold weights."""
    from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16 as flydsl_mm
    from sglang.kernels.ops.gemm import gfx1250_decode_gemm as g

    dev = torch.device("cuda:0")
    N, K = 2048, 40960
    torch.manual_seed(0)
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(4)]
    x = torch.randn(8, K, device=dev, dtype=torch.bfloat16)
    ref = x.float() @ ws[0].float().t()
    cfgs = {"gluon16x512_bitwise": None}
    for tn, tk, nw, sk in ((32, 256, 2, 4), (32, 256, 2, 8), (32, 128, 2, 8), (64, 128, 4, 4), (32, 256, 2, 16),
                           (16, 256, 1, 8), (16, 512, 1, 4)):
        cfgs[f"flydsl_n{tn}_k{tk}_w{nw}_sk{sk}"] = dict(tile_m=16, tile_n=tn, tile_k=tk, m_warp=1, n_warp=nw,
                                                        num_buffers=3, split_k=sk)
    for name, cfg in cfgs.items():
        try:
            if cfg is None:
                f = lambda i: g.mm(x, ws[i % 4])
            else:
                f = lambda i, cfg=cfg: flydsl_mm(x, ws[i % 4], None, torch.bfloat16, None, **cfg)
            y = f(0).float()
            err = float((y - ref).abs().max())
            us = sorted(graph_time([lambda i=i: f(i) for i in range(8)], reps=20) for _ in range(3))[1]
        except Exception as ex:
            print(name, "failed", str(ex)[:120], flush=True)
            continue
        res[name] = {"us_cold": us, "max_abs_err_vs_fp32": err}
        print(name, json.dumps(res[name]), flush=True)


GEMM_SHAPES = {"conv_proj": (2048, 8192), "qkv": (2560, 8192), "o_proj": (8192, 2048)}


def bench_gemm(args, res):
    """Draft small-M GEMMs at M = 8: the served table pick vs FlyDSL split-K / gluon variants, warm and cold."""
    from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16 as flydsl_mm
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as gluon_mm
    from sglang.kernels.ops.gemm import gfx1250_decode_gemm as g

    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    for name, (N, K) in GEMM_SHAPES.items():
        if args.shape and name not in args.shape:
            continue
        ncold = max(2, (400 << 20) // (N * K * 2) + 1)
        ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(ncold)]
        x = torch.randn(args.m[0], K, device=dev, dtype=torch.bfloat16)
        ref = x.float() @ ws[0].float().t()
        served_y = g.mm(x, ws[0])
        cfgs = {"served": None}
        for tn in (16, 32, 64):
            for tk in (128, 256, 512):
                for sk in (2, 4, 8, 16, 32):
                    if K // sk < tk or (K // sk) % tk:
                        continue
                    for nw in ((1, 2) if tn <= 32 else (2, 4)):
                        cfgs[f"fly_n{tn}_k{tk}_sk{sk}_w{nw}"] = ("fly", dict(tile_m=16, tile_n=tn, tile_k=tk, m_warp=1,
                                                                             n_warp=nw, num_buffers=3, split_k=sk))
        for bn in (16, 32, 64, 128):
            for bk in (128, 256, 512):
                for nw in (1, 2, 4):
                    cfgs[f"gluon_n{bn}_k{bk}_w{nw}"] = ("gluon", dict(BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, NUM_BUFFERS=4,
                                                                      kernel_type="bandwidth_bound", num_warps=nw))
        rows = []
        for cname, cfg in cfgs.items():
            if cfg is None:
                f = lambda i: g.mm(x, ws[i % ncold])
            elif cfg[0] == "fly":
                f = lambda i, c=cfg[1]: flydsl_mm(x, ws[i % ncold], None, torch.bfloat16, None, **c)
            else:
                f = lambda i, c=cfg[1]: gluon_mm(x, ws[i % ncold], None, torch.bfloat16, None, config=dict(c),
                                                 backend="gluon")
            try:
                y = f(0)
                torch.cuda.synchronize()
                err = float((y.float() - ref).abs().max())
                bit = bytes_diff(y, served_y)
                cold = graph_time([lambda i=i: f(i) for i in range(2 * ncold)], reps=15)
                warm = graph_time([lambda: f(0)] * 16, reps=15)
            except Exception as ex:
                continue
            rows.append((cold, warm, cname, err, bit))
        rows.sort()
        served_row = [r for r in rows if r[2] == "served"]
        res[name] = {"served": served_row[0][:2] if served_row else None,
                     "top": [dict(cfg=r[2], cold=round(r[0], 2), warm=round(r[1], 2), max_err=r[3], bytes_vs_served=r[4])
                             for r in rows[:8]]}
        print(name, json.dumps(res[name]), flush=True)


def bench_prefetch(args, res):
    """Can a weight touched by another kernel (or a side-stream prefetch beside a latency-bound predecessor)
    make the served draft GEMM run warm? Spans in graph replay, us per (step) over rotating cold weights."""
    import triton
    import triton.language as tl
    from sglang.kernels.ops.gemm import gfx1250_decode_gemm as g

    @triton.jit
    def touch(w_ptr, out_ptr, n_per_prog, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        base = pid.to(tl.int64) * n_per_prog
        acc = tl.zeros([BLOCK], dtype=tl.int32)
        for i in range(0, n_per_prog, BLOCK * STRIDE):
            offs = base + i + tl.arange(0, BLOCK) * STRIDE
            acc += tl.load(w_ptr + offs, mask=offs < base + n_per_prog, other=0)
        tl.store(out_ptr + pid, tl.sum(acc, axis=0))

    @triton.jit
    def spin(out_ptr, n_iter):
        # latency-bound stand-in for the draft attention / norm before a GEMM: one dependent chain per program
        pid = tl.program_id(0)
        x = pid
        for _ in range(n_iter):
            x = tl.load(out_ptr + (x % 64)) + 1
        tl.store(out_ptr + 64 + pid, x)

    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    sink = torch.zeros(4096, device=dev, dtype=torch.int32)
    spin_buf = torch.zeros(256, device=dev, dtype=torch.int32)
    side = torch.cuda.Stream()
    for name, (N, K) in GEMM_SHAPES.items():
        ncold = max(2, (400 << 20) // (N * K * 2) + 1)
        ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(ncold)]
        wi = [w.view(torch.int32).view(-1) for w in ws]
        x = torch.randn(8, K, device=dev, dtype=torch.bfloat16)
        nel = wi[0].numel()
        progs = 1024
        per = nel // progs

        def t_(i):
            touch[(progs,)](wi[i % ncold], sink, per, STRIDE=32, BLOCK=256, num_warps=8)  # one dword per 128 B line

        def sp():
            spin[(16,)](spin_buf, args.spin_iters, num_warps=1)

        def gm(i):
            g.mm(x, ws[i % ncold])

        def fork_join(i):
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                t_(i)
            sp()
            cur.wait_stream(side)
            gm(i)

        n = 2 * ncold
        arms = {
            "gemm_cold": [lambda i=i: gm(i) for i in range(n)],
            "gemm_warm": [lambda: gm(0)] * n,
            "touch_only": [lambda i=i: t_(i) for i in range(n)],
            "touch_then_gemm": [lambda i=i: (t_(i), gm(i)) for i in range(n)],
            "spin_only": [sp] * n,
            "spin_then_gemm_cold": [lambda i=i: (sp(), gm(i)) for i in range(n)],
            "spin||touch_then_gemm": [lambda i=i: fork_join(i) for i in range(n)],
        }
        out = {k: [] for k in arms}
        for t in range(args.trials):
            for k in (list(arms) if t % 2 == 0 else list(arms)[::-1]):
                out[k].append(graph_time(arms[k], reps=15))
        r = {k: round(sorted(v)[len(v) // 2], 2) for k, v in out.items()}
        r["gemm_after_touch"] = round(r["touch_then_gemm"] - r["touch_only"], 2)
        res[name] = r
        print(name, json.dumps(r), flush=True)


class _Src:
    def __init__(self, ids, rows):
        self.ids, self.row_ids, self.row_stride, self.entry_page_size = ids, rows, ids.stride(0), 1


def bench_meta(args, res):
    """Served _update_target_verify_buffers sequence (same torch ops + gather kernel) vs dflash_verify_meta_."""
    from sglang.kernels.ops.attention.kv_indices import create_flashinfer_kv_indices_triton
    from sglang.kernels.ops.speculative.dflash_verify_meta import dflash_verify_meta_

    dev = torch.device("cuda:0")
    W, T, WIDTH, POOL = 2048, 8, 1 << 20, 1 << 22
    torch.manual_seed(0)
    r2t = torch.randint(0, POOL, (16, WIDTH), device=dev, dtype=torch.int32)
    mapping = torch.randint(-1, POOL, (POOL,), device=dev, dtype=torch.int64)

    def bufs(bs):
        z = lambda n, dt: torch.full((n,), -7, device=dev, dtype=dt)
        return dict(qo=z(65, torch.int64), kv=z(65, torch.int32), kvi=z(64 * 400000, torch.int64),
                    wkv=z(65, torch.int32), wkvi=z(64 * W, torch.int64), woff=z(64, torch.int64),
                    mask=z(65, torch.int64))

    def served(b, seq, rows, bs, mp):
        b["qo"][: bs + 1] = torch.arange(0, (1 + bs) * T, step=T, dtype=torch.int32, device=dev)
        kv = b["kv"][: bs + 1]
        kv[1:] = torch.cumsum(seq, dim=0)
        create_flashinfer_kv_indices_triton[(bs, 1)](r2t, rows[:bs], seq, kv, None, b["kvi"], r2t.stride(0),
                                                     ENTRY_PAGE_SIZE=1, TOKEN_BLOCK_PARALLEL=False,
                                                     token_mapping=None)
        wl = torch.minimum(seq, torch.tensor(W))
        b["wkv"][1 : bs + 1] = torch.cumsum(wl, dim=0)
        wkv = b["wkv"][: bs + 1]
        start = seq - wl
        create_flashinfer_kv_indices_triton[(bs, 1)](r2t, rows[:bs], wl, wkv, start, b["wkvi"], r2t.stride(0),
                                                     ENTRY_PAGE_SIZE=1, TOKEN_BLOCK_PARALLEL=False,
                                                     token_mapping=mp)
        b["woff"][:bs] = start
        sm = T * (seq + T)
        b["mask"][1 : bs + 1] = torch.cumsum(sm, dim=0)

    def fused(b, seq, rows, bs, mp, full):
        dflash_verify_meta_(seq_lens=seq, tokens_per_req=T, window=W, full_src=_Src(r2t, rows),
                            win_src=_Src(r2t, rows), win_mapping=mp, qo_indptr=b["qo"], kv_indptr=b["kv"],
                            kv_indices=b["kvi"], win_indptr=b["wkv"], win_indices=b["wkvi"], win_offsets=b["woff"],
                            mask_indptr=b["mask"], gather_full=full)

    for bs in (1, 4):
        for ctx in args.ctx:
            for use_map in (False, True):
                mp = mapping if use_map else None
                seq = torch.tensor([ctx + T - 37 * i for i in range(bs)], device=dev, dtype=torch.int64)
                rows = torch.randperm(16, device=dev)[:bs].to(torch.int64)
                a, c, d = bufs(bs), bufs(bs), bufs(bs)
                served(a, seq, rows, bs, mp)
                fused(c, seq, rows, bs, mp, True)
                fused(d, seq, rows, bs, mp, False)
                torch.cuda.synchronize()
                diff = {k: bytes_diff(a[k], c[k]) for k in a}
                diff_skip = {k: bytes_diff(a[k], d[k]) for k in a if k != "kvi"}
                arms = {"served": lambda: served(a, seq, rows, bs, mp),
                        "fused": lambda: fused(c, seq, rows, bs, mp, True),
                        "fused_skip_full": lambda: fused(d, seq, rows, bs, mp, False)}
                out = {k: [] for k in arms}
                for t in range(args.trials):
                    for k in (list(arms) if t % 2 == 0 else list(arms)[::-1]):
                        out[k].append(graph_time([arms[k]] * 8, reps=20))
                key = f"meta_bs{bs}_{ctx // 1024}K_map{int(use_map)}"
                res[key] = {k: sorted(v)[len(v) // 2] for k, v in out.items()}
                res[key]["bytes_differing"] = sum(diff.values())
                res[key]["bytes_differing_skip_full_excl_kv_indices"] = sum(diff_skip.values())
                print(key, json.dumps(res[key]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fc", "fcsweep", "meta", "gemm", "prefetch"])
    ap.add_argument("--spin-iters", type=int, default=12)
    ap.add_argument("--shape", nargs="*", default=None)
    ap.add_argument("--ctx", type=int, nargs="+", default=[32768, 131072, 348160])
    ap.add_argument("--m", type=int, nargs="+", default=[8])
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--cold-sets", type=int, default=4)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    {"fc": bench_fc, "fcsweep": bench_fcsweep, "meta": bench_meta, "gemm": bench_gemm, "prefetch": bench_prefetch}[a.what](a, res)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
