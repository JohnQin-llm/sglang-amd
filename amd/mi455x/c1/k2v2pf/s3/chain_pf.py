"""Chain over rotated layer copies: the served per-layer sequence, with and without K2's next-weight prefetch.

Per layer slot i (copy c = i % copies; every copy has its own weights / KV pool / gate at other addresses, same
contents, so nothing stays cached across a cycle of ``--cold`` copies):

  kernel_unified_attention_3d -> reduce_segments -> _rms_norm_gate_fwd_kernel -> K2 (k2v2pf build) ->
  _fused_norm_residual_triton_kernel -> qkv_proj_qk_norm_rope_cache (layer L + 1's QKV of copy c)

Arms differ only in K2's build key (``pf_at`` / ``pf_cpol`` / ``pf_chunk`` / ``pf_rot``) and the prefetch regions
passed to it: ``a`` = this slot's QKV weight (75.5 MB, consumed by the qkv launch two launches later), ``b`` =
the next slot's K2 front weights (o_proj slice + router + pooler, one contiguous buffer a copy). ``stock`` is the
k2v2_land1 program (no prefetch code).

Timing: HIP graph of one epoch advance + ``--reps`` layer slots, replayed ``--iters`` times; per rank the median,
reported = slowest rank; ``--trials`` trials, arm order alternating. Bitwise: every arm's eager golden vs stock's
(K2 stream_out / branch_out, routes, new stream, attn_in, q / k / v / gate, both FP8 caches), graph replays vs
golden.

  /data/muse455/k2/k2safe.sh -t 1500 -w 'python3 /data/muse455/k2/cand/k2v2_poll/s3/mpol_exec.py' \
      /data/muse455/k2/cand/k2v2pf s3/chain_pf.py --layer 31 --ctx 131072 > .../logs/chain.log 2>&1
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TREE = os.path.abspath((os.environ.get("PYTHONPATH") or "/data/muse455/k2/cand/k2v2pf").split(":")[0])
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TREE)
sys.path.insert(0, HERE)
sys.path.insert(0, "/data/muse455/k2/cand/k2lm/snap/k1att")  # k1att.common (fixtures, snapshot copy)
OUT = os.path.join(TREE, "results")
from layer_common import ROWS, HQ, HK, D, EPS_POST_FFN, EPS_RMS, Tail, config, served_ops  # noqa: E402

EPS_O = 1e-5
QKV_CHUNK = 8192  # 128 B lines of one _dense_partial N tile (64 rows x 8192 K bf16 = 1 MB)

# arm -> (K2Build kwargs, prefetch a (QKV), prefetch b (next K2 front))
ARMS = {
    "stock": (dict(), False, False),
    "fin": (dict(pf_at="fin"), True, False),
    "finx": (dict(pf_at="fin", pf_chunk=QKV_CHUNK), True, False),
    "finx_rot4": (dict(pf_at="fin", pf_chunk=QKV_CHUNK, pf_rot=4), True, False),
    "finx_sys": (dict(pf_at="fin", pf_chunk=QKV_CHUNK, pf_cpol=24), True, False),
    "endx": (dict(pf_at="end", pf_chunk=QKV_CHUNK), True, False),
    "routex": (dict(pf_at="route", pf_chunk=QKV_CHUNK), True, False),
    "finx_k2": (dict(pf_at="fin", pf_chunk=QKV_CHUNK), True, True),
    "endx_k2": (dict(pf_at="end", pf_chunk=QKV_CHUNK), True, True),
    "fin_k2only": (dict(pf_at="fin"), False, True),
    "finx_wo": (dict(pf_at="fin", pf_chunk=QKV_CHUNK, pf_bchunk=512), True, "w_o"),
    "fint": (dict(pf_at="fin", pf_touch=1), True, False),
    "endt": (dict(pf_at="end", pf_touch=1), True, False),
    "routet": (dict(pf_at="route", pf_touch=1), True, False),
    "fintx": (dict(pf_at="fin", pf_touch=2, pf_chunk=QKV_CHUNK), True, False),
    "rt_finx": (dict(pf_at="fin", pf_touch=2, pf_tat="route", pf_chunk=QKV_CHUNK), True, False),
    "rt_endx": (dict(pf_at="end", pf_touch=2, pf_tat="route", pf_chunk=QKV_CHUNK), True, False),
    "rt_finx5": (dict(pf_at="fin", pf_touch=2, pf_tat="route", pf_chunk=QKV_CHUNK, pf_nj=5), True, False),
    "rt_finx3": (dict(pf_at="fin", pf_touch=2, pf_tat="route", pf_chunk=QKV_CHUNK, pf_nj=3), True, False),
    "rt_fin24x": (dict(pf_at="fin", pf_touch=2, pf_tat="route", pf_chunk=QKV_CHUNK, pf_ctas=24), True, False),
    "rt_finx_nt": (dict(pf_at="fin", pf_touch=2, pf_tat="route", pf_chunk=QKV_CHUNK, pf_cpol=17), True, False),
    "rt_finx_nt_wo": (dict(pf_at="fin", pf_touch=2, pf_tat="route", pf_chunk=QKV_CHUNK, pf_cpol=17, pf_bchunk=512,
                           pf_btouch=32), True, "w_o"),
    "rt_split4": (dict(pf_at="fin", pf_touch=2, pf_tat="route", pf_chunk=QKV_CHUNK, pf_cpol=17, pf_split=4), True, False),
    "rt_split6": (dict(pf_at="fin", pf_touch=2, pf_tat="route", pf_chunk=QKV_CHUNK, pf_cpol=17, pf_split=6), True, False),
    "rt_routex": (dict(pf_at="route", pf_touch=2, pf_tat="route", pf_chunk=QKV_CHUNK), True, False),
    # diagnostics (an extra launch, s3/pfqkv.hip): every QKV program's own weight lines, after K2 / before K2
    "kpf": (dict(), False, False),
    "kld": (dict(), False, False),
    "kpf_pre": (dict(), False, False),
    "kld_row": (dict(), False, False),
    "kld_64k": (dict(), False, False),
    "kld_tile": (dict(), False, False),
    "ksld_row": (dict(), False, False),
    "kld_pre": (dict(), False, False),
    "kld_row_pre": (dict(), False, False),
    "ktp": (dict(), False, False),
    "kttp": (dict(), False, False),
    "endx_wo": (dict(pf_at="end", pf_chunk=QKV_CHUNK, pf_bchunk=512), True, "w_o"),
}


POST = {"kpf": 0, "kld": 1, "kld_row": 2, "kld_64k": 3, "kld_tile": 4, "ksld_row": 5,
        "ktp": (2, 0), "kttp": (4, 0)}  # pfqkv mode(s) after K2
PRE = {"kpf_pre": 0, "kld_pre": 1, "kld_row_pre": 2}  # ... before K2 (after the gated norm)


class PfQkv:
    """s3/pfqkv.co through hipModuleLaunchKernel on the current stream (captured into the graph like any launch)."""

    def __init__(self):
        import ctypes

        lib = None
        for line in open("/proc/self/maps"):
            if "libamdhip64.so" in line:
                lib = line.split()[-1]
                break
        self.h = ctypes.CDLL(lib)
        self.ct = ctypes
        mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        assert self.h.hipModuleLoad(ctypes.byref(mod), os.path.join(HERE, "pfqkv.co").encode()) == 0
        assert self.h.hipModuleGetFunction(ctypes.byref(fn), mod, b"pf_qkv") == 0
        self.fn = fn
        self.sink = torch.zeros(16, dtype=torch.int32, device="cuda")

    def __call__(self, w, mode, stream):
        ct = self.ct
        args = [ct.c_void_p(w.data_ptr()), ct.c_int(w.shape[1]), ct.c_int(64), ct.c_int(8), ct.c_int(mode),
                ct.c_void_p(self.sink.data_ptr())]
        arr = (ct.c_void_p * len(args))(*[ct.cast(ct.pointer(a), ct.c_void_p) for a in args])
        grid = (w.shape[0] // 64) * 8
        rc = self.h.hipModuleLaunchKernel(self.fn, grid, 1, 1, 256, 1, 1, 0, ct.c_void_p(stream), arr, None)
        assert rc == 0, rc


def capture_attention(fx, seqused, dev):
    """The served verify call's 3D + reduce launch arguments (aiter unified_attention, both stubbed once)."""
    import aiter.ops.triton.attention.unified_attention as UA
    from k1att import common as KC

    got = {}
    o3, orr = UA._unified_attention_3d_triton, UA._reduce_segments_triton
    UA._unified_attention_3d_triton = lambda *a, **k: got.setdefault("3d", (a, k))
    UA._reduce_segments_triton = lambda *a, **k: got.setdefault("red", (a, k))
    out = torch.empty(ROWS, HQ, D, dtype=torch.bfloat16, device=dev)
    try:
        KC.native_call(fx, out, seqused)
    finally:
        UA._unified_attention_3d_triton, UA._reduce_segments_triton = o3, orr
    assert "3d" in got and "red" in got, f"served verify call did not take the 3D split + Triton reduce: {list(got)}"
    return dict(a3=got["3d"], ar=got["red"], f3=o3, fr=orr, out=out)


def worker(rank, args, _):
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("K2G_GPUS", "0,1,2,3")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"file:///tmp/k2pf_pg_{args.tag}", rank=rank, world_size=4)
    from k1att import common as KC
    from port import consts as K
    from port import k2
    from port import validation as V
    from port.sync.debug import given_up_waits
    from port.sync.epoch import advance_epoch_
    from port.sync.peer import PeerRuntime
    import run_k2 as H
    from sglang.srt.rocm_muse_spark.fused_norm_gate import (
        FusedRMSNormGate, _as_grouped_gate_view, _rms_norm_gate_config, _rms_norm_gate_fwd_kernel)
    from sglang.srt.rocm_muse_spark.layers import MuseSpark1xRMSNorm

    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    cfg = config()
    qkv_op, fnr_kernel, (fnr_nw, fnr_ns, fnr_extra) = served_ops()
    keys = {a: k2.K2Build(**ARMS[a][0]) for a in args.arms}
    t0 = time.time()
    for kk in keys.values():
        k2.build(kk)
    rt = PeerRuntime(k2.peer_bytes(), dist.group.WORLD, rank, K.TP, dev, blocks=K.BLOCKS)
    scratch = torch.zeros(k2.scratch_bytes() // 4, dtype=torch.int32, device=dev)
    slay = k2.scratch_layout()
    empty_f32 = torch.empty(0, dtype=torch.float32, device=dev)
    ng_op = FusedRMSNormGate(MuseSpark1xRMSNorm(D, eps=EPS_O, has_weight=False), True, D)
    log(f"K2 builds {list(keys)} in {time.time() - t0:.0f} s")

    def gather(x):
        got = [None] * 4
        dist.all_gather_object(got, x)
        return got

    def health(what):
        torch.cuda.synchronize()
        waits = given_up_waits(rt.health.cpu(), k2.REGIONS)
        failed = False
        try:
            rt.check()
        except RuntimeError as ex:
            failed = str(ex)[:200]
        got = gather((len(waits), waits[:3], failed))
        bad = [g for g in got if g[0] or g[2]]
        if bad:
            log(f"HEALTH {what}: {got}")
            raise RuntimeError(f"gave up at {what}: {bad[:2]}")

    def graph_time(body, reps, iters):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            advance_epoch_(rt.epoch)
            for i in range(reps):
                body(1 + i)
        torch.cuda.current_stream().wait_stream(side)
        health("graph warm-up")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            advance_epoch_(rt.epoch)
            for i in range(reps):
                body(1 + i)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            dist.barrier()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1000 / reps)
        health("graph replays")
        times.sort()
        med = times[len(times) // 2]
        ranks = gather(med)
        del g
        return {"rank_median_us": ranks, "slowest_rank_median_us": max(ranks), "min_us": times[0],
                "p90_us": times[(9 * len(times)) // 10]}

    def run_case(layer, ctx, mode):
        tc = time.time()
        kind = "full" if cfg["layer_types"][layer] == "full_attention" else "swa"
        lay = V.LayerData(layer, dev)
        meta = [None]
        if rank == 0:
            corr, nd = V.calibrate(lay, args.target_distinct, 0)
            case0 = V.make_case(lay, corr, 0, dev=dev)
            meta = [(corr, nd)]
            stream_a = case0.stream_a.cpu()
        else:
            stream_a = torch.empty(ROWS, K.H, dtype=torch.float32)
        dist.broadcast_object_list(meta, src=0)
        dist.broadcast(stream_a, src=0)
        stream_a = stream_a.to(dev)
        ncopy = args.cold if mode == "cold" else 1
        fxs, caps = [], []
        for ci in range(ncopy):
            if ci == 0:
                fx = KC.make_fixture(kind, ctx, rank + 1, dev=dev)
            else:  # the same request and cache contents at other addresses
                fx = dict(fxs[0], k=fxs[0]["k"].clone(), v=fxs[0]["v"].clone(), table=fxs[0]["table"].clone())
            fx["seqused"] = (fx["seq_lens"] + ROWS).to(torch.int32)
            fxs.append(fx)
            caps.append(capture_attention(fx, fx["seqused"], dev))
        a3, k3 = caps[0]["a3"]
        nseg, tile = int(a3[4]), int(a3[5])
        g = torch.Generator().manual_seed(7919 * (layer + 1) + 13 * rank)
        gates = [torch.randn(ROWS, HQ * D, generator=g).to(torch.bfloat16).to(dev)]
        gates += [gates[0].clone() for _ in range(ncopy - 1)]
        y = torch.empty(ROWS, HQ * D, dtype=torch.bfloat16, device=dev)

        def attn3d(i):
            a, k_ = caps[i % ncopy]["a3"]
            caps[0]["f3"](*a, **k_)

        def reduce(i):
            a, k_ = caps[i % ncopy]["ar"]
            caps[0]["fr"](*a, **k_)

        gv0 = _as_grouped_gate_view(caps[0]["out"].view(ROWS, HQ * D), gates[0], D)
        ng_strides = (gv0.stride(0), gv0.stride(1), gv0.shape[-1] // D)
        ng_cfg = _rms_norm_gate_config(ROWS * HQ, HQ, D, gv0.stride(0), torch.bfloat16, True, True, False, dev.index)

        def normgate(i):
            x = caps[i % ncopy]["out"].view(ROWS, HQ * D)
            bm, nw, extra = ng_cfg
            s0, s1, hpg = ng_strides
            n = ROWS * HQ
            _rms_norm_gate_fwd_kernel[((n + bm - 1) // bm,)](
                X=x.view(n, D), G=_as_grouped_gate_view(x, gates[i % ncopy], D), Y=y.view(n, D), W=None,
                eps=ng_op._eps, N=n, H=HQ, stride_g0=s0, stride_g1=s1, HEADS_PER_GROUP=hpg, D=D, BD=D,
                BLOCK_M=bm, HAS_WEIGHT=False, HAS_NORM=True, HAS_GATE=True, num_warps=nw, **extra)

        attn3d(0)
        reduce(0)
        normgate(0)
        torch.cuda.synchronize()
        y_ref = ng_op.forward_triton(caps[0]["out"].view(ROWS, HQ * D), gates[0])
        assert torch.equal(y_ref, y), "direct norm-gate launch != FusedRMSNormGate.forward_triton"
        full = torch.cat([t_.to(torch.bfloat16) for t_ in gather(y.cpu())], 1)
        case = V.Case(full.to(dev), stream_a)
        experts = torch.unique(V.seam_and_route(lay, case)["ids"]).tolist()
        assert all(e == experts for e in gather(experts)), "ranks disagree on the routed experts"
        ops0 = H.k2_operands(lay, rank, experts, dev)
        tail = Tail(lay, layer, rank, dev, cfg, 0, False, pos_base=ctx)

        # K2 operand copies; the routing-independent front weights (o_proj slice, router, pooler) of a copy in
        # one contiguous buffer (prefetch region b)
        front = ("w_o", "w_router", "w_pool")
        opss, fronts = [], []
        for ci in range(ncopy):
            nb = [(ops0[n].numel() * ops0[n].element_size() + 4095) // 4096 * 4096 for n in front]
            buf = torch.empty(sum(nb), dtype=torch.uint8, device=dev)
            o = dict(ops0) if ci == 0 else {n: (v.clone() if torch.is_tensor(v) else v) for n, v in ops0.items()}
            off = 0
            for n, b_ in zip(front, nb):
                src = ops0[n]
                assert src.is_contiguous(), n
                view = buf[off:off + src.numel() * src.element_size()].view(src.dtype).view(src.shape)
                view.copy_(src)
                o[n] = view
                off += b_
            opss.append(o)
            fronts.append(buf)
        del ops0
        k2_bytes = {n: opss[0][n].numel() * opss[0][n].element_size() for n in opss[0] if torch.is_tensor(opss[0][n])}
        log(f"L{layer} ctx {ctx} {mode}: setup {time.time() - tc:.0f} s, {kind}, nseg {nseg} tile {tile}, distinct "
            f"experts {len(experts)} (corr {meta[0][0]:.4f}), copies {ncopy}, front {fronts[0].numel() / 2**20:.1f} MB, "
            f"routed expert bytes {len(experts) * (k2_bytes['w13'] + k2_bytes['w13s'] + k2_bytes['w2'] + k2_bytes['w2s']) / 256 / 2**20:.0f} MB")

        wcopies = [dict(w_qkv=tail.w_qkv, w_post=tail.w_post, w_pre=tail.w_pre, alpha=tail.alpha, beta=tail.beta)]
        for _ in range(ncopy - 1):
            wcopies.append({k_: v_.clone() for k_, v_ in wcopies[0].items()})
        qkv_bytes = tail.w_qkv.numel() * 2

        so_s = torch.empty(ROWS, K.H, dtype=torch.float32, device=dev)
        br_s = torch.empty(ROWS, K.H, dtype=torch.bfloat16, device=dev)
        attn_s = torch.empty(ROWS, K.H, dtype=torch.bfloat16, device=dev)
        stream_s = torch.empty(ROWS, K.H, dtype=torch.float32, device=dev)
        kc_s, vc_s = tail.new_caches()
        ts = [dict(o, core=y, stream_a=stream_a, stream_out=so_s, branch_out=br_s) for o in opss]
        qkv_out = {}

        def k2_call(arm, slot):
            _, pa, pb = ARMS[arm]
            c = slot % ncopy
            if pb == "w_o":  # the next slot's o_proj slice (XCD-matched whole 64 KB tiles)
                wo = opss[(slot + 1) % ncopy]["w_o"]
                pbr = (wo, wo.numel() * wo.element_size())
            else:
                pbr = (fronts[(slot + 1) % ncopy], fronts[0].numel()) if pb else (0, 0)
            pf = ((wcopies[c]["w_qkv"], qkv_bytes) if pa else (0, 0), pbr)
            k2.k2_launch(keys[arm], ts[c], rank, slot, rt.epoch, rt.peers, scratch, rt.health, rt.alarm,
                         torch.cuda.current_stream().cuda_stream, pf=pf)

        def boundary(slot):
            w_ = wcopies[slot % ncopy]
            fnr_kernel[(ROWS,)](
                br_s, so_s, attn_s, stream_s, empty_f32, w_["w_post"], w_["w_pre"], w_["alpha"], w_["beta"],
                EPS_POST_FFN, EPS_RMS, K.H, 8192, True, True, True, True, True, False, False, 0,
                num_warps=fnr_nw, num_stages=fnr_ns, **fnr_extra)

        def qkv(slot):
            q, k_, v, gt = qkv_op(attn_s, wcopies[slot % ncopy]["w_qkv"], tail.positions, tail.cos_sin, kc_s, vc_s,
                                  tail.locations, tail.mapping, EPS_RMS, tail.scale_q, tail.k_scale, tail.v_scale,
                                  num_q_heads=HQ, num_kv_heads=HK)
            qkv_out.update(q=q, k=k_, v=v, g=gt)

        pfq = PfQkv() if any(a in PRE or a in POST for a in args.arms) else None

        def body_of(arm, parts="all"):
            def body(slot):
                st = torch.cuda.current_stream().cuda_stream
                if parts in ("all", "attn_k2"):
                    attn3d(slot)
                    reduce(slot)
                    normgate(slot)
                if arm in PRE:
                    pfq(wcopies[slot % ncopy]["w_qkv"], PRE[arm], st)
                k2_call(arm, slot)
                if arm in POST:
                    for md in (POST[arm] if isinstance(POST[arm], tuple) else (POST[arm],)):
                        pfq(wcopies[slot % ncopy]["w_qkv"], md, st)
                if parts in ("all", "k2_tail"):
                    boundary(slot)
                    qkv(slot)
            return body

        def route_ids():
            return H.route_ids(scratch, slay["route"], K.MAX_TOKENS, K.TOPK)

        def golden(arm):
            dist.barrier()
            advance_epoch_(rt.epoch)
            for s_ in range(ncopy):
                body_of(arm)(s_)
            ids = route_ids()
            health(f"{arm} golden")
            return dict(so=so_s.clone(), br=br_s.clone(), ids=ids, stream=stream_s.clone(), attn=attn_s.clone(),
                        q=qkv_out["q"].clone(), k=qkv_out["k"].clone(), v=qkv_out["v"].clone(),
                        g=qkv_out["g"].clone(), kc=kc_s.view(torch.uint8).clone(), vc=vc_s.view(torch.uint8).clone())

        golds = {a: golden(a) for a in args.arms}
        bitwise = {a: {n: int((golds["stock"][n].contiguous().view(torch.uint8)
                               != golds[a][n].contiguous().view(torch.uint8)).sum()) for n in golds["stock"]}
                   for a in args.arms if a != "stock"}
        allb = gather(bitwise)
        ok = all(all(all(v == 0 for v in d.values()) for d in b.values()) for b in allb)
        log(f"L{layer} ctx {ctx} {mode}: bitwise vs stock, all ranks, all arms: {ok}; rank 0 {bitwise}")

        times = {a: [] for a in args.arms}
        for trial in range(1, args.trials + 1):
            order = args.arms if trial % 2 else args.arms[::-1]
            for a in order:
                times[a].append(graph_time(body_of(a), args.reps, args.iters))
            log(f"L{layer} ctx {ctx} {mode} trial {trial}: "
                + ", ".join(f"{a} {times[a][-1]['slowest_rank_median_us']:.2f}" for a in args.arms) + " us/layer")
        rep_ok = {}
        for a in args.arms:  # graph replays vs golden (the last arm timed left its outputs; re-check every arm)
            graph_time(body_of(a), ncopy, 1)
            rep_ok[a] = all(torch.equal(x_, golds[a][n_]) for n_, x_ in (("so", so_s), ("br", br_s),
                                                                        ("stream", stream_s), ("q", qkv_out["q"]),
                                                                        ("g", qkv_out["g"])))
        rep = gather(rep_ok)
        parts = {}
        if args.parts:
            for a in args.parts_arms:
                for p_ in ("attn_k2", "k2_tail", "k2"):
                    parts[f"{a}:{p_}"] = graph_time(body_of(a, p_), args.reps, args.iters)["slowest_rank_median_us"]
            log(f"L{layer} ctx {ctx} {mode} partial chains (us/slot): { {k_: round(v_, 2) for k_, v_ in parts.items()} }")
        med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
        doc = {"layer": layer, "ctx": ctx, "mode": mode, "kind": kind, "copies": ncopy,
               "distinct_experts": len(experts), "qkv_bytes": qkv_bytes, "front_bytes": fronts[0].numel(),
               "bitwise_all_ranks": ok, "bytes_differ_per_rank": allb,
               "replay_matches_golden": all(all(r.values()) for r in rep),
               "us": {a: [t["slowest_rank_median_us"] for t in times[a]] for a in args.arms},
               "median_us": {a: med([t["slowest_rank_median_us"] for t in times[a]]) for a in args.arms},
               "trials": times, "partial_chains_us": parts}
        doc["delta_vs_stock_us"] = {a: doc["median_us"][a] - doc["median_us"]["stock"] for a in args.arms}
        log(f"L{layer} ctx {ctx} {mode} median us/layer: {doc['median_us']}; delta vs stock {doc['delta_vs_stock_us']}; "
            f"replays match {doc['replay_matches_golden']}")
        del wcopies, fxs, caps, opss, fronts, ts
        torch.cuda.empty_cache()
        return doc

    for mode in args.modes:
        d = run_case(args.layer, args.ctx, mode)
        if rank == 0:
            os.makedirs(OUT, exist_ok=True)
            path = os.path.join(OUT, f"chain_L{args.layer}_c{args.ctx}_{mode}{('_' + args.out_tag) if args.out_tag else ''}.json")
            json.dump(dict(d, args=vars(args), env={k: v for k, v in os.environ.items() if k.startswith("K2_")}),
                      open(path, "w"), indent=1, default=str)
            log(f"wrote {path}")
    rt.close()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--modes", default="cold,warm")
    ap.add_argument("--arms", default="stock,fin,finx,endx,finx_k2")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--cold", type=int, default=8)
    ap.add_argument("--parts", action="store_true", help="partial chains (attn..K2, K2..qkv, K2 alone) a parts arm")
    ap.add_argument("--parts-arms", default="stock,finx")
    ap.add_argument("--target-distinct", type=int, default=43)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    args.modes = args.modes.split(",")
    args.arms = args.arms.split(",")
    args.parts_arms = args.parts_arms.split(",")
    assert args.arms[0] == "stock"
    args.tag = f"{int(time.time())}_{os.getpid()}"
    mp.spawn(worker, args=(args, None), nprocs=4, join=True)


if __name__ == "__main__":
    main()
