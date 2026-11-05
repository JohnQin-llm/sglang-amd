"""S3 lead test: the gfx1250 K2 (port/k2.py) on 4 GPUs, one process a rank, real layer weights.

    numerics : one launch a layer; stream_out / branch_out / route ids vs validation.k2ref and stock (the bars)
    timing   : HIP graph of (epoch advance + K2) x R replays, per-layer time = median over replays / R, all ranks

Only the routed experts of the test case are loaded into the EP1 stacks (the others stay zero; K2 reads only
routed experts), so a layer loads in seconds.

Run (needs all four GPUs: take the four locks in order, check that no server / bench is running):
  flock /data/muse455/locks/gpu0.lock flock /data/muse455/locks/gpu1.lock flock /data/muse455/locks/gpu2.lock \
  flock /data/muse455/locks/gpu3.lock docker exec muse_mi455x bash -c 'cd /tmp && PYTHONSAFEPATH=1 \
  PYTHONPATH=/data/muse455/k2/cand/k2v6 timeout 900 python3 /data/muse455/k2/cand/k2v6/s3/run_k2.py --layers 31'
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, "/data/muse455/k2/cand/k2v6")
OUT = "/tmp/k2probe"


def ep1_stacks(lay, rank, experts, dev):
    from port import consts as K
    from port import layout as L
    import ep1_layout as P

    ri = K.RI
    w13 = torch.zeros(K.E, 2 * ri, K.MH // 2, dtype=torch.uint8)
    s13 = torch.zeros(K.E, 2 * ri, K.MH // 32, dtype=torch.uint8)
    w2 = torch.zeros(K.E, K.MH, ri // 2, dtype=torch.uint8)
    s2 = torch.zeros(K.E, K.MH, ri // 32, dtype=torch.uint8)
    for e in experts:
        a, b, c, d = P.slice_expert(*[t.cpu() for t in lay.expert(e)], rank)
        w13[e], s13[e], w2[e], s2[e] = a, b, c, d
    return tuple(t.to(dev).contiguous() for t in L.gfx1250_weights(w13, s13, w2, s2))


def k2_operands(lay, rank, experts, dev):
    from port import consts as K
    from port.gemv import gemv

    shuf = gemv.shuffle_rows if gemv.SHUFFLED else (lambda t: t)
    w = lay.w
    t = {
        "w_o": shuf(w["w_o"][:, rank * K.CORE:(rank + 1) * K.CORE].contiguous()),
        "alpha_a": w["alpha_a"], "beta_a": w["beta_a"], "w_fn": w["w_fn"].contiguous(),
        "w_router": w["w_router"].float().contiguous(), "expert_bias": w["bias"].float().contiguous(),
        "w_pool": shuf(w["w_pool"].contiguous()), "w_pre": w["w_pre"].contiguous(),
        "w_post": w["w_post"].contiguous(), "w_unpool": shuf(w["w_unpool"].contiguous()),
    }  # fmt: skip
    t["w13"], t["w13s"], t["w2"], t["w2s"] = ep1_stacks(lay, rank, experts, dev)
    return t


def route_ids(scratch, lay_route, s, topk):
    off, n = lay_route
    w = scratch.view(torch.int32)[off // 4:(off + n) // 4].view(-1, 2)[:, 0]
    return w.view(s, 2 * topk)[:, :topk].clone()


def worker(rank, args, q):
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("K2G_GPUS", "0,1,2,3")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"file:///tmp/k2g_pg_{args.tag}", rank=rank, world_size=4)
    from port import consts as K
    from port import k2
    from port import validation as V
    from port.sync.epoch import advance_epoch_
    from port.sync.peer import PeerRuntime

    key = k2.K2Build(act=args.act)
    t0 = time.time()
    k2.build(key)
    build_s = time.time() - t0
    rt = PeerRuntime(k2.peer_bytes(), dist.group.WORLD, rank, K.TP, dev, blocks=K.BLOCKS)
    if os.environ.get("K2_SCRATCH_UC") == "1":
        from port.sync.peer import _heap, _DeviceBytes
        _sp = _heap().alloc_uncached(k2.scratch_bytes())
        scratch = torch.as_tensor(_DeviceBytes(_sp, k2.scratch_bytes()), device=dev).view(torch.int32)
        scratch.zero_()
    else:
        scratch = torch.zeros(k2.scratch_bytes() // 4, dtype=torch.int32, device=dev)
    slay = k2.scratch_layout()
    res = {"build_s": build_s}
    for layer in args.layers:
        lay = V.LayerData(layer, dev)
        corr, nd = V.calibrate(lay, args.target_distinct, args.seed)
        case = V.make_case(lay, corr, args.seed, dev=dev)
        ref = V.k2ref(lay, case, args.act) if rank == 0 else None
        ids0 = V.seam_and_route(lay, case)["ids"]
        experts = torch.unique(ids0).tolist()
        ops = k2_operands(lay, rank, experts, dev)
        so = torch.zeros(K.MAX_TOKENS, K.H, dtype=torch.float32, device=dev)
        br = torch.zeros(K.MAX_TOKENS, K.H, dtype=torch.bfloat16, device=dev)
        t = dict(ops, core=case.core_rank(rank), stream_a=case.stream_a.contiguous(), stream_out=so, branch_out=br)
        stream = torch.cuda.current_stream()

        def launch():
            advance_epoch_(rt.epoch)
            k2.k2_launch(key, t, rank, layer, rt.epoch, rt.peers, scratch, rt.health, rt.alarm,
                         torch.cuda.current_stream().cuda_stream)

        def checked(what):
            from port.sync.debug import given_up_waits
            torch.cuda.synchronize()
            waits = given_up_waits(rt.health.cpu(), k2.REGIONS)
            got = [None] * 4
            dist.all_gather_object(got, (int(rt.epoch.item()), len(waits), waits[:64]))
            if rank == 0:
                for r, (ep, n, w) in enumerate(got):
                    print(f"[{what}] rank {r} epoch {ep}: {n} CTAs gave up" + ("".join("\n    " + x for x in w)), flush=True)
            dist.barrier()
            rt.check()

        dist.barrier()
        launch()
        torch.cuda.synchronize()
        dist.barrier()
        checked("first launch")
        row = {"distinct": nd, "corr": corr, "experts_loaded": len(experts)}
        if rank == 0:
            outs = {"stream_out": so.clone(), "branch_out": br.clone(),
                    "ids": route_ids(scratch, slay["route"], K.MAX_TOKENS, K.TOPK)}
            row["check"] = V.check(outs, case, lay, ref)
            print(f"layer {layer}: {json.dumps(row['check'])}", flush=True)
        if args.time_single:
            ts = []
            for _ in range(15):
                dist.barrier(); torch.cuda.synchronize(); dist.barrier()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); launch(); e1.record(); torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) * 1000)
            ts.sort()
            checked("single timed")
            got = [None] * 4
            dist.all_gather_object(got, (ts[len(ts) // 2], ts[0]))
            if rank == 0:
                print(f"SINGLE median/min us (epoch bump + K2, barrier-separated): {got}", flush=True)
            args.eager = 0
        if args.timeline:
            tlb = torch.zeros(K.BLOCKS * k2.TL_WORDS, dtype=torch.int32, device=dev)
            def launch_tl():
                advance_epoch_(rt.epoch)
                k2.k2_launch(key, t, rank, layer, rt.epoch, rt.peers, scratch, rt.health, rt.alarm,
                             torch.cuda.current_stream().cuda_stream, tl=tlb)
            for _ in range(10):
                launch_tl()
            tlb.zero_()
            launch_tl()
            torch.cuda.synchronize()
            checked("timeline")
            import statistics as stt
            T = tlb.view(K.BLOCKS, k2.TL_WORDS).cpu().tolist()
            names = k2.TL_STAGES
            per = {n: [] for n in names}
            total = []
            ex = [[], [], []]
            for cta in range(K.BLOCKS):
                d0, p1, ch = T[cta][8], T[cta][14], T[cta][15]
                if d0 and p1 and ch:
                    ex[0].append(((p1 - d0) % (1 << 31)) / 100); ex[1].append(((ch - p1) % (1 << 31)) / 100)
            for cta in range(K.BLOCKS):
                row = [(names[i], T[cta][i]) for i in range(len(names)) if T[cta][i] != 0]
                for (n, a), (_, b) in zip(row, row[1:]):
                    per[n].append(((b - a) % (1 << 31)) / 100)
                if len(row) > 1:
                    total.append(((row[-1][1] - row[0][1]) % (1 << 31)) / 100)
            summ = {n: (len(v), round(stt.median(v), 1), round(max(v), 1)) for n, v in per.items() if v}
            _ = 0
            ugr = []
            for i in range(6):
                v = [((T[cta][16 + i] - T[cta][7]) % (1 << 31)) / 100 for cta in range(K.BLOCKS) if T[cta][16 + i] and T[cta][7]]
                if v:
                    ugr.append((i, round(stt.median(v), 1), round(max(v), 1)))
            summ["ug_round_end_from_ug_start"] = ugr
            def _gap(i, j):
                v = [((T[cta][j] - T[cta][i]) % (1 << 31)) / 100 for cta in range(K.BLOCKS) if T[cta][i] and T[cta][j]]
                return (len(v), round(stt.median(v), 1), round(max(v), 1)) if v else None
            for nm, i, j in (("acq_down", 8, 24), ("acq_ug", 25, 7), ("poll_down", 24, 14), ("yr_drain", 15, 26)):
                if _gap(i, j):
                    summ[nm] = _gap(i, j)
            if ex[0]:
                summ["down1_wait"] = (len(ex[0]), round(stt.median(ex[0]), 1), round(max(ex[0]), 1))
                summ["down1_chain"] = (len(ex[1]), round(stt.median(ex[1]), 1), round(max(ex[1]), 1))
            got = [None] * 4
            dist.all_gather_object(got, (summ, round(stt.median(total), 1), round(max(total), 1)))
            if rank == 0:
                for r, (sm, tm, tx) in enumerate(got):
                    print(f"TL rank {r}: CTA span median {tm} us max {tx} us; UG round ends (round, median, max us after UG start) {sm.get('ug_round_end_from_ug_start')}", flush=True)
                    for n in names + ["down1_wait", "down1_chain", "acq_down", "acq_ug", "poll_down", "yr_drain"]:
                        if n in sm:
                            print(f"TL rank {r}   {n:11s} CTAs {sm[n][0]:3d}  median {sm[n][1]:8.1f}  max {sm[n][2]:8.1f} us", flush=True)
            args.eager = 0
        # ---- back-to-back eager launches (no host barrier between them)
        side_s = torch.cuda.Stream() if args.side else None
        side_x = torch.empty(args.side_mb * (1 << 20) // 4, dtype=torch.float32, device=dev) if args.side else None
        for _ in range(args.eager):
            if args.side:
                side_s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side_s):
                    for _r in range(args.side_reps):
                        side_x.mul_(1.0000001)  # an independent elementwise kernel on another stream, as in serving
            launch()
            if args.side and args.side_sync:
                torch.cuda.current_stream().wait_stream(side_s)
            if args.sync_between:
                torch.cuda.synchronize()
                h = rt.health.cpu().view(-1, 8)
                want = int(rt.epoch.item())
                eps = h[1:257, 7].tolist()
                bad = [(cta, e) for cta, e in enumerate(eps) if e != want]
                failed = int(h[0, 0].item()) != 0
                if bad or failed:
                    import collections
                    print(f"EPOCH rank {rank} launch epoch {want}: {len(bad)} CTAs read another epoch "
                          f"{dict(collections.Counter(e for _, e in bad))}, CTAs {[c for c, _ in bad][:40]}, failed={failed}",
                          flush=True)
                    hl = h[1:257].tolist()
                    st = [r[5] for r in hl]; t0 = min(st)
                    late_start = sorted(((r[5] - t0) / 100, i) for i, r in enumerate(hl))[-6:]
                    upd = [(i, (r[6] - r[5]) / 100, (r[4] - r[6]) / 100) for i, r in enumerate(hl[:128])]
                    slow = sorted(upd, key=lambda x: -x[2])[:8]
                    print(f"TIMES rank {rank}: start spread (us, latest CTAs) {late_start}; "
                          f"unpool CTA (bid, start->up us, up wait+compute us) slowest {slow}", flush=True)
                    if failed:
                        break
        torch.cuda.synchronize()
        dist.barrier()
        checked(f"{args.eager} eager back-to-back")
        if args.eager:
            row["eager_matches_first"] = bool(torch.equal(so, outs["stream_out"])) if rank == 0 else None
        # ---- probe: repeated launches
        import time as _t
        for mode in ("nobarrier",):
            for i in range(5):
                if mode == "barrier":
                    dist.barrier()
                launch()
                if mode == "barrier":
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
            dist.barrier()
            try:
                rt.check()
                ok = True
            except RuntimeError as ex:
                ok = str(ex)[:60]
            from port.sync.debug import given_up_waits
            from port import k2 as _k2
            torch.cuda.synchronize()
            waits = given_up_waits(rt.health.cpu(), _k2.REGIONS)
            got = [None] * 4
            dist.all_gather_object(got, waits[:6])
            if rank == 0 and any(got):
                for r, w in enumerate(got):
                    print(f"PROBE {mode} rank {r}: " + " | ".join(w), flush=True)
            if rank == 0:
                print(f"PROBE {mode}: ok={ok} same_out={bool(torch.allclose(so, so))}", flush=True)
            if ok is not True:
                break
    rt.close()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="31")
    ap.add_argument("--act", default="a8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-distinct", type=int, default=43)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--eager", type=int, default=4)
    ap.add_argument("--sync-between", action="store_true")
    ap.add_argument("--time-single", action="store_true")
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--side", action="store_true", help="run elementwise kernels on a second stream during each launch")
    ap.add_argument("--side-mb", type=int, default=256)
    ap.add_argument("--side-reps", type=int, default=4)
    ap.add_argument("--side-sync", action="store_true")
    args = ap.parse_args()
    args.layers = [int(x) for x in args.layers.split(",")]
    args.tag = f"{int(time.time())}"
    mp.spawn(worker, args=(args, None), nprocs=4, join=True)


if __name__ == "__main__":
    main()
