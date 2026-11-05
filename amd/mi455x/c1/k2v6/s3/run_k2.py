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
OUT = "/data/muse455/k2/cand/k2v6/s3"


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
            import re

            from port.sync.abi import PEER_HEAD
            from port.sync.debug import given_up_waits

            torch.cuda.synchronize()
            waits = given_up_waits(rt.health.cpu(), k2.REGIONS)
            ep = int(rt.epoch.item())
            info = f"rank {rank} {what}: epoch {ep}, {len(waits)} CTAs gave up"
            if waits:
                info += "; first:\n  " + "\n  ".join(waits[:12])
                play, half = k2.peer_layout(), k2._half()
                words = rt.buffer.bytes.view(torch.int32)
                for w in waits[:6]:
                    m = re.search(r": (\w+) pair (\d+)", w)
                    if m and m.group(1) in play:
                        at = (PEER_HEAD + (ep % 2) * half + play[m.group(1)][0]) // 4 + 2 * int(m.group(2))
                        info += f"\n  now {m.group(1)}[{m.group(2)}] = {words[at:at + 4].tolist()}"
            allinfo = [None] * 4
            dist.all_gather_object(allinfo, info)
            failed = False
            try:
                rt.check()
            except RuntimeError:
                failed = True
            failed_any = [None] * 4
            dist.all_gather_object(failed_any, failed)
            if any(failed_any):
                if rank == 0:
                    print("\n".join(allinfo), flush=True)
                raise RuntimeError(f"K2 failed at {what}")

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
            import hashlib

            row["digest"] = hashlib.sha256(so.cpu().numpy().tobytes() + br.view(torch.int16).cpu().numpy().tobytes()
                                           + outs["ids"].cpu().numpy().tobytes()).hexdigest()[:16]
            print(f"layer {layer}: digest {row['digest']}", flush=True)
        # ---- eager launches (back-to-back unless --sync-each), on the null stream or a side stream
        es = torch.cuda.Stream() if args.eager_side else stream
        es.wait_stream(stream)
        with torch.cuda.stream(es):
            for i in range(args.eager):
                launch()
                if args.sync_each:
                    torch.cuda.synchronize()
                    dist.barrier()
                    checked(f"eager {i} ({'side' if args.eager_side else 'null'} stream, synced)")
        stream.wait_stream(es)
        torch.cuda.synchronize()
        dist.barrier()
        checked(f"{args.eager} eager ({'side' if args.eager_side else 'null'} stream)")
        if args.eager_only:
            continue
        if args.eager:
            row["eager_matches_first"] = bool(torch.equal(so, outs["stream_out"])) if rank == 0 else None
        # ---- timing: graph of R launches
        reps = args.reps
        g = torch.cuda.CUDAGraph()
        s2 = torch.cuda.Stream()
        s2.wait_stream(stream)
        dist.barrier()
        hw = []
        with torch.cuda.stream(s2):
            for _ in range(2):
                h0 = time.time()
                launch()
                hw.append(round((time.time() - h0) * 1e3, 2))
        stream.wait_stream(s2)
        torch.cuda.synchronize()
        print(f"rank {rank}: side-stream warm-up host ms {hw}", flush=True)
        dist.barrier()
        checked("side-stream warm-up")
        with torch.cuda.graph(g):
            for _ in range(reps):
                launch()
        torch.cuda.synchronize()
        first = so.clone()
        so.zero_()
        times = []
        for _ in range(args.iters):
            dist.barrier()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1000 / reps)
        checked("graph replays")
        row["replay_matches_first"] = bool(torch.equal(so, first))
        assert row["replay_matches_first"], "graph replays did not rewrite stream_out identically"
        times.sort()
        row["us_per_layer_median"] = times[len(times) // 2]
        row["us_per_layer_min"] = times[0]
        gathered = [None] * 4
        dist.all_gather_object(gathered, row["us_per_layer_median"])
        row["us_per_layer_median_ranks"] = gathered
        if rank == 0:
            print(f"layer {layer}: K2 {row['us_per_layer_median']:.1f} us/layer (ranks {gathered})", flush=True)
            res[str(layer)] = row
            with open(os.path.join(OUT, f"run_k2_{args.act}.json"), "w") as f:
                json.dump(res, f, indent=1, default=str)
        del g
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
    ap.add_argument("--eager-side", action="store_true")
    ap.add_argument("--sync-each", action="store_true")
    ap.add_argument("--eager-only", action="store_true")
    args = ap.parse_args()
    args.layers = [int(x) for x in args.layers.split(",")]
    args.tag = f"{int(time.time())}"
    mp.spawn(worker, args=(args, None), nprocs=4, join=True)


if __name__ == "__main__":
    main()
