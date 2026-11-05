"""Slow-replay study for the served K2 (k2v2_land1): every iteration, every rank, with host timestamps.

Per layer (default 1, 31, 60) and cache mode (warm: one operand copy; cold: 8 rotated copies of every K2 weight),
real weights, 8 rows, all four GPUs, one process a rank. Three loops per case:

  sync    : the layer_ab / run_k2 shape: gloo barrier -> e0 -> replay (graph: epoch advance + --reps chained calls,
            slots 1 .. reps) -> e1 -> synchronize. Per iteration and rank: event us, host CLOCK_MONOTONIC stamps
            (before / after the barrier, after e0.record, after replay(), after the sync), rusage deltas of the
            thread (minor / major faults, voluntary / involuntary context switches), the CPU it ran on.
  queued  : no host sync between calls: --qchunk graph replays (graph: epoch advance + 1 call; cold rotates 8 graphs)
            queued back to back with an event between replays, one barrier + sync per chunk. Host launch timing
            cannot reach the GPU-side times here (the queue runs ahead); the first call of a chunk carries the
            chunk-start skew and is reported separately.
  sync1   : as sync with a one-call graph (per-call times including launch).

CLOCK_MONOTONIC is system-wide (the container shares the host kernel), so stamps compare across ranks and with the
host monitor (spike_mon.py). Output: s3/spike/<tag>_L<layer>_<mode>.json (all ranks gathered on rank 0).

Run (all four GPUs through k2safe):
  /data/muse455/k2/k2safe.sh -t 1200 /data/muse455/k2/cand/k2v2_poll s3/spike.py --tag base \
      > /data/muse455/k2/cand/k2v2_poll/logs/spike_base.log 2>&1
"""

import argparse
import gc
import json
import os
import resource
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "s3", "spike")
LAND1 = "/data/muse455/k2/cand/k2v2_land1"  # default --tree (read only)


def now():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def cpu_now():
    with open("/proc/thread-self/stat") as f:
        return int(f.read().rsplit(")", 1)[1].split()[36])


def k2_operands(lay, rank, experts, dev):
    """run_k2.k2_operands of the served tree (the routed experts only in the EP1 stacks)."""
    from port import consts as K
    from port import layout as L
    from port.gemv import gemv
    import ep1_layout as P

    shuf = gemv.shuffle_rows if gemv.SHUFFLED else (lambda t: t)
    w = lay.w
    t = {
        "w_o": shuf(w["w_o"][:, rank * K.CORE:(rank + 1) * K.CORE].contiguous()),
        "alpha_a": w["alpha_a"], "beta_a": w["beta_a"], "w_fn": w["w_fn"].contiguous(),
        "w_router": w["w_router"].float().contiguous(), "expert_bias": w["bias"].float().contiguous(),
        "w_pool": shuf(w["w_pool"].contiguous()), "w_pre": w["w_pre"].contiguous(),
        "w_post": w["w_post"].contiguous(), "w_unpool": shuf(w["w_unpool"].contiguous()),
    }  # fmt: skip
    ri = K.RI
    w13 = torch.zeros(K.E, 2 * ri, K.MH // 2, dtype=torch.uint8)
    s13 = torch.zeros(K.E, 2 * ri, K.MH // 32, dtype=torch.uint8)
    w2 = torch.zeros(K.E, K.MH, ri // 2, dtype=torch.uint8)
    s2 = torch.zeros(K.E, K.MH, ri // 32, dtype=torch.uint8)
    for e in experts:
        a, b, c, d = P.slice_expert(*[x.cpu() for x in lay.expert(e)], rank)
        w13[e], s13[e], w2[e], s2[e] = a, b, c, d
    st = L.gfx1250_weights(w13, s13, w2, s2)
    t["w13"], t["w13s"], t["w2"], t["w2s"] = (x.to(dev).contiguous() for x in st)
    return t


WEIGHTS = ("w_o", "alpha_a", "beta_a", "w_fn", "w_router", "expert_bias", "w_pool", "w_pre", "w_post", "w_unpool",
           "w13", "w13s", "w2", "w2s")


def worker(rank, args, _):
    sys.path.insert(0, args.tree)
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("K2G_GPUS", "0,1,2,3")
    if args.pin:
        cpus = []
        for part in args.pin.split("/")[rank].split(","):
            a, _, b = part.partition("-")
            cpus += list(range(int(a), int(b or a) + 1))
        os.sched_setaffinity(0, cpus)
    if args.nice:
        os.nice(args.nice)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"file:///tmp/k2spike_pg_{args.tag}_{args.stamp}", rank=rank,
                            world_size=4)
    from port import consts as K
    from port import k2
    from port import validation as V
    from port.sync.debug import given_up_waits
    from port.sync.epoch import advance_epoch_
    from port.sync.peer import PeerRuntime

    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    gcev = []
    gc.callbacks.append(lambda phase, info: gcev.append((phase[0], now(), info.get("generation"))))
    key = k2.K2Build()
    k2.build(key)
    rt = PeerRuntime(k2.peer_bytes(), dist.group.WORLD, rank, K.TP, dev, blocks=K.BLOCKS)
    scratch = torch.zeros(k2.scratch_bytes() // 4, dtype=torch.int32, device=dev)
    meta = {"pid": os.getpid(), "affinity": sorted(os.sched_getaffinity(0)), "cpu": cpu_now(),
            "tree": args.tree, "k2_file": k2.__file__, "numa_maps0": open("/proc/self/numa_maps").readline().strip()}
    allmeta = [None] * 4
    dist.all_gather_object(allmeta, meta)
    log(f"ranks: {allmeta}")

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
        if any(g[0] or g[2] for g in got):
            log(f"HEALTH {what}: {got}")
            raise RuntimeError(f"K2 gave up at {what}")

    def capture(body):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            body()
        torch.cuda.current_stream().wait_stream(side)
        health("graph warm-up")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        torch.cuda.synchronize()
        return g

    def sync_loop(g, n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        rows = []
        for _ in range(n):
            r0 = resource.getrusage(resource.RUSAGE_THREAD)
            tb0 = now()
            dist.barrier()
            tb1 = now()
            e0.record()
            te = now()
            g.replay()
            tr = now()
            e1.record()
            torch.cuda.synchronize()
            ts = now()
            r1 = resource.getrusage(resource.RUSAGE_THREAD)
            rows.append((round(e0.elapsed_time(e1) * 1000, 2), tb0, tb1, te, tr, ts,
                         r1.ru_minflt - r0.ru_minflt, r1.ru_majflt - r0.ru_majflt,
                         r1.ru_nvcsw - r0.ru_nvcsw, r1.ru_nivcsw - r0.ru_nivcsw, cpu_now()))
        return rows

    def queued_loop(graphs, n, chunk):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(chunk + 1)]
        us, chunks = [], []
        done = 0
        while done < n:
            m = min(chunk, n - done)
            dist.barrier()
            t0 = now()
            for j in range(m):
                ev[j].record()
                graphs[(done + j) % len(graphs)].replay()
            ev[m].record()
            tq = now()
            torch.cuda.synchronize()
            ts = now()
            us.append([round(ev[j].elapsed_time(ev[j + 1]) * 1000, 2) for j in range(m)])
            chunks.append((t0, tq, ts))
            done += m
        return us, chunks

    for layer in args.layers:
        tc = time.time()
        lay = V.LayerData(layer, dev)
        cm = [None]
        if rank == 0:
            corr, nd = V.calibrate(lay, args.target_distinct, args.seed)
            case = V.make_case(lay, corr, args.seed, dev=dev)
            cm = [(corr, nd)]
            core, stream_a = case.core.cpu(), case.stream_a.cpu()
        else:
            core = torch.empty(K.MAX_TOKENS, K.CORE * K.TP, dtype=torch.bfloat16)
            stream_a = torch.empty(K.MAX_TOKENS, K.H, dtype=torch.float32)
        dist.broadcast_object_list(cm, src=0)
        dist.broadcast(core, src=0)
        dist.broadcast(stream_a, src=0)
        case = V.Case(core.to(dev), stream_a.to(dev))
        experts = torch.unique(V.seam_and_route(lay, case)["ids"]).tolist()
        assert all(e == experts for e in gather(experts)), "ranks disagree on the routed experts"
        ops = k2_operands(lay, rank, experts, dev)
        so = torch.zeros(K.MAX_TOKENS, K.H, dtype=torch.float32, device=dev)
        br = torch.zeros(K.MAX_TOKENS, K.H, dtype=torch.bfloat16, device=dev)
        base = dict(ops, core=case.core_rank(rank), stream_a=case.stream_a.contiguous(), stream_out=so, branch_out=br)
        log(f"L{layer}: setup {time.time() - tc:.0f} s, corr {cm[0][0]:.4f}, distinct {cm[0][1]}")

        def call(t, slot):
            k2.k2_launch(key, t, rank, slot, rt.epoch, rt.peers, scratch, rt.health, rt.alarm,
                         torch.cuda.current_stream().cuda_stream)

        dist.barrier()
        advance_epoch_(rt.epoch)
        call(base, layer)
        health("golden")
        gold = so.clone()
        import hashlib

        off, n = k2.scratch_layout()["route"]
        ids = scratch.view(torch.int32)[off // 4:(off + n) // 4].view(-1, 2)[:, 0].view(K.MAX_TOKENS, 2 * K.TOPK)
        digest = hashlib.sha256(so.cpu().numpy().tobytes() + br.view(torch.int16).cpu().numpy().tobytes()
                                + ids[:, :K.TOPK].cpu().numpy().tobytes()).hexdigest()[:16]
        digests = gather(digest)
        log(f"L{layer}: golden digest per rank {digests}")
        for mode in args.modes:
            ncopy = 8 if mode == "cold" else 1
            copies = [base] + [dict(base, **{k: base[k].clone() for k in WEIGHTS}) for _ in range(ncopy - 1)]
            torch.cuda.synchronize()

            def chain():
                advance_epoch_(rt.epoch)
                for i in range(args.reps):
                    call(copies[i % ncopy], 1 + i)

            def one(k):
                def body():
                    advance_epoch_(rt.epoch)
                    call(copies[k], 1)
                return body

            res = {"layer": layer, "mode": mode, "copies": ncopy, "reps": args.reps, "meta": allmeta,
                   "corr_distinct": cm[0], "args": vars(args)}
            gc_was = gc.isenabled()
            if args.gc_off:
                gc.collect()
                gc.freeze()
                gc.disable()
            for loop in args.loops:
                gcev.clear()
                t0 = now()
                if loop == "sync":
                    g = capture(chain)
                    sync_loop(g, args.warmup)
                    rows = sync_loop(g, args.iters)
                    health("sync")
                    res["sync"] = gather(rows)
                    del g
                elif loop == "sync1":
                    g = capture(one(0))
                    sync_loop(g, args.warmup)
                    rows = sync_loop(g, args.iters1)
                    health("sync1")
                    res["sync1"] = gather(rows)
                    del g
                elif loop == "queued":
                    gs = [capture(one(k)) for k in range(ncopy)]
                    queued_loop(gs, min(args.qchunk, 200), min(args.qchunk, 200))
                    us, chunks = queued_loop(gs, args.qiters, args.qchunk)
                    health("queued")
                    res["queued"] = gather({"us": us, "chunks": chunks})
                    del gs
                res[f"{loop}_gc"] = gather(list(gcev))
                res[f"{loop}_t"] = (t0, now())
                if rank == 0:
                    if loop.startswith("sync"):
                        per = [[r[0] / (args.reps if loop == "sync" else 1) for r in rr] for rr in res[loop]]
                    else:
                        per = [[x for c in rr["us"] for x in c[1:]] for rr in res[loop]]
                    s = []
                    for p in per:
                        q = sorted(p)
                        s.append(f"p50 {q[len(q) // 2]:.1f} max {q[-1]:.1f} >150 {sum(x > 150 for x in q)}")
                    log(f"L{layer} {mode} {loop}: {len(per[0])} it/rank; " + " | ".join(s))
            if args.gc_off:
                gc.unfreeze()
                if gc_was:
                    gc.enable()
            res["replay_matches_golden"] = gather(bool(torch.equal(so, gold)))
            res["golden_digest"] = digests
            if rank == 0:
                os.makedirs(OUT, exist_ok=True)
                path = os.path.join(OUT, f"{args.tag}_L{layer}_{mode}.json")
                with open(path, "w") as f:
                    json.dump(res, f)
                log(f"wrote {path} (replay == golden: {res['replay_matches_golden']})")
            del copies
            torch.cuda.empty_cache()
        del ops, base
        torch.cuda.empty_cache()
    rt.close()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=LAND1)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--layers", default="1,31,60")
    ap.add_argument("--modes", default="warm,cold")
    ap.add_argument("--loops", default="sync,queued,sync1")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--iters1", type=int, default=2000)
    ap.add_argument("--qiters", type=int, default=20000)
    ap.add_argument("--qchunk", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-distinct", type=int, default=43)
    ap.add_argument("--pin", default="", help="per-rank CPU lists: 0,1/2,3/4,5/6,7")
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--gc-off", action="store_true")
    ap.add_argument("--mpol", default="", help="'local': MPOL_LOCAL task policy (opts the ranks out of NUMA balancing)")
    args = ap.parse_args()
    args.layers = [int(x) for x in args.layers.split(",")]
    args.modes = args.modes.split(",")
    args.loops = args.loops.split(",")
    args.stamp = f"{int(time.time())}_{os.getpid()}"
    if args.mpol:
        set_mempolicy(args.mpol)
    mp.spawn(worker, args=(args, None), nprocs=4, join=True)


def set_mempolicy(kind):
    """Task memory policy, inherited by the spawned ranks and their threads. A non-default policy without
    MPOL_F_NUMA_BALANCING has no MPOL_F_MOF, so NUMA balancing's task_numa_work skips this process's VMAs
    (no PROT_NONE scans, no MMU-notifier invalidations of GPU-mapped host memory). Placement: 'local' = MPOL_LOCAL."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    mode = {"local": 4}[kind]
    if libc.syscall(238, mode, None, 0) != 0:  # __NR_set_mempolicy (x86_64)
        raise OSError(ctypes.get_errno(), "set_mempolicy")
    with open("/proc/self/numa_maps") as f:
        print(f"set_mempolicy({kind}): {f.readline().strip()}", flush=True)


if __name__ == "__main__":
    main()
