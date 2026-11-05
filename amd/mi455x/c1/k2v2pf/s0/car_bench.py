"""S0: the image's custom all-reduce (aiter, gfx1250 kernel, IPC transport) on 4 ranks.

Times, per call (slowest rank, median of trials):
  * all_reduce [8,4096] / [8,8192] / [64,4096] bf16 in a CUDA graph (N calls back to back, replay / N) and eager;
  * reduce_scatter [64,4096] bf16 along dim 1 (the EP1 yr exchange shape) and all_gather [8,1024] dim 1
    (the FIN gather), eager, if the custom ops accept them.
Run: python3 car_bench.py --json out.json
"""
import argparse, json, os, sys, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world, port, args, q):
    os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    ca = CustomAllreduce(dist.group.WORLD, torch.device(f"cuda:{rank}"), max_size=64 * 1024 * 1024)
    assert not ca.disabled, "custom AR disabled"
    res = {}
    N = 50
    for shape in [(8, 4096), (8, 8192), (64, 4096)]:
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * (rank + 1)
        ref = None
        # correctness vs gloo sum (on CPU)
        xc = x.float().cpu()
        dist.all_reduce(xc)
        y = ca.custom_all_reduce(x)
        torch.cuda.synchronize()
        err = (y.float().cpu() - xc).abs().max().item()
        # eager timing
        ts = []
        for _ in range(20):
            dist.barrier(); torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(N):
                ca.custom_all_reduce(x)
            e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1e3 / N)
        eager = sorted(ts)[len(ts) // 2]
        # graph timing
        g = torch.cuda.CUDAGraph()
        with ca.capture():
            for _ in range(3):
                ca.custom_all_reduce(x)
            with torch.cuda.graph(g):
                for _ in range(N):
                    out = ca.custom_all_reduce(x)
        torch.cuda.synchronize()
        g.replay(); torch.cuda.synchronize()
        ts = []
        for _ in range(20):
            dist.barrier(); torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); g.replay(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1e3 / N)
        graph = sorted(ts)[len(ts) // 2]
        gerr = (out.float().cpu() - xc).abs().max().item()
        res[f"ar_{shape[0]}x{shape[1]}"] = dict(eager_us=eager, graph_us=graph, maxerr=err, graph_maxerr=gerr)
    # reduce-scatter / all-gather (eager)
    for name, fn in [
        ("rs_64x4096_dim1", lambda: _rs(ca, (64, 4096), 1, world)),
        ("ag_8x1024_dim1", lambda: _ag(ca, (8, 1024), 1)),
        ("ag_8x2048_dim1", lambda: _ag(ca, (8, 2048), 1)),
    ]:
        try:
            f = fn()
            ts = []
            for _ in range(20):
                dist.barrier(); torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(N):
                    f()
                e.record(); torch.cuda.synchronize()
                ts.append(s.elapsed_time(e) * 1e3 / N)
            res[name] = dict(eager_us=sorted(ts)[len(ts) // 2])
        except Exception as ex:  # noqa
            res[name] = dict(error=repr(ex)[:300])
    q.put((rank, res))
    dist.barrier()
    ca.close()
    dist.destroy_process_group()


def _rs(ca, shape, dim, world):
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    oshape = list(shape); oshape[dim] //= world
    out = torch.empty(oshape, device="cuda", dtype=torch.bfloat16)
    def f():
        ca.reduce_scatter(x, out, dim)
    f(); torch.cuda.synchronize()
    return f


def _ag(ca, shape, dim):
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    def f():
        ca.all_gather_unreg(x, dim=dim)
    f(); torch.cuda.synchronize()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    world = 4
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29000 + (os.getpid() % 1000)
    ps = [ctx.Process(target=worker, args=(r, world, port, args, q)) for r in range(world)]
    for p in ps:
        p.start()
    out = {}
    for _ in range(world):
        r, res = q.get(timeout=600)
        out[r] = res
    for p in ps:
        p.join(60)
    summary = {}
    for k in out[0]:
        if "error" in out[0][k]:
            summary[k] = out[0][k]
            continue
        summary[k] = {m: max(out[r][k][m] for r in out) for m in out[0][k]}
    print(json.dumps(summary, indent=1))
    if args.json:
        json.dump(dict(slowest_rank=summary, per_rank=out), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
