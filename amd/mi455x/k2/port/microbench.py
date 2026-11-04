"""K2 gfx1250 microbench: one K2 launch a layer on 4 ranks, HIP-graph replay, synthetic weights.

K2's time depends on the routes (the number of distinct experts U), not on weight values, so the weights are
random and the router bias steers the 64 picks of the 8 rows into a pool of ``--pool`` experts (pool 64 gives
U of about 43, the C1 verify mean). Prints the per-layer time on every rank and the U it ran with.

    python3 -m sglang.srt.rocm_muse_spark.mono_gfx1250.microbench [--pool 64] [--reps 20] [--iters 20]
    (dev tree: PYTHONPATH=/data/muse455/k2/ep1 python3 -m port.microbench)

Needs the four GPUs of the TP group and no server running on them.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from . import consts as K
from . import k2


def _operands(dev, pool, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def bf(*shape, scale=0.02):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(dev)

    def u8(*shape, fill=None):
        if fill is not None:
            return torch.full(shape, fill, dtype=torch.uint8, device=dev)
        return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8).to(dev)

    experts = torch.randperm(K.E, generator=g)[:pool]
    bias = torch.full((K.E,), -30.0)
    bias[experts] = 0.0
    t = {
        "w_o": bf(K.H, K.CORE), "alpha_a": torch.ones(K.H, device=dev), "beta_a": torch.ones(K.H, device=dev),
        "w_fn": bf(K.H), "w_router": (torch.randn(K.E, K.H, generator=g) * 0.02).to(dev),
        "expert_bias": bias.to(dev), "w_pool": bf(K.MH, K.H), "w_pre": bf(K.MH), "w_post": bf(K.MH),
        "w_unpool": bf(K.H, K.MH),
        "w13": u8(K.E, 2 * K.RI, K.MH // 2), "w13s": u8(K.E, 2 * K.RI, K.MH // 32, fill=127),
        "w2": u8(K.E, K.MH, K.RI // 2), "w2s": u8(K.E, K.MH, K.RI // 32, fill=127),
        "core": bf(K.MAX_TOKENS, K.CORE, scale=1.0),
        "stream_a": torch.randn(K.MAX_TOKENS, K.H, generator=g).to(dev),
        "stream_out": torch.zeros(K.MAX_TOKENS, K.H, device=dev),
        "branch_out": torch.zeros(K.MAX_TOKENS, K.H, dtype=torch.bfloat16, device=dev),
    }  # fmt: skip
    return t


def _distinct(scratch):
    off, n = k2.scratch_layout()["route"]
    ids = scratch.view(torch.int32)[off // 4:(off + n) // 4].view(-1, 2)[:, 0].view(K.MAX_TOKENS, 2 * K.TOPK)
    return int(torch.unique(ids[:, :K.TOPK]).numel())


def _worker(rank, args):
    from .sync.epoch import advance_epoch_
    from .sync.peer import PeerRuntime

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"file:///tmp/k2g_mb_{args.tag}", rank=rank, world_size=K.TP)
    key = k2.K2Build(act=args.act)
    k2.build(key)
    rt = PeerRuntime(k2.peer_bytes(), dist.group.WORLD, rank, K.TP, dev, blocks=K.BLOCKS)
    scratch = torch.zeros(k2.scratch_bytes() // 4, dtype=torch.int32, device=dev)
    t = _operands(dev, args.pool, args.seed)  # one seed: replicated tensors agree across ranks
    stream = torch.cuda.current_stream()

    def launch():
        advance_epoch_(rt.epoch)
        k2.k2_launch(key, t, rank, args.layer, rt.epoch, rt.peers, scratch, rt.health, rt.alarm,
                     torch.cuda.current_stream().cuda_stream)

    dist.barrier()
    launch()
    torch.cuda.synchronize()
    dist.barrier()
    rt.check()
    nu = _distinct(scratch)
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(stream)
    with torch.cuda.stream(side):
        launch()
    stream.wait_stream(side)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(args.reps):
            launch()
    times = []
    for _ in range(args.iters):
        dist.barrier()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1000 / args.reps)
    rt.check()
    times.sort()
    row = {"rank": rank, "distinct": nu, "us_median": times[len(times) // 2], "us_min": times[0]}
    rows = [None] * K.TP
    dist.all_gather_object(rows, row)
    if rank == 0:
        out = {"act": args.act, "pool": args.pool, "reps": args.reps, "iters": args.iters, "ranks": rows,
               "us_per_layer_max_rank_median": max(r["us_median"] for r in rows)}
        print(json.dumps(out, indent=1), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
    del g
    rt.close()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="a8")
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--layer", type=int, default=31, help="only the mailbox slot (128 + layer)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.tag = f"{os.getpid()}_{int(time.time())}"
    mp.spawn(_worker, args=(args,), nprocs=K.TP, join=True)


if __name__ == "__main__":
    main()
