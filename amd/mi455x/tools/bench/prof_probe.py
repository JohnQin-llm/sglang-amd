"""Which server condition stops torch.profiler from recording GPU activity on this ROCm stack.

Each case profiles a few matmuls and reports how many device and runtime events the trace holds.
    python3 prof_probe.py eager | graph | nccl | spawn
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity, profile


def count(p):
    ev = p.events()
    device = sum(1 for e in ev if e.device_type.name != "CPU")
    runtime = sum(1 for e in ev if e.name.startswith(("hip", "cuda")))
    return device, runtime


def run(case, rank=0):
    torch.cuda.set_device(rank)
    x = torch.randn(1024, 1024, device="cuda")
    if case == "graph":
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            y = x @ x
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                y = x @ x
        work = g.replay
    else:
        def work():
            return x @ x
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as p:
        for _ in range(3):
            work()
        torch.cuda.synchronize()
    device, runtime = count(p)
    print(f"[{case} rank {rank}] device events {device}, runtime events {runtime}", flush=True)


def worker(rank, case, world):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29531")
    torch.cuda.set_device(rank)
    if case == "nccl":
        dist.init_process_group("nccl", rank=rank, world_size=world)
        t = torch.ones(16, device="cuda")
        dist.all_reduce(t)
    run("graph" if case == "nccl" else "eager", rank)
    if case == "nccl":
        dist.destroy_process_group()


if __name__ == "__main__":
    case = sys.argv[1]
    if case in ("eager", "graph"):
        run(case)
    else:
        world = 2 if case == "nccl" else 1
        mp.spawn(worker, args=(case, world), nprocs=world)
