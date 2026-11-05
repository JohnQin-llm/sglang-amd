"""4-rank aiter gfx1250 custom all-reduce at the draft's shape ([8, 8192] bf16), graph replay.

  served : CustomAllreduce.custom_all_reduce under capture (copy into the registered pool, then ar_ll)
  direct : CustomAllreduce.all_reduce(registered_input=True) (SGLANG_MUSE_AR_LL_DIRECT): ar_ll on the input itself
Each graph holds CALLS all-reduces on distinct inputs; us per all-reduce, slowest rank, alternating trials.
"""
import sys

sys.path.insert(0, "/data/muse455/draft2/sglang/python")
import json
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CALLS, M, H, TRIALS, REPS = 20, 8, 8192, 4, 20


def worker(rank, world, port, out_path):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    dev = torch.device(f"cuda:{rank}")
    ca = CustomAllreduce(dist.group.WORLD, dev)
    assert not ca.disabled, "custom all-reduce disabled"
    torch.manual_seed(1000 + rank)
    xs = [torch.randn(M, H, device=dev, dtype=torch.bfloat16) for _ in range(CALLS)]
    outs = {}

    def run(arm):
        if arm == "served":
            return [ca.custom_all_reduce(x) for x in xs]
        return [ca.all_reduce(x, registered_input=True) for x in xs]

    graphs = {}
    for arm in ("served", "direct"):
        for _ in range(3):
            run(arm)  # eager warmup (same launch sequence on every rank)
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with ca.capture():
            with torch.cuda.graph(g, stream=s):
                outs[arm] = run(arm)
        torch.cuda.synchronize()
        dist.barrier()
        graphs[arm] = g
    # numerics: one replay each
    res = {}
    for arm in ("served", "direct"):
        dist.barrier()
        graphs[arm].replay()
        torch.cuda.synchronize()
    diff = sum(int((a.view(torch.uint8) != b.view(torch.uint8)).sum()) for a, b in zip(outs["served"], outs["direct"]))
    ts = {"served": [], "direct": []}
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for t in range(TRIALS):
        for arm in (("served", "direct") if t % 2 == 0 else ("direct", "served")):
            per = []
            for _ in range(REPS):
                dist.barrier()
                e0.record()
                graphs[arm].replay()
                e1.record()
                torch.cuda.synchronize()
                per.append(e0.elapsed_time(e1) * 1000 / CALLS)
            ts[arm].append(sorted(per)[REPS // 2])
    res = {arm: sorted(v)[len(v) // 2] for arm, v in ts.items()}
    res["bytes_differing"] = diff
    gathered = [None] * world
    dist.all_gather_object(gathered, res)
    if rank == 0:
        summary = {
            "per_rank": gathered,
            "slowest_rank_us": {arm: max(r[arm] for r in gathered) for arm in ("served", "direct")},
            "bytes_differing_total": sum(r["bytes_differing"] for r in gathered),
        }
        print(json.dumps(summary), flush=True)
        json.dump(summary, open(out_path, "w"), indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/data/muse455/draft2/logs/ar.json"
    mp.spawn(worker, args=(4, 29511, out), nprocs=4, join=True)
