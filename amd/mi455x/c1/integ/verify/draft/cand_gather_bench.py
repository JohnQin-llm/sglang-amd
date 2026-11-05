"""4-rank microbenchmark + numerics: the DFlash candidate top-k after the candidate-logits GEMM, TP4.

Arms, each from the rank's [8, 50512] BF16 logits to the global top-16 (ids int64, values FP32):
  served  torch.topk (radix) -> FP32 value all-gather + int64 id all-gather (NCCL) -> torch.topk -> gather
  packed  torch.topk -> dflash._packed_candidate_gather (one custom FP32 all-gather) -> torch.topk -> gather
  triton  dflash_candidate_topk: split top-k + packed merge -> one custom FP32 all-gather -> merge
Timing: CUDA graph captured under the TP group's graph_capture, replays alternate between arms, median of trials,
slowest rank. Numerics: ids and value bits of every arm vs served, on logits with many BF16 ties.
"""
import statistics
import sys

sys.path.insert(0, "/data/muse455/integ/sglang/python")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

N, K, SHARD, TP = 8, 16, 50512, 4


def worker(rank):
    torch.cuda.set_device(rank)
    from sglang.srt.distributed import parallel_state as ps

    ps.init_distributed_environment(world_size=TP, rank=rank, local_rank=rank, backend="nccl",
                                    distributed_init_method="tcp://127.0.0.1:29571")
    tp = ps.init_model_parallel_group([list(range(TP))], rank, backend="nccl", use_custom_allreduce=True,
                                      group_name="bench_tp")
    import sglang.srt.models.dflash as dflash
    from sglang.kernels.ops.speculative.dflash_candidate_topk import local_topk_packed, merge_gathered_topk

    ag = lambda x, dim=-1: tp.all_gather(x, dim=dim)
    dflash.tensor_model_parallel_all_gather = ag
    start = rank * SHARD
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    logits = (torch.randn(N, SHARD, device="cuda", generator=g) * 2).to(torch.bfloat16)
    logits[0, :40] = 9.0  # ties inside and across the top-16 boundary on every rank
    logits[1, 100:110] = 8.5

    def served():
        vals, ids = torch.topk(logits, K, dim=-1)
        global_ids = ids.long() + start
        gv = ag(vals.float(), dim=-1)
        gi = ag(global_ids, dim=-1)
        tv, sel = torch.topk(gv, K, dim=-1)
        return torch.gather(gi, -1, sel).long(), tv

    def packed():
        vals, ids = torch.topk(logits, K, dim=-1)
        i, v, _ = dflash._packed_candidate_gather(vals, ids, start, K)
        return i, v

    def triton_arm():
        return merge_gathered_topk(ag(local_topk_packed(logits, K, start), dim=-1), K)

    arms = {"served": served, "packed": packed, "triton": triton_arm}
    # tie diagnostics, eager: forced ties (rows 0-1) and natural BF16 ties only
    nat = (torch.randn(N, SHARD, device="cuda", generator=g) * 2).to(torch.bfloat16)
    diag = []
    for lg in (logits, nat):
        saved = logits.clone()
        logits.copy_(lg)
        (si, sv), (ti_, tv_) = served(), triton_arm()
        lv, li = torch.topk(logits, K, dim=-1)
        lp = local_topk_packed(logits, K, start)
        diag += [float(torch.equal(sv.view(torch.int32), tv_.view(torch.int32))),
                 float((si != ti_).any(-1).sum()),
                 float(torch.equal(lp[:, :K], lv.float())),
                 float((lp[:, K:].view(torch.int32).long() - start != li).any(-1).sum())]
        logits.copy_(saved)
    dg = torch.tensor(diag, device="cuda")
    alld = [torch.zeros_like(dg) for _ in range(TP)]
    dist.all_gather(alld, dg)
    if rank == 0:
        d = torch.stack(alld).cpu()
        for j, name in enumerate(("forced ties", "natural ties")):
            x = d[:, 4 * j: 4 * j + 4]
            print(f"[ties] {name}: global values bitwise equal {bool(x[:, 0].min() == 1)}, rows with different ids "
                  f"{x[:, 1].tolist()} (of {N}); rank-local values equal {bool(x[:, 2].min() == 1)}, rows with "
                  f"different ids {x[:, 3].tolist()}")
    # rank-local tie order: torch.topk vs the split top-k
    tv, ti = torch.topk(logits, K, dim=-1)
    p = local_topk_packed(logits, K, start)
    local_same = torch.equal(p[:, :K], tv.float()) and torch.equal(p[:, K:].view(torch.int32).long() - start, ti)
    graphs, outs = {}, {}
    for name, fn in arms.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with tp.graph_capture() as ctx:
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=ctx.stream):
                for _ in range(10):
                    o = fn()
        graphs[name], outs[name] = gr, o
        torch.cuda.synchronize()
    for gr in graphs.values():
        gr.replay()
    torch.cuda.synchronize()
    eq = {n: float(torch.equal(outs["served"][0], o[0]) and torch.equal(outs["served"][1].view(torch.int32),
                                                                       o[1].view(torch.int32)))
          for n, o in outs.items()}
    times = {n: [] for n in graphs}
    for trial in range(9):
        order = list(graphs) if trial % 2 == 0 else list(graphs)[::-1]
        for name in order:
            for _ in range(3):
                graphs[name].replay()
            dist.barrier()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20):
                graphs[name].replay()
            e1.record()
            torch.cuda.synchronize()
            times[name].append(e0.elapsed_time(e1) * 1000 / 200)
    names = list(graphs)
    med = torch.tensor([statistics.median(times[n]) for n in names] + [eq[n] for n in names] + [float(local_same)],
                       device="cuda")
    allm = [torch.zeros_like(med) for _ in range(TP)]
    dist.all_gather(allm, med)
    if rank == 0:
        a = torch.stack(allm).cpu()
        nn = len(names)
        print(f"[numerics] rank-local split top-k == torch.topk (values and ids, with ties) on all ranks: "
              f"{bool(a[:, -1].min() == 1)}")
        for j, n in enumerate(names):
            print(f"[numerics] {n}: global ids and value bits equal to served on all ranks: "
                  f"{bool(a[:, nn + j].min() == 1)}")
        for j, n in enumerate(names):
            print(f"[time] {n:7s}: {a[:, j].max():6.2f} us per draft step (slowest rank; per rank "
                  f"{[round(x, 2) for x in a[:, j].tolist()]})")
    dist.barrier()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=TP, join=True)
