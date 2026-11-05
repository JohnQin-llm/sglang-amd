"""Does the AITER custom all-reduce hang on gfx1250 when eager (non-graph) collectives at prefill
sizes run between replays of graphs captured with the copy-in path (SGLANG_AITER_CAPTURE_COPY_IN=1)?
Iteration 17's server hit 'HW Exception ... GPU Hang' in an eager 590-token prefill after about
3.5 minutes of graph-replayed decode.

--arm eager:      eager all-reduce only, token counts drawn from the served prefill mix.
--arm interleave: replay captured graphs (8 and 80 tokens: all-reduce + reduce-scatter + all-gather,
                  copy-in) then eager all-reduce (+ unregistered all-gather) at a prefill size, repeated.
--arm twostream:  as interleave, but the eager collectives go on a second stream with no ordering
                  against the replay stream (the race a side-stream path would create).

Run: timeout 300 torchrun --nproc-per-node 4 eager_stress.py --arm A --iters N --out <json>
A hang shows up as the timeout killing the run (no json written).
"""
import argparse, json, os, time
import torch
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--out", required=True)
args = ap.parse_args()
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dev = torch.device("cuda", torch.cuda.current_device())
dist.init_process_group("nccl", device_id=dev)
cpu = dist.new_group(backend="gloo")
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
ca = CustomAllreduce(group=cpu, device=dev, enable_register_for_capturing=True)
H = 8192
SIZES = [1, 7, 151, 330, 590, 593, 777, 981, 1100, 2048, 4096, 6417, 8192]
res = {"arm": args.arm, "iters": args.iters, "max_size": int(ca.max_size), "car_max": int(ca._car_max_size)}


def body(x, rs_out, ag_out):
    y = ca.custom_all_reduce(x)
    ca.reduce_scatter(y, rs_out, registered=False)
    ca.all_gather_unreg(rs_out * 0.5, out=ag_out, dim=0)
    return y


graphs = []
if args.arm in ("interleave", "twostream"):
    for T in (8, 80):
        x = torch.randn(T, H, device=dev).to(torch.bfloat16)
        rs_out = torch.empty(T // world, H, dtype=torch.bfloat16, device=dev)
        ag_out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
        g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream()); torch.cuda.synchronize()
        with ca.capture():
            with torch.cuda.stream(s):
                body(x, rs_out, ag_out); torch.cuda.synchronize()
                with torch.cuda.graph(g, stream=s):
                    y = body(x, rs_out, ag_out)
        graphs.append((g, x, y))
    torch.cuda.synchronize(); dist.barrier()

side = torch.cuda.Stream()
bufs = {T: torch.randn(T, H, device=dev).to(torch.bfloat16) for T in SIZES}
counts = {"custom": 0, "fallback": 0, "ag": 0}
errs = []
t0 = time.time()
for i in range(args.iters):
    T = SIZES[(i * 7 + 3) % len(SIZES)]
    for g, x, y in graphs:
        g.replay()
    stream = side if args.arm == "twostream" else torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        inp = bufs[T]
        out = ca.custom_all_reduce(inp)
        if out is None:
            counts["fallback"] += 1
            out = inp.clone(); dist.all_reduce(out)
        else:
            counts["custom"] += 1
        if T % world == 0 and ca.should_custom_ag(inp[: T // world]):
            counts["ag"] += 1
            ca.all_gather_unreg(inp[: T // world].contiguous(), dim=0)
    if i % 200 == 0:
        torch.cuda.synchronize()
        ref = inp.float().clone(); dist.all_reduce(ref)
        errs.append([i, T, (out.float() - ref).abs().max().item()])
        if rank == 0:
            print("iter", i, "T", T, "err", errs[-1][2], flush=True)
torch.cuda.synchronize(); dist.barrier()
res.update(counts=counts, seconds=round(time.time() - t0, 1), errs=errs, hung=False)
if rank == 0:
    json.dump(res, open(args.out, "w"), indent=1); print(json.dumps(res))
