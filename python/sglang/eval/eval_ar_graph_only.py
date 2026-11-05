"""Graph-only gate for the AITER custom collectives (SGLANG_AITER_AR_GRAPH_ONLY=1), through SGLang's
real GroupCoordinator on four ranks (tensor parallel 4, hidden 8192, bf16).

eager: all-reduce at 8/128/592/1100/4096 tokens; counts AITER calls (should be 0 with the gate) and
       the error against a float32 reference.
graph: per width (8/32/80/128) capture 65 all-reduces + 1 reduce-scatter + 1 all-gather inside
       tp_group.graph_capture(); count captured AITER calls, replay with new inputs, check error, and
       time one replay (microseconds per collective).
Arm set by SGLANG_USE_AITER_AR, SGLANG_AITER_CAPTURE_COPY_IN, SGLANG_AITER_AR_GRAPH_ONLY.
Run: timeout 600 torchrun --nproc-per-node 4 eval_ar_graph_only.py --out <json>
"""
import argparse, json, os
import torch

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
from sglang.srt.distributed import parallel_state as ps
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
ps.init_distributed_environment(world_size=world, rank=rank, local_rank=lr, backend="nccl",
                                distributed_init_method="env://")
from sglang.test.test_utils import publish_build_topology
publish_build_topology(tp_size=world, ep_size=1, pp_size=1)
ps.initialize_model_parallel()
tp = ps.get_tp_group(); ca = tp.ca_comm; dev = torch.device("cuda", lr)
H = 8192
calls = {"ca_ar": 0, "ca_ag": 0, "ca_rs": 0}
def wrap(name, key):
    if ca is None or not hasattr(ca, name): return
    f = getattr(ca, name)
    def g(*a, **k):
        calls[key] += 1; return f(*a, **k)
    setattr(ca, name, g)
wrap("custom_all_reduce", "ca_ar"); wrap("all_gather_unreg", "ca_ag"); wrap("all_gather_reg", "ca_ag")
wrap("reduce_scatter", "ca_rs")
def snap():
    d = dict(calls); [calls.__setitem__(k, 0) for k in calls]; return d
res = {"env": {k: os.environ.get(k) for k in ("SGLANG_USE_AITER_AR", "SGLANG_AITER_CAPTURE_COPY_IN",
                                              "SGLANG_AITER_AR_GRAPH_ONLY")},
       "ca_class": None if ca is None else f"{type(ca).__module__}.{type(ca).__name__}",
       "graph_only": ps._AITER_AR_GRAPH_ONLY, "eager": {}, "graph": {}}
def ref(x):
    r = x.float().clone(); torch.distributed.all_reduce(r, group=tp.device_group); return r
torch.manual_seed(rank)
for T in (8, 128, 592, 1100, 4096):
    x = torch.randn(T, H, device=dev).to(torch.bfloat16); snap()
    y = tp.all_reduce(x.clone()); torch.cuda.synchronize()
    res["eager"][str(T)] = {"calls": snap(), "err_ar": float((y.float() - ref(x)).abs().max())}
for T in (8, 32, 80, 128):
    x = torch.randn(T, H, device=dev).to(torch.bfloat16)
    rs = torch.empty(T // world, H, dtype=torch.bfloat16, device=dev)
    ag = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    def body():
        y = tp.all_reduce(x)
        first = y
        for _ in range(64): y = tp.all_reduce(y * (1.0 / world))
        tp.reduce_scatter_tensor(rs, y); tp.all_gather_into_tensor(ag, rs)
        return first
    g = torch.cuda.CUDAGraph(); snap()
    with tp.graph_capture() as ctx:
        body(); torch.cuda.synchronize(); warm = snap()
        with torch.cuda.graph(g, stream=ctx.stream):
            out = body()
    cap = snap(); torch.cuda.synchronize()
    x.copy_(torch.randn(T, H, device=dev).to(torch.bfloat16)); r = ref(x)
    g.replay(); torch.cuda.synchronize(); err = float((out.float() - r).abs().max())
    for _ in range(20): g.replay()
    torch.cuda.synchronize(); torch.distributed.barrier(group=tp.device_group)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(50): g.replay()
    e.record(); torch.cuda.synchronize(); us = s.elapsed_time(e) * 1000 / 50
    res["graph"][str(T)] = {"warm_calls": warm, "captured_calls": cap, "err_ar_replay": err,
                            "graph_us": round(us, 1), "us_per_collective": round(us / 67, 2)}
if rank == 0:
    json.dump(res, open(args.out, "w"), indent=1); print(json.dumps(res["graph"]))
torch.distributed.barrier(); ps.destroy_model_parallel(); ps.destroy_distributed_environment()
