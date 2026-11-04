"""Does importing a server module first stop torch.profiler from recording GPU kernels?"""
import importlib, sys
mods = sys.argv[1:]
for m in mods:
    importlib.import_module(m)
import torch
from torch.profiler import ProfilerActivity, profile
x = torch.randn(1024, 1024, device="cuda")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(3):
        x @ x
    torch.cuda.synchronize()
print(f"after importing {mods or 'nothing'}: device events {sum(1 for e in p.events() if e.device_type.name != 'CPU')}", flush=True)
