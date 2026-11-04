"""Pinned host -> GPU copies at full bandwidth on the visible GPU (the weight-loading data path)."""
import sys, time
import torch
gib, reps = int(sys.argv[1]), int(sys.argv[2])
src = torch.empty(gib * (1 << 30), dtype=torch.uint8).pin_memory()
dst = torch.empty_like(src, device="cuda")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(reps):
    dst.copy_(src, non_blocking=True)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"ok: {gib * reps} GiB host->GPU in {dt:.1f} s, {gib * reps / dt:.1f} GiB/s")
