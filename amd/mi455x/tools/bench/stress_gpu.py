"""Sustained matmul load plus a large allocation on the visible GPU, for SECONDS."""
import sys, time
import torch
secs, gib = float(sys.argv[1]), int(sys.argv[2])
hold = torch.empty(gib * (1 << 30), dtype=torch.uint8, device="cuda")
hold.fill_(1)
x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
t0, n = time.time(), 0
while time.time() - t0 < secs:
    y = x @ x
    n += 1
    if n % 50 == 0:
        torch.cuda.synchronize()
torch.cuda.synchronize()
print(f"ok: {n} matmuls in {time.time() - t0:.0f} s, {gib} GiB held, finite={bool(torch.isfinite(y).all())}")
