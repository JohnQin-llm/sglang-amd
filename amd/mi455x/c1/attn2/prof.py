"""Eager dispatches of bench.py variants for rocprofv3 (kernel trace / counters per dispatch).

usage (under k2safe -w '... rocprofv3 --pmc ... -d DIR -o NAME --'): prof.py --ctx 340000 --variants served,v2_T64_S128_Q0 --n 5
"""
import argparse

import torch

import bench as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="340000")
    ap.add_argument("--variants", default="served")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    for ctx in [int(c) for c in a.ctx.split(",")]:
        fx = B.C.make_fixture("full", ctx, 1, dev=dev)
        seqused = fx["seq_lens"] + B.C.ROWS
        out = torch.empty(B.C.ROWS, B.C.NQ, B.C.D, dtype=torch.bfloat16, device=dev)
        for n in a.variants.split(","):
            call, _ = B.make(n, fx, out, seqused)
            for _ in range(2):
                call()
            torch.cuda.synchronize()
            for _ in range(a.n):
                call()
            torch.cuda.synchronize()
            print(f"[{ctx}] {n} done", flush=True)


if __name__ == "__main__":
    main()
