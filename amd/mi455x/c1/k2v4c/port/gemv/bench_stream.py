"""Pure weight-stream microbench at the o_proj / pooler sizes: what the access pattern and grid alone cost on
gfx1250 (graph replay, ~1.5 GB of rotated buffers, launch included as in the copy sweep).

Each CTA reads its share with b128 loads, every load issued before any use, and writes one word a thread.
Patterns: ``rows`` = the plain-layout GEMV operand pattern (16 rows a wave, lanes r and r + 16 adjacent 16 B of
row r); ``flat`` = fully coalesced (an instruction reads 512 contiguous bytes; the preshuffled layout).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int64, T

from gemv.bench import time_graph
from gemv.common import rsrc


def offset(pattern, bid, tid, lane, wave, j, per_cta, threads, waves):
    if pattern == "flat":
        return (bid * per_cta + j * threads * 16 + tid * 16) // 4
    band = per_cta // 16  # CTA tile: 16 rows x band bytes; wave w a 256 B column band a step
    col = ((wave + waves * (j // 8)) * 256 + (j % 8) * 32 + (lane // 16) * 16) % band
    return (bid * per_cta + (lane % 16) * band + col) // 4


def make(pattern, grid, threads, loads, cm):
    waves = threads // 32
    key = (pattern, grid, threads, loads, cm)

    @flyc.kernel(name=f"k2g_stream_{pattern}_{grid}_{threads}_{loads}_{cm}", known_block_size=[threads, 1, 1])
    def kern(w: Int64, out: Int64):
        _ = key
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 32
        wave = tid // 32
        per_cta = threads * loads * 16
        r = rsrc(w)
        vs = []
        for j in range_constexpr(loads):
            off = offset(pattern, bid, tid, lane, wave, j, per_cta, threads, waves)
            vs.append(fx.Vector(bo.buffer_load(r, off, vec_width=4, dtype=T.i32, cache_modifier=cm)))
        acc = vs[0][0]
        for j in range_constexpr(loads):
            for e in range_constexpr(4):
                acc = acc ^ vs[j][e]
        bo.buffer_store(acc, rsrc(out), bid * threads + tid)

    @flyc.jit
    def launch(w: Int64, out: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = key
        kern(w, out).launch(grid=(grid,), block=(threads,), stream=stream)

    return launch


def main():
    res = {}
    out = torch.zeros(1 << 20, dtype=torch.int32, device="cuda")
    for nbytes in (16 << 20, 32 << 20):
        bufs = [torch.empty(nbytes // 4, dtype=torch.int32, device="cuda") for _ in range(1536 * 2**20 // nbytes)]
        for pattern in ("rows", "flat"):
            for cm in (0, 1):
                for grid, threads in ((256, 512), (512, 256), (512, 512), (1024, 256)):
                    loads = nbytes // (grid * threads * 16)
                    if loads < 1 or (pattern == "rows" and loads % 8):
                        continue
                    f = make(pattern, grid, threads, loads, cm)
                    calls = [
                        (lambda b=b: f(b.data_ptr(), out.data_ptr(),
                                       stream=fx.Stream(torch.cuda.current_stream().cuda_stream)))
                        for b in bufs
                    ]  # fmt: skip
                    t = time_graph(calls * max(1, 96 // len(calls)))
                    k = f"{nbytes >> 20}MB {pattern} cm{cm} grid{grid}x{threads} loads{loads}"
                    res[k] = {"us": round(t, 3), "TBs": round(nbytes / t / 1e6, 2)}
                    print(k, f"{t:.3f} us  {nbytes / t / 1e6:.2f} TB/s", flush=True)
        del bufs
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "bench_stream.json"), "w") as fo:
        json.dump(res, fo, indent=1)


if __name__ == "__main__":
    main()
