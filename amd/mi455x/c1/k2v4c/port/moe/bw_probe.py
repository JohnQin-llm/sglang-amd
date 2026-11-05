"""Achievable HBM read bandwidth on gfx1250 with the expert GEMMs' access shape: every wave streams
contiguous 1 KB steps (two b128 a lane, NT), ``inflight`` steps issued before the first use, xor-reduced.

Run: flock /data/muse455/locks/gpu1.lock docker exec muse_mi455x bash -c 'cd /tmp && PYTHONSAFEPATH=1
     HIP_VISIBLE_DEVICES=1 timeout 300 python3 /data/muse455/k2/cand/k2v4/port/moe/bw_probe.py'
"""

import functools
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int64, T


@functools.cache
def make(waves, chunk_kb, ctas, inflight, nt):
    steps = chunk_kb  # 1 KB a step a wave
    total = ctas * waves * chunk_kb * 1024

    @flyc.kernel(name=f"bw_w{waves}_c{chunk_kb}_n{ctas}_i{inflight}_nt{nt}", known_block_size=[32 * waves, 1, 1])
    def kern(src: Int64, out: Int64):
        tid = fx.thread_idx.x
        lane = tid % 32
        wave = tid // 32
        r = bo.create_buffer_resource_from_addr(src, num_records_bytes=total)
        base = ((fx.block_idx.x * waves + wave) * steps) * 256 + lane * 4
        acc = fx.Vector.filled(4, 0, fx.Int32)
        ring = []
        for st in range_constexpr(steps):
            ring.append([fx.Vector(bo.buffer_load(r, base + st * 256 + h * 128, vec_width=4, dtype=T.i32,
                                                  cache_modifier=nt)) for h in range(2)])
            if len(ring) > inflight:
                a, b = ring.pop(0)
                rocdl.sched_barrier(0)
                acc = acc ^ a ^ b
        for a, b in ring:
            acc = acc ^ a ^ b
        bo.buffer_store(acc, bo.create_buffer_resource_from_addr(out, num_records_bytes=ctas * waves * 32 * 16),
                        ((fx.block_idx.x * waves + wave) * 32 + lane) * 4)

    @flyc.jit
    def launch(src: Int64, out: Int64, stream: fx.Stream = fx.Stream(None)):
        kern(src, out).launch(grid=(ctas,), block=(32 * waves,), stream=stream)

    return launch, total


def main():
    src = torch.empty(2 << 30, dtype=torch.uint8, device="cuda")
    out = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    for waves, chunk_kb, inflight in ((2, 32, 8), (2, 64, 8), (4, 32, 8), (8, 32, 8), (2, 32, 16), (4, 64, 4)):
        ctas = (1 << 30) // (waves * chunk_kb * 1024)
        launch, total = make(waves, chunk_kb, ctas, inflight, 1)
        st = fx.Stream(torch.cuda.current_stream().cuda_stream)
        for i in range(3):
            launch(src.data_ptr() + (i % 2) * (1 << 30), out.data_ptr(), stream=st)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(10):
            launch(src.data_ptr() + (i % 2) * (1 << 30), out.data_ptr(), stream=st)
        e.record()
        e.synchronize()
        t = s.elapsed_time(e) / 10 * 1e-3
        print(f"waves {waves} chunk {chunk_kb} KB/wave inflight {inflight} steps: {total / t / 1e12:.2f} TB/s "
              f"({t * 1e6:.1f} us for {total >> 20} MB)", flush=True)


if __name__ == "__main__":
    main()
