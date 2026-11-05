"""Hardware probe of the gfx1250 primitives in ``common.py``: wave32 butterflies and ballots, LDS through
``fx.SharedAllocator``, the bf16 WMMA's operand symmetry (lane l of A and of B take the same K set, so any K
permutation both operands share leaves the dot unchanged), and a cross-CTA tagged-mailbox hand-off."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64, T

from gemv.common import ballot, lanes_below, popcount, rsrc, wave_max, wave_sum, wmma_bf16
from gemv.mailbox import Addr, Mailbox


@fx.struct
class Smem:
    buf: fx.Array[fx.Float32, 64, 16]


@flyc.kernel(name="gemv_probe", known_block_size=[64, 1, 1])
def probe(out: Int64, a: Int64, x: Int64, d: Int64, box: Int64, err: Int64):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    lane = tid % 32
    wave = tid // 32
    mb = Mailbox(7, err)
    poll, put = mb.poll, mb.put  # a method call on an object inside a traced branch makes it branch state
    if bid == 0:
        v = fx.Float32(lane * 3 + wave * 100) * fx.Float32(0.25)
        s = wave_sum(v)
        m = wave_max(v)
        bal = ballot(lane % 3 == 0)
        n = popcount(bal)
        lb = lanes_below(bal, lane)
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        fx.ptr_store(v, lds.buf.ptr + (63 - tid))
        gpu.barrier()
        back = fx.ptr_load(lds.buf.ptr + (63 - tid))
        bo.buffer_store(fx.Vector.from_elements([s, m, back, fx.Float32(0.0)], fx.Float32), rsrc(out), tid * 4)
        bo.buffer_store(fx.Vector.from_elements([bal, n, lb, fx.Int32(0)], fx.Int32), rsrc(out), 256 + tid * 4)
        # WMMA: A [16, 32] and X [16, 32] row-major bf16; lane l: row / column l % 16, K 16 (l // 16) .. + 16
        if wave == 0:
            g = lane // 16
            r = lane % 16
            av = fx.Vector(bo.buffer_load(rsrc(a), (r * 32 + g * 16) // 2, vec_width=4, dtype=T.i32)).shuffle(
                fx.Vector(bo.buffer_load(rsrc(a), (r * 32 + g * 16) // 2 + 4, vec_width=4, dtype=T.i32)),
                list(range(8)),
            )
            xv = fx.Vector(bo.buffer_load(rsrc(x), (r * 32 + g * 16) // 2, vec_width=4, dtype=T.i32)).shuffle(
                fx.Vector(bo.buffer_load(rsrc(x), (r * 32 + g * 16) // 2 + 4, vec_width=4, dtype=T.i32)),
                list(range(8)),
            )
            acc = wmma_bf16(av, xv, fx.Vector.filled(8, 0.0, fx.Float32))
            for vv in range_constexpr(8):
                bo.buffer_store(acc[vv], rsrc(d), (8 * g + vv) * 16 + r)
        got = poll([(Addr(box), tid, 1)])
        bo.buffer_store(got[0][0], rsrc(out), 512 + tid)
    else:
        put(Addr(box), tid, fx.Int32(1000 + tid))


@flyc.jit
def launch(out: Int64, a: Int64, x: Int64, d: Int64, box: Int64, err: Int64, stream: fx.Stream = fx.Stream(None)):
    probe(out, a, x, d, box, err).launch(grid=(2,), block=(64,), stream=stream)


def main():
    dev = "cuda"
    torch.manual_seed(0)
    out = torch.zeros(1024, dtype=torch.int32, device=dev)
    a = torch.randn(16, 32, device=dev).to(torch.bfloat16)
    x = torch.randn(16, 32, device=dev).to(torch.bfloat16)
    d = torch.zeros(16, 16, dtype=torch.float32, device=dev)
    box = torch.zeros(128, dtype=torch.int32, device=dev)
    err = torch.zeros(4, dtype=torch.int32, device=dev)
    launch(out.data_ptr(), a.data_ptr(), x.data_ptr(), d.data_ptr(), box.data_ptr(), err.data_ptr(),
           stream=fx.Stream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    f = out[:256].view(torch.float32).view(64, 4).cpu()
    i = out[256:512].view(64, 4).cpu()
    ok = True
    for w in range(2):
        v = [(l * 3 + w * 100) * 0.25 for l in range(32)]
        rows = f[32 * w: 32 * w + 32]
        ok &= bool((rows[:, 0] == sum(v)).all()) and bool((rows[:, 1] == max(v)).all())
    vals = torch.tensor([((t % 32) * 3 + (t // 32) * 100) * 0.25 for t in range(64)])
    ok_lds = bool((f[:, 2] == vals).all())
    bal = sum(1 << l for l in range(32) if l % 3 == 0)
    bal_s = bal - (1 << 32) if bal >= 1 << 31 else bal
    ok_bal = bool((i[:, 0] == bal_s).all()) and bool((i[:, 1] == 11).all())
    ok_lb = all(int(i[t, 2]) == sum(1 for l in range(t % 32) if l % 3 == 0) for t in range(64))
    werr = (d - a.float() @ x.float().T).abs().max().item()
    ok_mb = bool((out[512:576].cpu() == torch.arange(1000, 1064, dtype=torch.int32)).all()) and int(err[0]) == 0
    print(f"wave_sum/max {ok}  lds {ok_lds}  ballot {ok_bal}  lanes_below {ok_lb}  "
          f"wmma max|err| {werr:.3e} (symmetric K sets: {werr < 1e-3})  mailbox {ok_mb}")
    assert ok and ok_lds and ok_bal and ok_lb and werr < 1e-3 and ok_mb


if __name__ == "__main__":
    main()
