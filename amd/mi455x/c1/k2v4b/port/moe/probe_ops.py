"""gfx1250 probe of the primitives the MoE stage needs: cvt_scalef32_pk_fp4_f32 (E2M1 at a power-of-two
scale), cvt_pk_fp8_f32 (E4M3), wave32 xor shuffles, LDS, byte stores. Checks against the torch rules.

Run: flock /data/muse455/locks/gpu1.lock docker exec muse_mi455x bash -c 'cd /tmp && PYTHONSAFEPATH=1
     HIP_VISIBLE_DEVICES=1 timeout 300 python3 /data/muse455/k2/cand/k2v4/port/moe/probe_ops.py'
"""

import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T, as_ir_value

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref  # noqa: E402


def rs(a):
    return bo.create_buffer_resource_from_addr(a)


@flyc.kernel(name="moe_probe_cvt", known_block_size=[64, 1, 1])
def probe(x: Int64, code: Int64, o4: Int64, o8: Int64, osh: Int64, ob: Int64):
    tid = fx.thread_idx.x
    alloc = fx.SharedAllocator()

    @fx.struct
    class L:
        buf: fx.Array[fx.Float32, 64, 16]

    lds = alloc.allocate(L).peek()
    v = [fx.Float32(bo.buffer_load(rs(x), tid * 8 + j, vec_width=1, dtype=T.f32)) for j in range(8)]
    c = fx.Int32(bo.buffer_load(rs(code), tid, vec_width=1, dtype=T.i32))
    scale = (c << 23).bitcast(fx.Float32)
    pk = fx.Int32(0)
    for j in range_constexpr(4):
        pk = fx.Int32(rocdl.cvt_scalef32_pk_fp4_f32(T.i32, as_ir_value(pk), as_ir_value(v[2 * j]),
                                                   as_ir_value(v[2 * j + 1]), as_ir_value(scale), j))
    bo.buffer_store(pk, rs(o4), tid)
    inv = ((fx.Int32(254) - c) << 23).bitcast(fx.Float32)
    q = [fx.max(fx.min(a * inv, fx.Float32(448.0)), fx.Float32(-448.0)) for a in v]
    w0 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, as_ir_value(q[0]), as_ir_value(q[1]), as_ir_value(fx.Int32(0)), False))
    w0 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, as_ir_value(q[2]), as_ir_value(q[3]), as_ir_value(w0), True))
    w1 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, as_ir_value(q[4]), as_ir_value(q[5]), as_ir_value(fx.Int32(0)), False))
    w1 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, as_ir_value(q[6]), as_ir_value(q[7]), as_ir_value(w1), True))
    bo.buffer_store(fx.Vector.from_elements([w0, w1], fx.Int32), rs(o8), tid * 2)
    s = v[0]
    for off in (16, 8, 4, 2, 1):
        s = s + s.shuffle_xor(off, 32)
    fx.ptr_store(s, lds.buf.ptr + tid)
    gpu.barrier()
    bo.buffer_store(fx.ptr_load(lds.buf.ptr + (63 - tid)), rs(osh), tid)
    bo.buffer_store(fx.Int8(c), rs(ob), tid)


@flyc.jit
def launch(x: Int64, code: Int64, o4: Int64, o8: Int64, osh: Int64, ob: Int64, stream: fx.Stream = fx.Stream(None)):
    probe(x, code, o4, o8, osh, ob).launch(grid=(1,), block=(64,), stream=stream)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    x = (torch.randn(64, 8) * torch.exp2(torch.randint(-6, 6, (64, 1)).float())).to(torch.bfloat16).float()
    x[0, :] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0])  # E2M1 ties and saturation at scale 1
    x[1, :] = -x[0]
    # codes: the aiter rule per 8-group (the probe's group), A8 rule for o8
    xs = x.to(dev)
    code4 = ref.a4_code(xs.abs().amax(-1))
    code4[0] = code4[1] = 127
    o4 = torch.zeros(64, dtype=torch.int32, device=dev)
    o8 = torch.zeros(64, 2, dtype=torch.int32, device=dev)
    osh = torch.zeros(64, dtype=torch.float32, device=dev)
    ob = torch.zeros(64, dtype=torch.uint8, device=dev)
    code8 = ref.a8_code(xs.abs().amax(-1))
    for name, code in (("a4", code4), ("a8", code8)):
        launch(xs.data_ptr(), code.int().contiguous().data_ptr(), o4.data_ptr(), o8.data_ptr(), osh.data_ptr(),
               ob.data_ptr(), stream=fx.Stream(torch.cuda.current_stream().cuda_stream))
        torch.cuda.synchronize()
        if name == "a4":
            want = ref.e2m1_codes(xs / torch.exp2(code.float() - 127)[:, None])
            want = (want[:, 0::2] | (want[:, 1::2] << 4)).contiguous().view(torch.int32).view(-1)
            print("fp4 pack mismatches:", int((want != o4).sum()), "of 64 words")
            if (want != o4).any():
                i = int((want != o4).nonzero()[0])
                print("  row", i, x[i].tolist(), hex(int(want[i]) & 0xFFFFFFFF), hex(int(o4[i]) & 0xFFFFFFFF))
        else:
            want = ref.e4m3_bytes(xs, code)
            got = o8.view(torch.uint8).view(64, 8)
            print("fp8 pack mismatches:", int((want != got).sum()), "of 512 bytes")
            if (want != got).any():
                i = int((want != got).any(-1).nonzero()[0])
                print("  row", i, x[i].tolist(), want[i].tolist(), got[i].tolist())
    wsum = xs[:, 0].view(2, 32)
    s = wsum.clone()
    for off in (16, 8, 4, 2, 1):
        s = s[:, :off] + s[:, off:2 * off]
    want = s.expand(2, 32).reshape(64).flip(0)
    print("shuffle_xor butterfly matches halving order:", bool((want == osh).all()))
    print("byte store ok:", bool((ob == (code8.int() & 0xFF).to(torch.uint8)).all()))


if __name__ == "__main__":
    main()
