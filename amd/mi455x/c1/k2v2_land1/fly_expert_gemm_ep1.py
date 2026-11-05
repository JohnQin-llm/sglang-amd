"""EP1 copy of ``port/fly_expert_gemm.py`` with an A8W4 mode (``act_fp8``): MXFP8 e4m3 activations (fmtB=0),
16 dwords a lane, chunk c (dwords 4c..4c+3) = K 32c + 16 (lane // 16) .. +15 of the 128-k step (``s2/wmma_probe_fp8.py``).
Scales are unchanged (one e8m0 per 32 k; the lane's dword = 4 K blocks of step st + lane // 16, opsel = step parity).

gfx1250 (MI455X) FlyDSL expert GEMM: the TP4/EP4 K2's gate/up (and down) expert stage as a standalone
grouped kernel, A4W4 on V_WMMA_SCALE_F32_16X16X128_F8F6F4 (wave32).

Work units are the 16-row blocks of the Triton small align (``sorted_ids`` / ``expert_ids`` /
``num_post``, BLOCK_M = 16): unit u = local expert ``expert_ids[u]`` x the routes ``sorted_ids[16u ..]``.
CTA (u, nb) runs ``WAVES`` waves; wave w owns 16 output columns (tile ``nb * WAVES + w``): for stage 1
(``fuse_silu``) the gate rows of that tile and the matching up rows, one fp32 scaled-WMMA chain each over
K ascending (128 a step), then ``silu(g) * u`` (Triton's ``acc * sigmoid(acc) * up``) -> bf16 per route;
stage 2 one chain -> bf16 per route (unweighted, as the served path's ``y``).

Operands (``layout_gfx1250``): A = weights in the (16,16) preshuffle (the MI355X single-layout bytes), two
16 B loads a lane a step; A scale one dword a lane a step pair (gfx1250 scale order, opsel = step parity);
B = MXFP4 activations row-major (K chunks l//16 and l//16+2: two 16 B loads), B scale the row's dword of
the step pair (lanes 16..31 the odd step).  Activations come straight from global/L2 (16 KB a layer).
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int64, T

TOPK = 8
LOG2E = 1.4426950408889634


def _rsrc(addr, nbytes):
    """Sized: a masked buffer op moves its offset past ``nbytes`` (an unsized resource would fault)."""
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def _exp2(x):
    return fx.Float32(llvm.call_intrinsic(T.f32, "llvm.amdgcn.exp2.f32", [fx.Float32(x).ir_value()], [], []))


def _uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def _unit_body(xq, xs, w, ws, sorted_ids, expert_ids, out, tid, u, nb, *, n_out, k, slots, fuse_silu, waves,
               prefetch, nt, n_exp, units, act_fp8=False):
    """One work unit's tile (plain Python at trace time: static control flow only)."""
    steps = k // 128
    rows_w = 2 * n_out if fuse_silu else n_out
    w_dw_e = rows_w * k // 8
    tiles = n_out // 16
    a_dw_row = k // 4 if act_fp8 else k // 8
    a_s_row = k // 128
    lane = tid % 32
    wave = tid // 32
    a_rows = slots // TOPK if fuse_silu else slots
    e = _uniform(bo.buffer_load(_rsrc(expert_ids, units * 4), u, vec_width=1, dtype=T.i32))
    slot = fx.Int32(bo.buffer_load(_rsrc(sorted_ids, units * 64), u * 16 + lane % 16, vec_width=1, dtype=T.i32))
    live = slot < slots
    sl = fx.min(slot, slots - 1)
    arow = sl // TOPK if fuse_silu else sl
    rg = nb * waves + wave
    g = lane // 16
    rw, rws = _rsrc(w, n_exp * w_dw_e * 4), _rsrc(ws, n_exp * w_dw_e // 4)
    rx, rxs = _rsrc(xq, a_rows * k // (1 if act_fp8 else 2)), _rsrc(xs, a_rows * k // 32)
    wbase = [e * w_dw_e + (rg + j * tiles) * (2 * k) + g * 64 + (lane % 16) * 4
             for j in range(2 if fuse_silu else 1)]
    sbase = [(e * (rows_w // 16) + rg + j * tiles) * (k // 256) * 32 + lane
             for j in range(2 if fuse_silu else 1)]
    xbase = arow * a_dw_row + g * 4
    xsbase = arow * a_s_row + g

    def load(st):
        ops = {}
        ops["a"] = [fx.Vector(bo.buffer_load(rw, wb + st * 256, vec_width=4, dtype=T.i32, cache_modifier=nt))
                    .shuffle(fx.Vector(bo.buffer_load(rw, wb + st * 256 + 128, vec_width=4, dtype=T.i32,
                                                      cache_modifier=nt)), list(range(8)))
                    for wb in wbase]
        if act_fp8:
            bq = [fx.Vector(bo.buffer_load(rx, xbase + st * 32 + 8 * c, vec_width=4, dtype=T.i32)) for c in range(4)]
            ops["b"] = bq[0].shuffle(bq[1], list(range(8))).shuffle(bq[2].shuffle(bq[3], list(range(8))),
                                                                    list(range(16)))
        else:
            ops["b"] = fx.Vector(bo.buffer_load(rx, xbase + st * 16, vec_width=4, dtype=T.i32)).shuffle(
                fx.Vector(bo.buffer_load(rx, xbase + st * 16 + 8, vec_width=4, dtype=T.i32)), list(range(8)))
        if st % 2 == 0:
            ops["sa"] = [bo.buffer_load(rws, sb + (st // 2) * 32, vec_width=1, dtype=T.i32) for sb in sbase]
            ops["sb"] = bo.buffer_load(rxs, xsbase + st, vec_width=1, dtype=T.i32)
        return ops

    accs = [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(2 if fuse_silu else 1)]
    ring = [load(st) for st in range(min(prefetch, steps))]
    sa = sb = None
    for st in range_constexpr(steps):
        if st + prefetch < steps:
            ring.append(load(st + prefetch))
        ops = ring[st]
        if st % 2 == 0:
            sa, sb = ops["sa"], ops["sb"]
        for j in range_constexpr(len(accs)):
            accs[j] = fx.Vector(rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                T.vec(8, T.f32), ops["a"][j], ops["b"], accs[j], sa[j], sb, fmtA=4, fmtB=0 if act_fp8 else 4,
                scaleAType=st % 2, scaleBType=st % 2))
        ring[st] = None
    vals = []
    for v in range_constexpr(8):
        if fuse_silu:
            gt, up = accs[0][v], accs[1][v]
            r = fx.Float32(1.0) / (_exp2(gt * fx.Float32(-LOG2E)) + fx.Float32(1.0))
            vals.append((gt * r) * up)
        else:
            vals.append(accs[0][v])
    packed = fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    off = (sl * n_out + rg * 16 + 8 * g) // 2
    bo.buffer_store(packed, _rsrc(out, slots * n_out * 2), off, mask=live)


@functools.cache
def make(n_out: int, k: int, slots: int, units: int, fuse_silu: bool, waves: int = 4, prefetch: int = 8,
         nt: int = 0, n_exp: int = 64, act_fp8: bool = False):
    """Launcher for one shape. ``n_out``: output columns (I for stage 1, H for stage 2); ``k``: reduction
    dim; ``slots`` = M * TOPK; ``units``: grid x (max 16-row units); ``nt``: weight-load cache modifier."""
    steps = k // 128
    rows_w = 2 * n_out if fuse_silu else n_out
    w_dw_e = rows_w * k // 8
    s_dw_e = rows_w // 16 * (k // 256) * 32
    tiles = n_out // 16
    assert tiles % waves == 0 and steps % 2 == 0
    ctas_n = tiles // waves
    name = (f"k2g_expert_gemm_e{n_exp}_n{n_out}_k{k}_s{slots}_f{int(fuse_silu)}_w{waves}_p{prefetch}_nt{nt}"
            f"_a{8 if act_fp8 else 4}")

    @flyc.kernel(name=name, known_block_size=[32 * waves, 1, 1])
    def kern(xq: Int64, xs: Int64, w: Int64, ws: Int64, sorted_ids: Int64, expert_ids: Int64, num_post: Int64,
             out: Int64):
        tid = fx.thread_idx.x
        u = fx.block_idx.x
        nb = fx.block_idx.y
        npost = _uniform(bo.buffer_load(_rsrc(num_post, 4), 0, vec_width=1, dtype=T.i32))
        if u * 16 < npost:
            _unit_body(xq, xs, w, ws, sorted_ids, expert_ids, out, tid, u, nb, n_out=n_out, k=k, slots=slots,
                       fuse_silu=fuse_silu, waves=waves, prefetch=prefetch, nt=nt, n_exp=n_exp,
                       units=units, act_fp8=act_fp8)

    @flyc.jit
    def launch(xq: Int64, xs: Int64, w: Int64, ws: Int64, sorted_ids: Int64, expert_ids: Int64, num_post: Int64,
               out: Int64, stream: fx.Stream = fx.Stream(None)):
        kern(xq, xs, w, ws, sorted_ids, expert_ids, num_post, out).launch(grid=(units, ctas_n), block=(32 * waves,),
                                                                          stream=stream)

    return launch


def run(launch, xq, xs, w, ws, sorted_ids, expert_ids, num_post, out):
    import torch

    launch(xq.data_ptr(), xs.data_ptr(), w.data_ptr(), ws.data_ptr(), sorted_ids.data_ptr(), expert_ids.data_ptr(),
           num_post.data_ptr(), out.data_ptr(), stream=fx.Stream(torch.cuda.current_stream().cuda_stream))
