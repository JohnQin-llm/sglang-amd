# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# P252 parent-authored EP8 scaffold; arithmetic extracted from accepted Mono.
import functools
import hashlib
from pathlib import Path
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw
from flydsl.expr.typing import T, Int64, Int32
from sglang.srt.rocm_muse_spark.mono.ops.common.mx import FP4,UNIT_SCALE,mfma_scaled
from sglang.srt.rocm_muse_spark.mono.ops.common.ops import CM_DEV,CM_NT,bf16_round,bf16_pair,bf_lo,bf_hi,butterfly,rsrc,traced,uniform
from sglang.srt.rocm_muse_spark.mono.ops.common.plan import KernelAbi,source_digest
from sglang.srt.rocm_muse_spark.mono.ops.common.sync import Mailbox,publish,sreg
from sglang.srt.rocm_muse_spark.mono.ops.stages import mxq
E=32
RI=4096
MH=4096
TOPK=8
ROWS=16
UG_UNITS=1
UG_GROUPS=128
INTER_W=512
W13_K=4096
W13_STEPS=32
W13_SCALE_G=128
W13_BYTES=E*2*RI*W13_K//2
W13_SCALE_BYTES=E*2*RI*W13_SCALE_G
W2_STEPS=32
W2_SCALE_G=128
W2_BYTES=E*MH*RI//2
W2_SCALE_BYTES=E*MH*W2_SCALE_G
XQ_W=512
XQ_G=128
THREADS=512
WAVES=8
BLOCKS=256

EP8_ABI=KernelAbi((
    'hidden','ids','w13','s13','w2','s2','inter','scales','y','flags',
    'health','alarm','start',
))


def scale_dword(m, g, cols):
    """The dword of ``e8m0_shuffle``'d scales [rows, cols] holding (row m,
    column g); its byte is ``scale_byte(m, g)``."""
    return (m >> 5) * (8 * cols) + (g >> 3) * 64 + (g & 3) * 16 + (m & 15)

def scale_shift(m, g):
    """The bit shift of (row m, column g)'s byte in its ``scale_dword``."""
    return (((g >> 2) & 1) * 2 + ((m >> 4) & 1)) * 8

def ug_tile(c, task, units=UG_UNITS):
    """This wave's (slot, group, 16-row tile) of ug task ``task``: waves 4 u ..
    4 u + 3 take unit 2 task + u, wave 4 u + w its tile w (gate 2 g, gate 2 g +
    1, up 2 g, up 2 g + 1 of the 32-column group g)."""
    # A narrow-width build may schedule one 32-column group per CTA.  Clamp
    # the otherwise-idle waves onto the live unit so every derived address is
    # in range; their global loads are masked and their MFMA chain is skipped.
    wave = fx.min(c["wave"], 4 * units - 1)
    unit = task * units + wave // 4
    slot = unit // UG_GROUPS
    g = unit % UG_GROUPS
    wl = wave % 4
    rg = (wl // 2) * (RI // ROWS) + 2 * g + wl % 2
    return slot, g, rg

def ug_loads(c, task, units=UG_UNITS):
    """Task ``task``'s w13 tile for this wave (all 32 K steps) and its scale
    dwords (one a step pair)."""
    lane = c["lane"]
    a, rt = c["args"], c["rt"]
    slot, _g, rg = ug_tile(c, task, units)
    active = c["wave"] < 4 * units
    e = uniform(fx.ptr_load(rt["expert"] + slot))
    r_w = rsrc(a["w13"], W13_BYTES)
    base = e * (2 * RI * W13_K // 8)
    av = [
        fx.Vector(
            bo.buffer_load(
                r_w,
                base + rg * (2 * W13_K) + st * 256 + lane * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
                mask=active,
            )
        )
        for st in range(W13_STEPS)
    ]
    r_s = rsrc(a["w13s"], W13_SCALE_BYTES)
    m = e * (2 * RI) + rg * ROWS + lane % 16
    sw = [
        fx.Int32(
            bo.buffer_load(
                r_s,
                scale_dword(m, sp * 8 + lane // 16, W13_SCALE_G),
                vec_width=1,
                dtype=T.i32,
                mask=active,
            )
        )
        for sp in range(W13_STEPS // 2)
    ]
    return av + sw

@traced
def _stage_ug_epilogue(c, task, units=UG_UNITS):
    """Shared, bit-identical SiLU/BF16/MXFP4 epilogue for both W13 paths."""
    s, tid, red = c["S"], c["tid"], c["red"]
    rt = c["rt"]
    gpu.barrier()
    active_thread = tid < 256 * units
    u = fx.min(tid // 256, units - 1)
    tt = (tid % 256) // 32
    col = tid % 32
    live = active_thread & (tt < s)
    tt = fx.min(tt, s - 1)
    half = col // 16
    r = col % 16
    gv = fx.ptr_load(red + (((4 * u + half) * 64 + 16 * (r // 4) + tt) * 4 + r % 4))
    uv = fx.ptr_load(red + (((4 * u + 2 + half) * 64 + 16 * (r // 4) + tt) * 4 + r % 4))
    v = mxq.silu_mul(gv, uv)
    # Match the native two-stage q8 BF16 boundary before MXFP4 quantization.
    # Narrower widths preserve the original monokernel arithmetic.
    if s == 8:
        v = bf16_round(v)
    amax = butterfly(abs(v), (16, 8, 4, 2, 1), fx.max)
    code = mxq.inter_code(amax)
    gpu.barrier()
    if active_thread:
        fx.ptr_store(v, c["av"] + tid)
    gpu.barrier()
    uslot = (task * units + u) // UG_GROUPS
    ug = (task * units + u) % UG_GROUPS
    k = fx.ptr_load(rt["kof"] + (uslot * 8 + tt))
    if live & (col < 4) & (k >= 0):
        scale = mxq.code_scale(code)
        vals = [fx.ptr_load(c["av"] + (tid - col + col * 8 + j)) for j in range(8)]
        pk = mxq.pack8(vals, scale)
        pick = tt * TOPK + k
        bo.buffer_store(
            pk, rsrc(c["interb"]), pick * INTER_W + ug * 4 + col, cache_modifier=CM_DEV
        )
        if col == 0:
            bo.buffer_store(
                code, rsrc(c["iscb"]), pick * UG_GROUPS + ug, cache_modifier=CM_DEV
            )
    publish(c["put"], c["ugrdy"], task, fx.Int32(1), tid == 0)
    gpu.barrier()

@traced
def stage_ug(c, task, ops, units=UG_UNITS):
    """Task ``task``: one or two 32-column groups, a wave a 16-row tile against
    the MXFP4 xe (one MFMA chain over K); silu(g) u; MXFP4 a 32-group -> INTER /
    ISC of each picking token."""
    s, lane, wave, red = c["S"], c["lane"], c["wave"], c["red"]
    _slot, _g, rg = ug_tile(c, task, units)
    av = ops[:W13_STEPS]
    sw = ops[W13_STEPS:]
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    if wave < 4 * units:
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for st in range_constexpr(W13_STEPS):
            bw = fx.Vector(
                fx.ptr_load(
                    c["xq"] + (t * XQ_W + st * 16 + kb * 4),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            sb = fx.ptr_load(c["xsc"] + (t * XQ_G + st * 4 + kb))
            sa = (sw[st // 2] >> ((st % 2) * 16 + (rg % 2) * 8)) & 0xFF
            acc = mfma_scaled(av[st], bw, acc, sa, sb, FP4, FP4, 0, 0)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    _stage_ug_epilogue(c, task, units)

def expert_down_loads(c, slot, rgd):
    """Candidate W2 operands with wave-uniform terms in scalar offsets."""
    lane = c["lane"]
    r_w = rsrc(c["args"]["w2"], W2_BYTES)
    r_s = rsrc(c["args"]["w2s"], W2_SCALE_BYTES)
    e = uniform(fx.ptr_load(c["rt"]["expert"] + slot))
    av = [
        fx.Vector(
            bo.buffer_load(
                r_w,
                lane * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
                soffset_bytes=uniform(
                    (e * (MH * RI // 8) + rgd * (2 * RI) + st * 256) * 4
                ).ir_value(),
            )
        )
        for st in range(W2_STEPS)
    ]
    sw = [
        fx.Int32(
            bo.buffer_load(
                r_s,
                scale_dword(
                    e * MH + rgd * ROWS + lane % 16,
                    sp * 8 + lane // 16,
                    W2_SCALE_G,
                ),
                vec_width=1,
                dtype=T.i32,
            )
        )
        for sp in range(W2_STEPS // 2)
    ]
    return av + sw

@traced
def stage_down_expert(c, slot, rgd, ops):
    """One expert and one wave-owned W2 tile -> direct bf16 YR partials."""
    s, lane = c["S"], c["lane"]
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    k = fx.ptr_load(c["rt"]["kof"] + (slot * 8 + t))
    picked = (k >= 0) & (lane % 16 < s)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for st in range_constexpr(W2_STEPS):
        bw = fx.Vector(
            fx.ptr_load(
                c["interq"] + (t * INTER_W + st * 16 + kb * 4),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        sb = fx.ptr_load(c["intsc"] + (t * UG_GROUPS + st * 4 + kb))
        sw = ops[W2_STEPS + st // 2]
        sa = (sw >> ((st % 2) * 16 + (rgd % 2) * 8)) & 0xFF
        acc = mfma_scaled(ops[st], bw, acc, sa, sb, FP4, FP4, 0, 0)
    if picked:
        c["put_bf"](
            c["yr_owner"],
            (c["rank"] * s * TOPK + t * TOPK + k) * MH + rgd * ROWS + 4 * (lane // 16),
            [bf16_round(acc[i]) for i in range(4)],
        )
    gpu.barrier()

@traced
def load_input(c, hidden):
    tid=c['tid']
    for token in range_constexpr(8):
        v=fx.Vector(bo.buffer_load(rsrc(hidden),token*2048+tid*4,vec_width=4,dtype=T.i32))
        xs=[]
        for j in range_constexpr(4):
            xs.append(bf_lo(v[j]));xs.append(bf_hi(v[j]))
        amax=fx.Float32(0.)
        for v0 in xs:amax=fx.max(amax,abs(v0))
        amax=butterfly(amax,(2,1),fx.max)
        code=mxq.act_code(amax)
        pk=mxq.pack8(xs,mxq.code_scale(code))
        fx.ptr_store(pk,c['xq']+token*512+tid)
        if tid%4==0:fx.ptr_store(code,c['xsc']+token*128+tid//4)
    gpu.barrier()

@traced
def load_local_inter(c, expert):
    tid=c['tid']
    if tid<128:c['poll']([(c['ugrdy'],expert*128+tid,1)])
    gpu.barrier()
    for i in range_constexpr(8):
        off=tid+i*512
        token=off//512
        k=fx.ptr_load(c['rt']['kof']+expert*8+token)
        v=fx.Int32(bo.buffer_load(rsrc(c['interb'],64*512*4),(token*8+fx.max(k,0))*512+off%512,vec_width=1,dtype=T.i32,cache_modifier=CM_DEV,mask=k>=0))
        fx.ptr_store((k>=0).select(v,fx.Int32(0)),c['interq']+off)
    for i in range_constexpr(2):
        off=tid+i*512
        token=off//128
        k=fx.ptr_load(c['rt']['kof']+expert*8+token)
        v=fx.Int32(bo.buffer_load(rsrc(c['iscb'],64*128*4),(token*8+fx.max(k,0))*128+off%128,vec_width=1,dtype=T.i32,cache_modifier=CM_DEV,mask=k>=0))
        fx.ptr_store((k>=0).select(v,fx.Int32(UNIT_SCALE)),c['intsc']+off)
    gpu.barrier()

@traced
def put_bf_local(dst,offset,values):
    out=fx.Vector.from_elements([
        bf16_pair(values[0],values[1]).bitcast(fx.Int32),
        bf16_pair(values[2],values[3]).bitcast(fx.Int32),
    ],fx.Int32)
    bo.buffer_store(out,rsrc(dst),offset//2,cache_modifier=CM_DEV)


@traced
def run_experts(front):
    # Expert stages write complete local Y; their rank address must stay zero.
    c = dict(front)
    c["rank"] = fx.Int32(0)
    c["put_bf"] = put_bf_local
    c["yr_owner"] = front["expert_y"]
    tid,bid=c["tid"],c["bid"]
    rt=c["rt"]
    start=front["rank"]*32
    if tid<32:
        present=fx.Int32(0)
        for j in range_constexpr(64):
            e=fx.ptr_load(rt["route"]+(j//8)*16+j%8)
            present=(e==start+tid).select(fx.Int32(1),present)
        fx.ptr_store(present,rt["expert"]+tid)
    gpu.barrier()
    if tid==0:
        count=fx.Int32(0)
        for e in range_constexpr(32):
            is_active=fx.ptr_load(rt["expert"]+e)
            fx.ptr_store(fx.Int32(e),rt["active_ids"]+count)
            count=count+is_active
        fx.ptr_store(count,rt["count"])
    gpu.barrier()
    active=uniform(fx.ptr_load(rt["count"]))
    if active>0:
        team=uniform(bid%active)
        member=uniform(bid//active)
        team_size=uniform((255-team)//active+1)
        expert=uniform(fx.ptr_load(rt["active_ids"]+team))
        if tid<32:fx.ptr_store(tid,rt["expert"]+tid)
        if tid<8:
            k=fx.Int32(-1)
            for j in range_constexpr(8):
                e=fx.ptr_load(rt["route"]+tid*16+j)
                k=(e==start+expert).select(fx.Int32(j),k)
            fx.ptr_store(k,rt["kof"]+expert*8+tid)
        gpu.barrier()
        group=member
        while group<128:
            task=expert*128+group
            stage_ug(c,task,ug_loads(c,task,1),1)
            group=group+team_size
        load_local_inter(c,expert)
        rgd=member*8+c["wave"]
        while rgd<256:
            stage_down_expert(c,expert,rgd,expert_down_loads(c,expert,rgd))
            rgd=rgd+team_size*8

# normalize_route/local_sum/token_reduce are the narrow KernelForge task.

@traced
def normalize_route(c,slot):
    # Vendor normalization extraction; KernelForge tagged-mailbox bridge.
    tid,lane,wave=c['tid'],c['lane'],c['wave']
    if tid<256:c['poll']([(c['expertdone'],tid,1)])
    gpu.barrier()
    rank=c['rank']
    e=uniform(fx.ptr_load(c['rt']['route']+(slot//8)*16+slot%8))
    if (e>=rank*32)&(e<rank*32+32):
        active=tid<256
        safe_tid=tid%256
        values=[]
        ss=arith.constant(0.0,type=T.f32)
        for chunk in range_constexpr(4):
            h=safe_tid*4+chunk*1024
            yraw=bo.buffer_load(rsrc(c['expert_y'],64*4096*2),slot*4096+h,vec_width=4,dtype=T.bf16,cache_modifier=CM_DEV)
            wraw=bo.buffer_load(rsrc(c['args']['w_post'],4096*2),h,vec_width=4,dtype=T.bf16,cache_modifier=CM_DEV)
            yv=fx.Vector(yraw,shape=4,dtype=fx.BFloat16)
            wv=fx.Vector(wraw,shape=4,dtype=fx.BFloat16)
            for k in range_constexpr(4):
                scale=arith.extf(T.f32,arith.truncf(T.bf16,arith.addf(arith.extf(T.f32,_to_raw(wv[k])),arith.constant(1.0,type=T.f32))))
                value=arith.extf(T.f32,arith.truncf(T.bf16,arith.mulf(arith.extf(T.f32,_to_raw(yv[k])),scale)))
                values.append(value)
                ss=arith.addf(ss,arith.mulf(value,value))
        for shift in (32,16,8,4,2,1):
            peer=_to_raw(ArithValue(ss).shuffle_xor(shift,64))
            ss=arith.addf(ss,peer)
        if active&(lane==0):fx.ptr_store(fx.Float32(ss),c['red']+wave)
        gpu.barrier()
        total=arith.constant(0.0,type=T.f32)
        for w in range_constexpr(4):total=arith.addf(total,_to_raw(fx.ptr_load(c['red']+w)))
        mean=arith.addf(arith.mulf(total,arith.constant(1.0/4096.0,type=T.f32)),_to_raw(fx.Float32(c['eps_post'])))
        score=fx.ptr_load(c['rt']['route']+(slot//8)*16+8+slot%8).bitcast(fx.Float32)
        sc=arith.mulf(_to_raw(score),arith.divf(arith.constant(1.0,type=T.f32),fx.rocdl.sqrt(T.f32,mean)))
        for chunk in range_constexpr(4):
            h=safe_tid*4+chunk*1024
            out=fx.Vector.from_elements([arith.mulf(sc,values[chunk*4+k]) for k in range_constexpr(4)],fx.Float32)
            bo.buffer_store(out,rsrc(c['normy'],64*4096*4),slot*4096+h,cache_modifier=CM_DEV,mask=_to_raw(active))
    publish(c['put'],c['normrdy'],slot,fx.Int32(1),tid==0)

@traced
def local_sum(c,t):
    # KernelForge rank-ordered local contribution bridge.
    tid=c['tid']
    if tid<8:c['poll']([(c['normrdy'],t*8+tid,1)])
    gpu.barrier()
    rank=c['rank']
    h=tid*8
    sums=fx.Vector.filled(8,0.0,fx.Float32)
    for slot in range_constexpr(8):
        e=fx.ptr_load(c['rt']['route']+t*16+slot)
        if (e>=rank*32)&(e<rank*32+32):
            raw0=bo.buffer_load(rsrc(c['normy'],64*4096*4),(t*8+slot)*4096+h,vec_width=4,dtype=T.f32,cache_modifier=CM_DEV)
            raw1=bo.buffer_load(rsrc(c['normy'],64*4096*4),(t*8+slot)*4096+h+4,vec_width=4,dtype=T.f32,cache_modifier=CM_DEV)
            v0=fx.Vector(raw0,shape=4,dtype=fx.Float32)
            v1=fx.Vector(raw1,shape=4,dtype=fx.Float32)
            v=fx.Vector.from_elements([v0[k] for k in range_constexpr(4)]+[v1[k] for k in range_constexpr(4)],fx.Float32)
            sums=fx.Vector.from_elements([arith.addf(_to_raw(sums[k]),_to_raw(v[k])) for k in range_constexpr(8)],fx.Float32)
    offset=(rank*8+t)*4096+h
    owner=h//512
    for p in range_constexpr(8):
        if owner==p:
            c['put_bf'](c['peer']['epfin'][p],offset,[sums[k] for k in range_constexpr(4)])
            c['put_bf'](c['peer']['epfin'][p],offset+4,[sums[4+k] for k in range_constexpr(4)])
    gpu.barrier()

@traced
def token_reduce(c,j):
    # KernelForge rank-ordered tagged reduction bridge.
    t=c['wave']
    col=j*128+c['lane']*2
    lo=arith.constant(0.0,type=T.f32)
    hi=arith.constant(0.0,type=T.f32)
    for src in range_constexpr(8):
        packed=c['poll']([(c['peer']['epfin_own'],(src*8+t)*2048+col//2,1)])[0][0]
        lo=arith.addf(lo,_to_raw(bf_lo(packed)))
        hi=arith.addf(hi,_to_raw(bf_hi(packed)))
    out=bf16_pair(lo,hi).bitcast(fx.Int32)
    for p in range_constexpr(8):
        c['put_words'](c['peer']['owned_fin'][p],(t*4096+col)//2,[out])
    gpu.barrier()
