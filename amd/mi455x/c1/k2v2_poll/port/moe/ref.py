"""Torch reference of the gfx1250 TP4/EP1 MoE stage, the kernel's arithmetic step for step (so the
comparison is bit for bit), plus the quantization rules it shares with ``mxq``.

Rules (``act``):
  a4  : aiter ``_mxfp4_quant_op`` (production compact-A4, the TP4 K2's ``mxq.a4_code``): code =
        max(((bits(amax) + 0x200000) >> 23 & 0xFF) - 2, 0); E2M1 RNE saturating at 6; code 0 -> all-zero group.
  asm : the MI355X K2 (TP8 k2bundle ``mxq.act_code`` / ``inter_code``): code = e8m0_ceil(fp32(amax * fp32(1/6)))
        (inter: amax floored at 1e-6); E2M1 RNE saturating at 6.
  a8  : OCP MXFP8 E4M3 (the lead's EP1 decision): code = max(expfield(amax) - 8, 0); x * 2^(127 - code)
        clamped to +-448, RNE to E4M3.
"""

from __future__ import annotations

import torch

LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
RCP6 = torch.tensor([0x3E2AAAAB], dtype=torch.int32).view(torch.float32)
INTER_FLOOR = torch.tensor([0x358637BD], dtype=torch.int32).view(torch.float32)


def _bits(x):
    return x.float().contiguous().view(torch.int32)


def a4_code(amax):
    return torch.clamp(((((_bits(amax) + 0x200000) >> 23) & 0xFF) - 2), min=0)


def e8m0_ceil(t):
    b = _bits(t)
    return ((b >> 23) & 0xFF) + ((b & 0x7FFFFF) != 0).int()


def asm_code(amax, inter=False):
    a = amax.float()
    if inter:
        a = torch.maximum(a, INTER_FLOOR.to(a.device))
    return e8m0_ceil(a * RCP6.to(a.device))


def a8_code(amax):
    return torch.clamp(((_bits(amax) >> 23) & 0xFF) - 8, min=0)


def e2m1_codes(q):
    """fp32 already divided by the scale -> E2M1 codes (RNE, saturating), sign bit for negatives (incl. -0)."""
    a = q.abs()
    c = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int() + (a > 2.5).int()
         + (a >= 3.5).int() + (a > 5).int())
    sign = (q.float().contiguous().view(torch.int32) < 0).int()
    return (c | (sign << 3)).to(torch.uint8)


def e4m3_bytes(x, code):
    """x [.., n] fp32, code [..] (one per row of x) -> E4M3 bytes."""
    inv = ((254 - code.int()) << 23).view(torch.float32)
    q = torch.clamp(x.float() * inv[..., None], -448.0, 448.0)
    return q.to(torch.float8_e4m3fn).view(torch.uint8)


def quant(x, act, inter=False):
    """x [R, K] (fp32 holding the kernel's values) -> (packed bytes [R, K/2] (a4/asm) or [R, K] (a8), codes u8
    [R, K/32])."""
    r, k = x.shape
    xf = x.float().reshape(r, k // 32, 32)
    amax = xf.abs().amax(-1)
    if act == "a8":
        code = a8_code(amax)
        return e4m3_bytes(xf, code).reshape(r, k).contiguous(), code.to(torch.uint8).contiguous()
    code = a4_code(amax) if act == "a4" else asm_code(amax, inter)
    scale = (code << 23).view(torch.float32)
    q = xf / torch.where(scale > 0, scale, torch.ones_like(scale))[..., None]
    c = e2m1_codes(q)
    if act == "a4":
        c = torch.where(code[..., None] > 0, c, torch.zeros_like(c))
    c = c.reshape(r, k)
    return (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous(), code.to(torch.uint8).contiguous()


LUT_W = LUT


def dequant_w(w, s):
    """Native [.., N, K/2] e2m1x2 + [.., N, K/32] e8m0 -> fp32 [.., N, K]."""
    codes = torch.stack([w & 0xF, w >> 4], -1).reshape(*w.shape[:-1], w.shape[-1] * 2)
    return LUT_W.to(w.device)[codes.long()] * torch.exp2(s.float() - 127).repeat_interleave(32, -1)


def _fp32_cpu(x):
    return x.float().cpu()


def inv_sqrt(x):
    """1 / sqrt(x) with both steps correctly rounded in fp32 (the kernel's ``inv_sqrt_ieee``)."""
    sq = x.double().cpu().sqrt().float()
    return (torch.ones_like(sq) / sq).to(x.device)


def halve(x, n):
    """The wave32 butterfly sum over the last dim (size n = 32): [:off] + [off:2 off], off = 16, 8, 4, 2, 1."""
    off = n // 2
    while off >= 1:
        x = x[..., :off] + x[..., off:2 * off]
        off //= 2
    return x[..., 0]


def seq_sum(x, dim=-1, start=None):
    """Left-to-right fp32 sum over ``dim`` (from 0.0 unless ``start`` is False: then from the first element)."""
    x = x.movedim(dim, 0)
    acc = torch.zeros_like(x[0]) if start is None else x[0].clone()
    for i in range(0 if start is None else 1, x.shape[0]):
        acc = acc + x[i]
    return acc


def bf(x):
    return x.float().to(torch.bfloat16).float()


def xq(pool, w_pre, act, eps=1e-5):
    """``stage_xq``: pool [S, MH] bf16 -> (xe fp32 of bf16 values, packed, codes)."""
    s, mh = pool.shape
    x = pool.float()
    p = (x * x).view(s, mh // 256, 32, 8)  # [t][wave][lane][8]
    sq = halve(seq_sum(p, -1), 32)  # [t][wave]
    tot = seq_sum(sq, -1, start=False)
    r = inv_sqrt(tot * (1.0 / mh) + torch.tensor(eps, dtype=torch.float32))
    w1 = bf(w_pre.float() + 1.0)
    xe = bf((x * r[:, None]) * w1[None])
    q, c = quant(xe, act)
    return xe, q, c


def exp2_rn(x):
    return torch.exp2(x.double()).float()


def silu_mul(g, u):
    lg = torch.tensor([0x3FB8AA3B], dtype=torch.int32).view(torch.float32).to(g.device)
    r = torch.ones_like(g) / (exp2_rn(g * (-lg)) + 1.0)
    return (g * r) * u


def chain(w, x, steps, parts=1, single_round=False):
    """The scaled-WMMA chain: per 128-k step the exact dot (fp64), the fp32 accumulator over steps in order
    (``single_round``: acc + dot rounded once, else the dot rounded first); ``parts`` in-CTA K parts summed in
    part order. w [N, K] fp32, x [M, K] fp32 -> [M, N] fp32."""
    n, k = w.shape
    m = x.shape[0]
    d = torch.einsum("nsk,msk->mns", w.double().view(n, steps, k // steps), x.double().view(m, steps, k // steps))
    per = steps // parts
    out = None
    for p in range(parts):
        acc = torch.zeros(m, n, dtype=torch.float32, device=w.device)
        for st in range(p * per, (p + 1) * per):
            acc = (acc.double() + d[..., st]).float() if single_round else acc + d[..., st].float()
        out = acc if out is None else out + acc
    return out


def experts_rank(xe_q, xe_s, ids, slices, act, ri, ksplit=1, single_round=False):
    """One rank's ug + down from its (native) slices: ``slices[e] = (w13, s13, w2, s2)``; xe_q / xe_s the
    quantized xe [S, ..]. Returns (inter fp32 bf16-valued [S*TOPK, ri], inter packed, codes, yr bf16 [S*TOPK, MH])."""
    inter, iq, isc = ug_rank(xe_q, xe_s, ids, slices, act, ri, ksplit, single_round)
    return inter, iq, isc, down_rank(iq, isc, ids, slices, act, ri, single_round)


def ug_rank(xe_q, xe_s, ids, slices, act, ri, ksplit=1, single_round=False):
    s, topk = ids.shape
    xd = dequant(xe_q, xe_s, act)
    mh = xd.shape[1]
    inter = torch.zeros(s * topk, ri, dtype=torch.float32, device=xd.device)
    for e in sorted(set(ids.view(-1).tolist())):
        tok = (ids == e).nonzero()
        w13, s13, _, _ = slices[e]
        gu = chain(dequant_w(w13, s13), xd[tok[:, 0]], mh // 128, ksplit, single_round)
        inter[tok[:, 0] * topk + tok[:, 1]] = bf(silu_mul(gu[:, :ri], gu[:, ri:]))
    iq, isc = quant(inter, act, inter=True)
    return inter, iq, isc


def down_rank(iq, isc, ids, slices, act, ri, single_round=False):
    s, topk = ids.shape
    idq = dequant(iq, isc, act)
    mh = slices[int(ids[0, 0])][2].shape[0]
    yr = torch.zeros(s * topk, mh, dtype=torch.float32, device=idq.device)
    for e in sorted(set(ids.view(-1).tolist())):
        tok = (ids == e).nonzero()
        pick = tok[:, 0] * topk + tok[:, 1]
        _, _, w2, s2 = slices[e]
        yr[pick] = bf(chain(dequant_w(w2, s2), idq[pick], ri // 128, 1, single_round))
    return yr


def fin(yr_ranks, weights, w_post, eps=1e-8, fin_cols=128):
    """``stage_fin``: yr_ranks [TP][S*TOPK, MH] (bf16 values), weights [S, TOPK] fp32 -> fin [S, MH] (bf16
    values)."""
    tp = len(yr_ranks)
    s, topk = weights.shape
    mh = yr_ranks[0].shape[1]
    y = yr_ranks[0].float()
    for r in range(1, tp):
        y = y + yr_ranks[r].float()
    y = bf(y).view(s, topk, mh)
    w1 = bf(w_post.float() + 1.0)
    v = bf(y * w1)
    sl = mh // fin_cols
    p = (v * v).view(s, topk, sl, 32, fin_cols // 32)
    ss = halve(seq_sum(p, -1), 32)  # [t][k][slice]
    tot = halve(ss, sl)
    sc = weights.float() * inv_sqrt(tot * (1.0 / mh) + torch.tensor(eps, dtype=torch.float32))
    prod = sc[..., None] * v
    return bf(seq_sum(prod, 1))


def dequant(q, s, act):
    r = q.shape[0]
    if act == "a8":
        v = q.view(torch.float8_e4m3fn).float()
    else:
        lut = LUT.to(q.device)
        v = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], -1).reshape(r, -1)
    return (v.view(r, -1, 32) * torch.exp2(s.float() - 127)[..., None]).reshape(r, -1)
