"""attn_v2 with the GEMMs transposed: S^T = K Q^T, O^T = V^T P^T (same partials / split as attn_v2).

With BLOCK_M 64 as the GEMMs' N, the 4 waves split the tile's keys for QK^T and the head dim for PV, so each
FP8 K / V element is converted to BF16 once per program instead of once per wave. The cost: the per-row
max / sum reduce across waves, and P^T moves through LDS between the two GEMMs.
"""
import triton
import triton.language as tl

from .attn_v2 import _segment_range


@triton.jit
def _tile_t(
    t, QT, qk_scale, M, L, accT,
    k_ptr, v_ptr, bt_ptr, kv_off_k, kv_off_v, sk0, sk1, sv0, sv1, ctx, seq_len, qrow, qmask,
    TILE: tl.constexpr, PAGE: tl.constexpr, D: tl.constexpr, MASK: tl.constexpr,
):
    offs_t = t * TILE + tl.arange(0, TILE)
    offs_d = tl.arange(0, D)
    if MASK:
        kin = offs_t < seq_len
        pg = tl.load(bt_ptr + offs_t // PAGE, mask=kin, other=0).to(tl.int64)
    else:
        pg = tl.load(bt_ptr + offs_t // PAGE).to(tl.int64)
    in_pg = offs_t % PAGE
    k_row = pg * sk0 + in_pg * sk1 + kv_off_k
    v_row = pg * sv0 + in_pg * sv1 + kv_off_v
    if MASK:
        K = tl.load(k_ptr + k_row[:, None] + offs_d[None, :], mask=kin[:, None], other=0.0)  # [TILE, D]
        VT = tl.load(v_ptr + v_row[None, :] + offs_d[:, None], mask=kin[None, :], other=0.0)  # [D, TILE]
    else:
        K = tl.load(k_ptr + k_row[:, None] + offs_d[None, :])
        VT = tl.load(v_ptr + v_row[None, :] + offs_d[:, None])
    ST = tl.dot(K.to(tl.bfloat16), QT)  # [TILE, BLOCK_M]
    if MASK:
        vis = (offs_t[:, None] <= (ctx + qrow)[None, :]) & qmask[None, :]
        ST = tl.where(vis, ST, float("-inf"))
    m_j = tl.maximum(M, tl.max(ST, axis=0) * qk_scale)
    if MASK:
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    PT = tl.math.exp2(ST * qk_scale - m_j[None, :])
    l_j = tl.sum(PT, axis=0)
    alpha = tl.math.exp2(M - m_j)
    accT = accT * alpha[None, :]
    L = L * alpha + l_j
    accT = tl.dot(VT.to(tl.bfloat16), PT.to(tl.bfloat16), acc=accT)  # [D, BLOCK_M]
    return m_j, L, accT


@triton.jit
def attn_v2t_split(
    segm_out_ptr, segm_max_ptr, segm_sum_ptr,
    q_ptr, k_ptr, v_ptr, bt_ptr, seq_lens_ptr, cu_q_ptr, k_descale_ptr, v_descale_ptr,
    scale,
    q_stride0, q_stride1,
    sk0, sk1, sk2, sv0, sv1, sv2,
    NQ: tl.constexpr, G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    TILE: tl.constexpr, NSEG: tl.constexpr, BLOCK_M: tl.constexpr,
):
    seg = tl.program_id(0)
    kvh = tl.program_id(1)
    qb = tl.program_id(2)
    q_start = tl.load(cu_q_ptr)
    qlen = tl.load(cu_q_ptr + 1) - q_start
    seq_len = tl.load(seq_lens_ptr).to(tl.int32)
    num_tiles, tps, live = _segment_range(seq_len, TILE, NSEG)
    if seg >= live:
        return
    if qb * (BLOCK_M // G) >= qlen:
        return
    ctx = seq_len - qlen

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    qrow = qb * (BLOCK_M // G) + offs_m // G
    head = kvh * G + offs_m % G
    qmask = qrow < qlen
    QT = tl.load(
        q_ptr + (q_start + qrow)[None, :] * q_stride0 + head[None, :] * q_stride1 + offs_d[:, None],
        mask=qmask[None, :], other=0.0,
    )  # [D, BLOCK_M]
    qk_scale = scale * 1.4426950408889634 * tl.load(k_descale_ptr)

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    accT = tl.zeros([D, BLOCK_M], dtype=tl.float32)

    kv_off_k = kvh * sk2
    kv_off_v = kvh * sv2
    t0 = seg * tps
    t1 = tl.minimum(t0 + tps, num_tiles)
    t_full = tl.maximum(tl.minimum(t1, (ctx + qb * (BLOCK_M // G) + 1) // TILE), t0)
    for t in range(t0, t_full):
        M, L, accT = _tile_t(t, QT, qk_scale, M, L, accT, k_ptr, v_ptr, bt_ptr, kv_off_k, kv_off_v,
                             sk0, sk1, sv0, sv1, ctx, seq_len, qrow, qmask, TILE, PAGE, D, False)
    for t in range(t_full, t1):
        M, L, accT = _tile_t(t, QT, qk_scale, M, L, accT, k_ptr, v_ptr, bt_ptr, kv_off_k, kv_off_v,
                             sk0, sk1, sv0, sv1, ctx, seq_len, qrow, qmask, TILE, PAGE, D, True)

    accT = accT * tl.load(v_descale_ptr)
    so = (q_start + qrow) * (NQ * NSEG) + head * NSEG + seg
    tl.store(segm_out_ptr + so[None, :].to(tl.int64) * D + offs_d[:, None], accT, mask=qmask[None, :])
    tl.store(segm_max_ptr + so, M, mask=qmask)
    tl.store(segm_sum_ptr + so, L, mask=qmask)
