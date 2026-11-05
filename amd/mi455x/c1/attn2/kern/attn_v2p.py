"""attn_v2 with register prefetch instead of async-to-LDS pipelining (run with num_stages 1).

Tile t + 1's FP8 K / V are loaded into registers (coalesced blocked layout, 8 VGPRs each) while tile t
computes; the FP8 -> BF16 conversion then runs once per element in that layout, and Triton moves the BF16
operands to the WMMA layouts through LDS.
"""
import triton
import triton.language as tl

from .attn_v2 import _segment_range


@triton.jit
def _load_kv(t, k_ptr, v_ptr, bt_ptr, kv_off_k, kv_off_v, sk0, sk1, sv0, sv1, seq_len,
             TILE: tl.constexpr, PAGE: tl.constexpr, D: tl.constexpr):
    # keys past seq_len re-read the last key (finite values); the causal mask gives them P = 0
    offs_t = tl.minimum(t * TILE + tl.arange(0, TILE), seq_len - 1)
    offs_d = tl.arange(0, D)
    pg = tl.load(bt_ptr + offs_t // PAGE).to(tl.int64)
    in_pg = offs_t % PAGE
    k_row = pg * sk0 + in_pg * sk1 + kv_off_k
    v_row = pg * sv0 + in_pg * sv1 + kv_off_v
    K = tl.load(k_ptr + k_row[None, :] + offs_d[:, None])  # [D, TILE]
    V = tl.load(v_ptr + v_row[:, None] + offs_d[None, :])  # [TILE, D]
    return K, V


@triton.jit
def _compute(t, K, V, Q, qk_scale, M, L, acc, ctx, qrow, qmask, TILE: tl.constexpr, MASK: tl.constexpr):
    offs_t = t * TILE + tl.arange(0, TILE)
    S = tl.dot(Q, K.to(tl.bfloat16))
    if MASK:
        vis = (offs_t[None, :] <= (ctx + qrow)[:, None]) & qmask[:, None]
        S = tl.where(vis, S, float("-inf"))
    m_j = tl.maximum(M, tl.max(S, axis=1) * qk_scale)
    if MASK:
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    P = tl.math.exp2(S * qk_scale - m_j[:, None])
    l_j = tl.sum(P, axis=1)
    alpha = tl.math.exp2(M - m_j)
    acc = acc * alpha[:, None]
    L = L * alpha + l_j
    acc = tl.dot(P.to(tl.bfloat16), V.to(tl.bfloat16), acc=acc)
    return m_j, L, acc


@triton.jit
def attn_v2p_split(
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
    Q = tl.load(
        q_ptr + (q_start + qrow)[:, None] * q_stride0 + head[:, None] * q_stride1 + offs_d[None, :],
        mask=qmask[:, None], other=0.0,
    )
    qk_scale = scale * 1.4426950408889634 * tl.load(k_descale_ptr)
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    kv_off_k = kvh * sk2
    kv_off_v = kvh * sv2
    t0 = seg * tps
    t1 = tl.minimum(t0 + tps, num_tiles)
    t_full = tl.maximum(tl.minimum(t1, (ctx + qb * (BLOCK_M // G) + 1) // TILE), t0)
    Kc, Vc = _load_kv(t0, k_ptr, v_ptr, bt_ptr, kv_off_k, kv_off_v, sk0, sk1, sv0, sv1, seq_len, TILE, PAGE, D)
    for t in range(t0, t1):
        tn = tl.minimum(t + 1, t1 - 1)
        Kn, Vn = _load_kv(tn, k_ptr, v_ptr, bt_ptr, kv_off_k, kv_off_v, sk0, sk1, sv0, sv1, seq_len, TILE, PAGE, D)
        if t < t_full:
            M, L, acc = _compute(t, Kc, Vc, Q, qk_scale, M, L, acc, ctx, qrow, qmask, TILE, False)
        else:
            M, L, acc = _compute(t, Kc, Vc, Q, qk_scale, M, L, acc, ctx, qrow, qmask, TILE, True)
        Kc = Kn
        Vc = Vn

    acc = acc * tl.load(v_descale_ptr)
    so = (q_start + qrow) * (NQ * NSEG) + head * NSEG + seg
    tl.store(segm_out_ptr + so[:, None].to(tl.int64) * D + offs_d[None, :], acc, mask=qmask[:, None])
    tl.store(segm_max_ptr + so, M, mask=qmask)
    tl.store(segm_sum_ptr + so, L, mask=qmask)
