"""Verify-attention split kernel for one short query (<= 8 rows a block) over a paged FP8 E4M3 KV cache, gfx1250.

Specialized for the Muse Spark Open TP4 full-attention verify: 8 query rows x 8 q heads of one kv head packed as
BLOCK_M 64, head_dim 64, page 16, unshuffled cache [pages, 16, kv heads, 64], no sinks / alibi / softcap.
Writes the same fp32 partials (segm_output, segm_max, segm_expsum) and uses the same device-side segment
split as AITER's kernel_unified_attention_3d, so AITER's reduce_segments combines them unchanged.

Differences from the served loop:
- 32-bit positions; the causal mask is applied only to tiles past the first query's horizon (ctx + 1),
  so the steady-state loop has no compares / selects.
- QSPLIT 2 / 3: QK^T on FP8 WMMA with K used as stored (no FP8 -> BF16 conversion). Q is split per row into
  2 or 3 E4M3 terms after a power-of-two row scale (Q * 2^e = Qh + Ql / 16 [+ Qll / 256]); the terms are
  accumulated in one fp32 accumulator, and 2^-e folds into the per-row softmax scale. 3 terms represent
  every bf16 Q exactly; 2 terms keep ~8 significant bits.
- Grid (segments, kv heads, query blocks) with no upper-bound query blocks for a single sequence.
"""
import triton
import triton.language as tl

RCP_LN2 = 1.4426950408889634


@triton.jit
def _cdiv(x, y):
    return (x + y - 1) // y


@triton.jit
def _segment_range(seq_len, TILE: tl.constexpr, NSEG: tl.constexpr):
    num_tiles = _cdiv(seq_len, TILE)
    tps = tl.maximum(_cdiv(num_tiles, NSEG), 1)
    return num_tiles, tps, _cdiv(num_tiles, tps)


@triton.jit
def _touch_tile(tp, k_ptr, v_ptr, bt_ptr, kv_off_k, kv_off_v, sk0, sk1, sv0, sv1, seq_len,
                TILE: tl.constexpr, PAGE: tl.constexpr):
    """Read one byte of each of tile tp's K / V rows (64 bytes per key and head) so the rows are in L2 when the
    tile's async copy is issued; the caller carries the result and consumes it only after the loop."""
    offs_p = tl.minimum(tp * TILE + tl.arange(0, TILE), seq_len - 1)
    pg = tl.load(bt_ptr + offs_p // PAGE).to(tl.int64)
    in_pg = offs_p % PAGE
    kb = tl.load((k_ptr + (pg * sk0 + in_pg * sk1 + kv_off_k)).to(tl.pointer_type(tl.uint8), bitcast=True))
    vb = tl.load((v_ptr + (pg * sv0 + in_pg * sv1 + kv_off_v)).to(tl.pointer_type(tl.uint8), bitcast=True))
    return kb, vb


@triton.jit
def _split_q(Qf, QSPLIT: tl.constexpr):
    """Q (fp32, exact bf16 values) -> E4M3 terms and the per-row 2^-e that undoes the row scale."""
    amax = tl.max(tl.abs(Qf), axis=1)
    e = tl.floor(tl.log2(448.0 / tl.maximum(amax, 1e-30)))
    e = tl.minimum(e, 60.0)
    s = tl.exp2(e)
    Qs = Qf * s[:, None]
    Qh = Qs.to(tl.float8e4nv)
    R = (Qs - Qh.to(tl.float32)) * 16.0
    Ql = R.to(tl.float8e4nv)
    if QSPLIT == 3:
        R2 = (R - Ql.to(tl.float32)) * 16.0
        Qll = R2.to(tl.float8e4nv)
    else:
        Qll = Ql
    return Qh, Ql, Qll, tl.exp2(-e)


@triton.jit
def _tile(
    t, Q, Qh, Ql, Qll, qk_scale, row_scale, M, L, acc,
    k_ptr, v_ptr, bt_ptr, kv_off_k, kv_off_v, sk0, sk1, sv0, sv1, ctx, seq_len, qrow, qmask,
    TILE: tl.constexpr, PAGE: tl.constexpr, D: tl.constexpr, QSPLIT: tl.constexpr, MASK: tl.constexpr,
    FOLD: tl.constexpr,
):
    offs_t = t * TILE + tl.arange(0, TILE)
    offs_d = tl.arange(0, D)
    if MASK:
        kin = offs_t < seq_len
        pg = tl.load(bt_ptr + offs_t // PAGE, mask=kin, other=0).to(tl.int64)
    else:
        pg = tl.load(bt_ptr + offs_t // PAGE).to(tl.int64)
    in_pg = offs_t % PAGE
    k_row = pg * sk0 + in_pg * sk1 + kv_off_k  # [TILE] int64
    v_row = pg * sv0 + in_pg * sv1 + kv_off_v
    if MASK:
        K = tl.load(k_ptr + k_row[None, :] + offs_d[:, None], mask=kin[None, :], other=0.0)
        V = tl.load(v_ptr + v_row[:, None] + offs_d[None, :], mask=kin[:, None], other=0.0)
    else:
        K = tl.load(k_ptr + k_row[None, :] + offs_d[:, None])  # [D, TILE] e4m3
        V = tl.load(v_ptr + v_row[:, None] + offs_d[None, :])  # [TILE, D] e4m3

    if QSPLIT == 0:
        S = tl.dot(Q, K.to(tl.bfloat16))
        if not FOLD:
            S = S * qk_scale
    else:
        if QSPLIT == 3:
            Sl = tl.dot(Ql, K, acc=tl.dot(Qll, K) * 0.0625)
        else:
            Sl = tl.dot(Ql, K)
        S = tl.dot(Qh, K, acc=Sl * 0.0625) * row_scale[:, None]

    if MASK:
        vis = (offs_t[None, :] <= (ctx + qrow)[:, None]) & qmask[:, None]
        S = tl.where(vis, S, float("-inf"))
    if FOLD and QSPLIT == 0:
        # the scale is positive, so max(S) * scale = max(S * scale); P = exp2(fma(S, scale, -m))
        m_j = tl.maximum(M, tl.max(S, axis=1) * qk_scale)
        if MASK:
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.math.exp2(S * qk_scale - m_j[:, None])
    else:
        m_j = tl.maximum(M, tl.max(S, axis=1))
        if MASK:
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.math.exp2(S - m_j[:, None])
    l_j = tl.sum(P, axis=1)
    alpha = tl.math.exp2(M - m_j)
    acc = acc * alpha[:, None]
    L = L * alpha + l_j
    acc = tl.dot(P.to(tl.bfloat16), V.to(tl.bfloat16), acc=acc)
    return m_j, L, acc


@triton.jit
def attn_v2_split(
    segm_out_ptr, segm_max_ptr, segm_sum_ptr,
    q_ptr, k_ptr, v_ptr, bt_ptr, seq_lens_ptr, cu_q_ptr, k_descale_ptr, v_descale_ptr,
    scale,
    q_stride0, q_stride1,
    sk0, sk1, sk2, sv0, sv1, sv2,
    NQ: tl.constexpr, G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    TILE: tl.constexpr, NSEG: tl.constexpr, BLOCK_M: tl.constexpr, QSPLIT: tl.constexpr,
    FOLD: tl.constexpr = False, DUAL: tl.constexpr = False, PFD: tl.constexpr = 0, PF_SCOPE: tl.constexpr = 0,
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
    if QSPLIT == 0:
        Qh = Q
        Ql = Q
        Qll = Q
        row_scale = tl.full([BLOCK_M], 1.0, tl.float32) * qk_scale
    else:
        Qh, Ql, Qll, inv_s = _split_q(Q.to(tl.float32), QSPLIT)
        row_scale = inv_s * qk_scale

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    kv_off_k = kvh * sk2
    kv_off_v = kvh * sv2
    bt = bt_ptr
    t0 = seg * tps
    t1 = tl.minimum(t0 + tps, num_tiles)
    # keys < ctx + qb * BLOCK_Q + 1 are visible to every row of this block
    t_full = tl.maximum(tl.minimum(t1, (ctx + qb * (BLOCK_M // G) + 1) // TILE), t0)
    if DUAL:
        # two independent online-softmax streams (even / odd tiles), merged after the loop
        M2 = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        L2 = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc2 = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        n_pairs = (t_full - t0) // 2
        for i in range(n_pairs):
            t = t0 + 2 * i
            M, L, acc = _tile(t, Q, Qh, Ql, Qll, qk_scale, row_scale, M, L, acc, k_ptr, v_ptr, bt, kv_off_k,
                              kv_off_v, sk0, sk1, sv0, sv1, ctx, seq_len, qrow, qmask, TILE, PAGE, D, QSPLIT,
                              False, FOLD)
            M2, L2, acc2 = _tile(t + 1, Q, Qh, Ql, Qll, qk_scale, row_scale, M2, L2, acc2, k_ptr, v_ptr, bt,
                                 kv_off_k, kv_off_v, sk0, sk1, sv0, sv1, ctx, seq_len, qrow, qmask, TILE, PAGE,
                                 D, QSPLIT, False, FOLD)
        t_rest = t0 + 2 * n_pairs
    else:
        touch = tl.zeros([TILE], dtype=tl.int32)
        kb = tl.zeros([TILE], dtype=tl.uint8)
        vb = tl.zeros([TILE], dtype=tl.uint8)
        for t in range(t0, t_full):
            if PFD > 0:
                # consume the previous iteration's bytes, so their wait falls a tile after the loads
                touch += kb.to(tl.int32) + vb.to(tl.int32)
                kb, vb = _touch_tile(tl.minimum(t + PFD, t1 - 1), k_ptr, v_ptr, bt, kv_off_k, kv_off_v, sk0,
                                     sk1, sv0, sv1, seq_len, TILE, PAGE)
            M, L, acc = _tile(t, Q, Qh, Ql, Qll, qk_scale, row_scale, M, L, acc, k_ptr, v_ptr, bt, kv_off_k,
                              kv_off_v, sk0, sk1, sv0, sv1, ctx, seq_len, qrow, qmask, TILE, PAGE, D, QSPLIT,
                              False, FOLD)
        if PFD > 0:
            touch += kb.to(tl.int32) + vb.to(tl.int32)
            # never true (seq_len > 0); keeps the touch loads alive without a dependency in the loop
            L = tl.where(seq_len < 0, L + tl.sum(touch).to(tl.float32), L)
        t_rest = t_full
    for t in range(t_rest, t1):
        M, L, acc = _tile(t, Q, Qh, Ql, Qll, qk_scale, row_scale, M, L, acc, k_ptr, v_ptr, bt, kv_off_k,
                          kv_off_v, sk0, sk1, sv0, sv1, ctx, seq_len, qrow, qmask, TILE, PAGE, D, QSPLIT,
                          True, FOLD)
    if DUAL:
        m = tl.maximum(M, M2)
        m = tl.where(m > float("-inf"), m, 0.0)
        a1 = tl.math.exp2(M - m)
        a2 = tl.math.exp2(M2 - m)
        acc = acc * a1[:, None] + acc2 * a2[:, None]
        L = L * a1 + L2 * a2
        M = tl.maximum(M, M2)

    acc = acc * tl.load(v_descale_ptr)
    so = (q_start + qrow) * (NQ * NSEG) + head * NSEG + seg
    tl.store(segm_out_ptr + so[:, None].to(tl.int64) * D + offs_d[None, :], acc, mask=qmask[:, None])
    tl.store(segm_max_ptr + so, M, mask=qmask)
    tl.store(segm_sum_ptr + so, L, mask=qmask)
