"""Gluon split-KV verify attention for gfx1250 (MI455X), Muse Spark Open TP4 rank.

Shape: 8 verify rows x 8 q heads of one kv head packed as BLOCK_M 64, head_dim 64, FP8 E4M3 KV cache
[pages, 16, 4 kv heads, 64] (unshuffled, contiguous), page 16. Writes the same fp32 partials (segm_output
[T, NQ, NSEG, D], segm_max / segm_expsum [T, NQ, NSEG]) over the same device-side segment split as AITER's
kernel_unified_attention_3d (segment_range, incl. WINDOW_SPLIT for sliding layers), so AITER's reduce_segments
(TILE_SIZE = TILE) combines them unchanged.

Program = (segment, kv head), 4 waves; wave w owns query rows 16w..16w+15 of the packed 64. The segment's keys are
walked in 64-key units (4 pages):
- K and V pages move by TDM (tensor_load_to_lds) into an NBUF-deep LDS ring: one fused op each per unit, wave j
  loading page j, so each FP8 byte crosses HBM -> LDS once per program. Page ids are read one unit ahead of the
  issue, the issue runs NBUF - 1 units ahead of the compute; past the segment end the last unit is re-issued (an L2
  hit) instead of predicating the load off: a predicated-off TDM op is not counted by tensorcnt, which would break
  the static waits.
- S = Q K^T and O += P V on v_wmma_f32_16x16x32_bf16 in the transposed WMMA layout: each lane holds one query
  row, so the row max / sum are in-lane plus one cross-lane step, and P is the A operand of P V as computed.
- 32-bit positions; masks only on the units past the first query's causal horizon (full) or on every unit
  (sliding window); masked units zero V past seq_len in registers. exp2 with the softmax scale folded into one
  fma; fp32 softmax and accumulation.
"""
import triton.experimental.gluon.language as gl
from triton.experimental import gluon

TDM = gl.amd.gfx1250.tdm

NW = 4
U = gl.constexpr(64)  # keys per compute / TDM unit
QK_L = gl.constexpr(gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=[(1, 0), (2, 0)], reg_bases=[],
                                         instr_shape=[16, 16, 32]))
PV_L = QK_L
Q_DOT = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=QK_L, k_width=16))
K_DOT = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=QK_L, k_width=16))
P_DOT = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=PV_L, k_width=8))
V_DOT = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=PV_L, k_width=8))
Q_LOAD = gl.constexpr(gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[NW, 1],
                                       order=[1, 0]))
Q_SH = gl.constexpr(gl.PaddedSharedLayout.with_identity_for(interval_padding_pairs=[[64, 8]], shape=[64, 64],
                                                           order=[1, 0]))
KV_SH = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0]))
# V by per-lane async copies (AV): 16 B per lane, 8 rows x 4 lanes per wave, 2 instructions per 64-key unit
VLD = gl.constexpr(gl.BlockedLayout(size_per_thread=[1, 16], threads_per_warp=[8, 4], warps_per_cta=[NW, 1],
                                    order=[1, 0]))
# 16 B of padding per 64 B row: 80 B row stride, so 16 lanes reading 16 rows hit distinct banks
KV_PAD = gl.constexpr(gl.PaddedSharedLayout.with_identity_for(interval_padding_pairs=[[64, 16]], shape=[16, 64],
                                                             order=[1, 0]))
KV_PADU = gl.constexpr(gl.PaddedSharedLayout.with_identity_for(interval_padding_pairs=[[64, 16]], shape=[64, 64],
                                                              order=[1, 0]))


@gluon.jit
def _cdiv(x, y):
    return (x + y - 1) // y


@gluon.jit
def _ids(bt_ptr, u, n_log):
    """Physical pages of 64-key unit u (clamped to the sequence's last page)."""
    b = u * 4
    p0 = gl.load(bt_ptr + gl.minimum(b, n_log - 1))
    p1 = gl.load(bt_ptr + gl.minimum(b + 1, n_log - 1))
    p2 = gl.load(bt_ptr + gl.minimum(b + 2, n_log - 1))
    p3 = gl.load(bt_ptr + gl.minimum(b + 3, n_log - 1))
    return p0, p1, p2, p3


@gluon.jit
def _issue(k_desc, v_desc, k_smem, v_smem, p0, p1, p2, p3, slot, kvh, pred, PAGE: gl.constexpr, D: gl.constexpr,
           V_TDM: gl.constexpr = True):
    # TDM ignores a .slice() offset of its LDS destination (gfx1250 lowering): address whole pages by .index()
    j0 = slot * 4
    c = kvh * D
    # Only whole-page fused loads (one 16-row block per wave). The cooperative form (a page split 4 rows per wave)
    # with the padded layout hung GPU 0 at 17:28 UTC (a TDM op that never completed); it is not to be used.
    TDM.async_load_fused([
        (TDM.update_tensor_descriptor(k_desc, add_offsets=[p0 * PAGE, c], pred=pred), k_smem.index(j0), 1),
        (TDM.update_tensor_descriptor(k_desc, add_offsets=[p1 * PAGE, c], pred=pred), k_smem.index(j0 + 1), 2),
        (TDM.update_tensor_descriptor(k_desc, add_offsets=[p2 * PAGE, c], pred=pred), k_smem.index(j0 + 2), 4),
        (TDM.update_tensor_descriptor(k_desc, add_offsets=[p3 * PAGE, c], pred=pred), k_smem.index(j0 + 3), 8)])
    if V_TDM:
        TDM.async_load_fused([
        (TDM.update_tensor_descriptor(v_desc, add_offsets=[p0 * PAGE, c], pred=pred), v_smem.index(j0), 1),
        (TDM.update_tensor_descriptor(v_desc, add_offsets=[p1 * PAGE, c], pred=pred), v_smem.index(j0 + 1), 2),
        (TDM.update_tensor_descriptor(v_desc, add_offsets=[p2 * PAGE, c], pred=pred), v_smem.index(j0 + 2), 4),
        (TDM.update_tensor_descriptor(v_desc, add_offsets=[p3 * PAGE, c], pred=pred), v_smem.index(j0 + 3), 8)])


@gluon.jit
def _issue_v_async(v_ptr, vb, p0, p1, p2, p3, kvh, PAGE: gl.constexpr, NKV: gl.constexpr, D: gl.constexpr):
    """V of one 64-key unit into vb ([64, D]) by global_load_async_to_lds, page ids from the unit's scalars."""
    r = gl.arange(0, 64, layout=gl.SliceLayout(1, VLD))
    cc = gl.arange(0, D, layout=gl.SliceLayout(0, VLD))
    grp = r // PAGE
    pg = gl.where(grp == 0, p0, gl.where(grp == 1, p1, gl.where(grp == 2, p2, p3)))
    off = pg.to(gl.int64) * (PAGE * NKV * D) + (r % PAGE) * (NKV * D) + kvh * D
    gl.amd.gfx1250.async_copy.global_to_shared(vb, v_ptr + off[:, None] + cc[None, :])
    gl.amd.gfx1250.async_copy.commit_group()


@gluon.jit
def _unit(Q, M, L, acc, kb, vb, u, c, ctx, seq_len, qrow_s, kfix, vfix,
          D: gl.constexpr, BLOCK_M: gl.constexpr, MASK: gl.constexpr, SW: gl.constexpr, ABL: gl.constexpr = 0):
    if ABL & 4:
        k = kfix
    else:
        k = kb.permute([1, 0]).load(layout=K_DOT).to(gl.bfloat16)
    S = gl.amd.gfx1250.wmma(Q, k, gl.zeros([BLOCK_M, U], gl.float32, QK_L))
    if MASK:
        offs_t = u * U + gl.arange(0, U, layout=gl.SliceLayout(0, QK_L))
        hi = ctx + qrow_s
        vis = offs_t[None, :] <= hi[:, None]
        if SW > 0:
            vis = vis & (offs_t[None, :] > (hi - SW)[:, None])
        S = gl.where(vis, S, float("-inf"))
    m_j = gl.maximum(M, gl.max(S, axis=1) * c)
    if MASK:
        m_j = gl.where(m_j > float("-inf"), m_j, 0.0)
    if ABL & 2:
        P = S * c - m_j[:, None]
    else:
        P = gl.exp2(S * c - m_j[:, None])
    alpha = gl.exp2(M - m_j)
    L = L * alpha + gl.sum(P, axis=1)
    acc = acc * alpha[:, None]
    if (ABL & 1) == 0:
        p = gl.convert_layout(P.to(gl.bfloat16), P_DOT)
        if ABL & 8:
            v = vfix
        else:
            v = vb.load(layout=V_DOT).to(gl.bfloat16)
        if MASK:
            offs_k = u * U + gl.arange(0, U, layout=gl.SliceLayout(1, V_DOT))
            v = gl.where(offs_k[:, None] < seq_len, v, 0.0)
        acc = gl.amd.gfx1250.wmma(p, v, acc)
    return m_j, L, acc


@gluon.jit
def attn3_split(
    segm_out_ptr, segm_max_ptr, segm_sum_ptr,
    q_ptr, k_ptr, v_ptr, bt_ptr, seq_lens_ptr, cu_q_ptr, k_descale_ptr, v_descale_ptr,
    scale, q_stride0, q_stride1, num_pages,
    NQ: gl.constexpr, NKV: gl.constexpr, G: gl.constexpr, D: gl.constexpr, PAGE: gl.constexpr,
    TILE: gl.constexpr, NSEG: gl.constexpr, NBUF: gl.constexpr, BLOCK_M: gl.constexpr, SW: gl.constexpr = 0,
    PAD: gl.constexpr = 0, ABL: gl.constexpr = 0, AV: gl.constexpr = 0,
):
    gl.static_assert(TILE % U == 0)
    gl.static_assert(U // PAGE == 4)
    gl.static_assert(gl.num_warps() == 4)  # 4 waves = 4 pages per fused TDM op
    gl.static_assert(NBUF >= 2 and NBUF <= 8)  # tensorcnt wait <= 12; ring <= 8 x 10 KB per K / V
    seg = gl.program_id(0)
    kvh = gl.program_id(1)
    q_start = gl.load(cu_q_ptr)
    qlen = gl.load(cu_q_ptr + 1) - q_start
    seq_len = gl.load(seq_lens_ptr).to(gl.int32)
    ctx = seq_len - qlen
    num_tiles = _cdiv(seq_len, TILE)
    first_tile = 0
    if SW > 0:
        first_tile = gl.maximum(ctx - SW + 1, 0) // TILE
    span = num_tiles - first_tile
    tps = gl.maximum(_cdiv(span, NSEG), 1)
    if seg >= _cdiv(span, tps):
        return
    n_log = _cdiv(seq_len, PAGE)
    t0 = first_tile + seg * tps
    t1 = gl.minimum(t0 + tps, num_tiles)
    u0 = t0 * (TILE // U)
    u1 = gl.minimum(t1 * (TILE // U), _cdiv(seq_len, U))

    if PAD:
        DL: gl.constexpr = KV_PAD
        SL: gl.constexpr = KV_PADU
    else:
        DL: gl.constexpr = KV_SH
        SL: gl.constexpr = KV_SH
    k_desc = TDM.make_tensor_descriptor(base=k_ptr, shape=(num_pages * PAGE, NKV * D), strides=(NKV * D, 1),
                                        block_shape=(PAGE, D), layout=DL)
    v_desc = TDM.make_tensor_descriptor(base=v_ptr, shape=(num_pages * PAGE, NKV * D), strides=(NKV * D, 1),
                                        block_shape=(PAGE, D), layout=DL)
    k_ring = gl.allocate_shared_memory(gl.float8e4nv, [NBUF, U, D], layout=SL)
    v_ring = gl.allocate_shared_memory(gl.float8e4nv, [NBUF, U, D], layout=SL)
    # page view of the same ring for the TDM destinations
    k_smem = k_ring.reinterpret(shape=[NBUF * 4, PAGE, D], layout=DL)
    v_smem = v_ring.reinterpret(shape=[NBUF * 4, PAGE, D], layout=DL)

    for ip in gl.static_range(NBUF - 1):
        p0, p1, p2, p3 = _ids(bt_ptr, gl.minimum(u0 + ip, u1 - 1), n_log)
        _issue(k_desc, v_desc, k_smem, v_smem, p0, p1, p2, p3, ip, kvh, True, PAGE, D, AV == 0)
        if AV:
            _issue_v_async(v_ptr, v_ring.index(ip), p0, p1, p2, p3, kvh, PAGE, NKV, D)
    p0, p1, p2, p3 = _ids(bt_ptr, gl.minimum(u0 + NBUF - 1, u1 - 1), n_log)

    offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_LOAD))
    offs_d = gl.arange(0, D, layout=gl.SliceLayout(0, Q_LOAD))
    qrow = offs_m // G
    head = kvh * G + offs_m % G
    qptr = q_ptr + (q_start + qrow)[:, None] * q_stride0 + head[:, None] * q_stride1 + offs_d[None, :]
    qv = gl.load(qptr, mask=(qrow[:, None] < qlen) & (offs_d[None, :] < D), other=0.0)
    q_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, D], layout=Q_SH)
    q_smem.store(qv)
    Q = q_smem.load(layout=Q_DOT)
    c = scale * 1.4426950408889634 * gl.load(k_descale_ptr)

    qrow_s = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, QK_L)) // G
    M = gl.full([BLOCK_M], float("-inf"), gl.float32, gl.SliceLayout(1, QK_L))
    L = gl.zeros([BLOCK_M], gl.float32, gl.SliceLayout(1, QK_L))
    acc = gl.zeros([BLOCK_M, D], gl.float32, PV_L)

    kfix = gl.zeros([D, U], gl.bfloat16, K_DOT)
    vfix = gl.zeros([U, D], gl.bfloat16, V_DOT)
    if ABL & 12:
        TDM.async_wait(0)
        gl.barrier()
        kfix = k_ring.index(0).permute([1, 0]).load(layout=K_DOT).to(gl.bfloat16)
        vfix = v_ring.index(0).load(layout=V_DOT).to(gl.bfloat16)
    WAIT: gl.constexpr = (1 if AV else 2) * (NBUF - 2)
    if SW > 0:
        u_full = u0
    else:
        # keys < ctx + 1 are visible to every row
        u_full = gl.maximum(gl.minimum(u1, (ctx + 1) // U), u0)
    for u in range(u0, u_full):
        i = u - u0
        TDM.async_wait(WAIT)
        if AV:
            gl.amd.gfx1250.async_copy.wait_group(NBUF - 2)
        gl.barrier()
        _issue(k_desc, v_desc, k_smem, v_smem, p0, p1, p2, p3, (i + NBUF - 1) % NBUF, kvh, True,
               PAGE, D, AV == 0)
        if AV:
            _issue_v_async(v_ptr, v_ring.index((i + NBUF - 1) % NBUF), p0, p1, p2, p3, kvh, PAGE, NKV, D)
        p0, p1, p2, p3 = _ids(bt_ptr, gl.minimum(u + NBUF, u1 - 1), n_log)
        b = i % NBUF
        if (ABL & 16) == 0:
            M, L, acc = _unit(Q, M, L, acc, k_ring.index(b), v_ring.index(b), u, c, ctx, seq_len, qrow_s, kfix,
                              vfix, D, BLOCK_M, False, SW, ABL)
    for u in range(u_full, u1):
        i = u - u0
        TDM.async_wait(WAIT)
        if AV:
            gl.amd.gfx1250.async_copy.wait_group(NBUF - 2)
        gl.barrier()
        _issue(k_desc, v_desc, k_smem, v_smem, p0, p1, p2, p3, (i + NBUF - 1) % NBUF, kvh, True,
               PAGE, D, AV == 0)
        if AV:
            _issue_v_async(v_ptr, v_ring.index((i + NBUF - 1) % NBUF), p0, p1, p2, p3, kvh, PAGE, NKV, D)
        p0, p1, p2, p3 = _ids(bt_ptr, gl.minimum(u + NBUF, u1 - 1), n_log)
        b = i % NBUF
        if (ABL & 16) == 0:
            M, L, acc = _unit(Q, M, L, acc, k_ring.index(b), v_ring.index(b), u, c, ctx, seq_len, qrow_s, kfix,
                              vfix, D, BLOCK_M, True, SW, ABL)
    TDM.async_wait(0)
    if AV:
        gl.amd.gfx1250.async_copy.wait_group(0)

    acc = acc * gl.load(v_descale_ptr)
    om = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, PV_L))
    od = gl.arange(0, D, layout=gl.SliceLayout(0, PV_L))
    orow = om // G
    so = (q_start + orow) * (NQ * NSEG) + (kvh * G + om % G) * NSEG + seg
    gl.store(segm_out_ptr + so[:, None] * D + od[None, :], acc, mask=(orow[:, None] < qlen) & (od[None, :] < D))
    sm = (q_start + qrow_s) * (NQ * NSEG) + (kvh * G + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, QK_L)) % G) \
        * NSEG + seg
    gl.store(segm_max_ptr + sm, M, mask=qrow_s < qlen)
    gl.store(segm_sum_ptr + sm, L, mask=qrow_s < qlen)
