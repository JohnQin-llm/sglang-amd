"""Fused kernels for the DFlash2 draft layers on gfx950 (``SGLANG_MUSE_DFLASH_DRAFT_FUSE``).

A draft layer runs at M = 8 x batch rows. With these kernels it issues 16 launches instead of 21:

* The grouped-conv kernel projections, the QKV projection and gate/up run as split-K partial GEMMs at the widths
  of ``PARTIAL_LAUNCH`` (``bf16_splitk_gemm._dense_partial``, FP32 partials [SK, M, N] over the weight as
  loaded), and their consumers sum the partials: the input-side conv, QK-norm/RoPE and SiLU-and-mul.
* QK-norm/RoPE also writes K/V to the FP8 cache, so the attention backend's own store is skipped.
* The output-side conv, the residual add and the next RMSNorm are one kernel.

Every consumer keeps the arithmetic of the kernel it replaces (the inductor-compiled ``_grouped_conv``,
``_table_qk_norm_rope_kernel``, aiter's add-RMSNorm, sgl's SiLU-and-mul) and rounds the summed projection to
BF16 where the library GEMM did, so only the K summation order of the projections and the RMSNorm's sum of
squares differ from the served path.

Launch tables: MI355X, cold weights (draft-fuse/RESULTS.md). The down and O projections feed the all-reduce,
where a separate partial sum would cost more than split K saves, so they stay on library GEMMs: down, and QKV
and gate/up above their partial widths, on the per-width pick of ``LIBRARY_GEMMS``; O on the served dispatch.
"""

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.bf16_splitk_gemm import _dense_partial

# (N, K) -> ((max rows, split K, warps), ...): BN 64 x BK 128 tiles over the [N, K] weight.
# Gate/up stops at 56 rows: at 64, F.linear + SiLU-and-mul is ~2 us faster.
PARTIAL_LAUNCH = {
    (2048, 8192): ((64, 16, 4), (128, 8, 8)),  # grouped-conv kernel projection
    (2560, 8192): ((32, 16, 4), (64, 8, 8)),  # QKV
    (14336, 8192): ((56, 2, 4),),  # gate/up
}
# (N, K, M) -> the library GEMM that beat the served dispatch by >= 1 us per call (draft-fuse/gemm_sweep.py).
# "hipblaslt" entries are hipBLASLt solution ids of the container's library, as aiter's tuned csv stores them.
LIBRARY_GEMMS = {
    **{(2560, 8192, m): ("hipblaslt", 438757) for m in (80, 96)},
    **{(2560, 8192, m): ("hipblaslt", 438567) for m in (112, 128)},
    **{(14336, 8192, m): ("torch", 0) for m in (64, 80)},
    **{(14336, 8192, m): ("hipblaslt", 439344) for m in (96, 112, 128)},
    (8192, 7168, 8): ("hipblaslt", 439380),
    **{(8192, 7168, m): ("triton", 0) for m in (16, 24, 32, 56, 64)},
    **{(8192, 7168, m): ("hipblaslt", 438550) for m in (40, 48)},
    **{(8192, 7168, m): ("hipblaslt", 439293) for m in (80, 96)},
    **{(8192, 7168, m): ("hipblaslt", 438913) for m in (112, 128)},
}
BLOCK_N, BLOCK_K = 64, 128


class PartialLaunch(NamedTuple):
    split_k: int
    num_warps: int


def partial_launch(m: int, weight: torch.Tensor) -> Optional[PartialLaunch]:
    """The partial GEMM's launch for ``weight`` at M = ``m`` rows, or None to keep the library GEMM."""
    for max_rows, split_k, num_warps in PARTIAL_LAUNCH.get(tuple(weight.shape), ()):
        if m <= max_rows:
            return PartialLaunch(split_k, num_warps)
    return None


def projection_partials(
    x: torch.Tensor, weight: torch.Tensor, launch: PartialLaunch
) -> torch.Tensor:
    """FP32 split-K partials [split_k, M, N] of ``x @ weight.T``; the consumer sums them."""
    m, k = x.shape
    n = weight.shape[0]
    split_k = launch.split_k
    partials = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
    _dense_partial[(n // BLOCK_N, 1, split_k)](
        x,
        weight,
        partials,
        m,
        n,
        k,
        max(16, triton.next_power_of_2(m)),
        BLOCK_N,
        BLOCK_K,
        split_k,
        ".cg",
        num_warps=launch.num_warps,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )
    return partials


def library_gemm(x: torch.Tensor, weight: torch.Tensor):
    """``x @ weight.T`` on the library GEMM tuned for this width, or None when the served dispatch is kept."""
    choice = LIBRARY_GEMMS.get((weight.shape[0], weight.shape[1], x.shape[0]))
    if choice is None:
        return None
    kind, solution = choice
    if kind == "torch":
        return F.linear(x, weight)
    from aiter.tuned_gemm import hipb_gemm, triton_gemm

    gemm = triton_gemm if kind == "triton" else hipb_gemm
    return gemm(x, weight, solution, otype=x.dtype)


@triton.jit
def _summed(P, offsets, SPLIT_STRIDE, SPLIT_K: tl.constexpr):
    # The split-K partials summed in a fixed order, rounded to BF16 like the GEMM output they replace.
    acc = tl.load(P + offsets)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(P + s * SPLIT_STRIDE + offsets)
    return acc.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _grouped_conv_row(
    X, t, c, coef0, coef1, BASE, HIDDEN: tl.constexpr, BLOCK_TOKENS: tl.constexpr
):
    # dflash._grouped_conv for one row: taps 0 and 1 over the row and the previous row of its block.
    x = tl.load(X + t * HIDDEN + c).to(tl.float32)
    has_prev = (t & (BLOCK_TOKENS - 1)) >= 1
    # (c >= 0) only broadcasts the row's flag to the tile.
    prev = tl.load(X + (t - 1) * HIDDEN + c, mask=has_prev & (c >= 0), other=0.0).to(
        tl.float32
    )
    base0 = tl.load(BASE + c).to(tl.float32)
    base1 = tl.load(BASE + HIDDEN + c).to(tl.float32)
    return (base0 + coef0) * x + (base1 + coef1) * prev


@triton.jit
def _conv_in_kernel(
    X,
    P,
    BASE,
    OUT,
    SPLIT_STRIDE,
    SPLIT_K: tl.constexpr,
    COEF_COLS: tl.constexpr,
    HIDDEN: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    c = g[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    # Kernel projection columns: [side, tap, group]; the input side is side 0.
    coef0 = _summed(P, t * COEF_COLS + g, SPLIT_STRIDE, SPLIT_K)
    coef1 = _summed(P, t * COEF_COLS + GROUPS + g, SPLIT_STRIDE, SPLIT_K)
    out = _grouped_conv_row(
        X, t, c, coef0[:, None], coef1[:, None], BASE, HIDDEN, BLOCK_TOKENS
    )
    tl.store(OUT + t * HIDDEN + c, out.to(OUT.dtype.element_ty))


def conv_in(
    x: torch.Tensor,
    *,
    coef_partials: torch.Tensor,
    base: torch.Tensor,
    block_tokens: int,
    group_size: int,
) -> torch.Tensor:
    """DFlashGroupedConv.prepare's convolution of ``x`` from the kernel projection's partials."""
    m, hidden = x.shape
    groups = hidden // group_size
    out = torch.empty_like(x)
    # 2048 channels per program: 2.1-2.7 us per call at M8-128 (1024: up to 0.6 us slower at M128).
    block_groups = 128
    _conv_in_kernel[(m, groups // block_groups)](
        x,
        coef_partials,
        base,
        out,
        coef_partials.stride(0),
        coef_partials.shape[0],
        coef_partials.shape[2],
        hidden,
        groups,
        group_size,
        block_groups,
        block_tokens,
        num_warps=4,
    )
    return out


@triton.jit
def _conv_out_add_rmsnorm_kernel(
    Y,
    P,
    BASE,
    RESIDUAL,
    WEIGHT,
    OUT,
    RESIDUAL_OUT,
    SPLIT_STRIDE,
    EPS,
    SPLIT_K: tl.constexpr,
    COEF_COLS: tl.constexpr,
    HIDDEN: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, GROUPS)
    c = g[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    # The output side is side 1 of the kernel projection.
    coef0 = _summed(P, t * COEF_COLS + 2 * GROUPS + g, SPLIT_STRIDE, SPLIT_K)
    coef1 = _summed(P, t * COEF_COLS + 3 * GROUPS + g, SPLIT_STRIDE, SPLIT_K)
    conv = _grouped_conv_row(
        Y, t, c, coef0[:, None], coef1[:, None], BASE, HIDDEN, BLOCK_TOKENS
    )
    # aiter add-RMSNorm on the BF16 conv output: FP32 sum, BF16 residual, (sum * rsqrt) * weight.
    s = conv.to(tl.bfloat16).to(tl.float32) + tl.load(RESIDUAL + t * HIDDEN + c).to(
        tl.float32
    )
    tl.store(RESIDUAL_OUT + t * HIDDEN + c, s.to(RESIDUAL_OUT.dtype.element_ty))
    inv = tl.rsqrt(tl.sum(tl.sum(s * s, 1), 0) / HIDDEN + EPS)
    w = tl.load(WEIGHT + c).to(tl.float32)
    tl.store(OUT + t * HIDDEN + c, (s * inv * w).to(OUT.dtype.element_ty))


def conv_out_add_rmsnorm(
    y: torch.Tensor,
    *,
    coef_partials: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    block_tokens: int,
    group_size: int,
):
    """DFlashGroupedConv.finish of ``y``, then RMSNorm(conv + residual): (normed, new residual)."""
    m, hidden = y.shape
    out = torch.empty_like(y)
    residual_out = torch.empty_like(residual)
    _conv_out_add_rmsnorm_kernel[(m,)](
        y,
        coef_partials,
        base,
        residual,
        weight,
        out,
        residual_out,
        coef_partials.stride(0),
        eps,
        coef_partials.shape[0],
        coef_partials.shape[2],
        hidden,
        hidden // group_size,
        group_size,
        block_tokens,
        # One program per row: 3.6-4.2 us per call at M8-128 with 8 warps, 4.3-5.0 with 4.
        num_warps=8,
    )
    return out, residual_out


@triton.jit
def _projected(
    P, QKV, t, cols, SPLIT_STRIDE, SPLIT_K: tl.constexpr, QKV_COLS: tl.constexpr
):
    if SPLIT_K == 0:
        return tl.load(QKV + t * QKV_COLS + cols).to(tl.float32)
    return _summed(P, t * QKV_COLS + cols, SPLIT_STRIDE, SPLIT_K)


@triton.jit
def _qk_norm_rope_kv_kernel(
    P,
    QKV,
    Q_WEIGHT,
    K_WEIGHT,
    COS_SIN,
    POSITIONS,
    K_CACHE,
    V_CACHE,
    LOC,
    SPLIT_STRIDE,
    EPS,
    SPLIT_K: tl.constexpr,
    NHQ: tl.constexpr,
    NHK: tl.constexpr,
    D: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    QKV_COLS: tl.constexpr = (NHQ + 2 * NHK) * D
    HALF: tl.constexpr = D // 2
    half = tl.arange(0, HALF)
    col0 = h * D
    slot = tl.load(LOC + t).to(tl.int64)
    if h < NHQ + NHK:
        # dflash_utils._table_qk_norm_rope_kernel: per-head RMSNorm, BF16 round, table neox RoPE.
        pos = tl.load(POSITIONS + t).to(tl.int64)
        cos = tl.load(COS_SIN + pos * D + half).to(tl.float32)
        sin = tl.load(COS_SIN + pos * D + HALF + half).to(tl.float32)
        x = _projected(
            P, QKV, t, col0 + tl.arange(0, D), SPLIT_STRIDE, SPLIT_K, QKV_COLS
        )
        inv = 1.0 / tl.sqrt(tl.sum(x * x, 0) / D + EPS)
        w_ptr = tl.where(h < NHQ, Q_WEIGHT.to(tl.int64), K_WEIGHT.to(tl.int64)).to(
            tl.pointer_type(tl.bfloat16)
        )
        w1 = tl.load(w_ptr + half).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + half).to(tl.float32)
        x1 = (
            _projected(P, QKV, t, col0 + half, SPLIT_STRIDE, SPLIT_K, QKV_COLS)
            * inv
            * w1
        )
        x2 = (
            _projected(P, QKV, t, col0 + HALF + half, SPLIT_STRIDE, SPLIT_K, QKV_COLS)
            * inv
            * w2
        )
        x1 = x1.to(tl.bfloat16).to(tl.float32)
        x2 = x2.to(tl.bfloat16).to(tl.float32)
        o1 = (x1 * cos - x2 * sin).to(tl.bfloat16)
        o2 = (x2 * cos + x1 * sin).to(tl.bfloat16)
        tl.store(QKV + t * QKV_COLS + col0 + half, o1)
        tl.store(QKV + t * QKV_COLS + col0 + HALF + half, o2)
        if h >= NHQ:
            # The backend's store: the BF16 K cast to the FP8 cache dtype at the slot.
            k_row = K_CACHE + slot * (NHK * D) + (h - NHQ) * D
            tl.store(k_row + half, o1.to(K_CACHE.dtype.element_ty))
            tl.store(k_row + HALF + half, o2.to(K_CACHE.dtype.element_ty))
    else:
        d = tl.arange(0, D)
        v = _projected(P, QKV, t, col0 + d, SPLIT_STRIDE, SPLIT_K, QKV_COLS).to(
            tl.bfloat16
        )
        if SPLIT_K > 0:
            tl.store(QKV + t * QKV_COLS + col0 + d, v)
        tl.store(
            V_CACHE + slot * (NHK * D) + (h - NHQ - NHK) * D + d,
            v.to(V_CACHE.dtype.element_ty),
        )


def qk_norm_rope_kv_write(
    qkv: torch.Tensor,
    *,
    qkv_partials: Optional[torch.Tensor],
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    loc: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
) -> None:
    """QK RMSNorm + RoPE into ``qkv`` [T, (NHQ + 2 NHK) D] and K/V into the FP8 cache rows ``loc``.

    ``qkv_partials`` is the QKV projection's split-K partials (then ``qkv`` is written from their sum) or None
    (``qkv`` already holds the projection and is updated in place).
    """
    split_k = 0 if qkv_partials is None else qkv_partials.shape[0]
    partials = qkv if qkv_partials is None else qkv_partials
    _qk_norm_rope_kv_kernel[(qkv.shape[0], num_q_heads + 2 * num_kv_heads)](
        partials,
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        k_cache,
        v_cache,
        loc,
        partials.stride(0) if split_k else 0,
        eps,
        split_k,
        num_q_heads,
        num_kv_heads,
        head_dim,
        num_warps=4,
    )


@triton.jit
def _silu_mul_kernel(
    P,
    OUT,
    SPLIT_STRIDE,
    SPLIT_K: tl.constexpr,
    INTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = _summed(P, t * 2 * INTER + c, SPLIT_STRIDE, SPLIT_K)
    up = _summed(P, t * 2 * INTER + INTER + c, SPLIT_STRIDE, SPLIT_K)
    # sgl act_and_mul: silu rounded to BF16, then the BF16 product.
    act = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + t * INTER + c, (act * up).to(OUT.dtype.element_ty))


def silu_mul(gate_up_partials: torch.Tensor) -> torch.Tensor:
    """SiluAndMul of the gate/up projection from its split-K partials [SK, M, 2 I]."""
    split_k, m, two_inter = gate_up_partials.shape
    inter = two_inter // 2
    out = torch.empty((m, inter), device=gate_up_partials.device, dtype=torch.bfloat16)
    block = 1024
    _silu_mul_kernel[(m, inter // block)](
        gate_up_partials,
        out,
        gate_up_partials.stride(0),
        split_k,
        inter,
        block,
        num_warps=4,
    )
    return out
