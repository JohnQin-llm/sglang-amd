"""DFlash grouped conv (output side) + residual add + RMSNorm in one launch (``SGLANG_MUSE_DFLASH_CONV_NORM_FUSE``).

Each draft sublayer ends with ``DFlashGroupedConv.finish`` (an inductor-compiled pointwise kernel) and the next
norm is aiter's Triton add-RMSNorm. This kernel computes the same: the conv in FP32 as inductor emits it,
``(base0 + delta0) * y + (base1 + delta1) * y_prev``, rounded to BF16, then ``_fused_add_rmsnorm_kernel``'s
non-blocked row (BF16 add, FP32 sum of squares over one 8192-wide block at 4 warps, rsqrt, weight).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_add_rmsnorm_kernel(
    Y,
    DELTA,
    BASE,
    RES_IN,
    G,
    OUT,
    RES_OUT,
    delta_row_stride,
    n_cols,
    epsilon,
    GROUP_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    t = tl.program_id(0)
    c = tl.arange(0, BLOCK_SIZE)
    g = c // GROUP_SIZE
    y = tl.load(Y + t * BLOCK_SIZE + c).to(tl.float32)
    d0 = tl.load(DELTA + t.to(tl.int64) * delta_row_stride + g).to(tl.float32)
    d1 = tl.load(DELTA + t.to(tl.int64) * delta_row_stride + GROUPS + g).to(tl.float32)
    b0 = tl.load(BASE + c).to(tl.float32)
    b1 = tl.load(BASE + BLOCK_SIZE + c).to(tl.float32)
    prev = tl.load(Y + t * BLOCK_SIZE - BLOCK_SIZE + c, mask=(t - 1 >= 0) & (c >= 0), other=0.0).to(tl.float32)
    prev = tl.where((t & (BLOCK_TOKENS - 1)) >= 1, prev, 0.0)
    conv = ((b0 + d0) * y + (b1 + d1) * prev).to(tl.bfloat16)

    res_in = tl.load(RES_IN + t * BLOCK_SIZE + c, cache_modifier=".cg")
    row = conv
    row += res_in
    tl.store(RES_OUT + t * BLOCK_SIZE + c, row)
    row = row.to(tl.float32)
    w = tl.load(G + c).to(tl.float32)
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)
    tl.store(OUT + t * BLOCK_SIZE + c, (row * norm_factor * w).to(OUT.dtype.element_ty))


def conv_out_add_rmsnorm(
    y: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    block_tokens: int,
    group_size: int,
):
    """``RMSNorm(DFlashGroupedConv.finish(y, delta) + residual)``: (normed, new residual).

    ``y`` [T, H] BF16 contiguous, ``delta`` the output-side coefficients view [T, taps=2, groups] of the kernel
    projection, ``base`` the output-side base kernel [2, H].
    """
    t, h = y.shape
    assert h == triton.next_power_of_2(h) and delta.stride(2) == 1 and delta.stride(1) == h // group_size
    out = torch.empty_like(y)
    res_out = torch.empty_like(y)
    _conv_add_rmsnorm_kernel[(t,)](
        y,
        delta,
        base,
        residual,
        weight,
        out,
        res_out,
        delta.stride(0),
        h,
        eps,
        GROUP_SIZE=group_size,
        GROUPS=h // group_size,
        BLOCK_TOKENS=block_tokens,
        BLOCK_SIZE=h,
    )
    return out, res_out
