# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton RMSNorm for MuseSpark1x's two norm variants.

With ``residual_stream`` the norm also performs the norm-preserving gated
residual add that follows it in every decoder layer, which is what lets the
branch stay in registers instead of making a round trip through memory.
"""

import torch
import triton
import triton.language as tl

try:
    from flashinfer.norm import gated_residual_rmsnorm as postnorm_gated_residual
except (ImportError, AttributeError):  # pragma: no cover - optional FlashInfer API
    postnorm_gated_residual = None


@triton.jit
def _fast2sum(a, b):
    """Fast2Sum error-free transformation: ``a + b == s + e`` exactly.

    Requires ``|a| >= |b|`` element-wise. See
    https://en.wikipedia.org/wiki/2Sum.
    """
    s = a + b
    z = s - a
    return s, b - z


@triton.jit
def _effective_weight(w_ptr, d, dm, gain_center):
    """``weight + gain_center``, in fp32.

    The centre is added in fp32 and the weight multiplies in fp32: the norm
    rounds once, at its store, and nowhere else.
    """
    return tl.load(w_ptr + d, mask=dm, other=0.0).to(tl.float32) + gain_center


@triton.jit
def _blend_and_store(
    y_ptr,
    s_ptr,
    alpha_ptr,
    beta_ptr,
    idx,
    d,
    mask,
    dm,
    y,
    alpha_scalar,
    GATE_STREAM: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
):
    """Store ``alpha * stream + beta * y``, the norm-preserving residual add.

    ``y`` arrives as the fp32 norm output. The eager path rounds it to the
    branch dtype on the way out of the norm and reads it back; keeping it in
    registers skips that round trip and the rounding with it, so this is one
    rounding step more accurate than the two-kernel path.
    """
    stream = tl.load(s_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    if PER_CHANNEL:
        beta = tl.load(beta_ptr + d, mask=dm, other=0.0)[None, :]
    else:
        beta = tl.load(beta_ptr)

    if GATE_STREAM:
        if PER_CHANNEL:
            alpha = tl.load(alpha_ptr + d, mask=dm, other=0.0)[None, :]
        else:
            alpha = tl.load(alpha_ptr)
        out = alpha * stream + beta * y
    else:
        out = alpha_scalar * stream + beta * y

    tl.store(y_ptr + idx, out.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    s_ptr,
    alpha_ptr,
    beta_ptr,
    T,
    D: tl.constexpr,
    eps,
    gain_center,
    alpha_scalar,
    HAS_WEIGHT: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
    KAHAN: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
    RESIDUAL: tl.constexpr,
    GATE_STREAM: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    bT: tl.constexpr,
    bD: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """RMSNorm over the last dim of a ``[T, D]`` contiguous fp/bf tensor.

    ``y_ptr`` need not have ``x_ptr``'s dtype: the row, the weight and the
    blend are fp32, and the store is the only rounding.

    ``SINGLE_TILE`` (``D <= bD``) keeps the row in registers: one load, one
    store. Otherwise the row is walked twice -- once to accumulate the sum of
    squares, once to normalize and store -- re-applying the weight from cache on
    each pass rather than spilling the scaled row to memory.

    ``KAHAN`` accumulates the cross-tile sum of squares in compensated fp32
    instead of fp64. Blackwell runs fp64 at 1/64 of fp32, so the compensated
    form is both faster and about as accurate there.
    """
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    t_idx = tl.program_id(0)
    t = t_idx * bT + tl.arange(0, bT)
    tm = t < T

    if SINGLE_TILE:
        d = tl.arange(0, bD)
        dm = d < D
        idx = t[:, None] * D + d[None, :]
        mask = tm[:, None] & dm[None, :]

        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        if HAS_WEIGHT:
            w = _effective_weight(w_ptr, d, dm, gain_center)

        if INPUT_SCALE and HAS_WEIGHT:
            x = x * w[None, :]

        sum_sq = tl.sum(x * x, axis=1)
        y = x * tl.rsqrt(sum_sq * (1.0 / D) + eps)[:, None]

        if RESIDUAL:
            if HAS_WEIGHT and not INPUT_SCALE:
                y = y * w[None, :]
            _blend_and_store(
                y_ptr,
                s_ptr,
                alpha_ptr,
                beta_ptr,
                idx,
                d,
                mask,
                dm,
                y,
                alpha_scalar,
                GATE_STREAM=GATE_STREAM,
                PER_CHANNEL=PER_CHANNEL,
            )
        else:
            if HAS_WEIGHT and not INPUT_SCALE:
                y = y * w[None, :]
            tl.store(y_ptr + idx, y.to(y_ptr.dtype.element_ty), mask=mask)
    else:
        if KAHAN:
            sum_sq = tl.zeros([bT], dtype=tl.float32)
            compensation = tl.zeros([bT], dtype=tl.float32)
        else:
            sum_sq = tl.zeros([bT], dtype=tl.float64)

        for do in tl.range(0, D, bD):
            d = do + tl.arange(0, bD)
            dm = d < D
            x = tl.load(
                x_ptr + t[:, None] * D + d[None, :],
                mask=tm[:, None] & dm[None, :],
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            if INPUT_SCALE and HAS_WEIGHT:
                w = _effective_weight(w_ptr, d, dm, gain_center)
                x = x * w[None, :]
            tile_sum = tl.sum(x * x, axis=1)
            if KAHAN:
                sum_sq, compensation = _fast2sum(sum_sq, tile_sum + compensation)
            else:
                sum_sq += tile_sum.to(tl.float64)

        # fp32 once, here: without KAHAN the accumulator is fp64, and
        # leaving the cast to the use sites would promote the row
        # multiply to fp64 -- 1/64 rate on Blackwell, and no longer what
        # the fp32 torch reference computes.
        rstd = tl.rsqrt(sum_sq * (1.0 / D) + eps).to(tl.float32)

        for do in tl.range(0, D, bD):
            d = do + tl.arange(0, bD)
            dm = d < D
            idx = t[:, None] * D + d[None, :]
            mask = tm[:, None] & dm[None, :]
            x = tl.load(
                x_ptr + idx, mask=mask, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
            if HAS_WEIGHT:
                w = _effective_weight(w_ptr, d, dm, gain_center)
            if INPUT_SCALE and HAS_WEIGHT:
                x = x * w[None, :]
            if RESIDUAL:
                y = x * rstd[:, None]
                if HAS_WEIGHT and not INPUT_SCALE:
                    y = y * w[None, :]
                _blend_and_store(
                    y_ptr,
                    s_ptr,
                    alpha_ptr,
                    beta_ptr,
                    idx,
                    d,
                    mask,
                    dm,
                    y,
                    alpha_scalar,
                    GATE_STREAM=GATE_STREAM,
                    PER_CHANNEL=PER_CHANNEL,
                )
            else:
                y = x * rstd[:, None]
                if HAS_WEIGHT and not INPUT_SCALE:
                    y = y * w[None, :]
                tl.store(y_ptr + idx, y.to(y_ptr.dtype.element_ty), mask=mask)


def fused_rmsnorm_fwd(
    x: torch.Tensor,
    w: torch.Tensor | None,
    eps: float,
    *,
    gain_center: float = 0.0,
    input_scale: bool = False,
    use_kahan_summation: bool = True,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_stream: torch.Tensor | None = None,
    alpha: torch.Tensor | float | None = None,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSNorm with a gain-centred weight, applied before or after the norm.

    Args:
        x: Input, contiguous, normalized over its last dim.
        w: Weight of shape ``[D]``, or ``None`` for an unweighted norm.
        eps: Added to the mean of squares before the reciprocal square root.
        gain_center: Added to ``w`` (zero-centred gamma). Requires ``w``.
        input_scale: Apply ``w`` before the norm (``rms_norm(x * w)``) rather
            than after it.
        use_kahan_summation: Accumulate the cross-tile sum of squares in
            compensated fp32 rather than fp64. Only affects rows wider than one
            tile. Blackwell runs fp64 at 1/64 of fp32, so leave it on there.
        out: Optional destination; allocated if omitted. Shaped like ``x``, or
            like ``residual_stream`` when one is given.
        out_dtype: dtype of the allocated result; defaults to the stream's
            dtype, else ``x``'s. The computation is fp32 either way, so this is
            the only rounding -- a norm reading the fp32 residual stream and
            feeding a bf16 GEMM rounds once, not twice around a separate cast.
            Ignored (but checked) when ``out`` is given.
        residual_stream: The residual stream to blend the norm output into,
            same shape as ``x``. Turns the epilogue into
            ``alpha * residual_stream + beta * norm(x)``, evaluated in fp32,
            and makes the result take the stream's dtype. Requires ``beta``.
        alpha: Stream coefficient of shape ``[D]`` or ``[1]``, or the float
            ``1.0`` for a connection with no stream gate.
        beta: Branch coefficient, same shape as a tensor ``alpha``.

    Returns:
        The normalized tensor -- blended into the residual stream when one is
        given -- in ``out``'s dtype if given, else ``out_dtype``, else the
        stream's, else ``x``'s.
    """
    input_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    assert x.is_contiguous(), "fused RMSNorm requires a contiguous input"

    T, D = x.shape
    if w is None:
        assert gain_center == 0.0, "gain_center requires a weight"
        assert not input_scale, "input_scale requires a weight"
    else:
        assert w.shape == (D,), f"weight must be [{D}], got {tuple(w.shape)}"
        assert w.stride(-1) == 1, "fused RMSNorm requires a contiguous weight"

    # Narrowed on the values, not on booleans derived from them, so the
    # optional arguments are statically known to be present inside the block.
    residual = residual_stream is not None
    gate_stream = False
    per_channel = False
    alpha_scalar = 1.0
    if residual_stream is not None:
        assert beta is not None, "residual_stream requires beta"
        assert alpha is not None, "residual_stream requires alpha"
        assert residual_stream.shape == input_shape, (
            f"residual_stream must have x's shape {tuple(input_shape)}, got "
            f"{tuple(residual_stream.shape)}"
        )
        residual_stream = residual_stream.reshape(-1, D)
        assert residual_stream.is_contiguous(), (
            "fused RMSNorm requires a contiguous residual_stream"
        )
        assert beta.numel() in (1, D), (
            f"beta must be [1] or [{D}], got {tuple(beta.shape)}"
        )
        assert beta.is_contiguous(), "fused RMSNorm requires a contiguous beta"
        if isinstance(alpha, torch.Tensor):
            gate_stream = True
            assert alpha.shape == beta.shape, (
                f"alpha {tuple(alpha.shape)} and beta {tuple(beta.shape)} must match"
            )
            assert alpha.is_contiguous(), "fused RMSNorm requires a contiguous alpha"
        else:
            alpha_scalar = float(alpha)
        per_channel = beta.numel() > 1

    # MuseSpark's two post-norm residual sites have one fixed B300 contract.
    # Route only that exact contract to the CuTe kernel; every other RMSNorm
    # variant retains the general Triton implementation below.
    if (
        postnorm_gated_residual is not None
        and residual_stream is not None
        and out is None
        and (out_dtype is None or out_dtype == torch.float32)
        and torch.version.hip is None
        and torch.cuda.get_device_capability() == (10, 3)
        and D == 8192
        and x.dtype == torch.bfloat16
        and residual_stream.dtype == torch.float32
        and isinstance(alpha, torch.Tensor)
        and alpha.dtype == torch.float32
        and alpha.numel() == D
        and beta is not None
        and beta.dtype == torch.float32
        and beta.numel() == D
        and ((w is None and not input_scale) or (w is not None and input_scale))
        and (w is None or w.dtype == torch.bfloat16)
    ):
        return postnorm_gated_residual(
            x,
            residual_stream,
            w,
            alpha,
            beta,
            eps,
            gain_center,
        ).view(input_shape)

    if out is None:
        y = torch.empty_like(residual_stream if residual else x, dtype=out_dtype)
    else:
        assert out_dtype is None or out.dtype == out_dtype, (
            f"out is {out.dtype}, out_dtype says {out_dtype}"
        )
        assert out.shape == input_shape, "out must have x's shape"
        y = out.reshape(-1, D)
        assert y.is_contiguous(), "out must be contiguous"

    # One tile is 8192 elements; a row wider than that is walked in bD-wide
    # chunks and bT collapses to a single row per program.
    tile_elems = 8192
    bD = min(triton.next_power_of_2(D), tile_elems)
    # The residual epilogue holds the stream and the blend in fp32 alongside
    # the row, so a full-width tile spills once there are enough rows to want
    # occupancy. Measured on B300 at D=8192 (us, median): T=512 wants bD 4096
    # (14.4 vs 18.1 at 8192), T=8192 wants 2048 (136.6 vs 170.1). Below that
    # the single tile still wins -- T=1 is 8.2 us against 12.3 at bD 2048.
    if residual:
        if T >= 2048:
            bD = min(bD, 2048)
        elif T >= 128:
            bD = min(bD, 4096)
    bT = max(1, triton.next_power_of_2(tile_elems // bD))
    # A single full-width tile is register-bound rather than latency-bound, so
    # it wants the extra warps to hide the spills; the chunked path does not.
    num_warps = 16 if residual and bD >= D else 8

    enable_pdl = False
    _rmsnorm_fwd_kernel[(triton.cdiv(T, bT),)](
        x,
        y,
        w,
        # Unused pointers still have to be valid; x stands in for them and the
        # RESIDUAL/GATE_STREAM constexprs keep them from being dereferenced.
        residual_stream if residual else x,
        alpha if gate_stream else (beta if residual else x),
        beta if residual else x,
        T,
        D=D,
        eps=eps,
        gain_center=gain_center,
        alpha_scalar=alpha_scalar,
        HAS_WEIGHT=w is not None,
        INPUT_SCALE=input_scale,
        KAHAN=use_kahan_summation,
        SINGLE_TILE=bD >= D,
        RESIDUAL=residual,
        GATE_STREAM=gate_stream,
        PER_CHANNEL=per_channel,
        bT=bT,
        bD=bD,
        num_warps=num_warps,
        USE_PDL=enable_pdl,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return y.view(input_shape)


# The whole row has to be live for the second norm, so this path is single-tile
# only: one program per row, no chunked walk.
_BOUNDARY_TILE_ELEMS = 8192


@triton.jit
def _norm_residual_norm_fwd_kernel(
    branch_ptr,
    stream_ptr,
    new_stream_ptr,
    out_ptr,
    post_w_ptr,
    pre_w_ptr,
    alpha_ptr,
    beta_ptr,
    T,
    D: tl.constexpr,
    post_eps,
    pre_eps,
    post_gain_center,
    pre_gain_center,
    alpha_scalar,
    POST_HAS_WEIGHT: tl.constexpr,
    POST_INPUT_SCALE: tl.constexpr,
    PRE_HAS_WEIGHT: tl.constexpr,
    PRE_INPUT_SCALE: tl.constexpr,
    GATE_STREAM: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    bT: tl.constexpr,
    bD: tl.constexpr,
):
    """One residual boundary: post-norm, gated add, and the next pre-norm.

    The row is loaded once and never leaves registers. ``new_stream`` is stored
    on the way past -- the next layer's residual add needs it -- but the
    pre-norm reads the register copy, so the fp32 stream is not walked a second
    time.

    Both norms take their own eps, weight and gain centre, and each can apply
    its weight before the norm (``rms_norm(x * w)``) or after it. That covers
    the two flavours MuseSpark1x mixes: the post-FFN norm is weight-first under
    ``use_ffn_learnable_scales``, every pre-norm is not.
    """
    t = tl.program_id(0) * bT + tl.arange(0, bT)
    tm = t < T
    d = tl.arange(0, bD)
    dm = d < D
    idx = t[:, None] * D + d[None, :]
    mask = tm[:, None] & dm[None, :]

    # All loads before the first reduction: at decode this is latency-bound.
    x = tl.load(branch_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    stream = tl.load(stream_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    if PER_CHANNEL:
        beta = tl.load(beta_ptr + d, mask=dm, other=0.0)[None, :]
    else:
        beta = tl.load(beta_ptr)
    if GATE_STREAM:
        if PER_CHANNEL:
            alpha = tl.load(alpha_ptr + d, mask=dm, other=0.0)[None, :]
        else:
            alpha = tl.load(alpha_ptr)
    if POST_HAS_WEIGHT:
        post_w = _effective_weight(post_w_ptr, d, dm, post_gain_center)
    if PRE_HAS_WEIGHT:
        pre_w = _effective_weight(pre_w_ptr, d, dm, pre_gain_center)

    # post_norm(branch)
    if POST_INPUT_SCALE and POST_HAS_WEIGHT:
        x = x * post_w[None, :]
    sum_sq = tl.sum(x * x, axis=1)
    y = x * tl.rsqrt(sum_sq * (1.0 / D) + post_eps)[:, None]
    if POST_HAS_WEIGHT and not POST_INPUT_SCALE:
        y = y * post_w[None, :]

    # alpha * stream + beta * post_norm(branch), in fp32
    if GATE_STREAM:
        new_stream = alpha * stream + beta * y
    else:
        new_stream = alpha_scalar * stream + beta * y
    tl.store(
        new_stream_ptr + idx,
        new_stream.to(new_stream_ptr.dtype.element_ty),
        mask=mask,
    )

    # pre_norm(new_stream), from registers
    z = new_stream
    if PRE_INPUT_SCALE and PRE_HAS_WEIGHT:
        z = z * pre_w[None, :]
    pre_sum_sq = tl.sum(z * z, axis=1)
    z = z * tl.rsqrt(pre_sum_sq * (1.0 / D) + pre_eps)[:, None]
    if PRE_HAS_WEIGHT and not PRE_INPUT_SCALE:
        z = z * pre_w[None, :]

    tl.store(out_ptr + idx, z.to(out_ptr.dtype.element_ty), mask=mask)


def supports_norm_residual_norm(
    branch: torch.Tensor,
    stream: torch.Tensor,
) -> bool:
    """Whether :func:`fused_norm_residual_norm_fwd` can serve this boundary.

    Args:
        branch: The branch output the post-norm consumes.
        stream: The residual stream it blends into.

    Returns:
        True when the fused kernel applies; the caller runs the norm and the
        residual add as separate ops otherwise.
    """
    if not (branch.is_cuda and stream.is_cuda):
        return False
    if branch.shape != stream.shape or branch.ndim < 2:
        return False
    if stream.dtype is not torch.float32:
        return False
    if not (branch.is_contiguous() and stream.is_contiguous()):
        return False
    return triton.next_power_of_2(branch.shape[-1]) <= _BOUNDARY_TILE_ELEMS


def fused_norm_residual_norm_fwd(
    branch: torch.Tensor,
    stream: torch.Tensor,
    post_weight: torch.Tensor | None,
    pre_weight: torch.Tensor | None,
    post_eps: float,
    pre_eps: float,
    *,
    alpha: torch.Tensor | float,
    beta: torch.Tensor,
    post_gain_center: float = 0.0,
    pre_gain_center: float = 0.0,
    post_input_scale: bool = False,
    pre_input_scale: bool = False,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A whole residual boundary in one kernel.

    Computes, in fp32::

        new_stream = alpha * stream + beta * post_norm(branch)
        out = pre_norm(new_stream)

    which is the shape of both residual sites in every MuseSpark1x decoder
    layer -- once inside the layer (post-attn norm into the pre-FFN norm) and
    once across the layer boundary (post-FFN norm into the next layer's
    pre-attention norm). Running the three steps together costs one read of the
    branch and one of the stream, against the two reads and the extra full-width
    fp32 write that the norm-then-norm pair does.

    Numerically equivalent to ``fused_rmsnorm_fwd(..., residual_stream=stream)``
    followed by ``fused_rmsnorm_fwd(new_stream, ..., out_dtype=out_dtype)``:
    same fp32 math and the same rounding points -- the branch stays in fp32
    across the blend in both, and the stream is fp32 in registers and in memory.
    Only the order of the two sums of squares differs, because the row cannot be
    walked in chunks here. That is a rounding difference the pair already has
    with itself: its chunk width narrows as the batch grows. Measured against an
    fp64 reference at D=8192 the two agree to the digit, and 9e-6 of the bf16
    outputs differ by one ulp.

    Args:
        branch: Branch output ``[..., D]``, contiguous. The post-norm's input.
        stream: Residual stream, fp32, same shape as ``branch``, contiguous.
        post_weight: Post-norm weight ``[D]``, or None for an unweighted norm.
        pre_weight: Pre-norm weight ``[D]``, or None.
        post_eps: Post-norm epsilon.
        pre_eps: Pre-norm epsilon.
        alpha: Stream coefficient ``[D]`` or ``[1]``, or the float ``1.0`` for a
            connection with no stream gate.
        beta: Branch coefficient, same shape as a tensor ``alpha``.
        post_gain_center: Added to ``post_weight`` (zero-centred gamma). Pass
            0.0 when the norm does not centre its gain.
        pre_gain_center: Added to ``pre_weight``, same convention.
        post_input_scale: Apply ``post_weight`` before the post-norm.
        pre_input_scale: Apply ``pre_weight`` before the pre-norm.
        out_dtype: dtype of the pre-norm output; defaults to ``branch``'s. The
            computation is fp32, so this is the only rounding on that output.

    Returns:
        ``(out, new_stream)``. ``out`` has ``branch``'s shape in ``out_dtype``;
        ``new_stream`` is the fp32 stream the next boundary blends into.
    """
    input_shape = branch.shape
    assert supports_norm_residual_norm(branch, stream), (
        "fused norm/residual/norm boundary: unsupported branch "
        f"{tuple(branch.shape)}:{branch.dtype} over stream "
        f"{tuple(stream.shape)}:{stream.dtype}; check "
        "supports_norm_residual_norm() first"
    )

    branch = branch.reshape(-1, input_shape[-1])
    stream = stream.reshape(-1, input_shape[-1])
    T, D = branch.shape

    for weight, gain_center, input_scale, name in (
        (post_weight, post_gain_center, post_input_scale, "post"),
        (pre_weight, pre_gain_center, pre_input_scale, "pre"),
    ):
        if weight is None:
            assert gain_center == 0.0, f"{name}_gain_center requires a weight"
            assert not input_scale, f"{name}_input_scale requires a weight"
        else:
            assert weight.shape == (D,), (
                f"{name}_weight must be [{D}], got {tuple(weight.shape)}"
            )
            assert weight.stride(-1) == 1, f"{name}_weight must be contiguous"

    assert beta.numel() in (1, D), f"beta must be [1] or [{D}], got {tuple(beta.shape)}"
    assert beta.is_contiguous(), "beta must be contiguous"
    per_channel = beta.numel() > 1
    gate_stream = isinstance(alpha, torch.Tensor)
    alpha_scalar = 1.0
    if gate_stream:
        assert isinstance(alpha, torch.Tensor)
        assert alpha.shape == beta.shape, (
            f"alpha {tuple(alpha.shape)} and beta {tuple(beta.shape)} must match"
        )
        assert alpha.is_contiguous(), "alpha must be contiguous"
    else:
        alpha_scalar = float(alpha)

    out = torch.empty_like(branch, dtype=out_dtype)
    new_stream = torch.empty_like(stream)

    bD = triton.next_power_of_2(D)
    bT = max(1, _BOUNDARY_TILE_ELEMS // bD)

    num_warps = 32 if bD >= 8192 else (16 if bD >= 4096 else 8)
    if torch.version.hip is not None:
        # Triton maps a warp to a 64-thread wavefront on ROCm, so 32 warps
        # would request 2048 threads and exceed the 1024-thread workgroup
        # limit. CUDA warps are 32 threads and can retain the wider launch.
        num_warps = min(num_warps, 16)
    _norm_residual_norm_fwd_kernel[(triton.cdiv(T, bT),)](
        branch,
        stream,
        new_stream,
        out,
        # Unused pointers still have to be valid; the constexprs below keep
        # them from being dereferenced.
        post_weight if post_weight is not None else branch,
        pre_weight if pre_weight is not None else branch,
        alpha if gate_stream else beta,
        beta,
        T,
        D=D,
        post_eps=post_eps,
        pre_eps=pre_eps,
        post_gain_center=post_gain_center,
        pre_gain_center=pre_gain_center,
        alpha_scalar=alpha_scalar,
        POST_HAS_WEIGHT=post_weight is not None,
        POST_INPUT_SCALE=post_input_scale,
        PRE_HAS_WEIGHT=pre_weight is not None,
        PRE_INPUT_SCALE=pre_input_scale,
        GATE_STREAM=gate_stream,
        PER_CHANNEL=per_channel,
        bT=bT,
        bD=bD,
        num_warps=num_warps,
    )
    return out.view(input_shape), new_stream.view(input_shape)
