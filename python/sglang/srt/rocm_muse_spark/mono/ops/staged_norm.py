# SPDX-License-Identifier: Apache-2.0
"""Reuse the native fused boundary kernel with caller-owned graph storage."""

from sglang.srt.rocm_muse_spark.native_ops.fused_rmsnorm import (
    _norm_residual_norm_fwd_kernel,
)


def norm_into(
    branch, stream, pre_weight, alpha, beta, post_eps, pre_eps, out, new_stream
):
    # Wide Muse uses an unweighted post-attention norm, zero-centered pre-FFN
    # norm, and per-channel FP32 residual gates. The native kernel retains the
    # original arithmetic; only output allocation is supplied by the caller.
    rows = branch.shape[0]
    _norm_residual_norm_fwd_kernel[(rows,)](
        branch,
        stream,
        new_stream,
        out,
        branch,
        pre_weight,
        alpha,
        beta,
        rows,
        D=8192,
        post_eps=post_eps,
        pre_eps=pre_eps,
        post_gain_center=0.0,
        pre_gain_center=1.0,
        alpha_scalar=1.0,
        POST_HAS_WEIGHT=False,
        POST_INPUT_SCALE=False,
        PRE_HAS_WEIGHT=True,
        PRE_INPUT_SCALE=False,
        GATE_STREAM=True,
        PER_CHANNEL=True,
        bT=1,
        bD=8192,
        num_warps=16,
    )
