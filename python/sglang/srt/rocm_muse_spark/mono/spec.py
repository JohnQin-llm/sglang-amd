# SPDX-License-Identifier: Apache-2.0
"""Shape constants and pure eligibility checks for original Muse K1+K2."""

from __future__ import annotations

import os

from sglang.srt.environ import envs

TP = 8
MAX_TOKENS = 8
WIDE_TOKENS = (16, 24)
# Freeze before registration, allocation, and graph capture.
WIDE_ENABLED = envs.SGLANG_MUSE_MONO_WIDE_STAGED.get()
SERVING_WIDTHS = (1, 8) + (WIDE_TOKENS if WIDE_ENABLED else ())
ALLOCATION_WIDTHS = tuple(range(1, MAX_TOKENS + 1)) + (
    WIDE_TOKENS if WIDE_ENABLED else ()
)
BLOCKS = 256
THREADS = 512
HIDDEN = 8192
HEADS, KV_HEADS, HEAD_DIM = 16, 2, 64
QKVG = 2 * (HEADS + KV_HEADS) * HEAD_DIM
MOE_HIDDEN = 4096
EXPERTS, TOP_K = 256, 8
INTER = 512
WINDOW = 2048
KV_BLOCK = 16
MAX_FULL_CONTEXT = 32768


def native_attention_enabled() -> bool:
    """All selected K2 layers retain the native attention and cache backend."""
    return os.getenv("SGLANG_MUSE_ORIGINAL_NATIVE_ATTENTION", "0") == "1"


def kernel_layers(config) -> list[int]:
    """Sparse sliding layers except the first/last boundary owners."""

    config = getattr(config, "text_config", config)
    mlp = getattr(config, "mlp_layer_types", ())
    attention = getattr(config, "layer_types", ())
    return [
        i
        for i in range(1, int(config.num_hidden_layers) - 1)
        if i < len(mlp)
        and mlp[i] == "sparse"
        and (not attention or attention[i] == "sliding_attention")
    ]


def native_attention_layers(config) -> list[int]:
    """Interior sparse full-attention layers whose native core can feed K2.

    The released checkpoint leaves the first and last experts in BF16.
    Keep those boundary layers on their native path, as for fused K1+K2.
    """
    config = getattr(config, "text_config", config)
    mlp = getattr(config, "mlp_layer_types", ())
    attention = getattr(config, "layer_types", ())
    return [
        i
        for i in range(1, int(config.num_hidden_layers) - 1)
        if i < len(mlp)
        and mlp[i] == "sparse"
        and i < len(attention)
        and attention[i] == "full_attention"
    ]


def step_disable_reason(forward_batch, rows: int) -> str | None:
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()):
        return "not decode/target-verify"
    if rows not in ALLOCATION_WIDTHS:
        return f"width {rows} is not enabled for Muse monokernels"
    batch = int(forward_batch.batch_size)
    if batch <= 0 or rows % batch:
        return "rows do not hold whole uniform requests"
    q_len = rows // batch
    if mode.is_decode() and q_len != 1:
        return "decode query length is not one"
    return None
