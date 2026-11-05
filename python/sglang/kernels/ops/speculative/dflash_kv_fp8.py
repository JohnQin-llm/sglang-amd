"""FP8 draft-KV writes for the DFlash draft on ROCm (``SGLANG_MUSE_DFLASH_KV_FP8_FUSE``).

The served draft writes its FP8 KV cache with torch casts: per layer and step, ``k.to(fp8)`` and ``v.to(fp8)``
(~5 us each on gfx1250, they run on a handful of rows) and the store kernel; per committed context, the clamp,
cast and prefix-valid store once per draft layer. Here:

* ``table_qk_norm_rope_kv_`` is ``table_qk_norm_rope_`` that also writes K/V to the FP8 cache rows, so the
  backend's store is skipped (one launch instead of four per draft layer).
* ``commit_ctx_kv_fp8_prefix_valid`` writes the stacked K/V of the context commit for all draft layers in one
  launch (one launch instead of five per draft layer).

The FP8 bytes equal the eager path's: a clamp to +-448, then c10's float -> float8_e4m3fn conversion (round to
nearest even). The clamp also matches the unclamped ``.to(float8_e4m3fn)`` of this ROCm torch, which saturates.
"""

from typing import List, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3fn_bits(x, SATURATE: tl.constexpr):
    # c10::detail::fp8e4m3fn_from_fp32_value on an FP32 tensor; returns the byte as int32.
    if SATURATE:
        # saturate_to_fp8_range: clamp at the finite max, NaN passes through.
        x = tl.where(x > 448.0, 448.0, tl.where(x < -448.0, -448.0, x))
    b = x.to(tl.int32, bitcast=True)
    neg = b < 0
    a = b & 0x7FFFFFFF
    # Below 2**-6: add 2**14 so the FP32 adder rounds the mantissa to the denormal grid.
    d = (a.to(tl.float32, bitcast=True) + 16384.0).to(tl.int32, bitcast=True) - (141 << 23)
    n = (a - (120 << 23) + 0x7FFFF + ((a >> 20) & 1)) >> 20
    r = tl.where(a < (121 << 23), d, n)
    r = tl.where(a >= (1087 << 20), 0x7F, r & 0xFF)
    return tl.where(neg, r | 0x80, r)


@triton.jit
def _table_qk_norm_rope_kv_kernel(
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    pos_ptr,
    loc_ptr,
    k_cache_ptr,
    v_cache_ptr,
    row_stride,
    cache_row_stride,
    NHQ: tl.constexpr,
    NHK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    slot = tl.load(loc_ptr + t).to(tl.int64)
    if h < NHQ + NHK:
        # _table_qk_norm_rope_kernel, unchanged.
        pos = tl.load(pos_ptr + t).to(tl.int64)
        cos = tl.load(cos_sin_ptr + pos * D + half_ar).to(tl.float32)
        sin = tl.load(cos_sin_ptr + pos * D + HALF + half_ar).to(tl.float32)
        is_q = h < NHQ
        col0 = (h * D).to(tl.int64)
        w_ptr = tl.where(is_q, q_weight_ptr.to(tl.int64), k_weight_ptr.to(tl.int64)).to(
            tl.pointer_type(tl.bfloat16)
        )
        row = qkv_ptr + t * row_stride + col0
        x = tl.load(row + d_ar).to(tl.float32)
        ms = tl.sum(x * x, 0) / D
        inv = 1.0 / tl.sqrt(ms + EPS)
        w1 = tl.load(w_ptr + half_ar).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + half_ar).to(tl.float32)
        x1 = tl.load(row + half_ar).to(tl.float32) * inv * w1
        x2 = tl.load(row + HALF + half_ar).to(tl.float32) * inv * w2
        x1 = x1.to(tl.bfloat16).to(tl.float32)
        x2 = x2.to(tl.bfloat16).to(tl.float32)
        o1 = (x1 * cos - x2 * sin).to(tl.bfloat16)
        o2 = (x2 * cos + x1 * sin).to(tl.bfloat16)
        tl.store(row + half_ar, o1)
        tl.store(row + HALF + half_ar, o2)
        if h >= NHQ:
            k_row = k_cache_ptr + slot * cache_row_stride + (h - NHQ) * D
            tl.store(k_row + half_ar, _e4m3fn_bits(o1.to(tl.float32), True).to(tl.uint8))
            tl.store(k_row + HALF + half_ar, _e4m3fn_bits(o2.to(tl.float32), True).to(tl.uint8))
    else:
        hv = h - NHQ - NHK
        v = tl.load(qkv_ptr + t * row_stride + (NHQ + NHK + hv) * D + d_ar).to(tl.float32)
        tl.store(
            v_cache_ptr + slot * cache_row_stride + hv * D + d_ar,
            _e4m3fn_bits(v, True).to(tl.uint8),
        )


def table_qk_norm_rope_kv_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
    eps: float,
) -> None:
    """``table_qk_norm_rope_`` on ``qkv`` [T, (NHQ + 2 NHK) D], plus K and V into the FP8 cache rows ``loc``.

    ``k_cache`` / ``v_cache`` are one layer's [slots, NHK, D] FP8 (or uint8) buffers; the bytes equal
    ``set_kv_buffer``'s BF16 -> float8_e4m3fn cast of the same K and V.
    """
    T = qkv.shape[0]
    if T == 0:
        return
    _table_qk_norm_rope_kv_kernel[(T, num_q_heads + 2 * num_k_heads)](
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        loc,
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        qkv.stride(0),
        k_cache.stride(0),
        NHQ=num_q_heads,
        NHK=num_k_heads,
        D=head_dim,
        EPS=eps,
    )


@triton.jit
def _commit_ctx_kv_fp8_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptrs,
    v_cache_ptrs,
    loc_ptr,
    commit_lens_ptr,
    src_layer_stride,
    src_row_stride,
    cache_row_stride,
    BLOCK: tl.constexpr,
    ROW: tl.constexpr,
):
    r = tl.program_id(0)
    layer = tl.program_id(1)
    b = r // BLOCK
    j = r - b * BLOCK
    if j >= tl.load(commit_lens_ptr + b):
        return
    slot = tl.load(loc_ptr + r).to(tl.int64)
    c = tl.arange(0, ROW)
    src = layer.to(tl.int64) * src_layer_stride + r.to(tl.int64) * src_row_stride + c
    k_base = tl.load(k_cache_ptrs + layer).to(tl.pointer_type(tl.uint8))
    v_base = tl.load(v_cache_ptrs + layer).to(tl.pointer_type(tl.uint8))
    k = tl.load(k_ptr + src).to(tl.float32)
    v = tl.load(v_ptr + src).to(tl.float32)
    tl.store(k_base + slot * cache_row_stride + c, _e4m3fn_bits(k, True).to(tl.uint8))
    tl.store(v_base + slot * cache_row_stride + c, _e4m3fn_bits(v, True).to(tl.uint8))


class CtxKVCommitFP8:
    """One-launch prefix-valid FP8 commit of the stacked context K/V into every draft layer's cache."""

    def __init__(self, k_caches: List[torch.Tensor], v_caches: List[torch.Tensor]):
        device = k_caches[0].device
        self.n_layers = len(k_caches)
        self.row = k_caches[0].shape[1] * k_caches[0].shape[2]
        self.cache_row_stride = k_caches[0].stride(0)
        for c in list(k_caches) + list(v_caches):
            if c.stride(0) != self.cache_row_stride or c[0].numel() != self.row or not c[0].is_contiguous():
                raise ValueError("draft KV buffers must share one contiguous row layout")
        self.k_ptrs = torch.tensor([c.data_ptr() for c in k_caches], dtype=torch.int64, device=device)
        self.v_ptrs = torch.tensor([c.data_ptr() for c in v_caches], dtype=torch.int64, device=device)

    def __call__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """``k`` / ``v``: [n_layers, rows, NHK, D] BF16 (rows = loc_2d.numel(), row-major over loc_2d)."""
        bs, block = loc_2d.shape
        rows = bs * block
        if rows == 0:
            return
        assert k.shape[0] == self.n_layers and k.shape[1] >= rows and k.stride() == v.stride()
        assert k.stride(2) * k.shape[2] == k.stride(1) and k.stride(3) == 1
        _commit_ctx_kv_fp8_kernel[(rows, self.n_layers)](
            k,
            v,
            self.k_ptrs,
            self.v_ptrs,
            loc_2d,
            commit_lens,
            k.stride(0),
            k.stride(1),
            self.cache_row_stride,
            BLOCK=block,
            ROW=self.row,
            num_warps=1,
        )
