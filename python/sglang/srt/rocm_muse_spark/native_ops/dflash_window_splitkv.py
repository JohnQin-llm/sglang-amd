# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""C1 DFlash8 sliding-window attention with persistent split-KV scratch.

The online-softmax structure and cache quantization boundaries are adapted from
SGLang's ``extend_attention.py`` and ``verify_splitkv.py``.  The specialized
schedule supports only q8/h8/d128, one KV head, and window 2047.
"""

import torch
import triton
import triton.language as tl

N_HEADS = 8
N_ROWS = 8
N_SPLITS = 32
BLOCK_M = 16
BLOCK_N = 64
HEAD_DIM = 128
WINDOW = 2047


@triton.jit
def _draft_window_stage1(
    Q,
    K,
    V,
    KBuffer,
    VBuffer,
    QOIndptr,
    KVIndptr,
    KVIndices,
    Partial,
    LSE,
    sm_scale,
    k_scale,
    v_scale,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_kbs,
    stride_kbh,
    stride_kbd,
    stride_vbs,
    stride_vbh,
    stride_vbd,
    stride_pb,
    stride_ph,
    stride_ps,
    stride_pm,
    stride_lb,
    stride_lh,
    stride_ls,
    WINDOW_SIZE: tl.constexpr,
    N_ROWS: tl.constexpr = 8,
    N_SPLITS: tl.constexpr = 32,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 64,
    HEAD_DIM: tl.constexpr = 128,
):
    """One program owns one (batch, head, split), including 16 scratch rows."""
    head = tl.program_id(0)
    split = tl.program_id(1)
    batch = tl.program_id(2)
    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    cols = tl.arange(0, BLOCK_N)
    real_rows = rows < N_ROWS

    query_begin = tl.load(QOIndptr + batch)
    prefix_begin = tl.load(KVIndptr + batch)
    prefix_len = tl.load(KVIndptr + batch + 1) - prefix_begin
    # At most one 64-key prefix tile per split for the 2047-token window.
    per_split = tl.cdiv(tl.cdiv(prefix_len, N_SPLITS), BLOCK_N) * BLOCK_N
    split_begin = split * per_split
    split_end = tl.minimum(split_begin + per_split, prefix_len)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # q can be non-contiguous; all three q strides participate in addressing.
    program_live = (split_begin < split_end) | (split == N_SPLITS - 1)
    q = tl.load(
        Q
        + (query_begin + rows[:, None]) * stride_qm
        + head * stride_qh
        + dims[None, :] * stride_qd,
        mask=real_rows[:, None] & program_live,
        other=0.0,
    )
    q_prefix = q.to(KBuffer.dtype.element_ty)

    # Prefix cache. A row/key is live iff prefix_len + row <= key + window.
    for start in tl.range(split_begin, split_end, BLOCK_N):
        logical_k = start + cols
        key_live = logical_k < split_end
        physical_k = tl.load(
            KVIndices + prefix_begin + logical_k, mask=key_live, other=0
        )
        kb = tl.load(
            KBuffer
            + physical_k[None, :] * stride_kbs
            + 0 * stride_kbh
            + dims[:, None] * stride_kbd,
            mask=key_live[None, :],
            other=0.0,
        )
        scores = tl.dot(q_prefix, kb) * (sm_scale * k_scale)
        visible = (
            real_rows[:, None]
            & key_live[None, :]
            & (prefix_len + rows[:, None] <= logical_k[None, :] + WINDOW_SIZE)
        )
        scores = tl.where(visible, scores, -float("inf"))

        row_max = tl.max(scores, axis=1)
        # A fully masked tile must leave zero mass and must not form inf-inf.
        row_max_safe = tl.where(row_max == -float("inf"), -1.0e20, row_max)
        m_new = tl.maximum(m_i, row_max_safe)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        vb = tl.load(
            VBuffer
            + physical_k[:, None] * stride_vbs
            + 0 * stride_vbh
            + dims[None, :] * stride_vbd,
            mask=key_live[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(vb.dtype), vb) * v_scale
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Only the last split owns the fresh, noncausal 8-key block.
    if split == N_SPLITS - 1:
        fresh_live = cols < N_ROWS
        kf = tl.load(
            K
            + (query_begin + cols[None, :]) * stride_km
            + 0 * stride_kh
            + dims[:, None] * stride_kd,
            mask=fresh_live[None, :],
            other=0.0,
        )
        scores = tl.dot(q.to(kf.dtype), kf) * sm_scale
        visible = real_rows[:, None] & fresh_live[None, :]
        scores = tl.where(visible, scores, -float("inf"))

        row_max = tl.max(scores, axis=1)
        row_max_safe = tl.where(row_max == -float("inf"), -1.0e20, row_max)
        m_new = tl.maximum(m_i, row_max_safe)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        vf = tl.load(
            V
            + (query_begin + cols[:, None]) * stride_vm
            + 0 * stride_vh
            + dims[None, :] * stride_vd,
            mask=fresh_live[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(vf.dtype), vf)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    has_mass = real_rows & (l_i > 0.0)
    safe_l = tl.where(has_mass, l_i, 1.0)
    partial = tl.where(has_mass[:, None], acc / safe_l[:, None], 0.0)
    lse = tl.where(has_mass, m_i + tl.log(safe_l), -float("inf"))

    tl.store(
        Partial
        + batch * stride_pb
        + head * stride_ph
        + split * stride_ps
        + rows[:, None] * stride_pm
        + dims[None, :],
        partial,
    )
    tl.store(
        LSE + batch * stride_lb + head * stride_lh + split * stride_ls + rows,
        lse,
    )


@triton.jit
def _draft_window_stage2(
    Partial,
    LSE,
    Out,
    QOIndptr,
    stride_pb,
    stride_ph,
    stride_ps,
    stride_pm,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_om,
    stride_oh,
    stride_od,
    N_SPLITS: tl.constexpr = 32,
    HEAD_DIM: tl.constexpr = 128,
):
    """One program per (batch, row, head); reduce its 32 private splits."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    splits = tl.arange(0, N_SPLITS)
    dims = tl.arange(0, HEAD_DIM)

    lse = tl.load(
        LSE
        + batch * stride_lb
        + head * stride_lh
        + splits * stride_ls
        + row
    )
    max_lse = tl.max(lse, axis=0)
    finite = lse != -float("inf")
    safe_max = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    weights = tl.where(finite, tl.exp(lse - safe_max), 0.0)
    denom = tl.sum(weights, axis=0)
    partial = tl.load(
        Partial
        + batch * stride_pb
        + head * stride_ph
        + splits[:, None] * stride_ps
        + row * stride_pm
        + dims[None, :]
    )
    numer = tl.sum(partial * weights[:, None], axis=0)
    result = numer / tl.where(denom > 0.0, denom, 1.0)
    tl.store(
        Out
        + (tl.load(QOIndptr + batch) + row) * stride_om
        + head * stride_oh
        + dims * stride_od,
        result.to(Out.dtype.element_ty),
    )


class DraftWindowSplitKV:
    """Persistent scratch for dense batches of eight-token DFlash requests."""

    def __init__(self, device="cuda", max_batch_size=1):
        if (
            not isinstance(max_batch_size, int)
            or isinstance(max_batch_size, bool)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        self.partial = torch.empty(
            (max_batch_size, N_HEADS, N_SPLITS, BLOCK_M, HEAD_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self.lse = torch.empty(
            (max_batch_size, N_HEADS, N_SPLITS, BLOCK_M),
            dtype=torch.float32,
            device=self.device,
        )
        # Resolve an unspecified "cuda" device to the concrete allocation
        # device (for example cuda:0) before metadata-only hot-call checks.
        self.device = self.partial.device

    @staticmethod
    def _check_tensor(name, tensor, shape, dtype=None):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if dtype is not None and tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")

    @staticmethod
    def _check_positive_strides(name, tensor):
        if any(
            size > 1 and stride <= 0
            for size, stride in zip(tensor.shape, tensor.stride())
        ):
            raise ValueError(
                f"{name} must have positive strides on non-singleton axes"
            )

    @staticmethod
    def _check_non_overlapping_output(name, tensor):
        # A conservative metadata-only test which permits transposes and holes.
        required_stride = 1
        for stride, size in sorted(
            (stride, size)
            for size, stride in zip(tensor.shape, tensor.stride())
            if size > 1
        ):
            if stride < required_stride:
                raise ValueError(f"{name} must not have internally overlapping rows")
            required_stride = stride * size

    def __call__(
        self,
        q,
        k,
        v,
        out,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        *,
        sm_scale,
        k_scale=1.0,
        v_scale=1.0,
        sliding_window_size=WINDOW,
    ):
        if not isinstance(q, torch.Tensor):
            raise TypeError("q must be a torch.Tensor")
        if q.ndim != 3 or q.shape[1:] != (N_HEADS, HEAD_DIM):
            raise ValueError(
                f"q must have shape [8 * batch, {N_HEADS}, {HEAD_DIM}], "
                f"got {tuple(q.shape)}"
            )
        if q.shape[0] == 0 or q.shape[0] % N_ROWS != 0:
            raise ValueError("q must contain exactly eight contiguous rows per request")
        batch_size = q.shape[0] // N_ROWS
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch size {batch_size} exceeds max_batch_size={self.max_batch_size}"
            )
        rows = batch_size * N_ROWS
        self._check_tensor("q", q, (rows, N_HEADS, HEAD_DIM), torch.bfloat16)
        self._check_tensor("k", k, (rows, 1, HEAD_DIM), torch.bfloat16)
        self._check_tensor("v", v, (rows, 1, HEAD_DIM), torch.bfloat16)
        self._check_tensor("out", out, (rows, N_HEADS, HEAD_DIM), torch.bfloat16)
        self._check_tensor("qo_indptr", qo_indptr, (batch_size + 1,))
        self._check_tensor("kv_indptr", kv_indptr, (batch_size + 1,))
        if not isinstance(kv_indices, torch.Tensor):
            raise TypeError("kv_indices must be a torch.Tensor")
        if not isinstance(k_buffer, torch.Tensor):
            raise TypeError("k_buffer must be a torch.Tensor")
        if not isinstance(v_buffer, torch.Tensor):
            raise TypeError("v_buffer must be a torch.Tensor")
        if kv_indices.ndim != 1 or not kv_indices.is_contiguous():
            raise ValueError("kv_indices must be a contiguous 1-D buffer")
        if not qo_indptr.is_contiguous() or not kv_indptr.is_contiguous():
            raise ValueError("indptr buffers must be contiguous")
        if k_buffer.ndim != 3 or k_buffer.shape[1:] != (1, HEAD_DIM):
            raise ValueError("k_buffer must have shape [slots, 1, 128]")
        if v_buffer.shape != k_buffer.shape:
            raise ValueError("v_buffer must have the same shape as k_buffer")
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("out", out),
            ("k_buffer", k_buffer),
            ("v_buffer", v_buffer),
        ):
            self._check_positive_strides(name, tensor)
        self._check_non_overlapping_output("out", out)
        cache_types = {torch.bfloat16}
        fp8 = getattr(torch, "float8_e4m3fn", None)
        if fp8 is not None:
            cache_types.add(fp8)
        if k_buffer.dtype not in cache_types or v_buffer.dtype != k_buffer.dtype:
            raise TypeError("cache K/V must share BF16 or FP8 e4m3fn dtype")
        if qo_indptr.dtype not in (torch.int32, torch.int64):
            raise TypeError("qo_indptr must be int32 or int64")
        if kv_indptr.dtype not in (torch.int32, torch.int64):
            raise TypeError("kv_indptr must be int32 or int64")
        if kv_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("kv_indices must be int32 or int64")
        tensors = (q, k, v, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices)
        if any(t.device != self.device for t in tensors):
            raise ValueError("all inputs and output must be on the launcher's device")
        if sliding_window_size != WINDOW:
            raise ValueError("only sliding_window_size=2047 is supported")

        _draft_window_stage1[(N_HEADS, N_SPLITS, batch_size)](
            q,
            k,
            v,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.partial,
            self.lse,
            sm_scale,
            k_scale,
            v_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            k_buffer.stride(0),
            k_buffer.stride(1),
            k_buffer.stride(2),
            v_buffer.stride(0),
            v_buffer.stride(1),
            v_buffer.stride(2),
            self.partial.stride(0),
            self.partial.stride(1),
            self.partial.stride(2),
            self.partial.stride(3),
            self.lse.stride(0),
            self.lse.stride(1),
            self.lse.stride(2),
            WINDOW_SIZE=WINDOW,
            num_warps=4,
            num_stages=1,
        )
        _draft_window_stage2[(N_ROWS, N_HEADS, batch_size)](
            self.partial,
            self.lse,
            out,
            qo_indptr,
            self.partial.stride(0),
            self.partial.stride(1),
            self.partial.stride(2),
            self.partial.stride(3),
            self.lse.stride(0),
            self.lse.stride(1),
            self.lse.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            num_warps=4,
            num_stages=1,
        )
        return out
