"""DFlash candidate top-k over the vocab-sharded head in three launches (``SGLANG_MUSE_DFLASH_TRITON_TOPK``).

``candidate_topk`` under TP takes the rank-local top-K of the [N, V/TP] logits, gathers values and global ids
from every rank and keeps the global top-K. Served, that is torch's radix top-k (8 launches at N = 8, K = 16),
an FP32 value gather, an int64 id gather on NCCL, a second top-k and a gather. Here:

* ``local_topk_packed``: split top-K over column chunks, then a merge that writes one [N, 2K] FP32 row per token:
  the K values and the K global ids bit-cast from int32, the payload of a single custom FP32 all-gather;
* ``merge_gathered_topk``: the global top-K of the gathered [N, TP * 2K] rows, values and int64 ids.

Order: value descending, the smaller vocabulary id first among equal values.
"""

import torch
import triton
import triton.language as tl

CHUNK = 2048


@triton.jit
def _take_topk(vals, idx, K: tl.constexpr, BIG: tl.constexpr):
    # K rounds of (max value, smallest index holding it); returns them in order.
    out_v = tl.zeros([K], dtype=tl.float32)
    out_i = tl.zeros([K], dtype=tl.int32)
    slot = tl.arange(0, K)
    for r in tl.static_range(K):
        m = tl.max(vals, 0)
        i = tl.min(tl.where(vals == m, idx, BIG), 0)
        out_v = tl.where(slot == r, m, out_v)
        out_i = tl.where(slot == r, i, out_i)
        vals = tl.where(idx == i, float("-inf"), vals)
    return out_v, out_i


@triton.jit
def _split_topk_kernel(X, PV, PI, n_cols, x_row_stride, SPLITS: tl.constexpr, K: tl.constexpr,
                       CHUNK: tl.constexpr):
    row = tl.program_id(0)
    s = tl.program_id(1)
    c = s * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + row.to(tl.int64) * x_row_stride + c, mask=c < n_cols, other=float("-inf")).to(tl.float32)
    v, i = _take_topk(x, c, K, 2147483647)
    o = (row * SPLITS + s) * K + tl.arange(0, K)
    tl.store(PV + o, v)
    tl.store(PI + o, i)


@triton.jit
def _merge_splits_kernel(PV, PI, OUT, vocab_start, SPLITS: tl.constexpr, SPLITS_P2: tl.constexpr,
                         K: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, SPLITS_P2 * K)
    m = j < SPLITS * K
    v = tl.load(PV + row * SPLITS * K + j, mask=m, other=float("-inf"))
    i = tl.load(PI + row * SPLITS * K + j, mask=m, other=2147483647)
    tv, ti = _take_topk(v, i, K, 2147483647)
    k = tl.arange(0, K)
    tl.store(OUT + row * 2 * K + k, tv)
    tl.store(OUT.to(tl.pointer_type(tl.int32)) + row * 2 * K + K + k, ti + vocab_start)


def local_topk_packed(logits: torch.Tensor, k: int, vocab_start: int) -> torch.Tensor:
    """Rank-local top-``k`` of ``logits`` [N, V] as packed [N, 2k] FP32: values, then global ids as int32 bits."""
    n, v = logits.shape
    splits = triton.cdiv(v, CHUNK)
    pv = torch.empty((n, splits, k), dtype=torch.float32, device=logits.device)
    pi = torch.empty((n, splits, k), dtype=torch.int32, device=logits.device)
    _split_topk_kernel[(n, splits)](logits, pv, pi, v, logits.stride(0), splits, k, CHUNK, num_warps=4)
    out = torch.empty((n, 2 * k), dtype=torch.float32, device=logits.device)
    _merge_splits_kernel[(n,)](pv, pi, out, vocab_start, splits, triton.next_power_of_2(splits), k,
                               num_warps=4)
    return out


@triton.jit
def _merge_gathered_kernel(G, OUT_V, OUT_I, TP: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    # Position j = rank * K + slot: among equal values, a smaller j is a smaller global id.
    j = tl.arange(0, TP * K)
    col = (j // K) * 2 * K + (j % K)
    v = tl.load(G + row * TP * 2 * K + col)
    tv, tj = _take_topk(v, j, K, 2147483647)
    tid = tl.load(G.to(tl.pointer_type(tl.int32)) + row * TP * 2 * K + (tj // K) * 2 * K + (tj % K) + K)
    k = tl.arange(0, K)
    tl.store(OUT_V + row * K + k, tv)
    tl.store(OUT_I + row * K + k, tid.to(tl.int64))


def merge_gathered_topk(gathered: torch.Tensor, k: int):
    """Global top-``k`` of the all-gathered packed rows [N, TP * 2k]: (ids int64 [N, k], values FP32 [N, k])."""
    gathered = gathered.contiguous()
    n = gathered.shape[0]
    tp = gathered.shape[1] // (2 * k)
    vals = torch.empty((n, k), dtype=torch.float32, device=gathered.device)
    ids = torch.empty((n, k), dtype=torch.int64, device=gathered.device)
    _merge_gathered_kernel[(n,)](gathered, vals, ids, tp, k, num_warps=1)
    return ids, vals
