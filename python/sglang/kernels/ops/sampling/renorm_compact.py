"""Compact top-p selection with a FlashInfer-derived HIP fallback."""

from pathlib import Path
from typing import Union

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.sampling.renorm_triton import (
    _prepare_probs,
    top_p_renorm_probs_triton,
)


@cache_once
def _fallback_module():
    from aiter.jit.core import AITER_CSRC_DIR

    csrc = Path(AITER_CSRC_DIR)
    return load_jit(
        "compact_top_p_hip",
        cuda_files=["sampling/compact_top_p.cuh"],
        cuda_wrappers=[("run", "sglang::compact_top_p_fallback")],
        extra_include_paths=[str(csrc / "cpp_itfs/sampling"), str(csrc / "include")],
    )


@triton.jit
def _count(
    P, TopP, Counts, Partial, Out, V: tl.constexpr, CH: tl.constexpr, B: tl.constexpr
):
    row, chunk = tl.program_id(0), tl.program_id(1)
    i = chunk * B + tl.arange(0, B)
    v = tl.load(P + row.to(tl.int64) * V + i, i < V, 0.0)
    tl.store(Out + row.to(tl.int64) * V + i, 0.0, i < V)
    cutoff = (1.0 - tl.load(TopP + row)).to(tl.float64) / V
    keep = (i < V) & (v.to(tl.float64) >= cutoff) & (v > 0.0)
    tl.store(Counts + row * CH + chunk, tl.sum(keep.to(tl.int32)))
    tl.store(
        Partial + (row * CH + chunk) * 2, tl.sum(tl.where(keep, 0.0, v).to(tl.float64))
    )
    tl.store(Partial + (row * CH + chunk) * 2 + 1, tl.sum(v.to(tl.float64)))


@triton.jit
def _prefix(
    Counts,
    Offsets,
    Count,
    Partial,
    TopP,
    CAP: tl.constexpr,
    CH: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BC)
    n = tl.load(Counts + row * CH + c, c < CH, 0)
    inclusive = tl.cumsum(n)
    tl.store(Offsets + row * CH + c, inclusive - n, c < CH)
    total_mass = tl.sum(tl.load(Partial + (row * CH + c) * 2 + 1, c < CH, 0.0))
    # The sort-based reference clamps a cutoff beyond total mass to the
    # largest value. Let the vendor path handle this uncommon case.
    fallback = (1.0 - tl.load(TopP + row)).to(tl.float64) >= total_mass
    tl.store(Count + row, tl.where(fallback, CAP + 1, tl.sum(n)))


@triton.jit
def _pack(
    P,
    TopP,
    Counts,
    Offsets,
    Count,
    Compact,
    Ids,
    V: tl.constexpr,
    CH: tl.constexpr,
    CAP: tl.constexpr,
    B: tl.constexpr,
):
    row, chunk = tl.program_id(0), tl.program_id(1)
    n = tl.load(Counts + row * CH + chunk)
    total = tl.load(Count + row)
    if (n > 0) & (total <= CAP):
        i = chunk * B + tl.arange(0, B)
        v = tl.load(P + row.to(tl.int64) * V + i, i < V, 0.0)
        cutoff = (1.0 - tl.load(TopP + row)).to(tl.float64) / V
        keep = (i < V) & (v.to(tl.float64) >= cutoff) & (v > 0.0)
        start = tl.load(Offsets + row * CH + chunk)
        dst = start + tl.cumsum(keep.to(tl.int32)) - 1
        tl.store(Compact + row * CAP + dst, v, keep & (dst < CAP))
        tl.store(Ids + row * CAP + dst, i, keep & (dst < CAP))


@triton.jit
def _pivot(
    TopP,
    Count,
    Compact,
    Ids,
    Partial,
    Out,
    V: tl.constexpr,
    CH: tl.constexpr,
    CAP: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.program_id(0)
    p = tl.load(TopP + row)
    n = tl.load(Count + row)
    if (p < 1.0) & (n <= CAP):
        chunks = tl.arange(0, BC)
        low = tl.sum(tl.load(Partial + (row * CH + chunks) * 2, chunks < CH, 0.0))
        j = tl.arange(0, CAP)
        original = tl.load(Compact + row * CAP + j, j < n, float("inf"))
        values = tl.sort(original, descending=False)
        cdf = tl.cumsum(tl.where(j < n, values, 0.0).to(tl.float64))
        pivot = tl.min(
            tl.where(
                (j < n) & (cdf >= (1.0 - p).to(tl.float64) - low), values, float("inf")
            )
        )
        norm = tl.sum(tl.where((j < n) & (values >= pivot), values, 0.0))
        ids = tl.load(Ids + row * CAP + j, j < n, 0)
        tl.store(
            Out + row.to(tl.int64) * V + ids,
            original / norm,
            (j < n) & (original >= pivot),
        )


def top_p_renorm_probs_compact(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Select an ascending-CDF pivot and retain all probability ties.

    Small vocabularies and layouts outside the HIP vector contract use the
    existing implementation. Scratch belongs to each invocation and remains
    valid for graph replay without persistent mutable operator state.
    """
    probs = _prepare_probs(probs)
    rows, vocab = probs.shape
    if not rows or not vocab:
        return probs
    if vocab % 4 or vocab < 8192:
        return top_p_renorm_probs_triton(probs, top_p)
    if isinstance(top_p, torch.Tensor):
        top_ps = top_p.to(device=probs.device, dtype=torch.float32).reshape(-1)
        if top_ps.numel() == 1:
            top_ps = top_ps.expand(rows)
        elif top_ps.numel() != rows:
            raise ValueError("top_p must be scalar or have one value per row")
        top_ps = top_ps.contiguous()
    else:
        if not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p values must be in (0, 1]")
        top_ps = torch.full(
            (rows,), float(top_p), dtype=torch.float32, device=probs.device
        )
    # Resolve the JIT before launching kernels; warmup populates this cache.
    module = _fallback_module()
    block, cap = 8192, 1024
    chunks = triton.cdiv(vocab, block)
    count = torch.empty(rows, device=probs.device, dtype=torch.int32)
    counts = torch.empty((rows, chunks), device=probs.device, dtype=torch.int32)
    offsets = torch.empty_like(counts)
    compact = torch.empty((rows, cap), device=probs.device, dtype=torch.float32)
    ids = torch.empty((rows, cap), device=probs.device, dtype=torch.int32)
    partial = torch.empty((rows, chunks, 2), device=probs.device, dtype=torch.float64)
    out = torch.empty_like(probs)
    _count[(rows, chunks)](
        probs, top_ps, counts, partial, out, vocab, chunks, block, num_warps=4
    )
    _prefix[(rows,)](
        counts,
        offsets,
        count,
        partial,
        top_ps,
        cap,
        chunks,
        triton.next_power_of_2(chunks),
        num_warps=4,
    )
    _pack[(rows, chunks)](
        probs,
        top_ps,
        counts,
        offsets,
        count,
        compact,
        ids,
        vocab,
        chunks,
        cap,
        block,
        num_warps=4,
    )
    _pivot[(rows,)](
        top_ps,
        count,
        compact,
        ids,
        partial,
        out,
        vocab,
        chunks,
        cap,
        triton.next_power_of_2(chunks),
        num_warps=4,
    )
    module.run(probs, out, top_ps, count, cap)
    return out
