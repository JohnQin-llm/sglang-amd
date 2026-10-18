"""Stable local-expert alignment for small Muse verification batches."""

import torch
import triton
import triton.language as tl


@triton.jit
def _align_local(
    Ids,
    Sorted,
    Experts,
    Total,
    N: tl.constexpr,
    START: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BLOCK: tl.constexpr,
    BINS: tl.constexpr,
    NB: tl.constexpr,
):
    expert = tl.program_id(0)
    slots = tl.arange(0, BLOCK)
    ids = tl.load(Ids + slots, slots < N, other=-1) - START
    valid = (slots < N) & (ids >= 0) & (ids < E)
    histogram = tl.histogram(tl.where(valid, ids, E), BINS)
    bins = tl.arange(0, BINS)
    blocks = (histogram + BM - 1) // BM
    first_block = tl.sum(tl.where(bins < expert, blocks, 0), 0)
    total_blocks = tl.sum(tl.where(bins < E, blocks, 0), 0)
    count = tl.sum(tl.where(bins == expert, histogram, 0), 0)
    hit = valid & (ids == expert)
    position = tl.cumsum(hit.to(tl.int32), 0) - 1
    tl.store(Sorted + first_block * BM + position, slots, hit)
    # Padding and occupied slots are disjoint, so no initialization barrier.
    tl.store(
        Sorted + first_block * BM + slots,
        N,
        (slots >= count) & (slots < tl.cdiv(count, BM) * BM),
    )
    block_ids = tl.arange(0, NB)
    tl.store(
        Experts + block_ids,
        expert,
        (block_ids >= first_block) & (block_ids < first_block + tl.cdiv(count, BM)),
    )
    if expert == 0:
        tl.store(Total, total_blocks * BM)
        tl.store(
            Experts + block_ids,
            -1,
            (block_ids >= total_blocks) & (block_ids < tl.cdiv(N + E * (BM - 1), BM)),
        )


def align_local(
    ids: torch.Tensor, experts: int, start: int, block_m: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group local route slots by expert and pad each group to ``block_m``.

    Nonlocal routes are omitted. Padding slots contain ``ids.numel()``; unused
    expert blocks contain -1. The total padded length stays on the device so
    the same graph can replay with different routing, including empty ranks.
    """
    assert ids.is_contiguous() and ids.dtype == torch.int32
    n = ids.numel()
    assert block_m <= n <= 2048 and experts == 32 and block_m in (16, 32)
    capacity = n + experts * (block_m - 1)
    blocks = triton.cdiv(capacity, block_m)
    sorted_ids = torch.empty(capacity, device=ids.device, dtype=torch.int32)
    expert_ids = torch.empty(blocks, device=ids.device, dtype=torch.int32)
    total = torch.empty(1, device=ids.device, dtype=torch.int32)
    _align_local[(experts,)](
        ids,
        sorted_ids,
        expert_ids,
        total,
        n,
        start,
        experts,
        block_m,
        triton.next_power_of_2(n),
        triton.next_power_of_2(experts + 1),
        triton.next_power_of_2(blocks),
        num_warps=4,
    )
    return sorted_ids, expert_ids, total
