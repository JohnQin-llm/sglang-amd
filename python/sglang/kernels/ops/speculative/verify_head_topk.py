"""Top-k of temperature-scaled verify logits, the same tokens and values as torch.topk.

``torch.topk(logits / t, k)`` copies the logits to divide them and then makes several radix
passes over the vocabulary. Here the division happens in registers and each row is read
twice, split into segments so that a small batch still fills the GPU:
  1. every segment keeps the largest value of each of its token groups (1024 disjoint groups
     per row); the k-th largest group maximum is a lower bound for the row's k-th value;
  2. every segment appends its tokens at or above that bound to the row's candidates
     (usually a little over k);
  3. rows with more than ``CAP`` candidates (heavy ties) find the exact k-th value by
     bisection and keep only the k winners;
  4. each row sorts its candidates and writes the first k.

Values compare in torch's radix order (-0.0 below +0.0) and ties at the k-th value keep the
lowest token ids, which is torch.topk's selection. The output is sorted by value, ties by
token id, so tied values can come out in a different order than torch.topk's, but the
selected tokens and the sorted values are the same.
"""

import torch
import triton
import triton.language as tl

_INT32_MIN = tl.constexpr(-(2**31))
_INT32_MAX = tl.constexpr(2**31 - 1)
_INT64_MIN = tl.constexpr(-(2**63))
_LOW_32 = tl.constexpr(2**32 - 1)

_GROUPS_PER_ROW = 1024
_BLOCK = 4096
# Candidate room per row: the k winners plus what the group bound lets through; arbitrary.
_MIN_CAP = 1024


@triton.jit
def _order_key(x):
    """An int32 whose signed order is torch's radix order of the fp32 ``x``."""
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _scaled_keys(row_ptr, offsets, temperature, VOCAB_SIZE: tl.constexpr):
    in_row = offsets < VOCAB_SIZE
    x = tl.load(row_ptr + offsets, mask=in_row, other=0.0)
    # The same fp32 division as torch's ``logits / temperature``.
    return tl.where(in_row, _order_key(x / temperature), _INT32_MIN)


@triton.jit
def _pack(keys, offsets):
    """Sort keys for "larger value first, then lower token id"."""
    return (keys.to(tl.int64) << 32) | (_LOW_32 - offsets.to(tl.int64))


@triton.jit
def _row_and_temperature(logits_ptr, stride_row, temperatures_ptr, rows_per_request):
    row = tl.program_id(0)
    row_ptr = logits_ptr + row.to(tl.int64) * stride_row
    temperature = tl.load(temperatures_ptr + row // rows_per_request).to(tl.float32)
    return row, row_ptr, temperature


@triton.jit
def _group_max_kernel(
    logits_ptr,
    stride_row,
    temperatures_ptr,
    rows_per_request,
    group_max_ptr,
    counts_ptr,
    VOCAB_SIZE: tl.constexpr,
    SEGMENT: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, row_ptr, temperature = _row_and_temperature(
        logits_ptr, stride_row, temperatures_ptr, rows_per_request
    )
    segment = tl.program_id(1)
    if segment == 0:
        tl.store(counts_ptr + row, 0)
    acc = tl.full((BLOCK,), _INT32_MIN, tl.int32)
    for start in range(segment * SEGMENT, (segment + 1) * SEGMENT, BLOCK):
        keys = _scaled_keys(
            row_ptr, start + tl.arange(0, BLOCK), temperature, VOCAB_SIZE
        )
        acc = tl.maximum(acc, keys)
    # Group g of this segment: its tokens at positions congruent to g modulo GROUPS.
    groups = tl.max(tl.reshape(acc, (BLOCK // GROUPS, GROUPS)), axis=0)
    num_groups = tl.num_programs(1) * GROUPS
    tl.store(
        group_max_ptr + row * num_groups + segment * GROUPS + tl.arange(0, GROUPS),
        groups,
    )


@triton.jit
def _bound_kernel(group_max_ptr, bounds_ptr, K: tl.constexpr, NUM_GROUPS: tl.constexpr):
    """The row's k-th largest group maximum, at most its k-th largest value."""
    row = tl.program_id(0)
    ranks = tl.arange(0, NUM_GROUPS)
    ranked = tl.sort(tl.load(group_max_ptr + row * NUM_GROUPS + ranks), descending=True)
    tl.store(
        bounds_ptr + row, tl.max(tl.where(ranks == K - 1, ranked, _INT32_MIN), axis=0)
    )


@triton.jit
def _collect_candidates_kernel(
    logits_ptr,
    stride_row,
    temperatures_ptr,
    rows_per_request,
    bounds_ptr,
    counts_ptr,
    candidates_ptr,
    VOCAB_SIZE: tl.constexpr,
    SEGMENT: tl.constexpr,
    BLOCK: tl.constexpr,
    CAP: tl.constexpr,
):
    row, row_ptr, temperature = _row_and_temperature(
        logits_ptr, stride_row, temperatures_ptr, rows_per_request
    )
    segment = tl.program_id(1)
    bound = tl.load(bounds_ptr + row)
    # Lane j of a tile collects its candidates from every tile of the segment and owns a
    # run of slots for them, so the tile loop stays elementwise (per-tile scans, atomics
    # or scattered stores cost 2-4 us per tile on MI355X). A lane keeps its first two
    # candidates in registers; only a lane with three or more needs the second pass.
    per_lane = tl.zeros((BLOCK,), tl.int32)
    first = tl.zeros((BLOCK,), tl.int64)
    second = tl.zeros((BLOCK,), tl.int64)
    for start in range(segment * SEGMENT, (segment + 1) * SEGMENT, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        keys = _scaled_keys(row_ptr, offsets, temperature, VOCAB_SIZE)
        take = keys >= bound
        packed = _pack(keys, offsets)
        first = tl.where(take & (per_lane == 0), packed, first)
        second = tl.where(take & (per_lane == 1), packed, second)
        per_lane += take.to(tl.int32)
    base = tl.atomic_add(counts_ptr + row, tl.sum(per_lane, axis=0))
    slot = base + tl.cumsum(per_lane, axis=0) - per_lane
    row_candidates = candidates_ptr + row.to(tl.int64) * CAP
    tl.store(row_candidates + slot, first, mask=(per_lane > 0) & (slot < CAP))
    tl.store(row_candidates + slot + 1, second, mask=(per_lane > 1) & (slot + 1 < CAP))
    if tl.max(per_lane, axis=0) > 2:
        seen = tl.zeros((BLOCK,), tl.int32)
        for start in range(segment * SEGMENT, (segment + 1) * SEGMENT, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            keys = _scaled_keys(row_ptr, offsets, temperature, VOCAB_SIZE)
            take = keys >= bound
            later = take & (seen >= 2)
            tl.store(
                row_candidates + slot + seen,
                _pack(keys, offsets),
                mask=later & (slot + seen < CAP),
            )
            seen += take.to(tl.int32)


@triton.jit
def _count_at_least(
    row_ptr, temperature, bound, VOCAB_SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    count = 0
    for start in range(0, VOCAB_SIZE, BLOCK):
        keys = _scaled_keys(
            row_ptr, start + tl.arange(0, BLOCK), temperature, VOCAB_SIZE
        )
        count += tl.sum((keys >= bound).to(tl.int32), axis=0)
    return count


@triton.jit
def _tied_rows_kernel(
    logits_ptr,
    stride_row,
    temperatures_ptr,
    rows_per_request,
    counts_ptr,
    candidates_ptr,
    K: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    CAP: tl.constexpr,
):
    row, row_ptr, temperature = _row_and_temperature(
        logits_ptr, stride_row, temperatures_ptr, rows_per_request
    )
    if tl.load(counts_ptr + row) > CAP:
        # The k-th key: the largest key with at least K tokens at or above it.
        lo = tl.full((), _INT32_MIN, tl.int64)
        hi = tl.full((), _INT32_MAX, tl.int64)
        for _ in range(32):
            mid = lo + (hi - lo + 1) // 2
            count = _count_at_least(
                row_ptr, temperature, mid.to(tl.int32), VOCAB_SIZE, BLOCK
            )
            lo = tl.where(count >= K, mid, lo)
            hi = tl.where(count >= K, hi, mid - 1)
        kth = lo.to(tl.int32)
        above = _count_at_least(row_ptr, temperature, kth + 1, VOCAB_SIZE, BLOCK)
        taken = 0
        ties = 0
        for start in range(0, VOCAB_SIZE, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            keys = _scaled_keys(row_ptr, offsets, temperature, VOCAB_SIZE)
            tied = keys == kth
            tie_rank = ties + tl.cumsum(tied.to(tl.int32), axis=0) - 1
            take = (keys > kth) | (tied & (tie_rank < K - above))
            slot = taken + tl.cumsum(take.to(tl.int32), axis=0) - 1
            tl.store(
                candidates_ptr + row.to(tl.int64) * CAP + slot,
                _pack(keys, offsets),
                mask=take,
            )
            taken += tl.sum(take.to(tl.int32), axis=0)
            ties += tl.sum(tied.to(tl.int32), axis=0)
        tl.store(counts_ptr + row, taken)


@triton.jit
def _select_kernel(
    counts_ptr,
    candidates_ptr,
    out_values_ptr,
    out_ids_ptr,
    K: tl.constexpr,
    CAP: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, CAP)
    count = tl.load(counts_ptr + row)
    candidates = tl.load(
        candidates_ptr + row.to(tl.int64) * CAP + slots,
        mask=slots < count,
        other=_INT64_MIN,
    )
    best = tl.sort(candidates, descending=True)
    keys = (best >> 32).to(tl.int32)
    bits = tl.where(keys < 0, keys ^ 0x7FFFFFFF, keys)
    out = slots < K
    tl.store(
        out_values_ptr + row * K + slots, bits.to(tl.float32, bitcast=True), mask=out
    )
    tl.store(out_ids_ptr + row * K + slots, _LOW_32 - (best & _LOW_32), mask=out)


def verify_head_top_k(
    *, logits: torch.Tensor, temperatures: torch.Tensor, rows_per_request: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.topk(logits / t, k)``'s values and ids for fp32 ``logits`` ``[rows, vocab]``,
    where row ``r`` takes ``temperatures[r // rows_per_request]``; see the module docstring
    for the order of tied values."""
    rows, vocab = logits.shape
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert 0 < k <= min(_GROUPS_PER_ROW, vocab)
    device = logits.device
    values = torch.empty((rows, k), dtype=torch.float32, device=device)
    ids = torch.empty((rows, k), dtype=torch.int64, device=device)
    if rows == 0:
        return values, ids
    cap = max(_MIN_CAP, triton.next_power_of_2(k))
    # About 512 programs per pass; segments are whole tiles. Measured on MI355X.
    segments = min(
        triton.next_power_of_2(triton.cdiv(512, rows)),
        triton.cdiv(vocab, _BLOCK),
        _GROUPS_PER_ROW // 64,
    )
    segments = 1 << (segments.bit_length() - 1)
    segment = triton.cdiv(triton.cdiv(vocab, segments), _BLOCK) * _BLOCK
    temperatures = temperatures.reshape(-1).to(torch.float32)
    group_max = torch.empty((rows, _GROUPS_PER_ROW), dtype=torch.int32, device=device)
    counts = torch.empty((rows,), dtype=torch.int32, device=device)
    candidates = torch.empty((rows, cap), dtype=torch.int64, device=device)
    row_args = (logits, logits.stride(0), temperatures, rows_per_request)
    _group_max_kernel[(rows, segments)](
        *row_args,
        group_max,
        counts,
        VOCAB_SIZE=vocab,
        SEGMENT=segment,
        GROUPS=_GROUPS_PER_ROW // segments,
        BLOCK=_BLOCK,
        num_warps=8,
    )
    bounds = torch.empty((rows,), dtype=torch.int32, device=device)
    _bound_kernel[(rows,)](
        group_max, bounds, K=k, NUM_GROUPS=_GROUPS_PER_ROW, num_warps=8
    )
    _collect_candidates_kernel[(rows, segments)](
        *row_args,
        bounds,
        counts,
        candidates,
        VOCAB_SIZE=vocab,
        SEGMENT=segment,
        BLOCK=_BLOCK,
        CAP=cap,
        num_warps=8,
    )
    _tied_rows_kernel[(rows,)](
        *row_args,
        counts,
        candidates,
        K=k,
        VOCAB_SIZE=vocab,
        BLOCK=_BLOCK,
        CAP=cap,
        num_warps=8,
    )
    _select_kernel[(rows,)](counts, candidates, values, ids, K=k, CAP=cap, num_warps=8)
    return values, ids
