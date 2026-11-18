from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _dflash_accept_bonus_contig_kernel(
    candidates_ptr,
    target_top1_ptr,
    accept_lens_out_ptr,
    commit_lens_out_ptr,
    bonus_ids_out_ptr,
    out_tokens_ptr,
    prefix_lens_ptr,
    new_seq_lens_out_ptr,
    candidates_row_stride,
    target_row_stride,
    accept_stride,
    commit_stride,
    bonus_stride,
    out_tokens_row_stride,
    prefix_lens_stride,
    new_seq_lens_stride,
    block_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size
    draft_mask = cols < (block_size - 1)

    candidate_row_ptr = candidates_ptr + row * candidates_row_stride
    target_row_ptr = target_top1_ptr + row * target_row_stride
    candidate_tail = tl.load(candidate_row_ptr + cols + 1, mask=draft_mask, other=0)

    accept_len = tl.full((), 0, tl.int32)
    prefix_live = tl.full((), 1, tl.int32)
    for col in range(BLOCK_SIZE - 1):
        in_range = col < (block_size - 1)
        candidate_id = tl.load(candidate_row_ptr + (col + 1), mask=in_range, other=0)
        target_id = tl.load(target_row_ptr + col, mask=in_range, other=0)
        match_i32 = (candidate_id == target_id).to(tl.int32)
        keep = in_range & (prefix_live != 0) & (match_i32 != 0)
        accept_len += keep.to(tl.int32)
        prefix_live = tl.where(in_range, prefix_live & match_i32, prefix_live)

    commit_len = accept_len + 1
    bonus_id = tl.load(target_row_ptr + accept_len.to(tl.int64))
    new_seq_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride) + commit_len

    tl.store(accept_lens_out_ptr + row * accept_stride, accept_len)
    tl.store(commit_lens_out_ptr + row * commit_stride, commit_len)
    tl.store(bonus_ids_out_ptr + row * bonus_stride, bonus_id)
    tl.store(new_seq_lens_out_ptr + row * new_seq_lens_stride, new_seq_len)

    out_val = tl.where(draft_mask, candidate_tail, 0)
    out_val = tl.where(cols == accept_len, bonus_id, out_val)
    tl.store(
        out_tokens_ptr + row * out_tokens_row_stride + cols, out_val, mask=row_mask
    )


def _pick_num_warps(block_size: int) -> int:
    if block_size <= 16:
        return 1
    if block_size <= 32:
        return 2
    if block_size <= 64:
        return 4
    return 8


def _is_row_major_contiguous_2d(x: torch.Tensor) -> bool:
    return x.ndim == 2 and x.is_contiguous()


def _compute_dflash_accept_bonus_triton_unchecked(
    candidates: torch.Tensor,
    target_top1: torch.Tensor,
    accept_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    bonus_ids_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    prefix_lens: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
) -> None:
    batch_size, block_size = candidates.shape
    if batch_size == 0:
        return

    if not _is_row_major_contiguous_2d(candidates):
        raise ValueError("DFLASH Triton accept_bonus requires contiguous candidates.")
    if not _is_row_major_contiguous_2d(target_top1):
        raise ValueError("DFLASH Triton accept_bonus requires contiguous target_top1.")
    if not _is_row_major_contiguous_2d(out_tokens_out):
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous out_tokens_out."
        )
    if not accept_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous accept_lens_out."
        )
    if not commit_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous commit_lens_out."
        )
    if not bonus_ids_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous bonus_ids_out."
        )
    if prefix_lens.ndim != 1:
        raise ValueError("DFLASH Triton accept_bonus requires 1D prefix_lens.")
    if not new_seq_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous new_seq_lens_out."
        )

    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _dflash_accept_bonus_contig_kernel[(batch_size,)](
        candidates,
        target_top1,
        accept_lens_out,
        commit_lens_out,
        bonus_ids_out,
        out_tokens_out,
        prefix_lens,
        new_seq_lens_out,
        candidates.stride(0),
        target_top1.stride(0),
        accept_lens_out.stride(0),
        commit_lens_out.stride(0),
        bonus_ids_out.stride(0),
        out_tokens_out.stride(0),
        prefix_lens.stride(0),
        new_seq_lens_out.stride(0),
        block_size,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit
def _prepare_dflash_draft_block_contig_kernel(
    bonus_tokens_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    block_ids_out_ptr,
    positions_out_ptr,
    cache_loc_out_ptr,
    bonus_tokens_stride,
    prefix_lens_stride,
    req_pool_indices_stride,
    req_to_token_row_stride,
    block_ids_row_stride,
    positions_row_stride,
    cache_loc_row_stride,
    req_to_token_width,
    block_size,
    mask_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size

    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride)
    bonus_token = tl.load(bonus_tokens_ptr + row * bonus_tokens_stride)

    logical_pos = prefix_len.to(tl.int64) + cols
    valid = row_mask & (logical_pos < req_to_token_width)
    req_row_ptr = req_to_token_ptr + req_idx * req_to_token_row_stride
    slot_ids = tl.load(req_row_ptr + logical_pos, mask=valid, other=0)

    block_ids = tl.full((BLOCK_SIZE,), mask_token_id, tl.int64)
    block_ids = tl.where(cols == 0, bonus_token.to(tl.int64), block_ids)
    tl.store(
        block_ids_out_ptr + row * block_ids_row_stride + cols, block_ids, mask=row_mask
    )
    tl.store(
        positions_out_ptr + row * positions_row_stride + cols,
        logical_pos,
        mask=row_mask,
    )
    tl.store(
        cache_loc_out_ptr + row * cache_loc_row_stride + cols,
        slot_ids.to(tl.int64),
        mask=row_mask,
    )


def _prepare_dflash_draft_block_unchecked(
    bonus_tokens: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
    mask_token_id: int,
) -> None:
    batch_size = int(bonus_tokens.numel())
    if batch_size == 0:
        return

    if req_to_token.ndim != 2 or req_to_token.stride(1) != 1:
        raise ValueError("DFLASH Triton prepare_block requires row-major req_to_token.")
    if not _is_row_major_contiguous_2d(block_ids_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous block_ids_out."
        )
    if not _is_row_major_contiguous_2d(positions_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous positions_out."
        )
    if not _is_row_major_contiguous_2d(cache_loc_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous cache_loc_out."
        )

    block_size = int(block_ids_out.shape[1])
    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _prepare_dflash_draft_block_contig_kernel[(batch_size,)](
        bonus_tokens,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        block_ids_out,
        positions_out,
        cache_loc_out,
        bonus_tokens.stride(0),
        prefix_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        block_ids_out.stride(0),
        positions_out.stride(0),
        cache_loc_out.stride(0),
        int(req_to_token.shape[1]),
        block_size,
        int(mask_token_id),
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    uniforms_ptr,
    temperatures_ptr,
    greedy_ptr,
    tokens_ptr,
    q_ptr,
    slots: tl.constexpr,
    top_k: tl.constexpr,
):
    """One program per request: a slot's K scores stay in registers and the walk is a
    loop, so the slot-to-slot dependency costs nothing instead of one kernel each."""
    row = tl.program_id(0)
    offsets = tl.arange(0, top_k)
    temperature = tl.load(temperatures_ptr + row)
    greedy = tl.load(greedy_ptr + row) != 0
    previous = 0
    for slot in range(slots):
        base = (row * slots + slot) * top_k
        scores = tl.load(scores_ptr + (base + previous) * top_k + offsets).to(
            tl.float32
        )
        if greedy:
            # Same pick as torch.argmax: the first NaN if any, else the first max.
            is_nan = scores != scores
            best = tl.max(tl.where(is_nan, -float("inf"), scores), axis=0)
            has_nan = tl.max(is_nan.to(tl.int32), axis=0) > 0
            hit = tl.where(has_nan, is_nan, scores == best)
            index = tl.min(tl.where(hit, offsets, top_k), axis=0)
            probabilities = tl.where(offsets == index, 1.0, 0.0)
        else:
            scaled = scores / temperature
            exponentials = tl.exp(scaled - tl.max(scaled, axis=0))
            probabilities = exponentials / tl.sum(exponentials, axis=0)
            uniform = tl.load(uniforms_ptr + row * slots + slot)
            index = tl.sum(
                tl.where(uniform >= tl.cumsum(probabilities, axis=0), 1, 0), axis=0
            )
            index = tl.minimum(index, top_k - 1)
        tl.store(q_ptr + base + offsets, probabilities)
        tl.store(tokens_ptr + row * slots + slot, tl.load(candidate_ptr + base + index))
        previous = index


def selector_walk_triton(
    *,
    candidate_ids,
    scores,
    uniforms,
    temperatures,
    greedy_mask,
):
    batch, slots, top_k = candidate_ids.shape
    tokens = torch.empty((batch, slots), dtype=torch.int64, device=scores.device)
    q_rows = torch.empty(
        (batch, slots, top_k), dtype=torch.float32, device=scores.device
    )
    _selector_walk_kernel[(batch,)](
        scores.contiguous(),
        candidate_ids.contiguous(),
        uniforms.contiguous(),
        temperatures.contiguous(),
        greedy_mask.contiguous(),
        tokens,
        q_rows,
        slots=slots,
        top_k=top_k,
        num_warps=1,
    )
    return tokens, q_rows


@triton.jit
def _selector_collision_mass_kernel(
    scores_ptr,
    temperatures_ptr,
    out_ptr,
    slots: tl.constexpr,
    top_k: tl.constexpr,
):
    """One program per request: the sum over depths of the chance that two
    independent selector walks agree on the whole path to that depth."""
    row = tl.program_id(0)
    previous = tl.arange(0, top_k)
    candidate = tl.arange(0, top_k)
    temperature = tl.load(temperatures_ptr + row)
    base = scores_ptr + row * slots * top_k * top_k
    # Slot 0 follows the anchor, the lattice's row 0.
    first = tl.load(base + candidate).to(tl.float32) / temperature
    first = tl.exp(first - tl.max(first, axis=0))
    first = first / tl.sum(first, axis=0)
    agree = first * first
    mass = tl.sum(agree, axis=0)
    for slot in range(1, slots):
        scaled = (
            tl.load(
                base
                + slot * top_k * top_k
                + previous[:, None] * top_k
                + candidate[None, :]
            ).to(tl.float32)
            / temperature
        )
        exponentials = tl.exp(scaled - tl.max(scaled, axis=1)[:, None])
        probabilities = exponentials / tl.sum(exponentials, axis=1)[:, None]
        agree = tl.sum(agree[:, None] * probabilities * probabilities, axis=0)
        mass += tl.sum(agree, axis=0)
    tl.store(out_ptr + row, mass)


def selector_collision_mass(*, scores, temperatures, out=None):
    """Self-collision path mass of a selector lattice ``scores`` ([bs, slots, K, K]:
    slot, previous candidate, candidate) at ``temperatures`` ([bs]): sum over
    depths d of sum over paths of q(path to d)^2, from the lattice alone."""
    batch, slots, top_k, _ = scores.shape
    if out is None:
        out = torch.empty((batch,), dtype=torch.float32, device=scores.device)
    if not scores.is_cuda:
        q = torch.softmax(scores.float() / temperatures.view(-1, 1, 1, 1), dim=-1)
        squares = q * q
        agree = squares[:, 0, 0]
        mass = agree.sum(-1)
        for slot in range(1, slots):
            agree = torch.einsum("bp,bpc->bc", agree, squares[:, slot])
            mass = mass + agree.sum(-1)
        return out.copy_(mass)
    _selector_collision_mass_kernel[(batch,)](
        scores.contiguous(),
        temperatures.contiguous(),
        out,
        slots=slots,
        top_k=top_k,
        num_warps=1,
    )
    return out


@triton.jit
def _ragged_chain_rows_kernel(
    mass_ptr,
    edges_ptr,
    gains_ptr,
    draft_tokens_ptr,
    positions_ptr,
    verify_lens_ptr,
    qo_indptr_ptr,
    token_reqs_ptr,
    block_index_ptr,
    packed_tokens_ptr,
    packed_positions_ptr,
    bs,
    num_extra,
    num_rows,
    NUM_EDGES: tl.constexpr,
    GAINS: tl.constexpr,
    BLOCK: tl.constexpr,
    BS_PAD: tl.constexpr,
    EDGES_PAD: tl.constexpr,
    GAINS_PAD: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    ROW_CHUNK: tl.constexpr,
):
    """One program: each request verifies 1 + its share of the batch's num_extra
    largest marginal gains (ties to the earlier request), then the packed rows."""
    req = tl.arange(0, BS_PAD)
    req_ok = req < bs
    edge = tl.arange(0, EDGES_PAD)
    mass = tl.load(mass_ptr + req, mask=req_ok, other=0.0)
    edges = tl.load(edges_ptr + edge, mask=edge < NUM_EDGES, other=float("inf"))
    # torch.bucketize(mass, edges): the number of edges below the mass.
    bins = tl.sum((edges[None, :] < mass[:, None]).to(tl.int32), axis=1)
    col = tl.arange(0, GAINS_PAD)
    # Gains are >= 0, so the -1 padding is never among the num_extra largest.
    gains = tl.load(
        gains_ptr + bins[:, None] * GAINS + col[None, :],
        mask=req_ok[:, None] & (col[None, :] < GAINS),
        other=-1.0,
    )
    ranked = tl.sort(tl.reshape(gains, (BS_PAD * GAINS_PAD,)), descending=True)
    rank = tl.arange(0, BS_PAD * GAINS_PAD)
    threshold = tl.max(tl.where(rank == num_extra - 1, ranked, -2.0), axis=0)
    threshold = tl.where(num_extra > 0, threshold, float("inf"))
    above = tl.sum((gains > threshold).to(tl.int32), axis=1)
    ties = tl.sum((gains == threshold).to(tl.int32), axis=1)
    left = num_extra - tl.sum(above, axis=0)
    take = tl.minimum(tl.maximum(left - (tl.cumsum(ties, axis=0) - ties), 0), ties)
    verify_lens = tl.where(req_ok, 1 + above + take, 0)
    ends = tl.cumsum(verify_lens, axis=0)
    starts = ends - verify_lens
    tl.store(verify_lens_ptr + req, verify_lens, mask=req_ok)
    tl.store(qo_indptr_ptr + 1 + req, ends, mask=req_ok)
    tl.store(qo_indptr_ptr + req, starts, mask=req == 0)
    for chunk in tl.static_range(0, ROWS_PAD, ROW_CHUNK):
        row = chunk + tl.arange(0, ROW_CHUNK)
        row_ok = row < num_rows
        ended = (ends[None, :] <= row[:, None]) & req_ok[None, :]
        owner = tl.sum(ended.to(tl.int32), axis=1)
        owner_start = tl.sum(
            tl.where(req[None, :] == owner[:, None], starts[None, :], 0), axis=1
        )
        index = owner.to(tl.int64) * BLOCK + (row - owner_start)
        tl.store(token_reqs_ptr + row, owner, mask=row_ok)
        tl.store(block_index_ptr + row, index, mask=row_ok)
        tl.store(
            packed_tokens_ptr + row,
            tl.load(draft_tokens_ptr + index, mask=row_ok, other=0),
            mask=row_ok,
        )
        tl.store(
            packed_positions_ptr + row,
            tl.load(positions_ptr + index, mask=row_ok, other=0),
            mask=row_ok,
        )


def ragged_chain_rows(
    *,
    collision_mass: torch.Tensor,
    bin_edges: torch.Tensor,
    gains: torch.Tensor,
    rows_per_request: int,
    draft_tokens: torch.Tensor,
    positions: torch.Tensor,
):
    """Each request's verify rows from its collision-mass bin's marginal accept
    lengths (``gains`` [bins, block - 1], non-increasing), summing to
    rows_per_request * bs, and the packed rows: (verify_lens [bs] int32,
    qo_indptr [bs + 1] int32, token_reqs [rows] int32, block_index [rows] int64
    into the [bs, block] ``draft_tokens`` / ``positions``, packed tokens, packed
    positions)."""
    bs, block = draft_tokens.shape
    device = draft_tokens.device
    num_rows = rows_per_request * bs
    verify_lens = torch.empty(bs, dtype=torch.int32, device=device)
    qo_indptr = torch.empty(bs + 1, dtype=torch.int32, device=device)
    token_reqs = torch.empty(num_rows, dtype=torch.int32, device=device)
    block_index = torch.empty(num_rows, dtype=torch.int64, device=device)
    packed_tokens = torch.empty(num_rows, dtype=draft_tokens.dtype, device=device)
    packed_positions = torch.empty(num_rows, dtype=positions.dtype, device=device)
    rows_pad = triton.next_power_of_2(num_rows)
    _ragged_chain_rows_kernel[(1,)](
        collision_mass.contiguous(),
        bin_edges,
        gains,
        draft_tokens.contiguous(),
        positions.contiguous(),
        verify_lens,
        qo_indptr,
        token_reqs,
        block_index,
        packed_tokens,
        packed_positions,
        bs,
        (rows_per_request - 1) * bs,
        num_rows,
        NUM_EDGES=bin_edges.numel(),
        GAINS=block - 1,
        BLOCK=block,
        BS_PAD=triton.next_power_of_2(bs),
        EDGES_PAD=triton.next_power_of_2(bin_edges.numel()),
        GAINS_PAD=triton.next_power_of_2(block - 1),
        ROWS_PAD=rows_pad,
        ROW_CHUNK=min(rows_pad, 128),
        num_warps=4,
    )
    return (
        verify_lens,
        qo_indptr,
        token_reqs,
        block_index,
        packed_tokens,
        packed_positions,
    )


@triton.jit
def _pad_chain_rows_kernel(
    live_qo_indptr_ptr,
    live_token_reqs_ptr,
    qo_indptr_ptr,
    token_reqs_ptr,
    live_bs,
    bs,
    rows_per_request,
    BS_PAD: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    ROW_CHUNK: tl.constexpr,
):
    req = tl.arange(0, BS_PAD)
    live_rows = live_bs * rows_per_request
    is_live = req <= live_bs
    qo = tl.load(live_qo_indptr_ptr + req, mask=is_live & (live_bs > 0), other=0)
    qo = tl.where(is_live, qo, live_rows + (req - live_bs) * rows_per_request)
    tl.store(qo_indptr_ptr + req, qo, mask=req <= bs)
    for chunk in tl.static_range(0, ROWS_PAD, ROW_CHUNK):
        row = chunk + tl.arange(0, ROW_CHUNK)
        live = row < live_rows
        owner = tl.load(live_token_reqs_ptr + row, mask=live, other=0)
        owner = tl.where(live, owner, live_bs + (row - live_rows) // rows_per_request)
        tl.store(token_reqs_ptr + row, owner, mask=row < bs * rows_per_request)


def pad_chain_rows(
    *,
    live_qo_indptr,
    live_token_reqs,
    live_bs: int,
    bs: int,
    rows_per_request: int,
    qo_indptr_out,
    token_reqs_out,
):
    """The first live_bs requests' ragged rows, then rows_per_request rows for each
    of the bs - live_bs padding requests, into [bs + 1] / [bs * rows] buffers."""
    rows_pad = triton.next_power_of_2(max(1, bs * rows_per_request))
    _pad_chain_rows_kernel[(1,)](
        live_qo_indptr,
        live_token_reqs,
        qo_indptr_out,
        token_reqs_out,
        live_bs,
        bs,
        rows_per_request,
        BS_PAD=triton.next_power_of_2(bs + 1),
        ROWS_PAD=rows_pad,
        ROW_CHUNK=min(rows_pad, 256),
        num_warps=4,
    )


@triton.jit
def _fused_correct_drafts_and_bonus_kernel(
    candidates_ptr,
    target_predict_ptr,
    num_correct_drafts_ptr,
    bonus_tokens_ptr,
    block_size,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    in_row = offs < block_size - 1
    drafts = tl.load(candidates_ptr + b * block_size + 1 + offs, mask=in_row, other=-1)
    targets = tl.load(target_predict_ptr + b * block_size + offs, mask=in_row, other=-2)
    eq = (drafts == targets) & in_row
    # Leading-match count = index of the first mismatch lane; lanes past the
    # row and all-match rows both resolve to block_size - 1 via the min.
    num_correct = tl.min(tl.where(eq, BLOCK, offs), 0)
    bonus_token = tl.load(target_predict_ptr + b * block_size + num_correct)
    tl.store(num_correct_drafts_ptr + b, num_correct.to(tl.int32))
    tl.store(bonus_tokens_ptr + b, bonus_token.to(tl.int64))


def compute_dflash_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.

    Returns:
        correct_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index correct_len (the "bonus" token to append).

    Notes:
        Matches the reference implementation rule:
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    if candidates.is_cuda:
        num_correct_drafts = torch.empty(
            bs, dtype=torch.int32, device=candidates.device
        )
        bonus_tokens = torch.empty(bs, dtype=torch.int64, device=candidates.device)
        _fused_correct_drafts_and_bonus_kernel[(bs,)](
            candidates.contiguous(),
            target_predict.contiguous(),
            num_correct_drafts,
            bonus_tokens,
            block_size,
            BLOCK=triton.next_power_of_2(max(block_size - 1, 1)),
        )
        return num_correct_drafts, bonus_tokens

    matches = candidates[:, 1:] == target_predict[:, :-1]
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), correct_len]
    return correct_len.to(torch.int32), bonus.to(torch.int64)
