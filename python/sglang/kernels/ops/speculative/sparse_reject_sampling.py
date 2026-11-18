"""Chain verification (token-wise or block) over sparse target and draft supports, in Triton.

Same decisions and outputs as ``chain_speculative_sampling_triton``, but the target row of
each slot is its top-k support (token ids and probabilities, zero elsewhere) and the draft
row is the proposal's candidates and their q, so no dense ``[rows, vocab]`` probability
matrix is built and nothing loops over the vocabulary.

The dense kernel sums masses per vocabulary block (``BLOCK_V`` for the bonus draw,
``BLOCK_REDUCE`` for the block-verification residual) and adds the block sums in vocabulary
order. This kernel keeps that order: it walks the support in token order and closes a block
sum whenever the next token falls in a later block, so every sum with at most two support
tokens per block is bitwise the dense one. Three or more tokens in one block are added in
token order, where the dense ``tl.sum`` / ``tl.cumsum`` trees may round differently.
"""

import torch
import triton
import triton.language as tl

# A draw target no running sum of probabilities reaches: the walk then only sums.
_SUM_ONLY = tl.constexpr(2.0)


@triton.jit
def _row_support(ids_ptr, vals_ptr, row_offset, K: tl.constexpr, K_PAD: tl.constexpr):
    ranks = tl.arange(0, K_PAD)
    in_row = ranks < K
    ids = tl.load(ids_ptr + row_offset + ranks, mask=in_row, other=-1).to(tl.int64)
    vals = tl.load(vals_ptr + row_offset + ranks, mask=in_row, other=0.0)
    return ids, vals.to(tl.float32)


@triton.jit
def _lookup(ids, vals, token):
    """The dense row's entry at ``token``: its value if it is in the support, else 0."""
    return tl.sum(tl.where(ids == token, vals, 0.0), axis=0)


@triton.jit
def _lookup_many(ids, vals, tokens):
    """``_lookup`` for every entry of ``tokens``."""
    hit = tokens[:, None] == ids[None, :]
    return tl.sum(tl.where(hit, vals[None, :], 0.0), axis=1)


@triton.jit
def _walk_in_token_order(
    ids, vals, target_u, VOCAB_SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    """Sum ``vals`` the way the dense kernel does (per vocabulary block, blocks in order)
    and return the sum and the first token whose running sum exceeds ``target_u``
    (``VOCAB_SIZE - 1`` when none does, as in the dense kernel)."""
    positive = vals > 0.0
    # Only positive masses change a running sum; rank them by token id.
    order = tl.sum(
        (positive[None, :] & (ids[None, :] < ids[:, None])).to(tl.int32), axis=1
    )
    order = tl.where(positive, order, -1)
    num_positive = tl.sum(positive.to(tl.int32), axis=0)
    closed = tl.full((), 0.0, tl.float32)
    open_sum = tl.full((), 0.0, tl.float32)
    open_block = tl.full((), -1, tl.int64)
    token = tl.full((), VOCAB_SIZE - 1, tl.int64)
    found = tl.full((), 0, tl.int32)
    for rank in range(0, num_positive):
        at = order == rank
        tok = tl.sum(tl.where(at, ids, 0), axis=0)
        val = tl.sum(tl.where(at, vals, 0.0), axis=0)
        block = tok // BLOCK
        new_block = block != open_block
        closed = tl.where(new_block, closed + open_sum, closed)
        open_sum = tl.where(new_block, 0.0, open_sum)
        open_block = block
        open_sum = open_sum + val
        hit = (found == 0) & ((closed + open_sum) > target_u)
        token = tl.where(hit, tok, token)
        found = tl.where(hit, 1, found)
    return closed + open_sum, token


@triton.jit
def _sparse_chain_speculative_sampling_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetIds,
    TargetProbs,
    DraftIds,
    DraftProbs,
    stride_cand_b,
    stride_idx_b,
    stride_uni_b,
    stride_t_b,
    stride_t_s,
    stride_d_b,
    stride_d_s,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    KQ: tl.constexpr,
    KQ_PAD: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_REDUCE: tl.constexpr,
    BLOCK_VERIFICATION: tl.constexpr,
):
    pid = tl.program_id(0)
    cand_base = Candidates + pid * stride_cand_b
    idx_base = RetriveIndex + pid * stride_idx_b
    uni_base = UniformSamples + pid * stride_uni_b
    t_base = pid.to(tl.int64) * stride_t_b
    d_base = pid.to(tl.int64) * stride_d_b

    root_global_idx = tl.load(idx_base)
    tl.store(AcceptIndex + pid * stride_idx_b, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0
    residual_scale = 1.0
    norm_sum = -1.0

    if BLOCK_VERIFICATION:
        # Algorithm 2: https://arxiv.org/abs/2403.10444, as in the dense kernel.
        prefix_prob = 1.0
        prefix_probs = ()
        for step in tl.static_range(1, NUM_SLOTS):
            draft_token = tl.load(cand_base + step).to(tl.int64)
            t_ids, t_vals = _row_support(
                TargetIds, TargetProbs, t_base + (step - 1) * stride_t_s, K, K_PAD
            )
            d_ids, d_vals = _row_support(
                DraftIds, DraftProbs, d_base + (step - 1) * stride_d_s, KQ, KQ_PAD
            )
            p = _lookup(t_ids, t_vals, draft_token)
            q = _lookup(d_ids, d_vals, draft_token)
            q_is_prob = (q > 0.0) & (q <= 1.0)
            prefix_prob = tl.where(q_is_prob, tl.minimum(prefix_prob * p / q, 1.0), 0.0)
            prefix_probs += (prefix_prob,)

        for step in tl.static_range(NUM_SLOTS - 1, 0, -1):
            if num_accept == 0:
                prefix_prob = prefix_probs[step - 1]
                coin = tl.load(uni_base + (step - 1))
                if coin < prefix_prob:
                    accept_prob = prefix_prob
                    residual_mass = -1.0
                    if (
                        (step < NUM_SLOTS - 1)
                        & (prefix_prob > 0.0)
                        & (prefix_prob < 1.0)
                    ):
                        t_ids, t_vals = _row_support(
                            TargetIds, TargetProbs, t_base + step * stride_t_s, K, K_PAD
                        )
                        d_ids, d_vals = _row_support(
                            DraftIds, DraftProbs, d_base + step * stride_d_s, KQ, KQ_PAD
                        )
                        q_next = _lookup_many(d_ids, d_vals, t_ids)
                        residual = tl.maximum(prefix_prob * t_vals - q_next, 0.0)
                        residual_mass, _ = _walk_in_token_order(
                            t_ids, residual, _SUM_ONLY, VOCAB_SIZE, BLOCK_REDUCE
                        )
                        accept_prob = residual_mass / (
                            residual_mass + 1.0 - prefix_prob
                        )
                    if coin < accept_prob:
                        num_accept = step
                        residual_scale = prefix_prob
                        norm_sum = residual_mass

        for step in range(1, num_accept + 1):
            draft_token = tl.load(cand_base + step)
            tl.store(Predicts + last_accepted_global_idx, draft_token)
            last_accepted_global_idx = tl.load(idx_base + step)
            tl.store(AcceptIndex + pid * stride_idx_b + step, last_accepted_global_idx)
    else:
        step = 1
        continue_verifying = 1
        while (step < NUM_SLOTS) and (continue_verifying == 1):
            draft_token = tl.load(cand_base + step).to(tl.int64)
            t_ids, t_vals = _row_support(
                TargetIds, TargetProbs, t_base + (step - 1) * stride_t_s, K, K_PAD
            )
            d_ids, d_vals = _row_support(
                DraftIds, DraftProbs, d_base + (step - 1) * stride_d_s, KQ, KQ_PAD
            )
            p = _lookup(t_ids, t_vals, draft_token)
            q = _lookup(d_ids, d_vals, draft_token)
            coin = tl.load(uni_base + (step - 1))
            q_is_prob = (q > 0.0) & (q <= 1.0)
            if q_is_prob & (coin * q < p):
                num_accept += 1
                tl.store(Predicts + last_accepted_global_idx, draft_token)
                last_accepted_global_idx = tl.load(idx_base + step)
                tl.store(
                    AcceptIndex + pid * stride_idx_b + num_accept,
                    last_accepted_global_idx,
                )
                step += 1
            else:
                continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Bonus token from the slot after the last accepted draft.
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    coin_final = tl.load(UniformSamplesFinal + pid)
    t_ids, t_vals = _row_support(
        TargetIds, TargetProbs, t_base + num_accept * stride_t_s, K, K_PAD
    )
    if all_drafts_accepted:
        vals = t_vals
    else:
        d_ids, d_vals = _row_support(
            DraftIds, DraftProbs, d_base + num_accept * stride_d_s, KQ, KQ_PAD
        )
        q_vals = _lookup_many(d_ids, d_vals, t_ids)
        # The dense kernel's guard: a non-probability q counts as 0.
        q_vals = tl.where((q_vals >= 0.0) & (q_vals <= 1.0), q_vals, 0.0)
        diff = residual_scale * t_vals - q_vals
        vals = tl.where(diff > 0.0, diff, 0.0)
    if norm_sum < 0.0:
        norm_sum, _ = _walk_in_token_order(t_ids, vals, _SUM_ONLY, VOCAB_SIZE, BLOCK_V)
    _, final_token = _walk_in_token_order(
        t_ids, vals, coin_final * norm_sum, VOCAB_SIZE, BLOCK_V
    )
    tl.store(Predicts + last_accepted_global_idx, final_token)


def chain_speculative_sampling_sparse_triton(
    *,
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    vocab_size: int,
    block_verification: bool = False,
) -> None:
    """``chain_speculative_sampling_triton`` with ``target_probs`` given as each slot's
    support ``target_token_ids`` / ``target_probs`` ``[bs, num_slots, K]`` (unique ids,
    zero outside) and ``draft_probs`` as each draft slot's candidates
    ``draft_token_ids`` / ``draft_probs`` ``[bs, num_slots - 1, KQ]``."""
    batch_size, num_slots = candidates.shape
    if batch_size == 0:
        return
    k = target_token_ids.shape[-1]
    kq = draft_token_ids.shape[-1]
    target_token_ids = target_token_ids.contiguous()
    target_probs = target_probs.contiguous()
    draft_token_ids = draft_token_ids.contiguous()
    draft_probs = draft_probs.contiguous()
    candidates = candidates.contiguous()
    retrive_index = retrive_index.contiguous()
    uniform_samples = uniform_samples.contiguous()
    _sparse_chain_speculative_sampling_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_token_ids,
        target_probs,
        draft_token_ids,
        draft_probs,
        candidates.stride(0),
        retrive_index.stride(0),
        uniform_samples.stride(0),
        target_token_ids.stride(0),
        target_token_ids.stride(1),
        draft_token_ids.stride(0),
        draft_token_ids.stride(1),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        K=k,
        K_PAD=triton.next_power_of_2(k),
        KQ=kq,
        KQ_PAD=triton.next_power_of_2(kq),
        # The dense kernel's reduction blocks; see the module docstring.
        BLOCK_V=4096,
        BLOCK_REDUCE=min(triton.next_power_of_2(vocab_size), 32768),
        BLOCK_VERIFICATION=block_verification,
        num_warps=1,
    )
