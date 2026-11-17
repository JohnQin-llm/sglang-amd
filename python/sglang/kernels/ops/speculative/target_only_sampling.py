"""Target-only tree speculative sampling in Triton.

Same contract as the CUDA kernel in `kernels/jit/csrc/speculative/sampling.cuh`, for platforms
that do not build it (ROCm). Each level walks the sibling list of the last accepted node and
accepts a child when the coin falls inside the cumulative target mass of the children seen so
far; the bonus token is drawn from `relu(target - draft)`, where `draft` holds the mass of the
rejected children.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _tree_target_only_kernel(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    retrive_index_ptr,
    retrive_next_token_ptr,
    retrive_next_sibling_ptr,
    uniform_samples_ptr,
    uniform_samples_final_ptr,
    target_probs_ptr,
    draft_probs_ptr,
    stride_tp_b,
    stride_tp_n,
    stride_tp_v,
    stride_dp_b,
    stride_dp_n,
    stride_dp_v,
    threshold_single,
    threshold_acc,
    NUM_SPEC_TOKENS: tl.constexpr,
    NUM_DRAFT_TOKENS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * NUM_DRAFT_TOKENS
    tp_row = target_probs_ptr + pid.to(tl.int64) * stride_tp_b
    dp_row = draft_probs_ptr + pid.to(tl.int64) * stride_dp_b

    # prob_node is the tree node whose target distribution verifies the current level.
    prob_node = 0
    coin = tl.load(uniform_samples_ptr + row)
    last_accept_slot = tl.load(retrive_index_ptr + row).to(tl.int32)
    tl.store(accept_index_ptr + pid * NUM_SPEC_TOKENS, last_accept_slot)
    num_correct_drafts = 0
    prob_acc = 0.0
    node = 0
    level = 1
    walking = 1
    while (level < NUM_SPEC_TOKENS) & (walking == 1):
        node = tl.load(retrive_next_token_ptr + row + node).to(tl.int32)
        accept = 0
        while (node != -1) & (accept == 0):
            draft_token = tl.load(candidates_ptr + row + node)
            tp_offset = prob_node.to(tl.int64) * stride_tp_n + draft_token * stride_tp_v
            target_prob = tl.load(tp_row + tp_offset)
            prob_acc += target_prob
            if (target_prob > 0.0) & (
                (coin < prob_acc / threshold_acc) | (target_prob >= threshold_single)
            ):
                prob_acc = 0.0
                prob_node = node
                coin = tl.load(uniform_samples_ptr + row + node)
                tl.store(predicts_ptr + last_accept_slot, draft_token.to(tl.int32))
                num_correct_drafts += 1
                last_accept_slot = tl.load(retrive_index_ptr + row + node).to(tl.int32)
                tl.store(
                    accept_index_ptr + pid * NUM_SPEC_TOKENS + num_correct_drafts,
                    last_accept_slot,
                )
                accept = 1
            else:
                dp_offset = (
                    prob_node.to(tl.int64) * stride_dp_n + draft_token * stride_dp_v
                )
                tl.store(dp_row + dp_offset, target_prob)
                node = tl.load(retrive_next_sibling_ptr + row + node).to(tl.int32)
        # A level with no accepted child ends the walk.
        walking = accept
        level += 1
    tl.store(accept_token_num_ptr + pid, num_correct_drafts)

    # The bonus row has no draft distribution, so it samples the target directly.
    has_draft_mass = num_correct_drafts != NUM_SPEC_TOKENS - 1
    tp_base = tp_row + prob_node.to(tl.int64) * stride_tp_n
    dp_base = dp_row + prob_node.to(tl.int64) * stride_dp_n

    total = 0.0
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        offsets = v_start + tl.arange(0, BLOCK_V)
        mask = offsets < VOCAB_SIZE
        q = tl.load(tp_base + offsets * stride_tp_v, mask=mask, other=0.0)
        p = tl.load(
            dp_base + offsets * stride_dp_v, mask=mask & has_draft_mass, other=0.0
        )
        total += tl.sum(tl.maximum(q - p, 0.0))

    u = tl.load(uniform_samples_final_ptr + pid) * total
    cum = 0.0
    sampled = VOCAB_SIZE
    last_valid = -1
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if sampled == VOCAB_SIZE:
            offsets = v_start + tl.arange(0, BLOCK_V)
            mask = offsets < VOCAB_SIZE
            q = tl.load(tp_base + offsets * stride_tp_v, mask=mask, other=0.0)
            p = tl.load(
                dp_base + offsets * stride_dp_v, mask=mask & has_draft_mass, other=0.0
            )
            residual = tl.maximum(q - p, 0.0)
            hit = ((cum + tl.cumsum(residual, axis=0)) > u) & (residual > 0.0)
            if tl.max(hit.to(tl.int32), axis=0) > 0:
                sampled = v_start + tl.argmax(hit.to(tl.int32), axis=0)
            last_valid = tl.maximum(
                last_valid, tl.max(tl.where(residual > 0.0, offsets, -1), axis=0)
            )
            cum += tl.sum(residual)
    # u can exceed the summed mass by rounding; fall back like the CUDA kernel.
    if sampled == VOCAB_SIZE:
        sampled = tl.where(last_valid == -1, VOCAB_SIZE - 1, last_valid)
    tl.store(predicts_ptr + last_accept_slot, sampled.to(tl.int32))


def tree_speculative_sampling_target_only_triton(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,  # the scans here are sequential per request
) -> None:
    batch_size, num_draft_tokens = candidates.shape
    if batch_size == 0:
        return
    for name, tensor in (
        ("candidates", candidates),
        ("retrive_index", retrive_index),
        ("retrive_next_token", retrive_next_token),
        ("retrive_next_sibling", retrive_next_sibling),
        ("uniform_samples", uniform_samples),
        ("accept_index", accept_index),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    target_probs = target_probs.view(batch_size, num_draft_tokens, -1)
    draft_probs = draft_probs.view(batch_size, num_draft_tokens, -1)
    _tree_target_only_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        float(threshold_single),
        max(float(threshold_acc), 1e-9),
        NUM_SPEC_TOKENS=accept_index.shape[1],
        NUM_DRAFT_TOKENS=num_draft_tokens,
        VOCAB_SIZE=target_probs.shape[-1],
        BLOCK_V=4096,
        num_warps=8,
    )
