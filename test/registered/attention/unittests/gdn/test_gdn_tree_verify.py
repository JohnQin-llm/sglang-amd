"""GDN tree verify without per-node snapshots, and its accepted-path commit fold.

With an fp32 state pool both must be bitwise those of the snapshot tree verify
(``fused_sigmoid_gating_delta_rule_update`` with ``intermediate_states_buffer``)
and its accepted-node scatter, on trees from the DFLASH builder at Qwen3.8 shapes.
"""

import unittest

import torch

from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_fold import (
    commit_gdn_replayssm_fold_all_layers,
)
from sglang.kernels.ops.attention.linear.gdn_tree_verify import gdn_tree_verify
from sglang.kernels.ops.speculative.dflash_tree import build_dflash_tree_proposal
from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-large-amd")

H, HV, K, V = 16, 48, 128, 128
DEPTH, TOPK = 7, 16
DEVICE = "cuda"


def _random_trees(batch_size, num_nodes, seed, temperature):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    candidate_ids = torch.randint(
        0, 1000, (batch_size, DEPTH, TOPK), device=DEVICE, generator=gen
    )
    logits = (
        torch.randn(batch_size, DEPTH, TOPK, TOPK, device=DEVICE, generator=gen)
        / temperature
    )
    proposal = build_dflash_tree_proposal(
        candidate_ids, torch.log_softmax(logits, dim=-1), max_nodes=num_nodes
    )
    seq_lens = torch.full((batch_size,), 37, dtype=torch.long, device=DEVICE)
    outputs = build_tree_kernel_efficient(
        torch.zeros(batch_size, dtype=torch.long, device=DEVICE),
        proposal.parent_list,
        proposal.top_scores_index,
        proposal.draft_tokens,
        seq_lens,
        int(seq_lens.sum()),
        TOPK,
        DEPTH,
        num_nodes,
    )
    next_token, next_sibling = outputs[3].cpu(), outputs[4].cpu()
    # The parent links the GDN conv kernel derives from the same tree links.
    parents = torch.zeros(batch_size, num_nodes, dtype=torch.long)
    for row in range(batch_size):
        for node in range(num_nodes):
            child = int(next_token[row, node])
            if child != -1:
                parents[row, child] = node
            sibling = int(next_sibling[row, node])
            if sibling != -1:
                parents[row, sibling] = parents[row, node]
    return parents


def _accept_paths(parents, num_nodes, seed):
    """A random accepted prefix of a random root-to-leaf path per request, as
    the flat verify-token indices of ``accept_index``."""
    gen = torch.Generator().manual_seed(seed)
    batch_size = parents.shape[0]
    accept_index = torch.full((batch_size, DEPTH + 1), -1, dtype=torch.int32)
    accept_lens = torch.zeros(batch_size, dtype=torch.int32)
    last_nodes = []
    for row in range(batch_size):
        node = int(torch.randint(0, num_nodes, (1,), generator=gen))
        path = [node]
        while node != 0:
            node = int(parents[row, node])
            path.append(node)
        path.reverse()
        length = int(torch.randint(1, len(path) + 1, (1,), generator=gen))
        accept_index[row, :length] = torch.tensor(path[:length]) + row * num_nodes
        accept_lens[row] = length
        last_nodes.append(path[length - 1])
    return accept_index.to(DEVICE), accept_lens.to(DEVICE), last_nodes


def _inputs(batch_size, num_nodes, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    num_tokens = batch_size * num_nodes

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

    # Packed like the backend's mixed_qkv, so q/k/v are token-strided views.
    qkv = rand(num_tokens, 2 * H * K + HV * V)
    q, k, v = qkv.split([H * K, H * K, HV * V], dim=-1)
    return {
        "A_log": rand(HV, dtype=torch.float32) * 0.5,
        "dt_bias": rand(HV) * 0.5,
        "q": q.view(1, num_tokens, H, K),
        "k": k.view(1, num_tokens, H, K),
        "v": v.view(1, num_tokens, HV, V),
        "a": rand(num_tokens, HV),
        "b": rand(num_tokens, HV),
    }


def _snapshot_verify(inputs, state, slots, parents, num_nodes):
    batch_size = slots.shape[0]
    snapshots = torch.zeros(
        batch_size, num_nodes, HV, K, V, device=DEVICE, dtype=state.dtype
    )
    out = fused_sigmoid_gating_delta_rule_update(
        softplus_beta=1.0,
        softplus_threshold=20.0,
        initial_state_source=state,
        initial_state_indices=slots,
        cu_seqlens=torch.arange(
            0, batch_size * num_nodes + 1, num_nodes, dtype=torch.int32, device=DEVICE
        ),
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
        disable_state_update=True,
        intermediate_states_buffer=snapshots,
        intermediate_state_indices=torch.arange(
            batch_size, dtype=torch.int32, device=DEVICE
        ),
        cache_steps=num_nodes,
        retrieve_parent_token=parents.to(DEVICE),
        **{name: inputs[name].contiguous() for name in ("q", "k", "v")},
        **{name: inputs[name] for name in ("A_log", "dt_bias", "a", "b")},
    )
    return out, snapshots


def _rings(num_rows, num_nodes):
    return {
        "rawv": torch.zeros(1, num_rows, HV, num_nodes, V, device=DEVICE).bfloat16(),
        "rawk": torch.zeros(1, num_rows, H, num_nodes, K, device=DEVICE).bfloat16(),
        "g": torch.zeros(1, num_rows, HV, num_nodes, device=DEVICE),
        "beta": torch.zeros(1, num_rows, HV, num_nodes, device=DEVICE),
    }


def _tree_verify(inputs, state, slots, parents, rings, ring_rows, num_nodes):
    return gdn_tree_verify(
        initial_state_source=state,
        initial_state_indices=slots,
        retrieve_parent_token=parents.to(DEVICE),
        num_draft_tokens=num_nodes,
        rawv_ring=rings["rawv"][0],
        rawk_ring=rings["rawk"][0],
        g_ring=rings["g"][0],
        beta_ring=rings["beta"][0],
        ring_indices=ring_rows,
        **inputs,
    )


class TestGdnTreeVerify(CustomTestCase):
    def _check(self, *, batch_size, num_nodes, seed, temperature):
        parents = _random_trees(batch_size, num_nodes, seed, temperature)
        inputs = _inputs(batch_size, num_nodes, seed)
        num_slots = batch_size + 3
        state = torch.randn(
            num_slots,
            HV,
            V,
            K,
            device=DEVICE,
            generator=torch.Generator(device=DEVICE).manual_seed(seed),
        )
        slots = torch.randperm(num_slots, device=DEVICE)[:batch_size].int()
        ring_rows = torch.arange(batch_size, dtype=torch.int32, device=DEVICE)
        tag = f"{batch_size=} {num_nodes=} {seed=} {temperature=}"

        ref_out, snapshots = _snapshot_verify(inputs, state, slots, parents, num_nodes)
        rings = _rings(batch_size, num_nodes)
        out = _tree_verify(inputs, state, slots, parents, rings, ring_rows, num_nodes)
        self.assertTrue(torch.equal(out, ref_out), f"verify output {tag}")

        accept_index, accept_lens, last_nodes = _accept_paths(parents, num_nodes, seed)
        # Request 0 crosses a track interval at its last accepted node.
        track_slots = torch.full((batch_size,), -1, dtype=torch.int32, device=DEVICE)
        track_slots[0] = min(set(range(num_slots)) - set(slots.tolist()))
        track_steps = torch.full_like(track_slots, -1)
        track_steps[0] = accept_index[0, accept_lens[0] - 1]
        folded = state.clone().unsqueeze(0)
        commit_gdn_replayssm_fold_all_layers(
            checkpoint_state=folded,
            rawv_cache=rings["rawv"],
            rawk_cache=rings["rawk"],
            g_cache=rings["g"],
            beta_cache=rings["beta"],
            ssm_state_indices=slots,
            accept_lens=accept_lens,
            max_cache_len=num_nodes,
            num_k_heads=H,
            mamba_track_indices=track_slots,
            mamba_steps_to_track=track_steps,
            ring_indices=ring_rows,
            accept_index=accept_index,
            num_draft_tokens=num_nodes,
        )
        for row, node in enumerate(last_nodes):
            self.assertTrue(
                torch.equal(folded[0, slots[row]], snapshots[row, node]),
                f"committed state row={row} {tag}",
            )
        self.assertTrue(
            torch.equal(folded[0, track_slots[0]], snapshots[0, last_nodes[0]]),
            f"tracked state {tag}",
        )

    def test_matches_snapshot_tree_verify(self):
        for num_nodes in (16, 32):
            for batch_size, seed, temperature in (
                (1, 0, 1.0),
                (6, 1, 0.3),
                (4, 2, 3.0),
            ):
                self._check(
                    batch_size=batch_size,
                    num_nodes=num_nodes,
                    seed=seed,
                    temperature=temperature,
                )

    def test_garbage_links_stay_in_bounds(self):
        """CUDA-graph pad rows carry stale tree links; they must not stall the
        kernel or leave real rows unwritten."""
        num_nodes, batch_size = 16, 3
        parents = _random_trees(batch_size, num_nodes, 5, 1.0)
        parents[1] = torch.randint(-5, 3 * num_nodes, (num_nodes,))
        inputs = _inputs(batch_size, num_nodes, 5)
        state = torch.randn(batch_size, HV, V, K, device=DEVICE)
        slots = torch.arange(batch_size, dtype=torch.int32, device=DEVICE)
        ref_out, _ = _snapshot_verify(
            inputs, state, slots, parents.clamp(0, num_nodes - 1), num_nodes
        )
        out = _tree_verify(
            inputs,
            state,
            slots,
            parents,
            _rings(batch_size, num_nodes),
            slots,
            num_nodes,
        )
        real = torch.tensor([0, 2], device=DEVICE)
        rows = out.view(batch_size, num_nodes, HV, V)
        self.assertTrue(
            torch.equal(rows[real], ref_out.view(batch_size, num_nodes, HV, V)[real])
        )
        self.assertTrue(torch.isfinite(rows[1]).all())


if __name__ == "__main__":
    unittest.main()
