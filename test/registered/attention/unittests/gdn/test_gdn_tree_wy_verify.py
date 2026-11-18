"""Chunked (UT/WY) GDN tree and chain verify against the replay kernels.

The chunked form reorders the fp32 arithmetic, so its outputs must stay within
fp32 rounding of an fp64 path replay on every tree shape. Its tree ring records
must be bitwise those of ``gdn_tree_verify`` (the commit fold is unchanged), and
its chain ring records those of ``gdn_replayssm_spec_decode`` up to fp32
rounding, at the same circular positions (the compact commit is unchanged).
"""

import unittest

import torch

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)
from sglang.kernels.ops.attention.linear.gdn_tree_verify import gdn_tree_verify
from sglang.kernels.ops.attention.linear.gdn_tree_wy_verify import (
    gdn_chain_wy_verify,
    gdn_tree_wy_verify,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-large-amd")

H, HV, K, V = 16, 48, 128, 128
DEVICE = "cuda"
# fp32 path replay sits near 2e-7 of max |output|; the chunked form near 6e-7.
MAX_RELATIVE_ERROR = 2e-6
# Compact records are bf16 high + low parts, about 16 mantissa bits each way.
MAX_RECORD_DIFFERENCE = 2 * 2**-16


def _tree(shape, num_nodes, gen):
    if shape == "chain":
        return list(range(-1, num_nodes - 1))
    if shape == "star":
        return [-1] + [0] * (num_nodes - 1)
    parents = [-1]
    for node in range(1, num_nodes):
        parents.append(int(torch.randint(0, node, (1,), generator=gen)))
    if shape == "broken":
        # CUDA-graph pad rows: stale links that point forward or out of range
        # are read as the previous node.
        for node in range(1, num_nodes, 3):
            parents[node] = int(torch.randint(-5, 3 * num_nodes, (1,), generator=gen))
    return parents


def _effective_parents(parents):
    return [p if 0 <= p < node else node - 1 for node, p in enumerate(parents)]


def _inputs(batch_size, num_nodes, gen):
    num_tokens = batch_size * num_nodes

    def rand(*shape, dtype=torch.float32):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

    # Packed like the backend's mixed_qkv, so q/k/v are token-strided views.
    qkv = rand(num_tokens, 2 * H * K + HV * V)
    q, k, v = qkv.split([H * K, H * K, HV * V], dim=-1)
    return {
        "A_log": rand(HV) * 0.5,
        "dt_bias": rand(HV, dtype=torch.bfloat16) * 0.5,
        "q": q.view(1, num_tokens, H, K),
        "k": k.view(1, num_tokens, H, K),
        "v": v.view(1, num_tokens, HV, V),
        "a": rand(num_tokens, HV),
        "b": rand(num_tokens, HV),
    }


def _fp64_path_replay(inputs, state, slots, parents, num_nodes):
    batch_size = len(parents)
    rep = HV // H

    def per_node(name, heads, dim):
        x = inputs[name][0].double().view(batch_size, num_nodes, heads, dim)
        return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    q = (per_node("q", H, K) * K**-0.5).repeat_interleave(rep, dim=2)
    k = per_node("k", H, K).repeat_interleave(rep, dim=2)
    v = inputs["v"][0].double().view(batch_size, num_nodes, HV, V)
    a = inputs["a"].double().view(batch_size, num_nodes, HV)
    g = -torch.exp(inputs["A_log"].double()) * torch.nn.functional.softplus(
        a + inputs["dt_bias"].double()
    )
    beta = torch.sigmoid(inputs["b"].double().view(batch_size, num_nodes, HV))
    out = torch.zeros(batch_size, num_nodes, HV, V, dtype=torch.float64, device=DEVICE)
    for row in range(batch_size):
        root_state = (
            state[slots[row]].double()
            if slots[row] >= 0
            else torch.zeros(HV, V, K, dtype=torch.float64, device=DEVICE)
        )
        node_states = []
        for node, parent in enumerate(_effective_parents(parents[row])):
            h = (root_state if parent < 0 else node_states[parent]) * torch.exp(
                g[row, node]
            )[:, None, None]
            update = beta[row, node][:, None] * (
                v[row, node] - torch.einsum("hvk,hk->hv", h, k[row, node])
            )
            h = h + update[:, :, None] * k[row, node][:, None, :]
            node_states.append(h)
            out[row, node] = torch.einsum("hvk,hk->hv", h, q[row, node])
    return out.view(1, batch_size * num_nodes, HV, V)


def _rings(batch_size, num_nodes):
    return {
        "rawv_ring": torch.zeros(
            batch_size, HV, num_nodes, V, device=DEVICE
        ).bfloat16(),
        "rawk_ring": torch.zeros(batch_size, H, num_nodes, K, device=DEVICE).bfloat16(),
        "g_ring": torch.zeros(batch_size, HV, num_nodes, device=DEVICE),
        "beta_ring": torch.zeros(batch_size, HV, num_nodes, device=DEVICE),
        "ring_indices": torch.arange(batch_size, dtype=torch.int32, device=DEVICE),
    }


class TestGdnTreeWyVerify(CustomTestCase):
    def test_matches_path_replay(self):
        batch_size = 3
        for shape in ("chain", "star", "random", "broken"):
            for num_nodes in (8, 16, 24, 32):
                tag = f"{shape=} {num_nodes=}"
                gen = torch.Generator(device=DEVICE).manual_seed(num_nodes)
                cpu_gen = torch.Generator().manual_seed(num_nodes)
                parents = [_tree(shape, num_nodes, cpu_gen) for _ in range(batch_size)]
                inputs = _inputs(batch_size, num_nodes, gen)
                state = 0.3 * torch.randn(
                    batch_size + 2, HV, V, K, device=DEVICE, generator=gen
                )
                # The last request is a pad row: no state slot, zero state.
                slots = torch.tensor([2, 0, -1], dtype=torch.int32, device=DEVICE)
                tree = dict(
                    initial_state_source=state,
                    initial_state_indices=slots,
                    retrieve_parent_token=torch.tensor(parents, device=DEVICE),
                    num_draft_tokens=num_nodes,
                    **inputs,
                )
                path_rings = _rings(batch_size, num_nodes)
                wy_rings = _rings(batch_size, num_nodes)
                gdn_tree_verify(**tree, **path_rings)
                out = gdn_tree_wy_verify(**tree, **wy_rings)

                ref = _fp64_path_replay(inputs, state, slots, parents, num_nodes)
                error = (out.double() - ref).abs().max() / ref.abs().max()
                self.assertLess(error.item(), MAX_RELATIVE_ERROR, tag)
                for name, ring in path_rings.items():
                    self.assertTrue(torch.equal(wy_rings[name], ring), f"{name} {tag}")

    def test_chain_matches_circular_kernel(self):
        """From an empty ring at a nonzero circular base, as every commit leaves
        it when the ring folds each step."""
        batch_size, num_nodes, ring_len, num_rows = 3, 8, 16, 5
        gen = torch.Generator(device=DEVICE).manual_seed(7)
        inputs = _inputs(batch_size, num_nodes, gen)
        state = 0.3 * torch.randn(
            batch_size + 2, HV, V, K, device=DEVICE, generator=gen
        )
        # The last request is a pad row whose request row aliases request 0's.
        slots = torch.tensor([2, 0, -1], dtype=torch.int32, device=DEVICE)
        rows = torch.tensor([3, 1, 3], dtype=torch.int32, device=DEVICE)
        cache_base = torch.tensor([0, 13, 0, 6, 0], dtype=torch.int32, device=DEVICE)

        def rings():
            return {
                "d_cache": torch.zeros(
                    num_rows, HV, ring_len, V, device=DEVICE
                ).bfloat16(),
                "k_cache": torch.zeros(
                    num_rows, H, ring_len, K, device=DEVICE
                ).bfloat16(),
                "g_cache": torch.zeros(num_rows, HV, ring_len, device=DEVICE),
                "d_residual_cache": torch.zeros(
                    num_rows, HV, ring_len, V, device=DEVICE
                ).bfloat16(),
                "k_residual_cache": torch.zeros(
                    num_rows, H, ring_len, K, device=DEVICE
                ).bfloat16(),
            }

        num_tokens = batch_size * num_nodes
        flat = {
            "q": inputs["q"].view(num_tokens, H, K),
            "k": inputs["k"].view(num_tokens, H, K),
            "v": inputs["v"].view(num_tokens, HV, V),
        }
        ref_rings = rings()
        ref_out = torch.empty(num_tokens, HV, V, device=DEVICE)
        gdn_replayssm_spec_decode(
            a=inputs["a"],
            b=inputs["b"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            checkpoint_state=state,
            rawv_cache=ref_rings["d_residual_cache"],
            rawk_cache=ref_rings["k_residual_cache"],
            beta_cache=None,
            d_cache=ref_rings["d_cache"],
            k_cache=ref_rings["k_cache"],
            g_cache=ref_rings["g_cache"],
            out=ref_out,
            query_start_loc=torch.arange(
                0, num_tokens + 1, num_nodes, dtype=torch.int32, device=DEVICE
            ),
            ssm_state_indices=slots,
            replay_indices=rows,
            write_pos=torch.zeros(num_rows, dtype=torch.int32, device=DEVICE),
            cache_base=cache_base,
            is_flush=torch.zeros(num_rows, dtype=torch.int8, device=DEVICE),
            max_cache_len=ring_len,
            max_spec_len=num_nodes,
            null_block_id=-1,
            launch_mode="verify",
            **flat,
        )
        wy_rings = rings()
        out = gdn_chain_wy_verify(
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            a=inputs["a"],
            b=inputs["b"],
            checkpoint_state=state,
            ssm_state_indices=slots,
            replay_indices=rows,
            cache_base=cache_base,
            num_draft_tokens=num_nodes,
            **wy_rings,
            **flat,
        )

        real = slice(0, 2 * num_nodes)
        ref = _fp64_path_replay(
            inputs,
            state,
            slots,
            [list(range(-1, num_nodes - 1))] * batch_size,
            num_nodes,
        )
        error = (out[0, real].double() - ref[0, real]).abs().max() / ref.abs().max()
        self.assertLess(error.item(), MAX_RELATIVE_ERROR)
        for hi, low in (
            ("d_cache", "d_residual_cache"),
            ("k_cache", "k_residual_cache"),
        ):
            want = ref_rings[hi].float() + ref_rings[low].float()
            got = wy_rings[hi].float() + wy_rings[low].float()
            error = (got - want).abs().max() / want.abs().max()
            self.assertLess(error.item(), MAX_RECORD_DIFFERENCE, hi)
        self.assertTrue(
            torch.allclose(wy_rings["g_cache"], ref_rings["g_cache"], rtol=1e-6, atol=0)
        )


if __name__ == "__main__":
    unittest.main()
