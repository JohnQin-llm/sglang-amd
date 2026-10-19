# SPDX-License-Identifier: Apache-2.0
"""Check q8 fused MoE reduction with independent rank/column contributions."""

import hashlib
import multiprocessing
import os
import socket
import time
import unittest

import torch
import torch.distributed as dist
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")

TP, ROWS, H, MH, RI, E = 8, 8, 8192, 4096, 512, 256
VALUES = (1.0, 2.0, -1.0)
PACKED_E2M1 = (0x22, 0x44, 0xAA)
ATOL = 0.008


def _expected(phase, omitted_rank=None):
    coefficients = torch.tensor(
        [
            [VALUES[(rank + 2 * owner + phase) % 3] for owner in range(TP)]
            for rank in range(TP)
        ],
        dtype=torch.float64,
    )
    if omitted_rank is not None:
        coefficients[omitted_rank].zero_()
    reduced = coefficients.sum(0)
    normalized = reduced / reduced.square().mean().sqrt()
    # Every owner supplies 512 FIN columns. The unpooler selects column i%MH.
    return normalized.repeat_interleave(MH // TP).repeat(2).bfloat16()


def _native_intermediate(args):
    """Use the current AITER two-stage path as an independent quantization oracle."""
    import importlib
    from unittest.mock import patch

    from aiter import ActivationType, QuantType
    from aiter.ops.flydsl.moe_common import GateMode

    native = importlib.import_module("aiter.fused_moe")
    quant = native.fused_dynamic_mxfp4_quant_moe_sort
    captured = []

    def observe(*values, **kwargs):
        result = quant(*values, **kwargs)
        source = kwargs.get("input", values[0] if values else None)
        if source is not None and source.shape[-1] == RI:
            assert source.dtype == torch.bfloat16
            captured.append(result[0].clone())
        return result

    device = args["w13"].device
    # The identity pool and zero-offset norms map the all-ones stream to
    # BF16 ones; the descending expert bias selects experts 0 through 7.
    expanded = torch.ones((ROWS * 8, MH), dtype=torch.bfloat16, device=device)
    ids = torch.arange(8, dtype=torch.int32, device=device).repeat(ROWS).view(-1, 1)
    w1 = args["w13"].view(torch.float4_e2m1fn_x2)
    w2 = args["w2"].view(torch.float4_e2m1fn_x2)
    w1.is_shuffled = w2.is_shuffled = True
    with patch.object(native, "fused_dynamic_mxfp4_quant_moe_sort", observe):
        native.fused_moe(
            hidden_states=expanded,
            w1=w1,
            w2=w2,
            topk_weight=torch.ones_like(ids, dtype=torch.float32),
            topk_ids=ids,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Silu,
            w1_scale=args["w13s"],
            w2_scale=args["w2s"],
            doweight_stage1=False,
            hidden_pad=0,
            intermediate_pad=0,
            gate_mode=GateMode.SEPARATED.value,
            dtype=torch.bfloat16,
        )
    assert len(captured) == 1, "The reference must exercise the BF16 two-stage path"
    return captured[0].view(torch.uint8).flatten()


def _run(
    rank,
    port,
    route_boundary=False,
    native_quant=False,
    router_scaling=False,
    rows=ROWS,
):
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    from sglang.srt.rocm_muse_spark.mono.kernel import compile_checked
    from sglang.srt.rocm_muse_spark.mono.ops import k2
    from sglang.srt.rocm_muse_spark.mono.ops.common.epoch import advance_epoch_
    from sglang.srt.rocm_muse_spark.mono.ops.common.plan import HEALTH_WORDS
    from sglang.srt.rocm_muse_spark.mono.runner import fail_if_unhealthy_
    from sglang.srt.rocm_muse_spark.mono.runtime.abi import PEER_HEAD
    from sglang.srt.rocm_muse_spark.mono.runtime.peer import PeerBuffer

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP
    )
    peer = None
    try:
        peer = PeerBuffer(
            PEER_HEAD + k2.peer_bytes(), dist.group.WORLD, rank, TP, device
        )
        key = k2.K2Build(
            tokens=rows, router_output_multiplier=32**-0.5 if router_scaling else 1.0
        )
        problems = compile_checked(key)
        assert not problems, problems
        peer.bytes.zero_()
        bases = peer.addresses
        alarm = int(bases[0].item()) + rank * 4
        peers = bases + PEER_HEAD
        health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=device)
        epoch = torch.zeros(1, dtype=torch.int32, device=device)
        scratch = torch.zeros(k2.scratch_bytes(key), dtype=torch.uint8, device=device)
        bf16 = torch.bfloat16

        def zeros(shape, dtype=bf16):
            return torch.zeros(shape, device=device, dtype=dtype)

        stream = torch.ones((rows, H), dtype=torch.float32, device=device)
        if route_boundary:
            # Equal-norm token markers let the router select distinct experts.
            stream[torch.arange(rows), torch.arange(rows)] += 1
        stream_out = torch.empty_like(stream)
        branch_out = torch.empty((rows, H), dtype=bf16, device=device)
        pool = zeros((MH, H))
        columns = torch.arange(MH, device=device)
        pool[columns, columns] = 1
        unpool = zeros((H, MH))
        columns = torch.arange(H, device=device)
        unpool[columns, columns % MH] = 1
        down = torch.empty((E, MH, RI // 2), dtype=torch.uint8, device=device)
        args = {
            "key": key,
            "core": zeros((rows, 1024)),
            "stream_a": stream,
            "w_o": zeros((H, 1024)),
            "alpha_a": torch.ones(H, device=device),
            "beta_a": zeros(H, torch.float32),
            "w_fn": zeros(H),
            "w_router": zeros((E, H), torch.float32),
            "expert_bias": torch.arange(E, 0, -1, dtype=torch.float32, device=device)
            * 0.001,
            "w_pool": pool,
            "w_pre": zeros(MH),
            "w13": torch.full(
                (E, 2 * RI, MH // 2), 0x22, dtype=torch.uint8, device=device
            ),
            "w13s": torch.full(
                (E, 2 * RI, MH // 32), 127, dtype=torch.uint8, device=device
            ),
            "w2": down,
            "w2s": torch.full((E, MH, RI // 32), 127, dtype=torch.uint8, device=device),
            "w_post": zeros(MH),
            "w_unpool": unpool,
            "stream_out": stream_out,
            "branch_out": branch_out,
            "scratch": scratch,
            "peers": peers,
            "rank": rank,
            "epoch": epoch,
            "layer": 3,
            "health": health,
            "alarm": alarm,
        }

        base_up = None
        if native_quant:
            generator = torch.Generator(device=device).manual_seed(2026 + rank)
            base_up = torch.randint(
                0,
                256,
                args["w13"].shape,
                dtype=torch.uint8,
                device=device,
                generator=generator,
            )
            args["w13s"].fill_(122)

        def set_weights(phase):
            if native_quant:
                args["w13"].copy_(base_up)
                args["w13"].bitwise_xor_(phase * 0x11)
            for owner in range(TP):
                down[:, owner * (MH // TP) : (owner + 1) * (MH // TP), :].fill_(
                    PACKED_E2M1[(rank + 2 * owner + phase) % 3]
                )

        def set_routes(unique):
            # Every expert has identical gate/up weights and the same rank/column
            # pattern in down weights, so the analytical result is independent
            # of the selected experts. Column rows cancels the all-ones input;
            # each token's marker then selects exactly its eight expert IDs.
            router = args["w_router"]
            router.zero_()
            for token in range(rows):
                for slot in range(8):
                    router[(token * 8 + slot) % unique, token] = 1
            router[:, rows] = -router[:, :rows].sum(dim=1)

        def set_router_scale_case(phase):
            router = args["w_router"]
            router.zero_()
            logits = torch.full((E,), -4.0, device=device)
            bias = torch.full((E,), -1.0, device=device)
            logits[:8] = 4.0 + torch.arange(8, device=device) * 0.1
            logits[8:16] = torch.arange(8, device=device) * 0.1
            bias[:8], bias[8:16] = 0.0, 0.25
            router[:, 0] = logits.roll(16 * phase)
            args["expert_bias"].copy_(bias.roll(16 * phase))
            # Unit BF16 FFN input makes each logit equal to column zero.
            scores = torch.sigmoid(router[:, 0] * key.router_output_multiplier)
            ids = torch.topk(scores + args["expert_bias"], 8).indices
            unscaled = torch.topk(
                torch.sigmoid(router[:, 0]) + args["expert_bias"], 8
            ).indices
            assert set(ids.tolist()).isdisjoint(unscaled.tolist())
            chosen = scores[ids]
            return ids, chosen / chosen.sum()

        def launch():
            advance_epoch_(epoch)
            k2.k2_launch(**args)
            fail_if_unhealthy_(health)

        set_weights(0)
        if router_scaling:
            set_router_scale_case(0)
        if route_boundary:
            set_routes(8)
        torch.cuda.synchronize()
        dist.barrier()
        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            launch()
        baseline = None
        quant_baseline = None
        cases = (
            ((8, 0), (32, 1), (33, 2), (64, 1), (8, 0))
            if route_boundary
            else ((8, 0), (8, 1), (8, 2), (8, 0))
        )
        for unique, phase in cases:
            set_weights(phase)
            if router_scaling:
                route_ids, route_scores = set_router_scale_case(phase)
            if route_boundary:
                set_routes(unique)
            if native_quant:
                quant_reference = _native_intermediate(args)
                if phase == 0:
                    if quant_baseline is None:
                        quant_baseline = quant_reference.clone()
                    else:
                        assert torch.equal(quant_reference, quant_baseline)
                else:
                    assert not torch.equal(quant_reference, quant_baseline)
            else:
                reference = _expected(phase).repeat(rows, 1)
                # The construction must distinguish omission of every rank.
                sensitivity = [
                    float(
                        (reference[0].float() - _expected(phase, r).float()).abs().max()
                    )
                    for r in range(TP)
                ]
                assert min(sensitivity) > 4 * ATOL, sensitivity
            for replay in range(3):
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                if route_boundary:
                    offset, size = k2.scratch_layout(rows)["route"]
                    pairs = scratch.narrow(0, offset, size).view(torch.int32)
                    routes = pairs.reshape(-1, 2)[:, 0].reshape(rows, 16)[:, :8]
                    assert routes.unique().numel() == unique
                    for token in range(rows):
                        expected_ids = {
                            (token * 8 + slot) % unique for slot in range(8)
                        }
                        assert set(routes[token].tolist()) == expected_ids
                if router_scaling:
                    offset, size = k2.scratch_layout(rows)["route"]
                    pairs = scratch.narrow(0, offset, size).view(torch.int32)
                    route = pairs.reshape(-1, 2)[:, 0].reshape(rows, 16)
                    torch.testing.assert_close(
                        route[:, :8].long(), route_ids.expand(rows, -1), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        route[:, 8:].contiguous().view(torch.float32),
                        route_scores.expand(rows, -1),
                        rtol=1e-6,
                        atol=1e-7,
                    )
                observed = branch_out.cpu()
                if native_quant:
                    offset, size = k2.scratch_layout(rows)["interb"]
                    packed = scratch.narrow(0, offset, size)
                    torch.testing.assert_close(packed, quant_reference, rtol=0, atol=0)
                    assert torch.isfinite(observed).all()
                else:
                    torch.testing.assert_close(observed, reference, rtol=0, atol=ATOL)
                assert torch.equal(stream_out, stream)
                assert int(torch.count_nonzero(health)) == 0
                if phase == 0 and baseline is None:
                    baseline = observed.clone()
                if phase == 0:
                    assert torch.equal(observed, baseline)
                digest = hashlib.sha256(
                    observed.view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                gathered = [None] * TP
                dist.all_gather_object(gathered, digest)
                assert len(set(gathered)) == 1
        graph.reset()
        dist.barrier()
        torch.cuda.synchronize()
    finally:
        if peer is not None:
            peer.close()
        dist.destroy_process_group()


class TestMuseMonokernelOwnedReduction(CustomTestCase):
    @unittest.skipUnless(
        is_gfx95_supported() and torch.cuda.device_count() >= TP,
        "needs eight gfx950 GPUs",
    )
    def test_rank_column_contributions_and_live_weight_replay(self):
        self._check_replays(route_boundary=False)

    @unittest.skipUnless(
        is_gfx95_supported() and torch.cuda.device_count() >= TP,
        "needs eight gfx950 GPUs",
    )
    def test_route_count_boundary_during_graph_replay(self):
        self._check_replays(route_boundary=True)

    @unittest.skipUnless(
        is_gfx95_supported() and torch.cuda.device_count() >= TP,
        "needs eight gfx950 GPUs",
    )
    def test_native_bf16_intermediate_with_live_weights(self):
        self._check_replays(route_boundary=False, native_quant=True)

    @unittest.skipUnless(
        is_gfx95_supported() and torch.cuda.device_count() >= TP,
        "needs eight gfx950 GPUs",
    )
    def test_router_scale_with_live_weight_replay(self):
        for rows in (1, ROWS):
            with self.subTest(rows=rows):
                self._check_replays(
                    route_boundary=False, router_scaling=True, rows=rows
                )

    def _check_replays(
        self, route_boundary, native_quant=False, router_scaling=False, rows=ROWS
    ):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(
                target=_run,
                args=(rank, port, route_boundary, native_quant, router_scaling, rows),
            )
            for rank in range(TP)
        ]
        try:
            for process in processes:
                process.start()
            deadline = time.monotonic() + 600
            for process in processes:
                process.join(max(0, deadline - time.monotonic()))
            self.assertFalse(
                any(process.is_alive() for process in processes), "K2 test timed out"
            )
            for rank, process in enumerate(processes):
                self.assertEqual(process.exitcode, 0, f"rank {rank} failed")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()


if __name__ == "__main__":
    unittest.main()
