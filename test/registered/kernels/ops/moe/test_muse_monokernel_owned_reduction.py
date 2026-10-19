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

register_amd_ci(est_time=90, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")

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


def _run(rank, port):
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
        key = k2.K2Build(tokens=ROWS)
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

        stream = torch.ones((ROWS, H), dtype=torch.float32, device=device)
        stream_out = torch.empty_like(stream)
        branch_out = torch.empty((ROWS, H), dtype=bf16, device=device)
        pool = zeros((MH, H))
        columns = torch.arange(MH, device=device)
        pool[columns, columns] = 1
        unpool = zeros((H, MH))
        columns = torch.arange(H, device=device)
        unpool[columns, columns % MH] = 1
        down = torch.empty((E, MH, RI // 2), dtype=torch.uint8, device=device)
        args = {
            "key": key,
            "core": zeros((ROWS, 1024)),
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

        def set_weights(phase):
            for owner in range(TP):
                down[:, owner * (MH // TP) : (owner + 1) * (MH // TP), :].fill_(
                    PACKED_E2M1[(rank + 2 * owner + phase) % 3]
                )

        def launch():
            advance_epoch_(epoch)
            k2.k2_launch(**args)
            fail_if_unhealthy_(health)

        set_weights(0)
        torch.cuda.synchronize()
        dist.barrier()
        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            launch()
        baseline = None
        for phase in (0, 1, 2, 0):
            set_weights(phase)
            reference = _expected(phase).repeat(ROWS, 1)
            # Prove the analytical construction distinguishes omission of every rank.
            sensitivity = [
                float((reference[0].float() - _expected(phase, r).float()).abs().max())
                for r in range(TP)
            ]
            assert min(sensitivity) > 4 * ATOL, sensitivity
            for replay in range(3):
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                observed = branch_out.cpu()
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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(target=_run, args=(rank, port)) for rank in range(TP)
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
