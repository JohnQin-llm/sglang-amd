# SPDX-License-Identifier: Apache-2.0
"""TP-to-EP weight preparation and real graph dispatch on eight MI35x GPUs."""

import os
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=180, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")

NAMES = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")


def _raw_weights(rank, tp, phase):
    """Independent global expert/row/column coordinates for either partition."""
    experts = torch.arange(256 if tp else 32, device="cuda")
    if not tp:
        experts += rank * 32
    tensors = []
    for index in range(4):
        if index < 2:
            rows = torch.arange(512 if tp else 4096, device="cuda")
            if tp:
                rows += rank * 512
            rows = torch.cat((rows, rows + 4096))
            cols = torch.arange(2048 if index == 0 else 128, device="cuda")
        else:
            rows = torch.arange(4096, device="cuda")
            width = (256 if tp else 2048) if index == 2 else (16 if tp else 128)
            cols = torch.arange(width, device="cuda")
            if tp:
                cols += rank * width
        if index % 2:
            values = (
                experts[:, None, None].remainder(2).byte()
                + rows[None, :, None].remainder(2).byte()
                + cols[None, None, :].remainder(2).byte()
            ).remainder(2) + 119
        else:
            values = (
                (experts[:, None, None] * 17 + phase * 3).byte()
                + (rows[None, :, None] * 5).byte()
                + (cols[None, None, :] * 11).byte()
            )
        tensors.append(values.contiguous())
    return tensors


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=8)
    # Match registration order used by the real model loader.
    from aiter.ops.flydsl.kernels import moe_2stage_a16wmix  # noqa: F401
    from sglang.kernels.ops.muse_spark import fused_muse_spark_routing  # noqa: F401
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.rocm_muse_spark import model
    from sglang.srt.runtime_context import get_parallel

    group = GroupCoordinator(
        [list(range(8))],
        rank,
        "nccl",
        use_pynccl=True,
        use_custom_allreduce=True,
        use_mscclpp=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="muse_ep_test",
    )
    get_parallel().override_permanently(
        tp_group=group,
        world_group=group,
        tp_size=8,
        tp_rank=rank,
        moe_ep_size=1,
    )
    with torch.device("meta"):
        candidate = model.MuseSpark1xMxfp4Experts(256, 4096, 512)
        reference = model.MuseSpark1xMxfp4Experts(32, 4096, 4096)
    candidate.original_tp = True
    torch.manual_seed(137)
    norm = SimpleNamespace(
        weight=torch.randn(4096, device="cuda", dtype=torch.bfloat16) * 0.1,
        zero_centered_gamma=True,
        gain_center=1.0,
        eps=1e-8,
    )
    addresses = None
    graphs = []
    first_outputs = None
    for phase in (0, 1, 0):
        for obj, tp in ((candidate, True), (reference, False)):
            for name, value in zip(NAMES, _raw_weights(rank, tp, phase)):
                old = getattr(obj, name)
                if old.is_meta:
                    setattr(obj, name, torch.nn.Parameter(value, requires_grad=False))
                else:
                    old.data.copy_(value)
            if tp:
                obj.original_layout_dirty = True
            obj.prepare_prepacked_weights()
        for name in ("w13", "w13_scale", "w2", "w2_scale"):
            assert torch.equal(
                getattr(candidate._prepacked_moe, name),
                getattr(reference._prepacked_moe, name),
            ), (rank, phase, name)
        current_addresses = [b.data_ptr() for b in candidate._prepacked_moe.buffers()]
        if addresses is None:
            addresses = current_addresses
        assert current_addresses == addresses
        assert len(candidate.state_dict()) == 4
        if not graphs:
            # The extra EP layout must not alter the existing TP representation
            # consumed by low-batch Mono kernels.
            with torch.device("meta"):
                tp_reference = model.MuseSpark1xMxfp4Experts(256, 4096, 512)
            tp_reference.original_tp = True
            for name, value in zip(NAMES, _raw_weights(rank, True, phase)):
                setattr(
                    tp_reference, name, torch.nn.Parameter(value, requires_grad=False)
                )
            with patch.dict(os.environ, {"SGLANG_MUSE_MONO_EP_FALLBACK": "0"}):
                tp_reference.prepare_prepacked_weights()
            assert tp_reference._prepacked_moe is None
            assert all(
                torch.equal(getattr(candidate, n), getattr(tp_reference, n))
                for n in NAMES
            )
            del tp_reference
            for rows in (8, 15, 16, 17, 64, 256, 264):
                x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
                ids = torch.arange(8, device="cuda", dtype=torch.int32).repeat(rows, 1)
                scores = torch.full((rows, 8), 0.125, device="cuda")
                selected = 16 <= rows <= 256 and rows % 8 == 0
                assert candidate.uses_prepacked_batch(x) == selected
                if not selected:
                    with patch.object(
                        candidate, "_forward_original_tp", return_value=x
                    ) as native:
                        assert candidate(x, scores, ids, 0, norm) is x
                        assert native.call_count == 1
            for rows in (16, 256):
                x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
                ids = (
                    (
                        torch.arange(rows, device="cuda")[:, None] * 17
                        + torch.arange(8, device="cuda")[None, :] * 31
                    )
                    .remainder(256)
                    .int()
                )
                scores = torch.rand(rows, 8, device="cuda").softmax(-1)
                pair = []
                for obj in (candidate, reference):

                    def run(obj=obj, x=x, scores=scores, ids=ids):
                        out = obj(x, scores, ids, rank * 32, norm)
                        return out if obj is candidate else group.all_reduce(out)

                    for _ in range(3):
                        run()
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with (
                        group.graph_capture() as context,
                        torch.cuda.graph(graph, stream=context.stream),
                    ):
                        out = run()
                    pair.append((graph, out))
                graphs.append(
                    (x, ids, scores, pair, x.clone(), ids.clone(), scores.clone())
                )
        phase_outputs = []
        for x, ids, scores, pair, x0, ids0, scores0 in graphs:
            x.copy_(x0 if phase == 0 else x0 * 0.5)
            ids.copy_(
                ids0
                if phase == 0
                else torch.arange(8, device="cuda").repeat(x.shape[0], 1)
            )
            scores.copy_(scores0 if phase == 0 else scores0.roll(1, -1))
            dist.barrier()
            for graph, _ in pair:
                graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(pair[0][1], pair[1][1]), (rank, phase, x.shape)
            phase_outputs.append(pair[0][1].clone())
        if first_outputs is None:
            first_outputs = phase_outputs
        elif phase == 0:
            assert all(torch.equal(a, b) for a, b in zip(first_outputs, phase_outputs))
        else:
            assert all(
                not torch.equal(a, b) for a, b in zip(first_outputs, phase_outputs)
            )
    dist.barrier()
    print(
        f"rank {rank}: EP layouts, dispatch, changed graph and weight reload PASS",
        flush=True,
    )
    group.destroy()
    dist.destroy_process_group()


@unittest.skipUnless(
    is_gfx95_supported() and torch.cuda.device_count() >= 8, "Requires eight MI35x GPUs"
)
class TestMuseMonoEpFallback(CustomTestCase):
    def test_weight_layout_dispatch_and_graph_reload(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with patch.dict(
            os.environ,
            {
                "MUSE_MOE_BACKEND": "flydsl",
                "SGLANG_MUSE_MOE_PREPACKED": "1",
                "SGLANG_MUSE_MONO_EP_FALLBACK": "1",
            },
        ):
            mp.spawn(_worker, args=(port,), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
