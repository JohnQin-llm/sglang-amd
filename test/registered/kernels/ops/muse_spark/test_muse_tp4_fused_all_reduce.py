"""Muse Spark TP4 fused all-reduces on 4 GPUs, in CUDA graphs at every TP4 verify
width. The MoE finalize + all-reduce is bitwise equal to FlyDSL's finalize followed
by AITER's all-reduce, on random inputs and on partials that expose the order in
which the ranks are added. The attention all-reduce + boundary matches AITER's
all-reduce followed by the Triton boundary to a row cosine of 0.99999, with at least
99.9% of the elements bitwise equal."""

import multiprocessing
import socket
import time
import unittest

import torch
import torch.distributed as dist

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")

WORLD = 4
HIDDEN, MOE_HIDDEN, TOP_K, EXPERTS, LOCAL = 8192, 4096, 8, 256, 64
ROWS = tuple(range(8, 129, 8))
REPLAYS = 20


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _moe_inputs(rank: int, rows: int, seed: int, padded: bool):
    """Per-slot expert rows of this rank and routes shared by all ranks (over every
    rank's experts); with ``padded`` the last 8 rows carry expert id -1."""
    own = torch.Generator(device="cuda").manual_seed(seed * 31 + rank)
    shared = torch.Generator(device="cuda").manual_seed(seed)
    y = (torch.randn(rows * TOP_K, MOE_HIDDEN, generator=own, device="cuda") * 0.7).to(
        torch.bfloat16
    )
    logits = torch.randn(rows, EXPERTS, generator=shared, device="cuda")
    ids = logits.topk(TOP_K, dim=-1).indices.to(torch.int32)
    if padded:
        ids[-8:] = -1
    weights = torch.rand(rows, TOP_K, generator=shared, device="cuda") + 0.1
    scores = weights / weights.sum(-1, keepdim=True) * 2.5
    return y, scores, ids.contiguous()


def _attention_boundary(seed: int):
    from sglang.srt.rocm_muse_spark.fused_norm_residual import (
        FusedPostNormResidualPreNorm,
    )
    from sglang.srt.rocm_muse_spark.layers import (
        MuseSpark1xNormPreservingResidualConnection,
        MuseSpark1xRMSNorm,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    post = MuseSpark1xRMSNorm(
        HIDDEN,
        eps=1e-8,
        has_weight=False,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=1,
    ).cuda()
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3).cuda()
    pre = MuseSpark1xRMSNorm(
        HIDDEN, eps=1e-5, has_weight=True, zero_centered_gamma=True, residual_id=2
    ).cuda()
    with torch.no_grad():
        gate.gate.copy_(torch.randn(HIDDEN, generator=generator, device="cuda"))
        pre.weight.copy_(torch.randn(HIDDEN, generator=generator, device="cuda") * 0.05)
    gate.precompute_coefficients()
    pre.cache_effective_weight()
    boundary = FusedPostNormResidualPreNorm(post, gate, pre)
    boundary._ensure_cached_weights(torch.bfloat16)
    return boundary


def _attention_inputs(rank: int, rows: int, seed: int):
    own = torch.Generator(device="cuda").manual_seed(seed * 31 + rank)
    shared = torch.Generator(device="cuda").manual_seed(seed)
    partial = (torch.randn(rows, HIDDEN, generator=own, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    stream = torch.randn(rows, HIDDEN, generator=shared, device="cuda") * 3.0
    return partial, stream


def _capture(communicator, fn):
    graph = torch.cuda.CUDAGraph()
    with communicator.capture():
        with torch.cuda.graph(graph):
            out = fn()
    return graph, out


def _replay_with(graph, static_inputs, inputs):
    for static, value in zip(static_inputs, inputs):
        static.copy_(value)
    graph.replay()


def _check_moe(rank, communicator, fused, rows):
    from sglang.srt.rocm_muse_spark.mxfp4_moe import _finalize_per_slot

    eps, gain, start = 1e-8, 1.0, rank * LOCAL
    gamma = (
        torch.randn(
            MOE_HIDDEN,
            generator=torch.Generator(device="cuda").manual_seed(rows),
            device="cuda",
        )
        * 0.05
    ).to(torch.bfloat16)
    sets = [
        _moe_inputs(rank, rows, seed=2 * rows, padded=False),
        _moe_inputs(rank, rows, seed=2 * rows + 1, padded=rows > 64),
    ]
    static_inputs = [t.clone() for t in sets[0]]

    y, scores, ids = static_inputs

    def separate():
        partial = _finalize_per_slot(y, gamma, scores, ids, start, LOCAL, eps, gain)
        return communicator.custom_all_reduce(partial)

    def fused_call():
        return fused.finalize_all_reduce(y, gamma, scores, ids, start, LOCAL, eps, gain)

    reference_graph, reference = _capture(communicator, separate)
    fused_graph, out = _capture(communicator, fused_call)
    for i in range(REPLAYS):
        inputs = sets[i % 2]
        _replay_with(reference_graph, static_inputs, inputs)
        _replay_with(fused_graph, static_inputs, inputs)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), (
            f"MoE finalize + all-reduce differs at {rows} rows, replay {i}, rank {rank}"
        )


def _check_moe_rank_order(rank, communicator, fused, rows):
    """Finalized partials P, Q, Q, -P on ranks 0..3 (Q = 2^-25 P) sum to 0 in rank
    order 0..3 and to a nonzero value from any other first rank, so the fused output
    is bitwise equal to the separate kernels only if it adds the ranks in AITER's
    order."""
    from sglang.srt.rocm_muse_spark.mxfp4_moe import _finalize_per_slot

    eps, gain, start = 1e-8, 1.0, rank * LOCAL
    g = torch.Generator(device="cuda").manual_seed(rows)
    base = torch.randn(rows, MOE_HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    gamma = (torch.randn(MOE_HIDDEN, generator=g, device="cuda") * 0.05).to(
        torch.bfloat16
    )
    y = (-base if rank == 3 else base).repeat_interleave(TOP_K, dim=0).contiguous()
    # One slot per rank (experts 0, 64, 128, 192) with scores 1, 2^-25, 2^-25, 1.
    slot_scores = [1.0, 2**-25, 2**-25, 1.0, 0.0, 0.0, 0.0, 0.0]
    scores = torch.tensor(slot_scores, device="cuda").repeat(rows, 1).contiguous()
    slot_ids = [0, 64, 128, 192, -1, -1, -1, -1]
    ids = torch.tensor(slot_ids, dtype=torch.int32, device="cuda").repeat(rows, 1)
    ids = ids.contiguous()
    args = (y, gamma, scores, ids, start, LOCAL, eps, gain)

    partial = _finalize_per_slot(*args).float().cpu()
    gathered = [torch.empty_like(partial) for _ in range(WORLD)]
    dist.all_gather(gathered, partial)
    in_order = ((gathered[0] + gathered[1]) + gathered[2]) + gathered[3]
    from_rank_1 = ((gathered[1] + gathered[2]) + gathered[3]) + gathered[0]
    assert not torch.equal(in_order.bfloat16(), from_rank_1.bfloat16())

    reference_graph, reference = _capture(
        communicator,
        lambda: communicator.custom_all_reduce(_finalize_per_slot(*args)),
    )
    fused_graph, out = _capture(communicator, lambda: fused.finalize_all_reduce(*args))
    reference_graph.replay()
    fused_graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), (
        "MoE finalize + all-reduce adds the ranks in another order than AITER at "
        f"{rows} rows, rank {rank}"
    )


def _check_attention(rank, communicator, fused, rows):
    boundary = _attention_boundary(seed=rows)
    sets = [_attention_inputs(rank, rows, seed=2 * rows + k) for k in (0, 1)]
    static_inputs = [t.clone() for t in sets[0]]

    def separate():
        reduced = communicator.custom_all_reduce(static_inputs[0])
        ffn_input, new_stream, _ = boundary.forward(
            static_inputs[1], reduced, branch_dtype=torch.bfloat16
        )
        return ffn_input, new_stream

    def fused_call():
        return fused.all_reduce_boundary(
            *static_inputs,
            boundary._cc_alpha,
            boundary._cc_beta,
            boundary._cc_pre_weight,
            boundary._post_norm.eps,
            boundary._pre_norm.eps,
        )

    reference_graph, reference = _capture(communicator, separate)
    fused_graph, out = _capture(communicator, fused_call)
    for i in range(REPLAYS):
        _replay_with(reference_graph, static_inputs, sets[i % 2])
        _replay_with(fused_graph, static_inputs, sets[i % 2])
        torch.cuda.synchronize()
        for name, a, b in zip(("FFN input", "stream"), out, reference):
            a, b = a.float(), b.float()
            cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1).min()
            bitwise = (a == b).float().mean()
            assert cosine >= 0.99999 and bitwise >= 0.999, (
                f"attention {name} at {rows} rows, replay {i}, rank {rank}: "
                f"min row cosine {cosine:.7f}, bitwise {bitwise:.5f}"
            )


def _run_rank(rank: int, port: int) -> None:
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from sglang.kernels.ops.muse_spark.tp4_fused_all_reduce_hip import (
        Tp4FusedAllReduce,
    )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD
    )
    communicator = CustomAllreduce(dist.group.WORLD, device)
    try:
        assert not communicator.disabled
        fused = Tp4FusedAllReduce(communicator)
        for rows in ROWS:
            _check_moe(rank, communicator, fused, rows)
            _check_moe_rank_order(rank, communicator, fused, rows)
            _check_attention(rank, communicator, fused, rows)
            dist.barrier()
    finally:
        communicator.close()
        dist.destroy_process_group()


@unittest.skipUnless(
    is_hip() and is_gfx95_supported() and torch.cuda.device_count() >= WORLD,
    "4 gfx950 GPUs",
)
class TestMuseTp4FusedAllReduce(CustomTestCase):
    def test_fused_all_reduces_match_separate_kernels(self):
        """Rows 8..128 in CUDA graphs, replayed with alternating inputs (padded
        rows with expert id -1 in the second MoE set above 64 rows)."""
        port = _open_port()
        context = multiprocessing.get_context("spawn")
        procs = [
            context.Process(target=_run_rank, args=(rank, port))
            for rank in range(WORLD)
        ]
        for proc in procs:
            proc.start()
        deadline = time.monotonic() + 600
        for proc in procs:
            proc.join(timeout=max(0.0, deadline - time.monotonic()))
        for proc in procs:
            if proc.is_alive():
                proc.kill()
        self.assertEqual([proc.exitcode for proc in procs], [0] * WORLD)


if __name__ == "__main__":
    unittest.main()
