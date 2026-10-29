"""Muse Spark TP4 split pooler/unpool on 4 GPUs at its verify widths (48, 56, 64, 80
and 96 rows): the front epilogue is bitwise equal to the joint router/pooler
epilogue on the same pooled values and logits, both kernels match the replicated
projections, every rank returns the same rows, and CUDA-graph replays equal eager
calls."""

import multiprocessing
import socket
import time
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")

WORLD = 4
HIDDEN, POOLED, EXPERTS, TOP_K = 8192, 4096, 256, 8
MULTIPLIER, ROUTE_SCALE, NORM_EPS = 1.5, 2.5, 1e-5
ROWS = (48, 56, 64, 80, 96)
# Rows whose 8th and 9th biased scores are closer than this may swap experts
# between the summation orders of the two GEMMs.
ROUTE_MARGIN = 1e-5


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _front_weights():
    from sglang.kernels.ops.moe.router_pooler_topk import (
        pack_joint_router_pooler_weight,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    router = torch.randn(HIDDEN, EXPERTS, generator=g, device="cuda") * 0.005
    pooler = (torch.randn(POOLED, HIDDEN, generator=g, device="cuda") * 0.01).bfloat16()
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.01
    norm = (1 + 0.1 * torch.randn(POOLED, generator=g, device="cuda")).bfloat16()
    return router, pack_joint_router_pooler_weight(router, pooler), bias, norm


def _ffn_boundary():
    from sglang.srt.rocm_muse_spark.fused_norm_residual import (
        FusedPostNormResidualPreNorm,
    )
    from sglang.srt.rocm_muse_spark.layers import (
        MuseSpark1xNormPreservingResidualConnection,
        MuseSpark1xRMSNorm,
        MuseSpark1xRMSNormWithInputScale,
    )

    g = torch.Generator(device="cuda").manual_seed(1)
    post = MuseSpark1xRMSNormWithInputScale(
        HIDDEN,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=4,
    ).cuda()
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3).cuda()
    pre = MuseSpark1xRMSNorm(
        HIDDEN, eps=1e-5, has_weight=True, zero_centered_gamma=True, residual_id=5
    ).cuda()
    with torch.no_grad():
        post.weight.copy_(torch.randn(HIDDEN, generator=g, device="cuda") * 0.05)
        gate.gate.copy_(torch.randn(HIDDEN, generator=g, device="cuda"))
        pre.weight.copy_(torch.randn(HIDDEN, generator=g, device="cuda") * 0.05)
    gate.precompute_coefficients()
    post.cache_effective_weight()
    pre.cache_effective_weight()
    boundary = FusedPostNormResidualPreNorm(post, gate, pre)
    boundary._ensure_cached_weights(torch.bfloat16)
    return boundary


def _sequential_sum(partial: torch.Tensor) -> torch.Tensor:
    """The kernels' split order: ((p0 + p1) + p2) + ..."""
    total = partial[0].clone()
    for split in partial[1:]:
        total += split
    return total


def _near_tie_rows(x, router, bias):
    keys = torch.sigmoid((x.float() @ router) * MULTIPLIER) + bias
    top = keys.topk(TOP_K + 1, dim=-1).values
    return (top[:, TOP_K - 1] - top[:, TOP_K]) < ROUTE_MARGIN


def _check_front(rank, split, rows, router, packed, bias, norm):
    from sglang.kernels.ops.moe.router_pooler_topk import (
        _joint_epilogue,
        joint_router_pooler_topk,
    )
    from sglang.kernels.ops.muse_spark.tp4_split_pool_unpool_hip import (
        FRONT_LAUNCH,
        POOL_SLICE,
        ROUTER_ROW,
        _partials,
    )

    g = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").bfloat16()
    num_real = torch.tensor([rows - 8], dtype=torch.int32, device="cuda")
    args = (bias, norm, MULTIPLIER, ROUTE_SCALE, NORM_EPS)
    scores, ids, expert_input = split.front(x, packed, *args, num_real=num_real)

    # The joint epilogue on this kernel's pooled values and logits: bitwise.
    launch = FRONT_LAUNCH[rows]
    pooled = [
        _sequential_sum(
            _partials(x, packed[r * POOL_SLICE : (r + 1) * POOL_SLICE], launch)
        )
        .bfloat16()
        .float()
        for r in range(WORLD)
    ]
    router_sum = _sequential_sum(_partials(x, packed[ROUTER_ROW:], launch))
    joint_partial = torch.cat(pooled + [router_sum], dim=-1)[None].contiguous()
    expected = (
        torch.empty_like(scores),
        torch.empty_like(ids),
        torch.empty_like(expert_input),
    )
    _joint_epilogue[(2 * rows,)](
        joint_partial,
        bias,
        norm,
        *expected,
        rows,
        1,
        *args[2:],
        num_real,
        True,
        num_warps=4,
    )
    for name, a, b in zip(
        ("scores", "ids", "expert input"), (scores, ids, expert_input), expected
    ):
        assert torch.equal(a, b), (
            f"front {name} differs from _joint_epilogue at {rows} rows, rank {rank}"
        )
    assert (ids[rows - 8 :] == -1).all()

    # Against the replicated joint router/pooler: same experts (but for near ties),
    # close scores and expert input.
    ref_scores, ref_ids, ref_input = joint_router_pooler_topk(
        x, packed, *args, num_real=num_real
    )
    differ = (ids.sort(-1).values != ref_ids.sort(-1).values).any(-1)
    assert not (differ & ~_near_tie_rows(x, router, bias)).any(), (
        f"front routes differ at {rows} rows, rank {rank}"
    )
    same = ~differ
    torch.testing.assert_close(
        scores[same].sort(-1).values,
        ref_scores[same].sort(-1).values,
        atol=1e-6,
        rtol=0,
    )
    cosine = F.cosine_similarity(expert_input.float(), ref_input.float(), dim=-1)
    assert cosine.min() >= 0.99999, (
        f"expert input cosine {cosine.min():.7f} at {rows} rows"
    )
    return x, num_real, (scores, ids, expert_input)


def _check_tail(rank, split, rows, last_layer):
    g = torch.Generator(device="cuda").manual_seed(100 + rows)
    expert_output = torch.randn(rows, POOLED, generator=g, device="cuda").bfloat16()
    unpool = (torch.randn(HIDDEN, POOLED, generator=g, device="cuda") * 0.02).bfloat16()
    stream = torch.randn(rows, HIDDEN, generator=g, device="cuda") * 3.0
    boundary = _ffn_boundary()
    pre_norm_dtype = torch.bfloat16 if last_layer else None
    args = (
        boundary._cc_post_weight,
        boundary._cc_pre_weight,
        boundary._cc_alpha,
        boundary._cc_beta,
        boundary._post_norm.eps,
        boundary._pre_norm.eps,
        last_layer,
    )
    out = split.tail(expert_output, unpool, stream, *args)
    reference = boundary.forward(
        stream,
        F.linear(expert_output, unpool),
        branch_dtype=torch.bfloat16,
        pre_norm_dtype=pre_norm_dtype,
    )[:2]
    for name, a, b in zip(("next input", "stream"), out, reference):
        cosine = F.cosine_similarity(a.float(), b.float(), dim=-1).min()
        assert cosine >= 0.99999, (
            f"tail {name} cosine {cosine:.7f} at {rows} rows, rank {rank}"
        )
    return (expert_output, unpool, stream, args), out


def _same_on_every_rank(tensors) -> None:
    for tensor in tensors:
        gathered = [torch.empty_like(tensor) for _ in range(WORLD)]
        dist.all_gather(gathered, tensor)
        for other in gathered:
            assert torch.equal(other, tensor), "ranks disagree"


def _graph_replays_match(communicator, fn, inputs, eager_outputs) -> None:
    static = [t.clone() for t in inputs]
    graph = torch.cuda.CUDAGraph()
    with communicator.capture():
        with torch.cuda.graph(graph):
            outputs = fn(*static)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(outputs, eager_outputs):
        assert torch.equal(a, b), "graph replay differs from the eager call"


def _run_rank(rank: int, port: int) -> None:
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from sglang.kernels.ops.muse_spark.tp4_split_pool_unpool_hip import (
        Tp4SplitPoolUnpool,
    )
    from sglang.srt.rocm_muse_spark.router_pooler import _create_hipblaslt_handle

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD
    )
    communicator = CustomAllreduce(dist.group.WORLD, device)
    try:
        assert not communicator.disabled
        _create_hipblaslt_handle()
        split = Tp4SplitPoolUnpool(communicator)
        router, packed, bias, norm = _front_weights()
        for rows in ROWS:
            x, num_real, front = _check_front(
                rank, split, rows, router, packed, bias, norm
            )
            tail_inputs, tail = _check_tail(
                rank, split, rows, last_layer=rows == ROWS[-1]
            )
            torch.cuda.synchronize()
            _same_on_every_rank([t.cpu() for t in (*front, *tail)])
            _graph_replays_match(
                communicator,
                lambda x, n: split.front(
                    x, packed, bias, norm, MULTIPLIER, ROUTE_SCALE, NORM_EPS, num_real=n
                ),
                (x, num_real),
                front,
            )
            _graph_replays_match(
                communicator,
                lambda e, w, s: split.tail(e, w, s, *tail_inputs[3]),
                tail_inputs[:3],
                tail,
            )
            dist.barrier()
    finally:
        communicator.close()
        dist.destroy_process_group()


@unittest.skipUnless(
    is_hip() and is_gfx95_supported() and torch.cuda.device_count() >= WORLD,
    "4 gfx950 GPUs",
)
class TestMuseTp4SplitPoolUnpool(CustomTestCase):
    def test_split_pool_unpool_matches_replicated_projections(self):
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
