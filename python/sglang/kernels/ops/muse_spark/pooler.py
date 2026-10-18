"""gfx950 Muse pooler: M8 split-K projection and TP8 prefill token sharding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


@triton.jit
def _partial(
    X,
    W,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
):
    ni, ki, mi = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = mi * BM + tl.arange(0, BM)
    n = ni * BN + tl.arange(0, BN)
    k = ki * (K // SPLITS) + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for offset in range(K // SPLITS // BK):
        kk = k + offset * BK
        x = tl.load(X + m[:, None] * K + kk[None, :], m[:, None] < M, 0)
        w = tl.load(W + n[None, :] * K + kk[:, None])
        acc = tl.dot(x, w, acc)
    tl.store(P + ki * M * N + m[:, None] * N + n[None, :], acc, m[:, None] < M)


@triton.jit
def _reduce(
    P, Y, M: tl.constexpr, N: tl.constexpr, SPLITS: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, SPLITS)
    x = tl.load(P + s[:, None] * M * N + i[None, :], i[None, :] < M * N, 0)
    tl.store(Y + i, tl.sum(x, axis=0), i < M * N)


def splitk_projection(
    x: torch.Tensor, weight: torch.Tensor, splits: int = 16
) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    assert x.is_contiguous() and weight.is_contiguous()
    assert k % (splits * 64) == 0 and n % 64 == 0
    part = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _partial[(n // 64, splits, triton.cdiv(m, 16))](
        x, weight, part, m, n, k, 16, 64, 64, splits, num_warps=4
    )
    _reduce[(triton.cdiv(m * n, 256),)](part, out, m, n, splits, 256, num_warps=4)
    return out


def try_splitk_pooler(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
    """Only the measured gfx950 M8 pooler; preserve every other dispatch."""
    if not (
        torch.version.hip is not None
        and x.is_cuda
        and x.dtype == weight.dtype == torch.bfloat16
        and x.device == weight.device
        and tuple(x.shape) == (8, 8192)
        and tuple(weight.shape) == (4096, 8192)
        and x.is_contiguous()
        and weight.is_contiguous()
        and torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
        == "gfx950"
    ):
        return None
    return splitk_projection(x, weight, splits=16)


def try_token_sharded_pooler(
    x: torch.Tensor,
    linear: torch.nn.Linear,
    norm: torch.nn.Module | None,
    group: GroupCoordinator,
) -> torch.Tensor | None:
    """Prefill-only projection/norm sharding; gather before expert dispatch.

    Inputs must be replicated across the TP group. The full expert path still
    sees every token. The measured unpooler remains replicated.
    """
    if not (
        torch.version.hip is not None
        and x.is_cuda
        and group.world_size == 8
        and x.ndim == 2
        and 2048 <= x.shape[0] <= 16384
        and x.shape[1] == 8192
        and tuple(linear.weight.shape) == (4096, 8192)
        and x.dtype == linear.weight.dtype == torch.bfloat16
        and x.device == linear.weight.device
        and linear.bias is None
        and x.is_contiguous()
        and torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
        == "gfx950"
    ):
        return None
    rows = triton.cdiv(x.shape[0], group.world_size)
    local = x[group.rank_in_group * rows : (group.rank_in_group + 1) * rows]
    # Equal-sized collective inputs preserve token order for ragged batches.
    # Pad only this rank's tail, not the replicated input on every rank.
    if local.shape[0] != rows:
        local = F.pad(local, (0, 0, 0, rows - local.shape[0]))
    pooled = F.linear(local, linear.weight)
    if norm is not None:
        pooled = norm(pooled)
    return group.all_gather(pooled, dim=0)[: x.shape[0]]
