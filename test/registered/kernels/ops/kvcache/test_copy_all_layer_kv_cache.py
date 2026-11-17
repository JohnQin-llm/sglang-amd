"""``copy_all_layer_kv_cache_func`` must move slots in place when sources and targets
overlap, as compacting an accepted tree path to the front of its verify window does.
One request's window (a 16-row tile, smaller than the kernel's layout) once came back
corrupted in about half of the moves on MI355X, which made greedy tree decoding differ
from run to run.

    python -m pytest test/registered/kernels/ops/kvcache/test_copy_all_layer_kv_cache.py -v
"""

import sys

import pytest
import torch
import triton

from sglang.kernels.ops.kvcache.cache_move import copy_all_layer_kv_cache_func
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")


def test_overlapping_tree_path_moves():
    torch.manual_seed(0)
    # A hybrid-SWA pool's window layers: 2 KV heads of 128 bf16, 512-byte slot rows.
    buffers = [
        torch.randn(4096, 2, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(78)
    ]
    data_ptrs = torch.tensor(
        [b.data_ptr() for b in buffers], dtype=torch.uint64, device="cuda"
    )
    strides = torch.tensor(
        [b.stride(0) * b.element_size() for b in buffers], device="cuda"
    )
    row_bytes = buffers[0].stride(0) * buffers[0].element_size()
    config = dict(bytes_per_tile=128, byte_tiles=row_bytes // 128, num_warps=4)
    window, num_nodes = 64, 16
    for _ in range(50):
        length = torch.randint(2, num_nodes + 1, ()).item()
        # Root, then increasing node ids: node path[k] moves to front slot k.
        path = [0] + (
            torch.randperm(num_nodes - 1)[: length - 1] + 1
        ).sort().values.tolist()
        # The accept lists are padded to the window's width with the dummy slot 0.
        pad = [0] * (num_nodes - length)
        target = torch.tensor([window + k for k in range(length)] + pad, device="cuda")
        source = torch.tensor([window + node for node in path] + pad, device="cuda")
        expected = [b.clone() for b in buffers]
        for b in expected:
            b[target] = b[source].clone()
        copy_all_layer_kv_cache_func(
            data_ptrs,
            strides,
            target,
            source,
            target.numel(),
            triton.next_power_of_2(target.numel()),
            config,
        )
        for b, e in zip(buffers, expected):
            assert torch.equal(b, e)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
