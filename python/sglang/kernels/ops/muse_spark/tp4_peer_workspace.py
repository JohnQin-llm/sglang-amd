"""Uncached peer buffers registered with AITER's custom all-reduce, for the Muse
Spark TP4 kernels that hand rows between the ranks (tp4_row_exchange_hip.cuh)."""

import torch
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from aiter.jit.core import AITER_CSRC_DIR

from sglang.kernels.jit.utils import cache_once, load_jit

# hipIpcGetMemHandle always exports uncached allocations of at least 2 MiB;
# smaller ones only when they are a whole number of pages.
_GRANULARITY = 2 * 1024 * 1024


@cache_once
def _module():
    return load_jit(
        "muse_tp4_peer_workspace_hip",
        cuda_files=["muse_spark/tp4_peer_workspace_hip.cuh"],
        cuda_wrappers=[
            ("addresses", "muse_tp4_peer_workspace::PeerWorkspace::addresses")
        ],
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
    )


def register_peer_workspace(
    communicator: CustomAllreduce, name: str, size: int
) -> tuple[int, int, torch.Tensor]:
    """Allocate this rank's zeroed, uncached workspace ``name`` of at least ``size``
    bytes in the communicator's IPC pool and register it. Collective: every rank
    calls it eagerly, in the same order, before graph capture. Returns the
    workspace address, its size and every rank's address of it (CPU int64), which
    the kernels take by value."""
    pool = communicator._pool
    size = -(-size // _GRANULARITY) * _GRANULARITY
    if name not in pool:
        pool.create(name, size, uncached=True)
        handles, offsets = pool.get_ipc_meta(name)
        communicator._ops_register_input_buffer(
            communicator._ptr,
            pool[name].data_ptr,
            [handle.data_ptr() for handle in handles],
            offsets,
        )
    buffer = pool[name]
    peers = torch.empty(communicator.world_size, dtype=torch.int64)
    _module().addresses(communicator._ptr, buffer.data_ptr, peers)
    return buffer.data_ptr, buffer.max_size, peers
