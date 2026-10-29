# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/runtime/peer_memory.py
# atom/mono/runtime/consensus.py
"""Peer memory for in-kernel TP collectives: one uncached buffer a rank, every
rank holding every peer's address, and the TP agreements that must precede
anything collective."""

import ctypes

import torch
import torch.distributed as dist
from sglang.srt.rocm_muse_spark.tp4mono.runtime.spec import MonoUnsupported

_HIP_DEVICE_MALLOC_UNCACHED = 0x3  # hipDeviceMallocUncached
_HIP_IPC_LAZY_ENABLE_PEER_ACCESS = 0x1  # hipIpcMemLazyEnablePeerAccess
_HIP_IPC_HANDLE_BYTES = 64  # sizeof(hipIpcMemHandle_t)


class _HipIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * _HIP_IPC_HANDLE_BYTES)]


class _HipIpcHeap:
    """Uncached device memory and its IPC handles through the HIP runtime: the
    same calls as aiter's ``UncachedIpcHeap`` (``hipExtMallocWithFlags`` with
    ``hipDeviceMallocUncached``, ``hipIpcGetMemHandle``, ``hipIpcOpenMemHandle``)
    for aiter builds that do not ship it. Uncached memory is what makes a peer's
    system-scope stores visible to a spinning load without fences."""

    _lib: ctypes.CDLL | None = None

    @classmethod
    def lib(cls) -> ctypes.CDLL:
        if cls._lib is None:
            lib = ctypes.CDLL("libamdhip64.so")
            lib.hipExtMallocWithFlags.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            lib.hipIpcGetMemHandle.argtypes = [
                ctypes.POINTER(_HipIpcMemHandle),
                ctypes.c_void_p,
            ]
            # the handle is passed by value
            lib.hipIpcOpenMemHandle.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                _HipIpcMemHandle,
                ctypes.c_uint,
            ]
            lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
            lib.hipFree.argtypes = [ctypes.c_void_p]
            lib.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
            lib.hipDeviceSynchronize.argtypes = []
            cls._lib = lib
        return cls._lib

    @classmethod
    def _check(cls, err: int, what: str) -> None:
        if err != 0:
            raise RuntimeError(f"{what} failed: hipError {err}")

    @classmethod
    def alloc_uncached(cls, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        cls._check(
            cls.lib().hipExtMallocWithFlags(
                ctypes.byref(ptr), nbytes, _HIP_DEVICE_MALLOC_UNCACHED
            ),
            "hipExtMallocWithFlags(uncached)",
        )
        cls._check(cls.lib().hipMemset(ptr, 0, nbytes), "hipMemset")
        cls._check(cls.lib().hipDeviceSynchronize(), "hipDeviceSynchronize")
        return int(ptr.value)

    @classmethod
    def get_mem_handle_bytes(cls, ptr: int) -> bytes:
        handle = _HipIpcMemHandle()
        cls._check(
            cls.lib().hipIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(ptr)),
            "hipIpcGetMemHandle",
        )
        return bytes(bytearray(handle.reserved))

    @classmethod
    def open_mem_handle(cls, handle: bytes) -> int:
        ptr = ctypes.c_void_p()
        h = _HipIpcMemHandle.from_buffer_copy(handle)
        cls._check(
            cls.lib().hipIpcOpenMemHandle(
                ctypes.byref(ptr), h, _HIP_IPC_LAZY_ENABLE_PEER_ACCESS
            ),
            "hipIpcOpenMemHandle",
        )
        return int(ptr.value)

    @classmethod
    def close_mem_handle(cls, ptr: int) -> None:
        cls.lib().hipIpcCloseMemHandle(ctypes.c_void_p(ptr))

    @classmethod
    def free_device_mem(cls, ptr: int) -> None:
        cls.lib().hipFree(ctypes.c_void_p(ptr))

    @staticmethod
    def gather_object_list_via_broadcast(group, obj) -> list:
        out = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(out, obj, group=group)
        return out


def _heap():
    try:
        from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

        return UncachedIpcHeap
    except ImportError:
        return _HipIpcHeap


class _DeviceBytes:
    """``__cuda_array_interface__`` of ``nbytes`` bytes at a raw device address, for a
    non-owning torch view."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class PeerBuffer:
    """One uncached buffer per rank, every rank holding every peer's address
    (``addresses``, int64, indexed by TP rank). What lives where, and why no rank
    overwrites a peer's unread data, is the model's layout's to say.

    ``bytes`` is the whole buffer: the kernels tag every pair with the step's
    epoch, so it is never zeroed between steps.

    The constructor is collective over ``group`` (the handle exchange, then an
    agreement that every rank mapped every peer): every rank must reach it, so a
    runner reaches it only after ``tp_agree``. A failed mapping on any rank raises
    ``MonoUnsupported`` on all of them.
    """

    def __init__(
        self,
        nbytes: int,
        group,
        rank: int,
        npes: int,
        device: torch.device,
    ):
        heap = _heap()
        self._heap = heap
        self.local = 0
        self.bytes = None
        self.rank = rank
        self._opened: list[int] = []
        handle, failure = None, None
        try:
            self.local = heap.alloc_uncached(nbytes)
            # a view, not an owner: close() frees the memory under it
            self.bytes = torch.as_tensor(
                _DeviceBytes(self.local, nbytes), device=device
            )
            handle = heap.get_mem_handle_bytes(self.local)
        except Exception as err:  # noqa: BLE001 - every local refusal is agreed
            failure = err
        if not tp_agree(failure is None, group):
            self.close()
            raise MonoUnsupported(
                "uncached peer allocation/handle failed on "
                f"{'this' if failure else 'another'} TP rank"
            ) from failure

        addresses = [self.local]
        if npes > 1:
            handles = heap.gather_object_list_via_broadcast(group, handle)
            addresses, failure = [], None
            try:
                for peer, handle in enumerate(handles):
                    if peer == rank:
                        addresses.append(self.local)
                    else:
                        base = heap.open_mem_handle(handle)
                        self._opened.append(base)
                        addresses.append(base)
            except Exception as err:  # noqa: BLE001 - agreement prevents a hang
                failure = err
            if not tp_agree(failure is None, group):
                self.close()
                raise MonoUnsupported(
                    "peer memory handshake failed on "
                    f"{'this' if failure else 'another'}"
                    " TP rank"
                ) from failure
        failure = None
        try:
            self.addresses = torch.tensor(
                addresses, dtype=torch.int64, device=device
            )
        except Exception as err:  # noqa: BLE001 - every local refusal is agreed
            failure = err
        if not tp_agree(failure is None, group):
            self.close()
            raise MonoUnsupported(
                "peer address table construction failed on "
                f"{'this' if failure else 'another'} TP rank"
            ) from failure

    def close(self) -> None:
        """Close the peer mappings and free the local buffer; idempotent. Only once no
        kernel of any rank can still touch it (the runner is being dropped)."""
        opened, self._opened = self._opened, []
        for base in opened:
            self._heap.close_mem_handle(base)
        if self.local:
            self.bytes = None
            self._heap.free_device_mem(self.local)
            self.local = 0


# Decisions every TP rank must take alike.
#
# A mono kernel waits in-kernel on every peer rank, so a rank that turned mono off
# alone -- or failed to build what the others launch -- leaves the others spinning
# on the GPU for ever. Each such decision is a local verdict agreed on over the TP
# group's CPU group before anything collective (peer handshake, launch) follows.
#


def tp_agree(ok: bool, group) -> bool:
    """True when every rank of ``group`` passed ``ok`` true. Every rank must call
    it at the same point; ``group`` is a CPU (gloo) process group, or None for a
    single rank."""
    if group is None:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def bind_agreed(bind, group) -> None:
    """``bind()`` on this rank (raising ``MonoUnsupported`` to refuse), then every
    rank's verdict agreed: a rank that raised re-raises, the others raise too."""
    try:
        bind()
    except Exception:
        tp_agree(False, group)
        raise
    if not tp_agree(True, group):
        raise MonoUnsupported("another TP rank refused mono")
