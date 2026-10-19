# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono decode's per-rank state: one scratch buffer every launch shares
(their mailbox tags differ), the device step epoch, the health words, the TP
group's peer memory and the K1 -> K2 hand-off buffers.

Mailbox tags carry the epoch the model bumps once a forward (``step_begin``,
which a CUDA graph captures), so no buffer is zeroed between steps.

The peer buffer's first ``PEER_HEAD`` bytes hold the output rank's flags, a
word a rank (``alarm``): a rank whose wait gives up sets its word there before it
stores anything else. The kernels see the buffer past the head.

Adapted from Kimi-K3's ``mono/runner.py`` (PR #60683).
"""

import os

import torch
from sglang.srt.rocm_muse_spark.mono.ops.common.epoch import advance_epoch_
from sglang.srt.rocm_muse_spark.mono.runtime.abi import PEER_HEAD
from sglang.srt.rocm_muse_spark.mono.runtime.peer import PeerBuffer, bind_agreed
from sglang.srt.rocm_muse_spark.mono.spec import HIDDEN, MAX_TOKENS, TP
from sglang.srt.runtime_context import get_parallel


def fail_if_unhealthy_(health: torch.Tensor) -> None:
    """Turn a terminal K2 health verdict into a graph-captured GPU failure.

    K2 does not return until every CTA and rank reached its terminal rendezvous,
    and each rank has imported the shared alarm words.  Consequently this
    asynchronous assertion makes a post-launch failure fatal on every rank;
    no rank can continue into a native fallback with partially written state.
    """
    from sglang.srt.rocm_muse_spark.mono.runtime.abi import FAILED

    torch._assert_async(
        health[FAILED] == 0,
        "Muse MonoKernel distributed fail-stop: a bounded device wait failed",
    )


class MonoRunner:
    """Collective: every TP rank builds it at the same point."""

    def __init__(self, device: torch.device):
        from sglang.srt.rocm_muse_spark.mono.ops import k1, k2
        from sglang.srt.rocm_muse_spark.mono.ops.common.plan import HEALTH_WORDS

        self.device = device
        tp = get_parallel().tp_group
        if tp.world_size != TP:
            raise RuntimeError(f"MonoRunner needs TP{TP}, got TP{tp.world_size}")
        self.rank = tp.rank_in_group
        self.peers = None

        def allocate_local() -> None:
            self.allocation_guard_bytes = int(
                os.environ.get("VLLM_ROCM_LAYER_MONOKERNEL_CANARY_BYTES", "0")
            )
            if not 0 <= self.allocation_guard_bytes <= 1024 * 1024:
                raise ValueError(
                    "VLLM_ROCM_LAYER_MONOKERNEL_CANARY_BYTES must be between "
                    "0 and 1048576"
                )
            guard = self.allocation_guard_bytes
            self._allocation_canary = 0xA5
            self._scratch_bytes = max(
                k1.max_scratch_bytes(MAX_TOKENS), k2.max_scratch_bytes(MAX_TOKENS)
            )
            self._scratch_storage = torch.empty(
                self._scratch_bytes + 2 * guard,
                dtype=torch.uint8,
                device=device,
            )
            if guard:
                self._scratch_storage.fill_(self._allocation_canary)
            self.scratch = self._scratch_storage[guard : guard + self._scratch_bytes]
            self.scratch.zero_()
            self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
            self.seq_lens = torch.zeros(MAX_TOKENS, dtype=torch.int32, device=device)
            self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=device)
            self.stream_a = torch.empty(
                MAX_TOKENS, HIDDEN, dtype=torch.float32, device=device
            )
            self.core = torch.empty(
                MAX_TOKENS, k1.CORE, dtype=torch.bfloat16, device=device
            )
            torch.cuda.synchronize(device)

        bind_agreed(allocate_local, tp.cpu_group)
        guard = self.allocation_guard_bytes
        self._peer_bytes = PEER_HEAD + k2.peer_bytes(MAX_TOKENS)
        self.peers = PeerBuffer(
            self._peer_bytes + 2 * guard,
            tp.cpu_group,
            self.rank,
            tp.world_size,
            device,
        )

        def bind_peer_views() -> None:
            if guard:
                self.peers.bytes.fill_(self._allocation_canary)
            peer_base = self.peers.addresses + guard
            peer_bytes = self.peers.bytes[guard : guard + self._peer_bytes]
            peer_bytes.zero_()
            self.peer_table = peer_base + PEER_HEAD
            self.alarm = int(peer_base[0].item()) + 4 * self.rank
            self.flags = peer_bytes[: 4 * tp.world_size].view(torch.int32)
            torch.cuda.synchronize(device)

        try:
            bind_agreed(bind_peer_views, tp.cpu_group)
        except Exception:
            self.close()
            raise
        self.fault: dict = {}  # K2Build's fault_* fields (tests only)
        self.first_layer = -1  # the step's first kernel layer: bumps the epoch
        self.q_len = 1  # rows a request
        self.layer_names: dict[int, str] = {}
        # K1 closes layer i-1's deferred FFN boundary.
        self.prev_post_eps: dict[int, float] = {}

    def close(self) -> None:
        """Release peer mappings after GPU quiescence; idempotent."""
        peers, self.peers = getattr(self, "peers", None), None
        if peers is not None:
            peers.close()

    def step_begin(self) -> None:
        """Once a forward, before its first mono launch (a graph captures it)."""
        advance_epoch_(self.epoch)

    def launch_args(self) -> dict:
        return {
            "scratch": self.scratch,
            "epoch": self.epoch,
            "health": self.health,
            "alarm": self.alarm,
        }

    def peer_args(self) -> dict:
        return {"peers": self.peer_table, "rank": self.rank}

    def allocation_canary_report(self, stage: str) -> dict:
        """Check redzones around the real scratch and peer allocations.

        This is an opt-in validation hook and must only be called outside graph
        capture.  The inner views and every kernel-visible offset are identical
        to the ordinary runner layout.
        """
        guard = self.allocation_guard_bytes
        if not guard:
            return {"enabled": False, "stage": stage, "local_passed": True}

        expected = self._allocation_canary
        scratch_prefix = bool((self._scratch_storage[:guard] == expected).all().item())
        scratch_suffix = bool(
            (self._scratch_storage[guard + self._scratch_bytes :] == expected)
            .all()
            .item()
        )
        peer_prefix = bool((self.peers.bytes[:guard] == expected).all().item())
        peer_suffix = bool(
            (self.peers.bytes[guard + self._peer_bytes :] == expected).all().item()
        )
        return {
            "enabled": True,
            "stage": stage,
            "guard_bytes": guard,
            "scratch_bytes": self._scratch_bytes,
            "peer_bytes": self._peer_bytes,
            "scratch_prefix": scratch_prefix,
            "scratch_suffix": scratch_suffix,
            "peer_prefix": peer_prefix,
            "peer_suffix": peer_suffix,
            "local_passed": all(
                (scratch_prefix, scratch_suffix, peer_prefix, peer_suffix)
            ),
        }
