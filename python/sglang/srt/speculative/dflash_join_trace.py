"""Debug instrumentation: a CPU and GPU timeline of DFLASH forward calls.

It shows what a request join (its prefill) costs the running decode batch: how long the
extend runs on the GPU, whether its launch is CPU-bound, and how long the GPU idles
before and after it. With SGLANG_DFLASH_JOIN_TRACE=<path>, TP rank 0 appends one line
per forward call:

    kind bs tokens cpu_begin cpu_end gpu_begin gpu_mid gpu_end rids

``kind`` is "extend" or "decode"; ``tokens`` the extend tokens (0 for decode). Times are
wall-clock seconds (``time.time()``, the clock of the benchmark client). The GPU times
come from events recorded on the current stream when the call begins, after the target
forward (``gpu_mid``, extend only, else -1) and when the call returns; one reference
event recorded at a device sync converts them to wall clock. ``rids`` is the
comma-separated request ids of an extend, and of a decode batch whose requests changed
since the last decode ("=" when unchanged). Events are read only once they have
completed, so tracing never waits on the GPU.
"""

from __future__ import annotations

import collections
import time
from typing import Optional

import torch

from sglang.srt.environ import envs


class JoinTrace:
    def __init__(self, path: str):
        # Line-buffered: the server is killed at the end of a benchmark, which drops a block buffer.
        self._file = open(path, "a", buffering=1)
        torch.cuda.synchronize()
        self._ref = torch.cuda.Event(enable_timing=True)
        self._ref.record()
        self._ref.synchronize()
        self._ref_wall = time.time()
        self._pending: collections.deque = collections.deque()
        self._open: Optional[list] = None
        self._decode_rids: Optional[tuple] = None

    @staticmethod
    def _event() -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def begin(self, *, kind: str, bs: int, tokens: int, rids: tuple) -> None:
        if kind == "decode":
            if rids == self._decode_rids:
                rids = ()
                label = "="
            else:
                self._decode_rids = rids
                label = ",".join(rids)
        else:
            label = ",".join(rids)
        self._open = [kind, bs, tokens, label, time.time(), self._event(), None]

    def mark(self) -> None:
        """The target forward of an extend has been launched."""
        if self._open is not None:
            self._open[6] = self._event()

    def end(self) -> None:
        if self._open is None:
            return
        self._pending.append((*self._open, time.time(), self._event()))
        self._open = None
        self._write_completed()

    def _wall(self, event: Optional[torch.cuda.Event]) -> float:
        if event is None:
            return -1.0
        return self._ref_wall + self._ref.elapsed_time(event) / 1e3

    def _write_completed(self) -> None:
        while self._pending and self._pending[0][-1].query():
            kind, bs, tokens, label, cpu_begin, ev_begin, ev_mid, cpu_end, ev_end = (
                self._pending.popleft()
            )
            self._file.write(
                f"{kind} {bs} {tokens} {cpu_begin:.6f} {cpu_end:.6f} "
                f"{self._wall(ev_begin):.6f} {self._wall(ev_mid):.6f} "
                f"{self._wall(ev_end):.6f} {label or '-'}\n"
            )


def make_join_trace(tp_rank: int) -> Optional[JoinTrace]:
    path = envs.SGLANG_DFLASH_JOIN_TRACE.get()
    return JoinTrace(path) if path and tp_rank == 0 else None
