"""Compile-only probe: a tiny kernel over every sync primitive, ISA dumped to
probe/asm/ (FLYDSL_DEBUG_DUMP_ASM). No GPU work is launched.

docker exec muse_mi455x bash -c 'cd /tmp && PYTHONSAFEPATH=1 python3 \
    /data/muse455/k2/cand/k2v4/port/sync/probe/compile_probe.py'
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # .../port
os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1250")
os.environ["COMPILE_ONLY"] = "1"
os.environ["FLYDSL_DEBUG_DUMP_ASM"] = "1"
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = os.path.join(HERE, "asm")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr import range_constexpr  # noqa: E402
from flydsl.expr.typing import Int32, Int64  # noqa: E402

from sync.abi import K2_DONE_BYTES, K2_DONE_OFFSET, TP  # noqa: E402
from sync.debug import region_ids  # noqa: E402
from sync.memmodel import load_ptr64  # noqa: E402
from sync.plan import THREADS  # noqa: E402
from sync.ranks import peer_bases  # noqa: E402
from sync.sync import Mailbox, import_alarms, preg, publish, sreg, stall  # noqa: E402


@flyc.kernel(name="sync_probe", known_block_size=[THREADS, 1, 1])
def probe(scratch: Int64, peers: Int64, rank: Int32, epoch: Int64, health: Int64, alarm: Int64):
    tid = fx.thread_idx.x
    stall(1, 3, 0, 1000, rank, epoch, 0)
    mb = Mailbox(fx.Int32(41), health, alarm, region_ids(["x", "d", "done"]), wait_ticks=25_000_000)
    bases = peer_bases(peers, TP)
    own = preg(load_ptr64(peers, rank), 0, "x", 4096)
    c = {"put": mb.put, "put_bf": mb.put_bf, "poll": mb.poll, "pt": mb.poll_terminal}
    for p in range_constexpr(TP):
        c["put_bf"](preg(bases[p], 0, "x", 4096), tid * 4, [fx.Float32(1.0)] * 4)
    got = c["poll"]([(own, tid * 2, 2)])
    d = sreg(scratch, 0, "d", 4096)
    publish(c["put"], d, 0, got[0][0], tid == 0)
    publish(c["put"], own, 9, got[0][1], tid == 0)
    head = alarm - fx.Int64(rank) * 4
    done = preg(head, K2_DONE_OFFSET, "done", K2_DONE_BYTES)
    if tid == 0:
        c["put"](done, rank, 1)
        c["pt"]([(done, r, 1) for r in range(TP)])
        import_alarms(health, alarm, rank, TP)


@flyc.jit
def launch(scratch: Int64, peers: Int64, rank: Int32, epoch: Int64, health: Int64, alarm: Int64,
           stream: fx.Stream = fx.Stream(None)):
    probe(scratch, peers, rank, epoch, health, alarm).launch(grid=(1,), block=(THREADS,), stream=stream)


if __name__ == "__main__":
    launch(0, 0, 0, 0, 0, 0)
    print("compiled; asm under", os.path.join(HERE, "asm"))
