"""4-rank test of the ported K2 sync layer on MI455X (gfx1250), inside a
persistent 256-CTA kernel: the tagged mailbox (device and system scope),
a 4-rank barrier, a ping-pong hop, a 128 KB-a-peer all-to-all (K2's EP1 yr
exchange shape), ``publish`` (device-scope plain data + flag), epoch tagging
across eager steps and HIP-graph replays, and fail-stop (a dead rank and a
stalled rank must make every kernel return with FAILED health, not hang).

One process a GPU (torch.distributed gloo + IPC peer buffers: ``PeerRuntime``).

flock /data/muse455/locks/gpu0.lock flock /data/muse455/locks/gpu1.lock \
  flock /data/muse455/locks/gpu2.lock flock /data/muse455/locks/gpu3.lock \
  docker exec muse_mi455x bash -c 'cd /tmp && PYTHONSAFEPATH=1 timeout 300 \
  python3 /data/muse455/k2/cand/k2v6/port/sync/test_sync_4rank.py'

Results: test_sync_4rank.json next to this file (and stdout).
"""

# No ``from __future__ import annotations``: FlyDSL resolves a string annotation
# in the module globals only, and these kernels import ``Int64`` inside
# ``make``; an unresolved annotation falls back to the value's type (a Python
# int -> i32), which truncates every pointer argument.
import json
import os
import socket
import statistics
import sys
import time
import traceback
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # .../port: ``import sync``
os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1250")

TP = 4
OUT = os.path.join(HERE, "test_sync_4rank.json")
RES_DIR = os.path.join(HERE, "results")


@dataclass(frozen=True)
class Build:
    wait_ticks: int = 200_000_000  # 2 s
    bar_rounds: int = 32
    pp_rounds: int = 32
    payload: int = 32768  # words a peer (= 128 KB of bf16: K2 EP1 yr per peer)
    publish_fence: bool = True
    fault_rank: int = -1
    fault_epoch: int = 0
    fault_ticks: int = 0


TL = 8  # int64 timeline words a CTA


def make(key: Build):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr import const_expr, gpu, range_constexpr
    from flydsl.expr.typing import Int32, Int64, T

    from sync.abi import K2_DONE_BYTES, K2_DONE_OFFSET
    from sync.debug import region_ids
    from sync.epoch import EPOCH_TAG_PERIOD, TAG_SLOTS
    from sync.memmodel import (
        CM_DEV,
        bf16_pair,
        ld_i32,
        load_ptr64,
        memrealtime,
        rsrc,
        uniform,
        wait_stores,
    )
    from sync.plan import BLOCKS, THREADS, key_tuple, pair_layout
    from sync.ranks import peer_bases
    from sync.sync import Mailbox, import_alarms, preg, publish, shift, sreg, stall

    G, P = BLOCKS, key.payload
    W = P // G  # words a peer a CTA
    assert P % (2 * G) == 0 and W // 2 <= 64
    BR, PP = key.bar_rounds, key.pp_rounds
    play = pair_layout(
        (("bar", max(BR, 1) * TP), ("pp", max(PP, 1)), ("x2x", TP * max(P, 1))), align=256
    )
    half = (max(o + n for o, n in play.values()) + 255) // 256 * 256
    slay = pair_layout((("go", 1), ("prdy", G), ("finish", G)), align=16)
    pdata_off = (max(o + n for o, n in slay.values()) + 255) // 256 * 256
    scratch_bytes = pdata_off + G * THREADS * 4
    rids = region_ids(("bar", "pp", "x2x", "go", "prdy", "finish", "done"))
    keyed = key_tuple(key, "test_sync_4rank")
    name = (
        f"k2sync_t4_w{key.wait_ticks}_b{BR}_p{PP}_x{P}_f{int(key.publish_fence)}"
        + (f"_fr{key.fault_rank}_{key.fault_epoch}_{key.fault_ticks}" if key.fault_rank >= 0 else "")
    )

    def expect(src, g, ep):
        """bf16-exact payload element g of rank src at epoch ep."""
        return (((src * 37 + g * 11 + ep * 5) & 255) - 128).to(fx.Float32)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k(
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        err: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        t_start = memrealtime()
        if const_expr(key.fault_rank >= 0):
            stall(key.fault_rank, key.fault_epoch, 0, key.fault_ticks, rank, epoch, layer)
        ep = uniform(ld_i32(epoch, 0))
        tag = (ep % EPOCH_TAG_PERIOD) * TAG_SLOTS + layer + 1
        mb = Mailbox(tag - 1, health, alarm, rids, wait_ticks=key.wait_ticks)
        par = ep % 2
        poff = fx.Int64(par) * half
        bases = peer_bases(peers, TP)
        own = load_ptr64(peers, rank)
        peer = {
            n: [shift(preg(b, play[n][0], n, play[n][1]), poff) for b in bases] for n in play
        }
        mine = {n: shift(preg(own, play[n][0], n, play[n][1]), poff) for n in play}
        sc = {n: sreg(scratch, slay[n][0], n, slay[n][1]) for n in slay}
        done = preg(alarm - fx.Int64(rank) * 4, K2_DONE_OFFSET, "done", K2_DONE_BYTES)
        pdata = scratch + fx.Int64(pdata_off)
        me = bid * THREADS + tid
        c = {
            "put": mb.put,
            "put_bf": mb.put_bf,
            "poll": mb.poll,
            "poll_terminal": mb.poll_terminal,
        }

        # ---- 4-rank barrier rounds, then a 0 <-> 1 ping-pong (CTA 0, one thread)
        if (bid == 0) & (tid == 0):
            bad = fx.Int32(0)
            t0 = memrealtime()
            for r in range_constexpr(BR):
                if const_expr(r == 1):
                    t0 = memrealtime()  # round 0 absorbs the ranks' launch skew
                for p in range_constexpr(TP):
                    c["put"](peer["bar"][p], r * TP + rank, fx.Int32(r + 1000))
                got = c["poll"]([(mine["bar"], r * TP + q, 1) for q in range(TP)])
                for q in range_constexpr(TP):
                    bad = bad + (got[q][0] != r + 1000).select(fx.Int32(1), fx.Int32(0))
            t1 = memrealtime()
            if rank == 0:
                for r in range_constexpr(PP):
                    c["put"](peer["pp"][1], r, fx.Int32(r))
                    got = c["poll"]([(mine["pp"], r, 1)])
                    bad = bad + (got[0][0] != r).select(fx.Int32(1), fx.Int32(0))
            if rank == 1:
                for r in range_constexpr(PP):
                    got = c["poll"]([(mine["pp"], r, 1)])
                    bad = bad + (got[0][0] != r).select(fx.Int32(1), fx.Int32(0))
                    c["put"](peer["pp"][0], r, fx.Int32(r))
            t2 = memrealtime()
            c["put"](sc["go"], 0, fx.Int32(1))
            bo.buffer_store(ld_i32(err, me) + bad, rsrc(err), me)
            bo.buffer_store(fx.Int64(t1 - t0), rsrc(tl), bid * TL + 3)
            bo.buffer_store(fx.Int64(t2 - t1), rsrc(tl), bid * TL + 4)
        if tid == 0:
            c["poll"]([(sc["go"], 0, 1)])
        gpu.barrier()

        # ---- all-to-all: W words a peer a CTA, 2 words (4 bf16) a thread
        tx0 = memrealtime()
        if const_expr(P > 0):
            if tid < TP * 64:
                pu = uniform(tid // 64)  # two waves a peer: wave-uniform
                kk = tid % 64
                if kk < W // 2:
                    w = bid * W + 2 * kk
                    vals = [expect(rank, 2 * w + e, ep) for e in range(4)]
                    dst = bases[0]
                    for p in range_constexpr(1, TP):
                        dst = (pu == p).select(bases[p], dst)
                    c["put_bf"](
                        shift(preg(dst, play["x2x"][0], "x2x", play["x2x"][1]), poff),
                        (rank * P + w) * 2,
                        vals,
                    )
                    got = c["poll"]([(mine["x2x"], pu * P + w, 2)])
                    e0 = bf16_pair(expect(pu, 2 * w, ep), expect(pu, 2 * w + 1, ep))
                    e1 = bf16_pair(expect(pu, 2 * w + 2, ep), expect(pu, 2 * w + 3, ep))
                    bad = (got[0][0] != e0.bitcast(fx.Int32)).select(
                        fx.Int32(1), fx.Int32(0)
                    ) + (got[0][1] != e1.bitcast(fx.Int32)).select(fx.Int32(1), fx.Int32(0))
                    bo.buffer_store(ld_i32(err, me) + bad, rsrc(err), me)
            gpu.barrier()
        tx1 = memrealtime()

        # ---- publish: plain stores, flag; the next CTA reads them at device scope
        val = bid * 65536 + tid + ep * 7
        bo.buffer_store(fx.Int32(val), rsrc(pdata), me)
        tp0 = memrealtime()
        if const_expr(key.publish_fence):
            publish(c["put"], sc["prdy"], bid, fx.Int32(1), tid == 0)
        else:  # publish without the release fence (latency comparison only)
            wait_stores()
            gpu.barrier()
            if tid == 0:
                c["put"](sc["prdy"], bid, fx.Int32(1))
        tp1 = memrealtime()
        src = (bid + G - 1) % G
        if tid == 0:
            c["poll"]([(sc["prdy"], src, 1)])
        gpu.barrier()
        tp2 = memrealtime()
        seen = fx.Int32(
            bo.buffer_load(
                rsrc(pdata), src * THREADS + tid, vec_width=1, dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )  # fmt: skip
        bad = (seen != src * 65536 + tid + ep * 7).select(fx.Int32(1), fx.Int32(0))
        bo.buffer_store(ld_i32(err, me) + bad, rsrc(err), me)

        # ---- the launch tail: local fan-in, then the 4-rank completion pairs
        # The realtime counter is not common to the CTAs of a GPU on gfx1250
        # (ms apart), so only CTA-local differences are recorded, plus tx0 as
        # an absolute value to report that spread.
        t_end = memrealtime()
        if tid == 0:
            bo.buffer_store(fx.Int64(t_end - t_start), rsrc(tl), bid * TL + 0)
            bo.buffer_store(fx.Int64(tx0), rsrc(tl), bid * TL + 1)
            bo.buffer_store(fx.Int64(tx1 - tx0), rsrc(tl), bid * TL + 2)
            bo.buffer_store(fx.Int64(tp1 - tp0), rsrc(tl), bid * TL + 5)
            bo.buffer_store(fx.Int64(tp2 - tp1), rsrc(tl), bid * TL + 6)
        publish(c["put"], sc["finish"], bid, fx.Int32(1), tid == 0)
        if bid == 0:
            if tid < G:
                c["poll_terminal"]([(sc["finish"], tid, 1)])
            gpu.barrier()
            if tid == 0:
                c["put"](done, rank, fx.Int32(1))
                wait_stores()
                c["poll_terminal"]([(done, r, 1) for r in range(TP)])
                import_alarms(health, alarm, rank, TP)
                bo.buffer_store(fx.Int64(memrealtime() - t_end), rsrc(tl), 7)

    @flyc.jit
    def launch(
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        health: Int64,
        alarm: Int64,
        err: Int64,
        tl: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = keyed
        k(scratch, peers, rank, epoch, layer, health, alarm, err, tl).launch(
            grid=(G,), block=(THREADS,), stream=stream
        )

    return launch, 2 * half, scratch_bytes


# ------------------------------------------------------------------ host
def _timeline(tl, key: Build):
    """Per-launch latencies (us) from one rank's [G][TL] int64 timeline."""
    t = tl.view(-1, TL).cpu().double()
    out = {
        "exchange_us_max_cta": float(t[:, 2].max()) / 100,
        "exchange_us_median_cta": float(t[:, 2].median()) / 100,
        "publish_us_median": float(t[:, 5].median()) / 100,
        "publish_wait_us_median": float(t[:, 6].median()) / 100,
        "cta_body_us_max": float(t[:, 0].max()) / 100,
        "terminal_us_cta0": float(t[0, 7]) / 100,
        "clock_spread_us": float(t[:, 1].max() - t[:, 1].min()) / 100,
    }
    if key.bar_rounds > 1:
        out["barrier_us_per_round"] = float(t[0, 3]) / (key.bar_rounds - 1) / 100
    if key.pp_rounds:
        out["pingpong_us_per_hop"] = float(t[0, 4]) / (2 * key.pp_rounds) / 100
    return out


def worker(rank: int, port: int):
    import torch
    import torch.distributed as dist

    os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0,1,2,3")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP
    )
    group = dist.group.WORLD
    res: dict = {"rank": rank}

    import flydsl.expr as fx

    from sync.debug import given_up_waits
    from sync.epoch import advance_epoch_
    from sync.peer import PeerRuntime
    from sync.plan import BLOCKS, THREADS

    regions = ("bar", "pp", "x2x", "go", "prdy", "finish", "done")

    def summarize(health):
        """The wait records: how many CTAs gave up on each region, CTA 0's."""
        lines = given_up_waits(health, regions)
        by = {}
        for ln in lines:
            reg = ln.split(": ", 1)[1].split(" pair")[0]
            by[reg] = by.get(reg, 0) + 1
        return {
            "ctas": len(lines),
            "by_region": by,
            "cta0": next((ln for ln in lines if " CTA 0:" in ln), None),
        }
    full = Build()
    lean = Build(wait_ticks=100_000_000, bar_rounds=1, pp_rounds=0)
    nofence = Build(wait_ticks=100_000_000, bar_rounds=1, pp_rounds=0, publish_fence=False)
    builds = {"full": full, "lean": lean, "nofence": nofence}
    t0 = time.time()
    made = {n: make(b) for n, b in builds.items()}
    peer_bytes = max(m[1] for m in made.values())
    scratch_bytes = max(m[2] for m in made.values())
    rt = PeerRuntime(peer_bytes, group, rank, TP, dev, blocks=BLOCKS)
    scratch = torch.zeros(scratch_bytes, dtype=torch.uint8, device=dev)
    err = torch.zeros(BLOCKS * THREADS, dtype=torch.int32, device=dev)
    tl = torch.zeros(BLOCKS * TL, dtype=torch.int64, device=dev)

    def run(launch, stream=None):
        s = stream if stream is not None else torch.cuda.current_stream()
        launch(
            scratch.data_ptr(), rt.peers.data_ptr(), rank, rt.epoch.data_ptr(), 0,
            rt.health.data_ptr(), rt.alarm, err.data_ptr(), tl.data_ptr(),
            stream=fx.Stream(s.cuda_stream),
        )  # fmt: skip

    def state():
        torch.cuda.synchronize()
        st = {
            "errors": int(err.sum()),
            "failed": int(rt.health[0]),
            "epoch": int(rt.epoch[0]),
        }
        if rank == 0:
            st["alarm_words"] = rt.error_words().cpu().tolist()
        return st

    def reset():
        torch.cuda.synchronize()
        dist.barrier()
        rt.health.zero_()
        err.zero_()
        if rank == 0:
            rt.error_words().zero_()
        torch.cuda.synchronize()
        dist.barrier()

    # compile every build once (a compile-only call keeps no GPU busy)
    os.environ["COMPILE_ONLY"] = "1"
    for n, (launch, _, _) in made.items():
        run(launch)
    del os.environ["COMPILE_ONLY"]
    res["compile_s"] = round(time.time() - t0, 1)
    dist.barrier()

    # ---- 1. eager steps of the full build: exactness + latencies
    lat = []
    for _ in range(20):
        dist.barrier()
        run(made["full"][0])
        torch.cuda.synchronize()
        lat.append(_timeline(tl, full))
        advance_epoch_(rt.epoch)
    st = state()
    res["eager_full"] = {
        **st,
        "steps": 20,
        **{k: statistics.median(d[k] for d in lat) for k in lat[0]},
        "pingpong_min": min(d.get("pingpong_us_per_hop", 0) for d in lat),
        "barrier_min": min(d["barrier_us_per_round"] for d in lat),
        "exchange_min": min(d["exchange_us_max_cta"] for d in lat),
    }
    if rank == 0:
        rt.check()  # raises on any alarm word

    # ---- 2. graph capture of the lean build (K2-shaped launch) + epoch bump
    reset()
    lean_launch = made["lean"][0]
    run(lean_launch)  # one eager step on every rank before capture
    advance_epoch_(rt.epoch)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    cap = torch.cuda.Stream()
    with torch.cuda.graph(g, stream=cap):
        run(lean_launch, cap)
        advance_epoch_(rt.epoch)
    e0 = int(rt.epoch[0])
    n_rep = 500
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dist.barrier()
    ev0.record()
    for _ in range(n_rep):
        g.replay()
    ev1.record()
    torch.cuda.synchronize()
    st = state()
    res["graph_lean"] = {
        **st,
        "replays": n_rep,
        "epoch_advanced": st["epoch"] - e0,
        "us_per_step": ev0.elapsed_time(ev1) * 1000 / n_rep,
        "last_step": _timeline(tl, lean),
    }
    if rank == 0:
        rt.check()
    del g
    # the epoch bump alone, to subtract from the step
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, stream=cap):
        advance_epoch_(rt.epoch)
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(n_rep):
        g2.replay()
    ev1.record()
    torch.cuda.synchronize()
    res["graph_lean"]["epoch_bump_only_us"] = ev0.elapsed_time(ev1) * 1000 / n_rep
    del g2

    # ---- 3. eager lean vs lean-without-fence: the publish release's cost
    for n in ("lean", "nofence"):
        reset()
        lat = []
        for _ in range(20):
            dist.barrier()
            run(made[n][0])
            torch.cuda.synchronize()
            lat.append(_timeline(tl, builds[n]))
            advance_epoch_(rt.epoch)
        res[f"eager_{n}"] = {
            **state(),
            **{k: statistics.median(d[k] for d in lat) for k in lat[0]},
        }

    # ---- 4. fail-stop: a dead rank (rank 3 never launches)
    reset()
    fa = Build(wait_ticks=25_000_000, bar_rounds=1, pp_rounds=0)
    fa_launch = make(fa)[0]
    os.environ["COMPILE_ONLY"] = "1"
    run(fa_launch)
    del os.environ["COMPILE_ONLY"]
    dist.barrier()
    w0 = time.time()
    if rank != TP - 1:
        run(fa_launch)
    torch.cuda.synchronize()
    wall = time.time() - w0
    advance_epoch_(rt.epoch)
    st = state()
    raised = None
    if rank == 0:
        try:
            rt.check()
        except RuntimeError as e:
            raised = str(e)
    res["failstop_dead_rank"] = {
        **st,
        "returned_after_s": round(wall, 3),
        "wait_budget_s": fa.wait_ticks / 1e8,
        "check_raised": raised,
        "given_up": summarize(rt.health.cpu()),
    }

    # ---- 5. fail-stop: CTA 0 of rank 1 stalls 0.4 s (budget 0.25 s)
    reset()
    fb = Build(
        wait_ticks=25_000_000, bar_rounds=1, pp_rounds=0,
        fault_rank=1, fault_epoch=int(rt.epoch[0]), fault_ticks=40_000_000,
    )  # fmt: skip
    fb_launch = make(fb)[0]
    os.environ["COMPILE_ONLY"] = "1"
    run(fb_launch)
    del os.environ["COMPILE_ONLY"]
    dist.barrier()
    w0 = time.time()
    run(fb_launch)
    torch.cuda.synchronize()
    wall = time.time() - w0
    advance_epoch_(rt.epoch)
    st = state()
    raised = None
    if rank == 0:
        try:
            rt.check()
        except RuntimeError as e:
            raised = str(e)
    res["failstop_stall"] = {
        **st,
        "returned_after_s": round(wall, 3),
        "stall_s": fb.fault_ticks / 1e8,
        "wait_budget_s": fb.wait_ticks / 1e8,
        "check_raised": raised,
        "given_up": summarize(rt.health.cpu()),
    }

    # ---- 6. a clean step after reset: nothing stale leaks across the incident
    reset()
    dist.barrier()
    run(made["full"][0])
    advance_epoch_(rt.epoch)
    res["after_reset"] = state()
    if rank == 0:
        rt.check()

    # ---- residency of the full build
    if rank == 0:
        try:
            from sync.residency import residency

            cus = torch.cuda.get_device_properties(rank).multi_processor_count
            res["residency"] = [
                {**r.__dict__, "fits_256": r.fits(BLOCKS, cus), "cus": cus}
                for r in residency(made["full"][0], THREADS)
            ]
        except Exception as e:  # noqa: BLE001 - reported, not fatal
            res["residency"] = f"error: {e!r}"

    torch.cuda.synchronize()
    dist.barrier()
    rt.close()
    with open(os.path.join(RES_DIR, f"rank{rank}.json"), "w") as f:
        json.dump(res, f, indent=1)
    dist.destroy_process_group()


def _main(rank, port):
    try:
        worker(rank, port)
    except Exception:
        traceback.print_exc()
        with open(os.path.join(RES_DIR, f"rank{rank}.json"), "w") as f:
            json.dump({"rank": rank, "error": traceback.format_exc()}, f, indent=1)
        os._exit(1)


def main():
    import torch.multiprocessing as mp

    os.makedirs(RES_DIR, exist_ok=True)
    for r in range(TP):
        p = os.path.join(RES_DIR, f"rank{r}.json")
        if os.path.exists(p):
            os.remove(p)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    t0 = time.time()
    mp.spawn(_main, args=(port,), nprocs=TP, join=True)
    ranks = []
    for r in range(TP):
        with open(os.path.join(RES_DIR, f"rank{r}.json")) as f:
            ranks.append(json.load(f))
    out = {"wall_s": round(time.time() - t0, 1), "ranks": ranks}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def compile_check():
    """Trace and compile every build without a GPU launch (no locks needed)."""
    os.environ["COMPILE_ONLY"] = "1"
    for key in (
        Build(),
        Build(bar_rounds=1, pp_rounds=0),
        Build(bar_rounds=1, pp_rounds=0, publish_fence=False),
        Build(wait_ticks=25_000_000, bar_rounds=1, pp_rounds=0, fault_rank=1,
              fault_epoch=5, fault_ticks=40_000_000),
    ):  # fmt: skip
        t0 = time.time()
        launch, pb, sb = make(key)
        launch(*([0] * 9))
        print(f"compiled {key} in {time.time() - t0:.1f}s: peer {pb} B, scratch {sb} B")


if __name__ == "__main__":
    if "--compile" in sys.argv:
        compile_check()
    else:
        main()
