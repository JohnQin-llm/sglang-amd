"""Serving-like K2 stress: many layers, varied inputs every step, a long-context attention-like kernel before every
K2 launch, graph replay, every give-up counted and decoded.

Per rank: the layers' real dense weights (validation.LayerData) and all 256 EP1 experts (read from the checkpoint
as the served runner does), a case pool per layer (make_cases.py: synthetic verify rows + adversarial cases with a
chosen number of distinct experts nu). Phases:

  sweep : eager, one launch per (sweep layer, case), every case of the pool (nu = 63 included); give-ups decoded.
  serve : one CUDA graph a step = epoch advance + for every layer (attention-like over --ctx tokens, then K2 on that
          layer's input buffer, then a weighted int64 digest of its outputs). Before each replay the layers' cases
          are picked (host RNG, same seed on every rank -> same picks) into the graph's index tensor; nu = 63 cases
          only at --inject chosen steps (one layer each). After each replay: synchronize, this rank's FAILED word
          and (rank 0) the output rank's flag words, a gloo all-reduce; on a give-up every rank decodes its records
          (port.sync.debug.given_up_waits), the step is logged, health and flags are reset and the run goes on.

Output: <out>/<tag>.json (counts, give-ups, step-time stats, digest of all non-failed steps) and <tag>_dig.pt.
Run under k2safe (with or without -w mpol_exec.py):
  k2safe.sh -t 1500 <tree> ../k2v4b/stress/serve_stress.py --tree <tree> --tag x ...
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/data/Muse-Spark-Open-816B-A42B-Quark-MXFP4"
PREFIX = "model.language_model.layers.{}.mlp.experts.{}."


def now():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def load_pools(layers):
    pools = {}
    for f in sorted(glob.glob(os.path.join(HERE, "pool_*.pt"))):
        for k, v in torch.load(f).items():
            pools[int(k)] = v
    missing = [l for l in layers if l not in pools]
    assert not missing, f"no case pool for layers {missing}"
    return {l: pools[l] for l in layers}


class Ckpt:
    def __init__(self, root=CKPT):
        from safetensors import safe_open

        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self._open, self._files = safe_open, {}

    def slice(self, name, dim, start, length):
        fn = self.map[name]
        if fn not in self._files:
            self._files[fn] = self._open(os.path.join(self.root, fn), "pt")
        sl = self._files[fn].get_slice(name)
        idx = [slice(None)] * len(sl.get_shape())
        idx[dim] = slice(start, start + length)
        return sl[tuple(idx)]


def ep1_experts(ck, layer, rank, dev, chunk=32):
    """All 256 experts of ``layer``, rank's intermediate quarter, gfx1250 layout (the served runner's loader)."""
    from port import consts as K
    from port import layout as L

    ri, E, MH = K.RI, K.E, K.MH
    shapes = ((E, 2 * ri, MH // 2), (E, 2 * ri, MH // 32), (E, MH, ri // 2), (E, MH, ri // 32))
    out = [torch.empty(s, dtype=torch.uint8, device=dev) for s in shapes]
    for c0 in range(0, E, chunk):
        ids = list(range(c0, min(E, c0 + chunk)))
        host = [torch.empty((len(ids),) + s[1:], dtype=torch.uint8) for s in shapes]
        for j, e in enumerate(ids):
            p = PREFIX.format(layer, e)
            host[0][j, :ri] = ck.slice(p + "gate_proj.weight", 0, rank * ri, ri)
            host[0][j, ri:] = ck.slice(p + "up_proj.weight", 0, rank * ri, ri)
            host[1][j, :ri] = ck.slice(p + "gate_proj.weight_scale", 0, rank * ri, ri)
            host[1][j, ri:] = ck.slice(p + "up_proj.weight_scale", 0, rank * ri, ri)
            host[2][j] = ck.slice(p + "down_proj.weight", 1, rank * ri // 2, ri // 2)
            host[3][j] = ck.slice(p + "down_proj.weight_scale", 1, rank * ri // 32, ri // 32)
        parts = L.gfx1250_weights(*(h.to(dev) for h in host))
        for dst, src in zip(out, parts):
            dst[c0:c0 + len(ids)].copy_(src)
        del parts
    return tuple(t.contiguous() for t in out)


def dense_operands(lay, rank):
    from port import consts as K
    from port.gemv import gemv

    shuf = gemv.shuffle_rows if gemv.SHUFFLED else (lambda t: t)
    w = lay.w
    return {
        "w_o": shuf(w["w_o"][:, rank * K.CORE:(rank + 1) * K.CORE].contiguous()),
        "alpha_a": w["alpha_a"], "beta_a": w["beta_a"], "w_fn": w["w_fn"].contiguous(),
        "w_router": w["w_router"].float().contiguous(), "expert_bias": w["bias"].float().contiguous(),
        "w_pool": shuf(w["w_pool"].contiguous()), "w_pre": w["w_pre"].contiguous(),
        "w_post": w["w_post"].contiguous(), "w_unpool": shuf(w["w_unpool"].contiguous()),
    }  # fmt: skip


def worker(rank, args, _):
    sys.path.insert(0, args.tree)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"file:///tmp/k2stress_pg_{args.tag}_{args.stamp}", rank=rank,
                            world_size=4)
    from port import consts as K
    from port import k2
    from port import validation as V
    from port.sync.debug import given_up_waits
    from port.sync.epoch import advance_epoch_
    from port.sync.peer import PeerRuntime
    from port.sync import sync as S

    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    T = K.MAX_TOKENS
    PF = dict(pf_at="fin", pf_touch=2, pf_tat="route", pf_chunk=8192, pf_cpol=17)  # k2v2pf's recommended set
    key = k2.K2Build(**PF) if args.pf else k2.K2Build()
    tb = time.time()
    k2.build(key)
    log(f"tree {args.tree} k2 {k2.__file__} sources {k2._SOURCES[:12]} build {time.time() - tb:.0f} s; "
        f"sleep {S._POLL_SLEEP} fast {getattr(S, '_FAST_SITES', None)} fast_polls {getattr(S, '_FAST_POLLS', None)} "
        f"ugw {getattr(k2, '_VR_UGW', None)} wait_ticks {S.WAIT_TICKS}")
    rt = PeerRuntime(k2.peer_bytes(), dist.group.WORLD, rank, K.TP, dev, blocks=K.BLOCKS)
    scratch = torch.zeros(k2.scratch_bytes() // 4, dtype=torch.int32, device=dev)
    pools = load_pools(args.layers)
    ck = Ckpt()

    # ---- per layer: operands, pool tensors, static input / output buffers
    tl0 = time.time()
    L = {}
    for n, layer in enumerate(args.layers):
        lay = V.LayerData(layer, dev)
        ops = dense_operands(lay, rank)
        ops["w13"], ops["w13s"], ops["w2"], ops["w2s"] = ep1_experts(ck, layer, rank, dev)
        del lay
        cases = pools[layer]
        reg = [i for i, c in enumerate(cases) if c["nu"] != 63]
        bad = [i for i, c in enumerate(cases) if c["nu"] == 63]
        core = torch.stack([c["core"][:, rank * K.CORE:(rank + 1) * K.CORE] for c in cases]).to(dev).contiguous()
        stream = torch.stack([c["stream_a"] for c in cases]).float().to(dev).contiguous()
        L[layer] = dict(ops=ops, core_pool=core, stream_pool=stream, nus=[c["nu"] for c in cases], reg=reg, bad=bad,
                        core=torch.empty(T, K.CORE, dtype=torch.bfloat16, device=dev),
                        stream=torch.empty(T, K.H, dtype=torch.float32, device=dev),
                        so=torch.zeros(T, K.H, dtype=torch.float32, device=dev),
                        br=torch.zeros(T, K.H, dtype=torch.bfloat16, device=dev),
                        # --pf: the next layer's QKV weight stand-in (bf16 [4608, 8192] = 75,497,472 B: 72 tiles of 64 rows), only read
                        qkv=torch.randn(4608, 8192, device=dev).to(torch.bfloat16) if args.pf else None)
        if (n + 1) % 4 == 0:
            log(f"{n + 1}/{len(args.layers)} layers loaded ({time.time() - tl0:.0f} s)")
    torch.cuda.synchronize()
    log(f"pools: " + ", ".join(f"L{l}: {len(L[l]['nus'])} cases nu {min(L[l]['nus'])}..{max(L[l]['nus'])}, "
                                f"{len(L[l]['bad'])} x nu63" for l in args.layers[:3]) + " ...")

    def launch(layer):
        d = L[layer]
        t = dict(d["ops"], core=d["core"], stream_a=d["stream"], stream_out=d["so"], branch_out=d["br"])
        kw = {"pf": ((d["qkv"], 75497472), (0, 0))} if args.pf else {}
        k2.k2_launch(key, t, rank, layer, rt.epoch, rt.peers, scratch, rt.health, rt.alarm,
                     torch.cuda.current_stream().cuda_stream, **kw)

    flags_host = torch.zeros(1 + K.TP, dtype=torch.int32, pin_memory=True)
    err = rt.error_words()

    def failed_now():
        """(after a sync) this rank's FAILED word, and on rank 0 the output rank's flags, all-reduced."""
        flags_host[:1].copy_(rt.health[:1])
        if rank == 0:
            flags_host[1:].copy_(err)
        f = torch.tensor([int(flags_host[0] != 0) | (int(flags_host[1:].any()) if rank == 0 else 0)])
        dist.all_reduce(f, op=dist.ReduceOp.MAX)
        return bool(f.item())

    giveups = []

    def handle(where, picks):
        """Every rank's records, then reset health / flags for the next step."""
        torch.cuda.synchronize()
        dist.barrier()
        recs = given_up_waits(rt.health.cpu(), k2.REGIONS)
        raw = rt.health.cpu().view(-1, 8)[1:]
        raw = raw[raw[:, 0] != 0].tolist()
        tags = sorted({r[3] for r in raw})
        info = {"rank": rank, "failed_word": int(rt.health[0].item()), "n_ctas": len(raw), "first": recs[:6],
                "tags": tags, "layers": sorted({(t - 1) % 256 - k2.K2_SLOT0 for t in tags}),
                "regions": sorted({r.split(": ")[1].split(" pair")[0] for r in recs}),
                "ctas": sorted({r[5] for r in raw})[:12]}
        allinfo = [None] * 4
        dist.all_gather_object(allinfo, info)
        if rank == 0:
            ev = {"where": where, "flags": err.cpu().tolist(), "picks": picks, "ranks": allinfo}
            giveups.append(ev)
            log(f"GIVE-UP {where}: flags {ev['flags']} picks {picks}")
            for a in allinfo:
                log(f"  rank {a['rank']}: {a['n_ctas']} CTAs, regions {a['regions']}, layers {a['layers']}, "
                    f"CTAs {a['ctas']}; first: {a['first'][:3]}")
        torch.cuda.synchronize()
        dist.barrier()
        rt.health.zero_()
        if rank == 0:
            err.zero_()
        torch.cuda.synchronize()
        dist.barrier()

    W = torch.randint(-(1 << 20), 1 << 20, (T * K.H,), generator=torch.Generator().manual_seed(5), dtype=torch.int64)
    W = W.to(dev)

    def digest_into(out, layer):
        d = L[layer]
        out[0] = (d["so"].view(torch.int32).view(-1).long() * W).sum()
        out[1] = (d["br"].view(torch.int16).view(-1).long() * W).sum()

    # ---- sweep (eager, every case of the sweep layers)
    sweep = []
    for layer in args.sweep_layers:
        d = L[layer]
        for i in sorted(range(len(d["nus"])), key=lambda i: d["nus"][i]):
            d["core"].copy_(d["core_pool"][i])
            d["stream"].copy_(d["stream_pool"][i])
            dist.barrier()
            advance_epoch_(rt.epoch)
            t0 = now()
            launch(layer)
            torch.cuda.synchronize()
            dt = (now() - t0) / 1e3
            dg = torch.zeros(2, dtype=torch.int64, device=dev)
            digest_into(dg, layer)
            f = failed_now()
            row = {"layer": layer, "case": i, "nu": d["nus"][i], "gave_up": f, "us": round(dt, 1),
                   "digest": dg.cpu().tolist()}
            if f:
                handle(f"sweep L{layer} case {i} nu {d['nus'][i]}", {layer: [i, d["nus"][i]]})
            sweep.append(row)
        log(f"sweep L{layer}: " + " ".join(f"{r['nu']}{'X' if r['gave_up'] else ''}" for r in sweep
                                            if r["layer"] == layer))

    # ---- serve loop
    res = {"args": vars(args), "sweep": sweep}
    if args.steps:
        nl = len(args.layers)
        kv = torch.randn(2, 16, max(args.ctx, 1), 128, generator=torch.Generator(device=dev).manual_seed(3),
                         device=dev, dtype=torch.bfloat16)
        idx_host = torch.zeros(nl, dtype=torch.int64, pin_memory=True)
        idx_dev = torch.zeros(nl, dtype=torch.int64, device=dev)
        chk = torch.zeros(nl, 2, dtype=torch.int64, device=dev)
        attn_out = torch.zeros(16, T, 128, dtype=torch.bfloat16, device=dev)

        def attention(layer):
            # decode attention of the 8 verify rows over --ctx cached tokens (16 heads of 128): ~2 x 16 ctx 256 B read
            q = L[layer]["core"].view(T, 16, 128).transpose(0, 1)
            s = torch.bmm(kv[0], q.transpose(1, 2)).float() * (128 ** -0.5)
            p = torch.softmax(s, dim=1).to(torch.bfloat16)
            attn_out.copy_(torch.bmm(p.transpose(1, 2), kv[1]))

        def step_body():
            advance_epoch_(rt.epoch)
            for j, layer in enumerate(args.layers):
                d = L[layer]
                d["core"].copy_(d["core_pool"].index_select(0, idx_dev[j:j + 1])[0])
                d["stream"].copy_(d["stream_pool"].index_select(0, idx_dev[j:j + 1])[0])
                if args.ctx:
                    attention(layer)
                launch(layer)
                digest_into(chk[j], layer)

        # first picks so the warm-up and capture run real cases
        for j, layer in enumerate(args.layers):
            idx_host[j] = L[layer]["reg"][0]
        idx_dev.copy_(idx_host)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            step_body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        assert not failed_now(), "give-up in the warm-up step"
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step_body()
        torch.cuda.synchronize()
        assert not failed_now(), "give-up in the capture"

        rng = torch.Generator().manual_seed(args.seed)
        inj_g = torch.Generator().manual_seed(args.seed + 1)
        inject = {}
        if args.inject:
            steps = torch.randperm(args.steps - 100, generator=inj_g)[:args.inject] + 100
            for s_ in sorted(steps.tolist()):
                inject[s_] = args.layers[int(torch.randint(nl, (1,), generator=inj_g))]
        dig = torch.zeros(args.steps, nl, 2, dtype=torch.int64, device=dev)
        st_us = []
        failed_steps = []
        sticky_until, sticky_done, sticky_steps = None, False, []
        dist.barrier()
        t_start = time.time()
        for step in range(args.steps):
            picks = {}
            for j, layer in enumerate(args.layers):
                d = L[layer]
                if inject.get(step) == layer and d["bad"]:
                    i = d["bad"][int(torch.randint(len(d["bad"]), (1,), generator=rng))]
                else:
                    i = d["reg"][int(torch.randint(len(d["reg"]), (1,), generator=rng))]
                idx_host[j] = i
                picks[layer] = i
            idx_dev.copy_(idx_host, non_blocking=True)
            t0 = now()
            g.replay()
            dig[step].copy_(chk)
            torch.cuda.synchronize()
            st_us.append((now() - t0) / 1e3)
            if sticky_until is not None:
                sticky_steps.append(step)  # serving: no reset, the sticky FAILED word stays set
                if step == sticky_until:
                    handle(f"end of sticky window at step {step}", {})
                    sticky_until = None
                continue
            if failed_now():
                failed_steps.append(step)
                if args.sticky and not sticky_done:
                    sticky_done = True
                    sticky_until = step + 1 + args.sticky
                    log(f"step {step}: give-up; FAILED left set for {args.sticky} steps (as the server does)")
                    continue
                handle(f"serve step {step}" + (f" (injected nu63 on L{inject[step]})" if step in inject else ""),
                       {l: [i, L[l]["nus"][i]] for l, i in picks.items()
                        if L[l]["nus"][i] >= 62 or (step in inject and l == inject[step])})
            if args.barrier_every and step % args.barrier_every == 0:
                dist.barrier()
            if rank == 0 and step and step % 2000 == 0:
                q = sorted(st_us[-2000:])
                log(f"step {step}: {time.time() - t_start:.0f} s, step p50 {q[1000]:.0f} us p99 {q[1980]:.0f} "
                    f"max {q[-1]:.0f}; give-ups so far {len(giveups)}")
        wall = time.time() - t_start
        digc = dig.cpu()
        ok = torch.ones(args.steps, dtype=torch.bool)
        for s_ in failed_steps + sticky_steps:
            ok[s_] = False
        for s_ in inject:  # an injected step digests differently on a tree that fixes it: compare separately
            ok[s_] = False
        h = hashlib.sha256(digc[ok].numpy().tobytes()).hexdigest()[:16]
        hin = hashlib.sha256(digc[sorted(inject)].numpy().tobytes()).hexdigest()[:16] if inject else ""
        q = sorted(st_us)
        med = q[len(q) // 2]
        stats = {"steps": args.steps, "calls": args.steps * nl, "wall_s": round(wall, 1), "p50_us": round(med, 1),
                 "p99_us": round(q[int(0.99 * len(q))], 1), "p999_us": round(q[int(0.999 * len(q))], 1),
                 "max_us": round(q[-1], 1), "steps_over_+1ms": sum(x > med + 1000 for x in q),
                 "steps_over_+10ms": sum(x > med + 10000 for x in q), "failed_steps": failed_steps,
                 "injected": {str(k): v for k, v in inject.items()}, "digest_ok_steps": h, "digest_injected": hin,
                 "sticky_steps": [sticky_steps[0], sticky_steps[-1]] if sticky_steps else []}
        allstats = [None] * 4
        dist.all_gather_object(allstats, stats)
        res["serve"] = allstats
        if rank == 0:
            torch.save(digc, os.path.join(args.out, f"{args.tag}_dig.pt"))
        log(f"serve: {args.steps} steps x {nl} layers = {args.steps * nl} K2 calls in {wall:.0f} s; give-ups "
            f"{len(failed_steps)} at {failed_steps[:20]}; rank0 step p50 {med:.0f} us p99.9 {stats['p999_us']:.0f} "
            f"max {q[-1]:.0f}, steps > +1 ms {stats['steps_over_+1ms']} (> +10 ms {stats['steps_over_+10ms']}); "
            f"digest(non-failed, non-injected steps) per rank {[s['digest_ok_steps'] for s in allstats]}, "
            f"injected {[s['digest_injected'] for s in allstats]}")
    res["giveups"] = giveups
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"{args.tag}.json"), "w") as f:
            json.dump(res, f)
        log(f"wrote {os.path.join(args.out, args.tag + '.json')}; total give-up events {len(giveups)}")
    dist.barrier()
    rt.close()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", default="1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,60")
    ap.add_argument("--sweep-layers", default="1,33,60")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--inject", type=int, default=0, help="steps with a nu = 63 case on one layer")
    ap.add_argument("--ctx", type=int, default=120000, help="attention-like kernel's cached tokens (0: none)")
    ap.add_argument("--barrier-every", type=int, default=0)
    ap.add_argument("--pf", action="store_true", help="K2Build with the touch-then-prefetch of a QKV-sized region")
    ap.add_argument("--sticky", type=int, default=0, help="after the first serve give-up, run this many steps "
                    "with FAILED left set (no reset: what the server does), then reset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    args.layers = [int(x) for x in args.layers.split(",")]
    sl = args.sweep_layers
    args.sweep_layers = [] if sl in ("", "none") else [int(x) for x in sl.split(",") if x]
    args.stamp = f"{int(time.time())}_{os.getpid()}"
    os.makedirs(args.out, exist_ok=True)
    mp.spawn(worker, args=(args, None), nprocs=4, join=True)


if __name__ == "__main__":
    main()
