"""Time per call of each stage alone at 8 rows (graph replay; weights rotated over ~1.5 GB of copies so every
call streams from HBM; mailbox inputs already published by the correctness run, so a call measures its weight
stream, its local reductions and its intra-launch hand-offs, not cross-GPU waits).  The empty stage kernel's
time on the same grid (launch + gap, ~1.28 us) is subtracted for the net number: inside the persistent K2 no
stage pays a launch."""

import os

import torch

from . import common as C
from . import moe_dense as M
from . import oproj

PEAK_TBS = 23.3
HBM_FILE = "/data/muse455/gemm/results_hbm_bw.txt"


def hbm_table():
    """(MB, TB/s, us) rows of the measured copy-bandwidth sweep."""
    rows = []
    if os.path.exists(HBM_FILE):
        for line in open(HBM_FILE):
            p = line.split()
            if len(p) > 6 and p[1] == "MB" and p[2] == "best":
                rows.append((float(p[0]), float(p[3]), float(p[5])))
    return rows


def achievable(nbytes, table):
    """The measured copy (MB, TB/s, us) at the nearest size at or above ``nbytes`` (the sweep starts at 16 MB)."""
    mb = nbytes / 2**20
    for row in table:
        if row[0] >= mb:
            return row
    return table[-1] if table else None


def time_graph(fn_list, reps=5):
    """Mean time a call (us) of the captured sequence ``fn_list`` (one launch each), best of ``reps``."""
    for f in fn_list[:4]:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for f in fn_list:
                f()
    torch.cuda.synchronize()
    best = 1e30
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3 / len(fn_list))
    return best


def copies(t, total_bytes=1536 * 2**20):
    n = max(2, total_bytes // (t.numel() * t.element_size()))
    return [t.clone() for _ in range(n)]


def bench_all(st, want):
    rig = st.rig
    table = hbm_table()
    out = st.res.setdefault("bench", {})
    n_calls = 96

    def empty(grid):
        return st.launch(f"empty{grid}", lambda c, bid: None, grid)

    def run(name, launch, grid, rotate, nbytes, rank=0, extra=None):
        keys = list(rotate)
        n = max(len(v) for v in rotate.values()) if rotate else 1
        calls = [
            (lambda i=i: rig.run(launch, rank, **{k: rotate[k][i % len(rotate[k])] for k in keys}, **(extra or {})))
            for i in range(n_calls)
        ]
        t = time_graph(calls)
        torch.cuda.synchronize()
        assert int(rig.err[0]) == 0, f"bench {name}: a poll timed out (inputs not published)"
        le = empty(grid)
        t0 = time_graph([lambda: rig.run(le, rank) for _ in range(n_calls)])
        r = {
            "us_per_call": round(t, 3), "empty_kernel_us": round(t0, 3), "net_us": round(t - t0, 3),
            "bytes": nbytes, "floor_us_23.3TBs": round(nbytes / PEAK_TBS / 1e6, 3),
            "achieved_TBs_net": round(nbytes / max(t - t0, 1e-3) / 1e6, 2),
            "copies_rotated": n, "grid": grid,
        }  # fmt: skip
        ach = achievable(nbytes, table)
        if ach:
            r["copy_kernel_at_%dMB" % ach[0]] = {"TBs": ach[1], "us": ach[2]}
            r["floor_us_at_copy_bw"] = round(nbytes / ach[1] / 1e6, 3)
        out[name] = r
        print("bench", name, r)
        st.save()

    s = st.rig.s
    if "oproj" in want:
        from .test_stages import l_oproj

        wo = copies(st.wo_k[0])
        nb = oproj.H * oproj.CORE * 2 + s * oproj.CORE * 2
        run("oproj", l_oproj(st), C.BLOCKS, {"w_o": wo}, nb, extra={"core": st.core[0]})
        del wo
    if "router" in want:
        from .test_stages import l_router

        wr = copies(st.w["w_router"])
        nb = M.EPR * C.H * 4 + s * C.H * 2
        run("router_q8", l_router(st), M.R_TASKS_Q8, {"w_router": wr}, nb)
        del wr
    if "pool" in want:
        from .test_stages import l_pool

        wp = copies(st.wpool_k)
        nb = M.POOL_N * C.H * 2 + s * C.H * 2
        run("pool", l_pool(st), M.P_TASKS, {"w_pool": wp}, nb)
        del wp
    if "up" in want:
        from .test_stages import l_up

        wu = copies(st.wunpool_k)
        nb = M.UP_N * M.MH * 2 + s * M.MH * 2
        run("up", l_up(st), M.UP_TASKS, {"w_unpool": wu}, nb)
        del wu
    if "route" in want:
        from .test_stages import l_route

        run("route", l_route(st), s, {}, s * M.E * M.R_PARTS * 8)
    if "owner" in want:
        from .test_stages import l_owner

        run("ar_owner_sum", l_owner(st), 4 * s, {}, C.TP * s * oproj.OWNER_COLS * 4)
    if "seam" in want:
        from .test_stages import l_seam

        run("seam_token", l_seam(st), s, {}, s * C.H * (4 + 4 + 4) + C.H * 10)
    if "gather" in want:
        from .test_stages import l_gather

        run("gather", l_gather(st), M.GATHER_TASKS, {}, s * C.H * 6)
    if "seama" in want:
        from .test_stages import l_seama

        run("seam_a", l_seama(st), 64, {}, s * C.H * (2 + 4 + 4 + 2) + C.H * 12,
            extra={"branch_in": st.br, "stream_in": st.stream_out, "stream_a": st.stream_a2})
