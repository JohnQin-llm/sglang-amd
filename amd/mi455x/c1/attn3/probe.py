"""Read-bandwidth probe of the paged FP8 KV cache layout [pages, 16, 4, 64] (no attention math).

Each program streams its segment's pages of K and V with plain Triton loads and keeps a byte sum alive:
  head : grid (NSEG, 4 kv heads), 16 rows x 64 B of one head per page (the attention access pattern)
  page : grid (NSEG, 1), the whole 4 KB page (all 4 heads, contiguous) -> same total bytes
"""
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["TRITON_CACHE_DIR"] = os.path.join(HERE, "tcache")
sys.path.insert(0, "/data/muse455/k2/cand/vagm")
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from vagm import attn_sweep as AS  # noqa: E402

C = AS.C


@triton.jit
def probe(out_ptr, k_ptr, v_ptr, bt_ptr, n_pages, PPS: tl.constexpr, W: tl.constexpr, NKV: tl.constexpr,
          PB: tl.constexpr):
    seg = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.arange(0, PB * 16)
    cols = tl.arange(0, W)
    acc = tl.zeros([PB * 16, W], tl.int32)
    for j in range(0, PPS, PB):
        pidx = tl.minimum(seg * PPS + j + r // 16, n_pages - 1)
        pg = tl.load(bt_ptr + pidx).to(tl.int64)
        off = pg[:, None] * 4096 + (r % 16)[:, None] * 256 + h * W + cols[None, :]
        kk = tl.load(k_ptr + off)
        vv = tl.load(v_ptr + off)
        acc += kk.to(tl.int32) + vv.to(tl.int32)
    tl.store(out_ptr + seg * NKV + h, tl.sum(tl.sum(acc, 1), 0))


def main():
    dev = torch.device("cuda", 0)
    ctx = 340000
    fx = C.make_fixture("full", ctx, 1, dev=dev)
    k = fx["k"].view(torch.uint8)
    v = fx["v"].view(torch.uint8)
    tab = fx["table"][0]
    n_pages = math.ceil((ctx + 8) / 16)
    flush = torch.empty(AS.FLUSH_MB * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    out = torch.zeros(1024, dtype=torch.int32, device=dev)
    res = {}
    for nseg in (128, 256, 512):
        pps = math.ceil(n_pages / nseg)
        for mode in ("head", "page"):
            for pb in (4, 8):
                pps_r = math.ceil(pps / pb) * pb
                W, nh = (64, 4) if mode == "head" else (256, 1)

                def call(W=W, nh=nh, pb=pb, pps_r=pps_r):
                    flush.fill_(1.0)
                    probe[(nseg, nh)](out, k, v, tab, n_pages, PPS=pps_r, W=W, NKV=4, PB=pb, num_warps=4)
                g = AS.graph_of(call, 10)
                gf = AS.graph_of(lambda: flush.fill_(1.0), 10)
                ts = []
                for _ in range(7):
                    ts.append((AS.time_graph(g) - AS.time_graph(gf)) / 10)
                t = st.median(ts)
                mb = 2 * n_pages * 4096 / 1e6
                print(f"nseg {nseg:4d} {mode} PB {pb}: {t:6.2f} us cold, {mb / t:5.2f} TB/s ({mb:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
