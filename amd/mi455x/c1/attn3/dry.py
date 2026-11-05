"""Bounds-checking dry run of ONE attn3 variant at ONE context (run under k2safe -t 60, one call per context).

usage: dry.py <variant, e.g. g3_T64_S128_B4_P1> <ctx> [full|swa]
1. Host-side replay of the kernel's index math for every program of the grid: the TDM page rows / columns issued
   (prologue, steady state and the clamped tail), against the cache tensor; the block-table indices; the LDS ring
   size; the tensorcnt wait against the ops issued. Any violation -> exit 3 before anything is launched.
2. One launch with sentinel partials, synchronize, then: every live (row, head, segment) partial written and finite,
   and (no ablation) the reduced output's max error vs the fp64 reference <= 1.5 x served's.
On success appends "<variant> <kind> <ctx> <kernel sha1>" to logs/dry_ok.txt; bench.py refuses unrecorded variants.
"""
import hashlib
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["TRITON_CACHE_DIR"] = os.path.join(HERE, "tcache")
sys.path.insert(0, "/data/muse455/k2/cand/vagm")
sys.path.insert(0, HERE)

KSRC = os.path.join(HERE, "kern", "attn3.py")
LDS_LIMIT = 160 * 1024  # per workgroup, well under the CU's LDS


def kernel_sha():
    return hashlib.sha1(open(KSRC, "rb").read()).hexdigest()[:12]


def parse(name):
    kv = {s[0]: s[1:] for s in name.split("_")[1:]}
    if "F" in kv or "X" in kv:
        raise SystemExit("non-fused / wide TDM variants are removed")
    return (int(kv["T"]), int(kv["S"]), int(kv.get("B", 4)), int(kv.get("P", 0)), int(kv.get("A", 0)),
            int(kv.get("V", 0)))


def static_check(T, S, B, P, ctx, kind, num_pages, table_pages, sw, rows=8, page=16, nkv=4, d=64):
    """Replay attn3_split's unit / page / descriptor arithmetic for every program; return a list of violations."""
    bad = []
    U = 64
    if T % U or not 2 <= B <= 8:
        bad.append(f"TILE {T} % 64 or NBUF {B} outside 2..8")
    seq = ctx + rows
    qlen = rows
    c = seq - qlen
    num_tiles = math.ceil(seq / T)
    first = max(c - sw + 1, 0) // T if sw > 0 else 0
    span = num_tiles - first
    tps = max(math.ceil(span / S), 1)
    live = math.ceil(span / tps)
    n_log = math.ceil(seq / page)
    if n_log > table_pages:
        bad.append(f"n_log {n_log} > block-table width {table_pages}")
    wait = 2 * (B - 2)
    if wait > 2 * (B - 1):
        bad.append(f"wait {wait} > ops in flight {2 * (B - 1)}")
    if wait > 63:
        bad.append(f"wait {wait} exceeds the tensorcnt field")
    max_row = num_pages * page
    for seg in range(live):
        t0 = first + seg * tps
        t1 = min(t0 + tps, num_tiles)
        u0 = t0 * (T // U)
        u1 = min(t1 * (T // U), math.ceil(seq / U))
        if u1 <= u0:
            bad.append(f"seg {seg}: empty unit range {u0}..{u1}")
            continue
        issued = [min(u0 + i, u1 - 1) for i in range(B - 1)]
        issued += [min(u + B - 1, u1 - 1) for u in range(u0, u1)]
        for u in set(issued):
            for j in range(4):
                pidx = min(u * 4 + j, n_log - 1)
                if not 0 <= pidx < table_pages:
                    bad.append(f"seg {seg}: block-table index {pidx}")
        # page ids are read from the fixture table below; rows checked there
    lds = 2 * B * U * (d + (16 if P else 0)) + rows * 8 * (2 * d + 16)
    if lds > LDS_LIMIT:
        bad.append(f"LDS {lds} B > {LDS_LIMIT}")
    if max_row >= 2 ** 31:
        bad.append("descriptor rows overflow i32")
    return bad, dict(live=live, tps=tps, n_log=n_log, lds=lds, programs=S * nkv, wait=wait)


def main():
    name, ctx = sys.argv[1], int(sys.argv[2])
    kind = sys.argv[3] if len(sys.argv) > 3 else "full"
    T, S, B, P, A, AVF = parse(name)
    import torch
    from vagm import attn_sweep as AS
    C = AS.C
    dev = torch.device("cuda", 0)
    fx = C.make_fixture(kind, ctx, 1, dev=dev)
    sw = (1 + fx["window"][0]) if kind == "swa" else 0
    k = fx["k"]
    assert tuple(k.stride()) == (16 * 4 * 64, 4 * 64, 64, 1) and fx["v"].stride() == k.stride()
    bad, info = static_check(T, S, B, P, ctx, kind, k.shape[0], fx["table"].shape[1], sw)
    seq = ctx + C.ROWS
    n_log = math.ceil(seq / C.PAGE)
    pages = fx["table"][0, :n_log]
    if int(pages.min()) < 0 or int(pages.max()) >= k.shape[0]:
        bad.append(f"page ids outside the pool: {int(pages.min())}..{int(pages.max())} of {k.shape[0]}")
    rows_max = (int(pages.max()) + 1) * C.PAGE
    if rows_max > k.shape[0] * C.PAGE:
        bad.append("TDM rows past the cache tensor")
    print(f"[dry] {name} {kind} ctx {ctx}: {info}, max TDM row {rows_max - 1} < {k.shape[0] * C.PAGE}, "
          f"cols {3 * 64}..{3 * 64 + 63} < 256", flush=True)
    if bad:
        print("[dry] STATIC CHECK FAILED:", *bad[:10], sep="\n  ", flush=True)
        sys.exit(3)

    import kern.attn3 as A3
    nseg = S
    so = torch.full((C.ROWS, C.NQ, nseg, C.D), 12345.0, device=dev)
    sm = torch.full((C.ROWS, C.NQ, nseg), 12345.0, device=dev)
    ss = torch.full((C.ROWS, C.NQ, nseg), 12345.0, device=dev)
    q = fx["q"]
    seqused = fx["seq_lens"] + C.ROWS
    h = A3.attn3_split[(nseg, C.NKV, 1)](
        so, sm, ss, q, k, fx["v"], fx["table"][0], seqused, fx["cu_q"], fx["k_descale"], fx["v_descale"], C.SCALE,
        q.stride(0), q.stride(1), k.shape[0],
        NQ=C.NQ, NKV=C.NKV, G=8, D=C.D, PAGE=C.PAGE, TILE=T, NSEG=nseg, NBUF=B, BLOCK_M=64, SW=sw, PAD=P, ABL=A,
        AV=AVF, num_warps=4)
    shared = h.metadata.shared
    if shared > LDS_LIMIT:
        print(f"[dry] compiled LDS {shared} > {LDS_LIMIT}", flush=True)
        sys.exit(3)
    torch.cuda.synchronize()
    live = info["live"]
    w = (sm[:, :, :live] != 12345.0).all().item() and (ss[:, :, :live] != 12345.0).all().item()
    finite = torch.isfinite(ss[:, :, :live]).all().item() and torch.isfinite(so[:, :, :live]).all().item()
    print(f"[dry] launched ok, LDS {shared} B; live partials written {w}, finite {finite}", flush=True)
    ok = w and finite
    if A == 0:
        import bench
        out = torch.empty(C.ROWS, C.NQ, C.D, dtype=torch.bfloat16, device=dev)
        call = bench.make(name, fx, out, seqused)[0]
        call()
        ref = AS.reference(fx)
        o2 = torch.empty_like(out)
        bench.make("served", fx, o2, seqused)[0]()
        torch.cuda.synchronize()
        e = (out.double() - ref).abs().max().item()
        es = (o2.double() - ref).abs().max().item()
        print(f"[dry] max err {e:.3e} (served {es:.3e}), nan {int(torch.isnan(out).sum())}", flush=True)
        ok = ok and e <= 1.5 * es and not torch.isnan(out).any().item()
    if not ok:
        print("[dry] FAILED", flush=True)
        sys.exit(4)
    with open(os.path.join(HERE, "logs", "dry_ok.txt"), "a") as f:
        f.write(f"{name} {kind} {ctx} {kernel_sha()}\n")
    print("[dry] PASS", flush=True)


if __name__ == "__main__":
    main()
