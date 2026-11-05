"""CPU only: emulate K2's vres2 down phase for every nu (1 .. 64) and both trees' splits: every CTA's run
(vr_down_range), every wave pair's tiles (vr_down_geom), the LDS slot of every tile (loc < VR_DN_LOCAL) and the YR
units that store it (vr_yr_out: unit = wave + 16 it, it < VR_YR_ITERS; tile l = 8 (unit // 8) + lane // 4).
A tile with no YR unit is a YR pair the finalize waits for forever (its 'yr' poll gives up at the deadline)."""
import sys

sys.path.insert(0, "/data/muse455/k2/cand/k2v4c")
from port.moe import moe as X  # noqa: E402

S, DT, CTAS, LOC = X.MAX_TOKENS, X.DOWN_TILES, X.VR_CTAS, X.VR_DN_LOCAL
YR_UNITS = 16 * X.VR_YR_ITERS


def split_land1(nu, ugw):
    m = nu % 8
    r = 32 * m
    return r, max((8 * nu + m * ugw - 8 * ugw) // 8, 0)


def split_v4b(nu, ugw):
    return X._down_split(nu, ugw, py=True)


def run(nu, ub, split, ugw):
    r, dh = split(nu, ugw)
    light, nl = nu * DT - r * dh, CTAS - r
    if ub < r:
        return light + ub * dh, light + (ub + 1) * dh
    return ((ub - r) * light) // nl, ((ub - r + 1) * light) // nl


def geom(j, d0, d1):  # vr_down_geom for wave pair j
    n = d1 - d0
    s0 = d0 // DT
    b = min(max((s0 + 1) * DT - d0, 0), n)
    m0 = min(max((16 * b + n) // (2 * max(n, 1)), 1), 7)
    m0 = 0 if b == 0 else m0
    m0 = 8 if b == n else m0
    first = j < m0
    l0 = j if first else b + j - m0
    stride = max(m0 if first else 8 - m0, 1)
    end = b if first else n
    nk = max((end - l0 + stride - 1) // stride, 0)
    return [l0 + stride * k for k in range(nk)]


def check(split, ugw):
    bad = {}
    for nu in range(1, 65):
        seen = [0] * (nu * DT)
        lds_over = no_yr = 0
        for ub in range(CTAS):
            d0, d1 = run(nu, ub, split, ugw)
            n = d1 - d0
            locs = sorted(l for j in range(8) for l in geom(j, d0, d1))
            assert locs == list(range(n)), (nu, ub, n)  # the wave pairs cover the run exactly once
            for l in locs:
                seen[d0 + l] += 1
            lds_over += sum(l >= LOC for l in locs)
            yr = {8 * (u // S) + i for u in range(YR_UNITS) if u < S * ((n + 7) // 8) for i in range(8)}
            no_yr += sum(l not in yr for l in locs)
        assert all(c == 1 for c in seen), nu
        if lds_over or no_yr:
            bad[nu] = (lds_over, no_yr)
    return bad


for ugw in (16, 24, 32):
    for name, sp in (("k2v4_land1 split", split_land1), ("k2v4b split", split_v4b)):
        b = check(sp, ugw)
        print(f"ugw {ugw:2d} {name:17s}: every tile computed exactly once for nu 1..64; "
              + ("no tile past the LDS sums / without a YR unit" if not b else
                 "nu with tiles past VR_DN_LOCAL / never YR-stored: "
                 + ", ".join(f"{k}: {v[0]} / {v[1]} tiles" for k, v in b.items())))
