"""CPU only: the k2v4b down split vs k2v4_land1's, every nu, ugw 16 / 24 / 32."""
import sys
sys.path.insert(0, "/data/muse455/k2/cand/k2v4b")
from port.moe import moe as X


def land1(nu, ub, ugw):
    m = nu % 8
    r = 32 * m
    dh = max((8 * nu + m * ugw - 8 * ugw) // 8, 0)
    light = nu * 256 - r * dh
    if ub < r:
        return light + ub * dh, light + (ub + 1) * dh
    return ((ub - r) * light) // (256 - r), ((ub - r + 1) * light) // (256 - r)


def v4b(nu, ub, ugw):
    r, dh = X._down_split(nu, ugw, py=True)
    light = nu * 256 - r * dh
    if ub < r:
        return light + ub * dh, light + (ub + 1) * dh
    return ((ub - r) * light) // (256 - r), ((ub - r + 1) * light) // (256 - r)


for ugw in (16, 24, 32):
    X.check_down_split(ugw)
    diff = [nu for nu in range(1, 65) if any(land1(nu, u, ugw) != v4b(nu, u, ugw) for u in range(256))]
    over = [(nu, max(b - a for a, b in (land1(nu, u, ugw) for u in range(256)))) for nu in range(1, 65)]
    over = [x for x in over if x[1] > X.VR_DN_LOCAL]
    print(f"ugw {ugw}: check ok; nu whose split changed {diff}; land1 runs > {X.VR_DN_LOCAL}: {over}")
