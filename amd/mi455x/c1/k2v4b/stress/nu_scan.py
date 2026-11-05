"""CPU only: distinct-expert counts (nu) of synthetic 8-row verify cases (validation.make_case) per layer / corr / seed."""
import json, sys, time
import torch
sys.path.insert(0, "/data/muse455/k2/cand/k2v4_land1")
from port import validation as V
torch.set_num_threads(16)
out = {}
for layer in [int(x) for x in sys.argv[1].split(",")]:
    t0 = time.time()
    lay = V.LayerData(layer, "cpu")
    rows = []
    for corr in (0.0, 0.2, 0.4, 0.6):
        for seed in range(int(sys.argv[2])):
            case = V.make_case(lay, corr, seed, dev="cpu")
            nu = int(torch.unique(V.seam_and_route(lay, case)["ids"]).numel())
            rows.append((corr, seed, nu))
    out[layer] = rows
    hist = {}
    for c, s, n in rows:
        hist.setdefault(c, []).append(n)
    print(layer, f"{time.time()-t0:.0f}s", {c: (min(v), sorted(v)[len(v)//2], max(v), sum(x == 63 for x in v)) for c, v in hist.items()}, flush=True)
json.dump(out, open(sys.argv[3], "w"))
