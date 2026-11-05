"""Price the dense decode GEMMs of this tree with the C1 MI455X per-shape table (CPU only).

The C1 campaign (same gfx1250 chip, same tensor parallel 4 shards) measured each dense weight shape at 8 rows,
cold weights, graph replay: the default kernel this tree still serves against the best AITER FlyDSL/gluon kernel.
Those numbers sit in the comments of C1's gfx1250_decode_gemm.py (iteration 9 patch). This script parses them,
multiplies by the layer count, and sets them against the byte floor of each shape at a given HBM rate.

    python3 -I eval_dense_decode_table.py --patch C1_iter9_patch.diff --layers 62 --hbm-tbps 13.3 --out OUT.json
"""
import argparse, json, re

p = argparse.ArgumentParser()
p.add_argument("--patch", required=True)
p.add_argument("--layers", type=int, default=62)
p.add_argument("--hbm-tbps", type=float, default=13.3)
p.add_argument("--out", required=True)
a = p.parse_args()

txt = open(a.patch).read()
seg = txt[txt.index("gfx1250_decode_gemm.py b/"):]
pat = re.compile(r"\((\d+), (\d+), _(BF16|FP32)\): _(flydsl|gluon)\(([^)]*)\),\s*#\s*([\w ]+?)(?: \([^)]*\))?(?:, [^:]*)?: ([\d.]+) vs ([\d.]+) \(([^)]+)\)")
rows, draft = [], False
for line in seg.splitlines():
    if "DFlash draft" in line:
        draft = True
    m = pat.search(line)
    if not m:
        continue
    n, k, dt, be, cfg, name, new, old, oldk = m.groups()
    n, k, new, old = int(n), int(k), float(new), float(old)
    mb = n * k * 2 / 1e6  # bf16 weights
    floor_us = mb / (a.hbm_tbps * 1e6) * 1e6
    rows.append(dict(name=name.strip(), model="draft" if draft else "target", n=n, k=k, out=dt,
                     tuned=f"{be}({cfg})", served=oldk, served_us=old, tuned_us=new, floor_us=round(floor_us, 2),
                     served_x_floor=round(old / floor_us, 2), tuned_x_floor=round(new / floor_us, 2)))

per_layer = [r for r in rows if r["model"] == "target" and r["n"] != 50512]
lm = [r for r in rows if r["model"] == "target" and r["n"] == 50512]
L = a.layers
tot = lambda key: sum(r[key] for r in per_layer) * L + sum(r[key] for r in lm)
out = dict(rows=rows, layers=L, hbm_tbps=a.hbm_tbps,
           target_step_ms=dict(served=round(tot("served_us") / 1e3, 3), tuned=round(tot("tuned_us") / 1e3, 3),
                               floor=round(tot("floor_us") / 1e3, 3)),
           pre_post_replicated_ms=dict(
               served=round(sum(r["served_us"] for r in per_layer if "expert" in r["name"]) * L / 1e3, 3),
               tuned=round(sum(r["tuned_us"] for r in per_layer if "expert" in r["name"]) * L / 1e3, 3)))
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
for r in rows:
    print(f'{r["model"]:6} {r["name"]:28} {r["n"]:>6}x{r["k"]:<6} served {r["served_us"]:7.1f} ({r["served"]:<16}) tuned {r["tuned_us"]:6.1f} floor {r["floor_us"]:6.2f}  x{r["served_x_floor"]:<5} -> x{r["tuned_x_floor"]}')
