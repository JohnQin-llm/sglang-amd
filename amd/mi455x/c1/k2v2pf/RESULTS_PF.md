# k2v2pf: K2 prefetches the next launch's weights (MI455X gfx1250, TP4, 8 verify rows)

Tree: copy of `k2v2_land1` (2026-11-05 03:5x UTC); only `port/k2.py` changed, every change behind `K2Build` fields
(default = the k2v2_land1 program). Harness `s3/chain_pf.py`, results `results/chain_*.json`, logs `logs/`.
Cache / attribute microbenchmarks: `/data/muse455/k2/cand/k2pf/s0/pfbench*.hip`, logs `/data/muse455/k2/cand/k2pf/logs/`.

## Chain measurement (`s3/chain_pf.py`)

Per layer slot: attention 3D -> reduce_segments -> gated norm -> K2 -> FFN boundary -> QKV (L + 1) (served kernels
and args, as `k2lm/s3/layer_lm.py` stock). Cold = 8 rotated copies of everything a slot reads (K2 operands incl.
the routed expert stacks, QKV / boundary weights, gate, KV pool), so nothing is reused within a cycle; warm = 1 copy.
HIP graph of 16 slots, 20 replays, slowest-rank median, trials alternate arm order. Bitwise: eager golden of every
arm vs stock, all ranks (stream_out, branch_out, routes, new stream, attn_in, q / k / v / gate, both FP8 caches);
graph replays vs golden. Every run: bitwise equal, replays match.

## Recommended knob set

`K2Build(pf_at="fin", pf_touch=2, pf_tat="route", pf_chunk=8192, pf_cpol=17)` with
`k2_launch(..., pf=((w_qkv_next, 75497472), (0, 0)))` (= arm `rt_finx_nt`; `pf_cpol=16` = `rt_finx`):
- at the route wait every CTA does <= 1 dword load (`global_load_b32` + `s_wait_loadcnt 0`, value discarded) per
  (row, K-slice) of the next QKV weight that its XCD's QKV programs read (36,864 loads a GPU);
- under the FIN all-gather (CTAs 0..191 after issuing their unpool weights, 192..255 after the finalize) every CTA
  issues `global_prefetch_b8 scope:SCOPE_DEV th:TH_LOAD_NT` for its 1/32 of its XCD's QKV tiles (N tile pn on
  XCD pn % 8: `_dense_partial` grid (72, 1, 8), program pn + 72 pk; K2 CTA bid on XCD bid % 8).
- ISA: 251 VGPRs (unchanged), 0 spills, LDS unchanged, SGPRs 89 -> 91.

## Results (us a layer, slowest rank, 128K context; tail = boundary + QKV from the partial chains)

| layer | mode | stock | rt_finx | rt_finx_nt | stock tail -> rt_finx tail | K2 alone stock -> rt_finx |
|---|---|---|---|---|---|---|
| 31 swa | cold | 102.8-104.2 (5 sessions) | -0.55, -1.61, -1.61 | -1.86, -2.05 | 19.3-20.5 -> 15.8-16.6 | 68.9-70.1 -> 70.7-71.7 |
| 31 swa | warm | 93.6-94.9 | -0.02, +0.69 | +0.35, +0.23 | 14.2-15.2 -> 13.8-14.7 | 66.7-67.5 -> 68.0-68.4 |
| 60 swa | cold | 102.7 | -0.53 | -0.60 | 19.2 -> 15.1 (nt) | 69.8 -> 72.6 (nt) |
| 60 swa | warm | 92.8 | +0.73 | +0.83 | | |
| 1 full | cold | 126.8 | -2.10 | - | 18.9 -> 16.7 | 71.4 -> 73.3 |
| 1 full | warm | 122.7 | -2.23 | - | | |

Rejected (L31 cold unless noted): prefetch only (`fin` +0.7, `finx` +0.2, `endx` +1.3; warm: misplaced XCD
`fin` / `finx_rot4` +3.4 / +4.8); touch only (`fint` -0.05); prefetch at K2's end (`rt_endx` -0.7, K2 +3.1);
partial prefetch (`rt_finx5` +0.6, `rt_finx3` -0.2: the QKV GEMM is as slow as its coldest program); prefetch
split between the route wait and the FIN hop (`rt_split4` / `rt_split6`: -1.2 / -1.25 cold, +0.8 / +0.5 warm, vs
-2.05 / +0.23 for `rt_finx_nt` in the same session); prefetch at the route wait (`rt_routex` -1.4 cold, +2.3 warm); next K2's
o_proj slice too (`rt_finx_nt_wo`, L60: cold +1.25, warm +3.9); contiguous next-K2 front (`finx_k2` warm +4.7).
Diagnostics with an extra launch (`s3/pfqkv.hip`): exact per-program prefetch right before the boundary: no tail
gain; dword-per-line loads: tail 19.9 -> 15.8 (warm level); sparse touches alone (per row / 64 KB / 1 MB): none.
