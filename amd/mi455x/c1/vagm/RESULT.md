# Verify attention + decode GEMMs outside K2 (MI455X gfx1250, TP4 rank, 8 verify rows): microbenchmarks

Tree: `/data/muse455/k2/cand/vagm` (served trees untouched; the served AITER / SGLang kernels are imported unchanged,
only their configs are overridden, plus kernel copies under `vagm/kern/`). Every GPU job ran through `k2safe.sh`
with the NUMA wrapper; all ended "GPUs drained, no new MES errors". Single GPU (GPU 0), HIP graph replay.

- warm: graph of R back-to-back calls / R. cold: graph of R x [512 MiB flush write + call] minus R x [flush], / R
  (attention); for GEMMs one graph over L distinct weight copies (> 2 GB, every call streams its weight from HBM).
- Trials alternate variants; medians. Run-to-run noise ~0.3-0.5 us on sliding, ~0.5-1 us on full layers.
- Numerics (attention): bf16 output vs an fp64 reference (dequantized FP8 KV, causal + window mask), 4 fixtures per
  context (seeds 1, 2 N(0,1); 101, 102 heavy-tailed K). Numerics (QKV): bytes differing vs the served op in all six
  outputs (Q, K, V, gate, K cache, V cache).
- Launch floor measured: an empty Triton kernel costs 1.28 us per launch in graph replay (1 or 256 programs).

Scripts: `vagm/attn_sweep.py`, `vagm/gemm_bench.py`, `vagm/gemm_bench2.py`; results `results/*.json`, logs `logs/`.

## A. Verify attention (`kernel_unified_attention_3d` + `reduce_segments`, Triton, gfx1250 DEFAULT.json)

The 3D kernel's grid is `(segments, 4 kv heads, 2 q blocks)`: BLOCK_M 64 = 8 rows x 8 q heads per kv head, so the
upper-bound grid carries one empty q block. Programs that do work = live segments x 4.

### Served kernels

| layer, ctx | us warm / cold | split + (reduce + gap) | programs (grid / working) | CUs used | KV bytes | % of 16.5 TB/s (cold) | BW floor |
|---|---|---|---|---|---|---|---|
| sliding, 32K-340K | 8.3-8.8 / 8.4-9.0 | 5.8 + 2.6-2.9 | 128 / 44 (11 segs x 3 tiles of 64) | 44 of 256 (17%) | 1.05 MB (window 2055 keys) | 0.8% | 0.06 us (latency floor ~2 x 1.28 launch) |
| full, 32K | 15.3 / 15.7 | 10.7 + 4.6 | 1024 / 344 (86 x 3 tiles of 128) | 256 | 16.8 MB | 6.5% | 1.0 us |
| full, 128K | 28.4 / 28.3 | 23.8 + 4.6 | 1024 / 456 (114 x 9) | 256 | 67.1 MB | 14% | 4.1 us |
| full, 340K | 54.2 / 54.2 | 49.6 + 4.6 | 1024 / 508 (127 x 21) | 256 | 174 MB | 19% | 10.6 us |

Reduce: grid (8 tokens, 32 heads, 1) = 256 one-warp programs; full layers take the 2-step loop path (128 segments >
SEGMENT_BLOCK 64), sliding the single-tile path. Cold ~= warm everywhere: neither kernel is bandwidth-bound.

What bounds them:
- Sliding: latency. 44 working programs of 3 tiles each; the split is a dependent chain (cu_seqlens -> seq_len ->
  block table -> KV -> store) plus two launches. More, shorter segments help until the reduce grows.
- Full: per-CU tile throughput, not HBM. The T128 kernel holds 402 VGPRs (4 waves); time follows
  rounds x tiles-per-segment (~1.2-1.4 us per 128-key tile per program): S256 (same total tiles, 820 programs) is
  slower than S128, BLOCK_M 16/32 (2-4x programs re-reading KV) is 1.7-2.3x slower, and 1- or 2-warp programs are
  slower. A register-double-buffered prefetch copy of the loop (`kern/ua_pf.py`, `ua_pf2.py`: K/V one or two tiles
  ahead, block table one further) is bit-identical but 30-40% slower (it drops the async-to-LDS pipelining), so the
  loop is not waiting on the block-table -> KV chain. Feeding FP8 Q (no FP8 -> BF16 K/V conversion, FP8 WMMA) is
  only 20-23% faster and wrecks numerics (max err 0.68-0.88), so conversion is not the main cost either. Reaching the
  10.6 us floor at 340K needs a redesigned loop (profiling with counters is the next step), not a config.

### Variants (us warm / cold; final confirmation run, 21 trials)

| variant | sliding 32K | sliding 128K | sliding 340K | full 32K | full 128K | full 340K |
|---|---|---|---|---|---|---|
| served | 8.33 / 8.67 | 8.38 / 8.97 | 8.78 / 8.40 | 15.28 / 15.71 | 28.44 / 28.34 | 54.15 / 54.20 |
| **proposed** DEFAULT.json (below) | 7.24 / 7.61 | 7.62 / 7.27 | 7.47 / 7.38 | 13.40 / 13.98 | 25.80 / 24.34 | 49.80 / 50.19 |
| proposed + reduce BLOCK_D 32 | 7.17 / 7.34 | 7.10 / 7.06 | 6.96 / 6.98 | 13.66 / 13.78 | 24.36 / 24.50 | 51.21 / 50.26 |
| sliding S32 T32 st3 | 6.84 / 7.41 | 6.97 / 7.30 | 6.82 / 7.07 | | | |
| full T128 st3, SEGMENT_BLOCK 128, BLOCK_D 32 (near-identical numerics) | | | | 14.64 / 14.41 | 25.88 / 25.41 | 51.57 / 50.86 |

Sliding proposed: grid 512 (2 x 4 x 64), 132 working programs (33 segs x 2 tiles of 32), split 4.0-4.2 us.
Full proposed: T64, 412 / 484 / 508 working programs (5 / 17 / 42 tiles), split 9.7 / 21.1 / 46.1 us, reduce single tile.

Also tried (128K unless noted; full table in `logs/`): sliding S32/S64/S128 x T16/T32/T64, BLOCK_M 16/32, warps 2/8,
stages 1/3, SEGMENT_GRID_FIRST, reduce warps 2, BLOCK_D 16/32 -- best cluster 6.8-7.2 us, worst w2/w8 at 11+ us;
single-sequence fast path in the kernel copy (`kern/ua_ss.py`, skips `find_seq_idx` when num_seqs == 1): <= 0.3 us,
within noise. Full: S64/S256/S512, T32/T64/T256, BLOCK_M 16/32, warps 1/2/8, stages 1/3/4, waves_per_eu 1/2, reduce
SEGMENT_BLOCK 32/128, BLOCK_D 16/32, warps 2 -- only T64, stages 3 and the single-tile reduce help; T256 spills to
1002 VGPRs.

Numerics (vs fp64, 4 fixtures x 65,536 outputs per context): max abs error is unchanged on every fixture (sliding
7.797e-3; full 7.778e-3 / 7.783e-3 / 7.805e-3 -- the bf16 output rounding of the largest value). Mean abs error
drops slightly (sliding 7.42e-4 -> 7.37e-4; full 6.613e-4 -> 6.606e-4 at 32K), and outputs that differ from
round-to-bf16(fp64) fall (sliding 3165 -> 1697; full 1603 -> 1260 at 32K, 1548 -> 1416 at 128K, 1500 -> 1493 at
340K). Proposed vs served differs in 2542-2680 (sliding) and 175-1201 (full) of 65,536 bf16 outputs, as expected
from a new split. The "T128 st3 + SEGMENT_BLOCK 128 + BLOCK_D 32" option differs in only 2-6 outputs.

### Best config and serving change

Edit `/sgl-workspace/aiter/aiter/ops/triton/configs/gfx1250/triton/attention/unified_attention/DEFAULT.json`
(the proposed file is `results/DEFAULT.proposed.json`, the bit-for-bit served copy `results/DEFAULT.served.json`;
the "json_proposed" timing rows above went through AITER's own config loader reading that file):

- `kv_split["D_LEQ_64.SW.DT_any_fp8"]`: SEGMENTS_PER_CU 1 -> 4, SEGMENT_TILE_MIN / TILE_SIZE_MIN / TILE_SIZE_MAX 64 -> 32
  (sliding: 16 segments x 64 -> 64 x 32)
- `kv_split["D_LEQ_64.DT_any_fp8"]`: TILE_SIZE_MIN / TILE_SIZE_MAX 128 -> 64 (full: 128 segments, tile 64)
- `attn_3d["D_LEQ_64.DT_any_fp8"]`: num_stages 2 -> 3 (shared by both layer types)
- `reduce["any"]`: SEGMENT_BLOCK 64 -> 128 (128-segment reduce takes the single-tile path)

No env, argv or code change. Expected per decode cycle at 340K (one request): sliding 46 x ~1.2 us (1.05 warm /
1.26 cold) = ~55 us, full 16 x ~4.2 us = ~67 us; ~120 us total (at 128K: ~55 + 16 x ~3.3 = ~110 us). The 3D path is only taken at
C1-sized verifies; re-check bs > 1 (sliding segment count falls with num_2d_prgms) before landing.

## B. Decode GEMMs at M = 8

### Fused QKV + gate projection (served `qkv_proj_qk_norm_rope_cache`, N 4608, K 8192, 75.5 MB weight)

| | us cold | us warm | programs | % of 16.5 TB/s | notes |
|---|---|---|---|---|---|
| served op | 13.17-13.24 | 7.98-8.08 | partials 576 (72 N tiles x split-K 8) + epilogue 256 | 35% of op | floor 4.58 us (weight) + 1.28 (2nd launch) |
| served `_dense_partial` alone | 9.63-9.65 | 4.28-4.32 | 576 (2.25 / CU), BN 64, BM 16, BK 128, 4 warps, 2 stages, .cg | 48% | |
| served `_qkv_epilogue` alone | 2.38 | | 8 x 32, 1 warp; K/V heads serialized after Q in 4 of 32 heads | | floor 1.28 |

Variants (all at split-K 8 = same K slices; every row below is bit-identical to served in all six outputs unless noted):

| variant | op cold | op warm | partials cold (TB/s) | epilogue |
|---|---|---|---|---|
| served | 13.24 | 7.98 | 9.65 (7.8) | 2.38 |
| Triton stages 3 | 10.88 | 8.07 | 7.11 (10.6) | 2.38 |
| Triton BK 256 | 11.04 | 7.98 | 7.20 (10.5) | 2.47 |
| Triton BN 32 / BN 128 w8 / BK 64 st3 | 13.5 / 14.2 / 13.6 | | 9.8 / 10.2 / 9.9 | |
| Triton split-K 16 st3 (not identical: 11 bytes differ) | 10.57 | 8.06 | 6.71 (11.3) | 2.45 |
| split epilogue (grid 8 x 36, K/V/cache in own programs) on served partials | 12.65 | 7.75 | 9.64 | 1.86 |
| **Triton stages 4 + split epilogue** | 9.95 | 7.21 | 6.73 (11.2) | 1.85 |
| Triton stages 3 + split epilogue | 10.24 | 7.15 | 7.06 (10.7) | 1.85 |
| **AITER gluon TDM batched GEMM (BN 32, BK 128, sk 8, 3 buf, 2 warps, .cg) + split epilogue** | 9.38-9.48 | 7.29 | 6.45 (11.7) | 1.86 |
| gluon BN 16 / BN 64 / w4 / BK 256 / 2 buf / no .cg (+ split epilogue) | 9.65 / 10.53* / 9.92 / 11.14 / 9.71 / 9.62 | | 6.5-7.9 | |

(*BN 64 measured with the served epilogue.) Not pursued: summing split-K inside the GEMM (needs a cross-CTA hand-off,
which on gfx1250 costs more than the 1.28 us launch it would remove) and stream-K (non-deterministic order).

Best: gluon TDM partials + split epilogue, 13.2 -> 9.4 us cold (-3.8 us), bit-identical. Pure-Triton alternative:
num_stages 4 + split epilogue, -3.3 us, bit-identical. Expected per decode cycle: 62 layers x 3.8 = ~235 us
(Triton-only: ~205 us).

Serving change (SGLang `python/sglang/kernels/ops/attention/qk_norm_rope_cache.py`, `qkv_proj_qk_norm_rope_cache`):
1. `_dense_partial` launch: `num_stages=2` -> `4` for the TP4 tile (or replace the launch with AITER's gfx1250
   `_batched_gemm_bf16_bandwidth_bound_kernel` writing `[8, 1, M, N]` FP32 partials, config above; launch code in
   `/data/muse455/decode_stack/qkv_bench_gluon.py:make_launch`).
2. `_qkv_epilogue[(m, hq)]` -> grid `(m, hq + hk)`: programs `head < hq` do gate + Q, programs `hq + kh` do K, V and
   the FP8 cache write (same per-element code; `_qkv_epilogue_split` in `vagm/gemm_bench.py`).
No env / argv change (path stays behind `SGLANG_MUSE_TP4_FUSED_QKV_CACHE=1`).

### Other dense decode GEMMs (served `gfx1250_decode_gemm` table, M = 8, cold)

| GEMM (per cycle) | kernel | us cold | weight | % of 16.5 TB/s | floor | best variant found |
|---|---|---|---|---|---|---|
| lm_head FP32 logits (1x) | FlyDSL 32x128, sk 1 | 52.7 | 828 MB | 95% | 50.2 | at floor (hipBLASLt 459) |
| draft logits (1x) | FlyDSL 128x128 | 51.5 | 828 MB | 97% | 50.2 | at floor |
| draft qkv 2560x8192 (5x) | FlyDSL 32x128 sk 16 | 5.8-5.9 | 42 MB | 43% | 2.5 | served is best of 10 |
| draft o 8192x2048 (5x) | gluon 64x256 | 4.7-5.0 | 34 MB | 41% | 2.0 | within noise of served (10 tried) |
| draft gate_up 14336x8192 (5x) | gluon 64x256 | 16.0-16.2 | 235 MB | 89% | 14.2 | served is best of 7 |
| draft down 8192x7168 (5x) | gluon 32x512 | 9.6-9.9 | 117 MB | 73% | 7.1 | within noise of served (8 tried) |

Router, o_proj, pre/post_expert_proj (pool/unpool) and the expert GEMMs are inside K2 (`mono_gfx1250.py` packs
`w_o`, `w_router`, `w_pool`, `w_unpool`, `w13`, `w2`), so they were out of scope. The small draft GEMMs sit at
~5-6 us against a 2-2.5 us bandwidth floor because one launch (1.28 us) plus split-K latency dominates; no tile in
the sweep beat the served table, and the remaining ~50 us of draft gap per cycle needs fusion across the draft
layers, not tiles.

## Summary of expected savings per decode cycle (340K, one request)

| change | where | numerics | us / cycle |
|---|---|---|---|
| QKV: gluon TDM partials (or Triton stages 4) + split epilogue | SGLang qk_norm_rope_cache.py | bit-identical | ~235 (~205) |
| Attention DEFAULT.json (4 entries) | AITER gfx1250 config | new split; max error unchanged, mean error slightly lower | ~120 (128K: ~110) |
| lm_head, draft GEMMs | -- | -- | 0 (at floor / no better tile) |
