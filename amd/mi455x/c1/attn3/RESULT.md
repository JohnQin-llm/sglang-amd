# Full / sliding verify attention on MI455X (gfx1250, TP4 rank, 8 verify rows): Gluon TDM split kernel

Tree: `/data/muse455/attn3` (served trees untouched). GPU jobs through `k2safe.sh -w mpol_exec.py`, GPU 0, HIP graph
replay; timing harness `bench.py` (fixtures, fp64 reference and numerics columns from k1att / vagm, as attn2).
Seeds 1 (N(0,1)) and 101 (heavy-tailed K) per context.

## Result

`kern2/muse_attn3.py:attn3_split` (Gluon) is correct and at served max error. It beats attn_v2 on full layers by
1.0-3.0 us, but it is far from the targets: 340K full is 37 us (target <= 20), and sliding is slower than served.
The TDM stream alone, with no compute, takes ~20 us at 340K (8.7 TB/s), so this load design cannot reach 20 us.

Full layers (split + served `reduce_segments` single tile; `logs/b_full1.log`, 11 alternating trials, medians, us):

| context | served warm / cold | attn_v2 T128 (served now) warm / cold | **attn3 T64 B4 pad** warm / cold | attn3 T128 B4 pad | split only: served / v2 / attn3 T64 |
|---|---|---|---|---|---|
| 32K | 15.27 / 15.87 | 13.09 / 13.10 | **12.05 / 11.90** | 12.33 / 12.44 | 10.55 / 9.90 / 8.37 |
| 128K | 28.27 / 27.48 | 22.70 / 22.13 | **19.92 / 19.11** | 20.80 / 20.65 | 23.19 / 18.85 / 16.49 |
| 340K | 53.69 / 54.65 | 40.15 / 39.91 | **36.77 / 37.17** | 38.58 / 37.89 | 50.02 / 35.94 / 33.02 |

Sliding layers (`logs/b_swa1.log`; served = AITER 3D split 64 x 32 as SGLANG_MUSE_SWA_CFG; attn3 with the window
range and the served reduce's WINDOW_SPLIT):

| context | served warm / cold (split) | attn3 T64 S64 B2 warm / cold (split) |
|---|---|---|
| 32K | 7.28 / 7.31 (4.10) | 8.52 / 8.34 (4.94) |
| 128K | 7.70 / 6.81 (4.10) | 8.60 / 8.41 (5.02) |
| 340K | 7.50 / 6.40 (3.99) | 8.62 / 8.47 (5.06) |

Numerics (bf16 output vs fp64 over dequantized FP8 KV, 2 fixtures x 16,384 outputs per context):

| layer, ctx | max abs err served / attn3 | mean abs err served / attn3 | outputs != served | NaN |
|---|---|---|---|---|
| full 32K | 7.778e-3 / 7.778e-3 | 7.6297e-4 / 7.6263e-4 | 586 | 0 |
| full 128K | 7.685e-3 / 7.685e-3 | 7.2661e-4 / 7.2662e-4 | 617 | 0 |
| full 340K | 7.805e-3 / 7.805e-3 | 7.3042e-4 / 7.3040e-4 | 98 | 0 |
| sliding 32K-340K | 7.784e-3 / 7.784e-3 | 7.953e-4 / 7.966-7.970e-4 | 483-1028 | 0 |

Max error equals served on every fixture. The differing outputs come from a different split: 64-token tiles, and
P rounded to bf16 per 64 keys. Per-segment partials at 4K match a torch fp32 computation to 2e-6 on max / sum
(`debug.py`).

## Design (as built)

- Grid (128 segments, 4 kv heads), 4 waves. Wave w owns packed rows 16w..16w+15 (8 tokens x 8 GQA heads = 64).
  The segment split is AITER's `segment_range`, including WINDOW_SPLIT, so the served reduce combines unchanged
  (TILE_SIZE 64).
- Keys are walked in 64-token units (4 pages). For each unit there is one fused TDM op for K and one for V, and
  wave j issues page j's `[16, 64]` FP8 block (`tensor_load_to_lds`, 2 per wave per unit). The units go into a
  4-deep LDS ring with 16 B of padding per 64 B row. Page ids are read one unit ahead of the issue, which runs 3
  units ahead of compute. Each unit has one `s_wait_tensorcnt 4` + one split barrier.
- QK^T and PV run on `v_wmma_f32_16x16x32_bf16` in the transposed WMMA layout. Each lane holds one query row:
  the row max is in-lane plus one `v_permlanex16`, and P feeds PV from registers with no shuffle. The softmax
  scale is folded into one `v_pk_fma`, with exp2 and fp32 softmax / accumulation. Positions are 32-bit, and masks
  apply only to units past ctx + 1 (full) or to every unit (sliding). Masked units zero V past seq_len in
  registers.
- Per wave per 64-key unit: 331 instructions (16 WMMA, 32 FP8 -> BF16 cvt, 33 exp, 24 LDS reads, 96 SALU),
  196 VGPRs, no spills, 50 KB LDS.

The FP8 -> BF16 conversion and the LDS reads are still done by all 4 waves: one conversion per element shared by
the waves is not implemented. Sharing it needs a second LDS pass, and the LDS read cost (below) argues against
that.

## Where the time goes (340K full, timing ablations, `logs/abl1.log`, `logs/abl2.log`, `logs/wide.log`)

| variant | split us |
|---|---|
| attn3 (base) | 33-37.6 (run to run) |
| no PV (no V read / convert / WMMA) | 26.3 |
| no exp2 | 34.4 |
| K read + convert hoisted out of the loop | 29.8 |
| V read + convert hoisted | 29.6 |
| K and V hoisted | 26.2 |
| hoisted K / V, no exp, no PV | 21.1 |
| **TDM stream only (no compute), ring 4 / 8 / 12 units** | **20.5 / 19.6 / 19.8** |
| TDM stream only, unpadded / 128 B wide blocks (2x the bytes) / 256 B (4x) | 24.7 / 23.6 / 35.5 |
| no padding (bank conflicts on the 64 B-row K reads) | 42.5-46.4 full kernel |

Plain-load bandwidth probe (`probe.py`, `logs/probe.log`; same 174 MB, one head's 64 B of each 256 B row):
6.1-6.9 TB/s at 512 programs, 10.5-11.0 at 1024, 13.4 at 2048. Whole-page reads are no faster, so 128-B line
granularity is not the limit; the memory system needs more loads in flight than 2 programs per CU provide.

What this says:
1. With this cache layout, TDM page loads stall at ~8.7 TB/s no matter how deep the ring. Doubling the bytes per
   op (full 128 B lines) is free, so the cost is per row or per op, not per byte. That alone is ~20 us at 340K.
2. On top of that, the 4x-redundant LDS reads + conversions (~11 us) and PV (~11 us) do not overlap with the
   stream.
3. Bank conflicts matter: the padded ring is worth 8 us.

To get to <= 20 us, the next design should:
- Move loads to many more waves (async copies from 8-16 waves per CU, or TDM for K plus async copies for V).
- Remove the 4x redundancy. Each wave should own all 64 rows over its own keys: an independent online softmax per
  wave, merged in LDS once per program, which also allows more waves per CU without more partials.

Sliding is latency-bound (one unit per program, 3-unit prologue). It needs the combine fused (or a leaner
reduce: it is ~3.2 us of the 7.3), which was not attempted.

## Integration patch

`/data/muse455/attn3/muse_attn3.patch`, against `/data/muse455/sglang` HEAD 9107618f1d (`git apply --check`
passes; the served tree was not modified):
- adds `python/sglang/kernels/ops/attention/muse_attn3.py` (the measured kernel without the ablation / probe
  paths; its gfx1250 ISA is instruction-for-instruction identical to the measured configuration);
- `rocm_muse_spark/verify_attention.py`: flag **`SGLANG_MUSE_ATTN3=1`**.
  - Scope: full-attention layers only. For the exact C1 shape (attn_v2's conditions plus a contiguous
    `[pages, 16, kv, 64]` cache, page 16), the kv split uses 64-token tiles with the same segment count. The 3D
    split runs `attn3_split` (ring 4), and the served `reduce_segments` runs on its single-tile path.
  - Fallback: anything else goes to attn_v2 (`SGLANG_MUSE_ATTN_V2`) or the served path.
  - Sliding layers are not changed (attn3 loses there).

Not yet verified for serving: the patched wrapper path end to end, odd lengths (1000 / 4093 / 100003 / 262147 /
340001), bs > 1 fallback, and in-model accuracy. All of these need the GPU (see the incident below). Expected
gain if landed: ~3 us per full layer at 340K vs attn_v2, ~16 layers -> ~0.05 ms per step.

## Gluon / gfx1250 findings

- `tdm.async_load(_fused)` into `smem.index(b).slice(...)` ignores the slice offset: all four pages land in
  slot 0 (`s25`, the LDS address in group0, never changes between the ops). Workaround: a page-shaped
  `reinterpret` view of the ring and `.index()`.
- A predicated-off TDM op (`pred=0`) is not counted by tensorcnt, so static `s_wait_tensorcnt` counts break.
  Re-issue a valid (clamped) unit instead.
- A padded layout (16 B per 64 B row) works for TDM destinations and transposing `ds_load_tr8_b64` reads.
- `memdesc.reshape` of a `[N*4, 16, 64]` ring to `[N, 64, 64]` asserts in LinearLayout.

## Incident (17:28-17:44 UTC)

Job `bench.py --kind full --ctx 340000 --variants g3_T128_S128_B4_P1_A16, ..._A16_F0, g3_T128_S256_B4_P1_A16,
g3_T64_S512_B4_P1_A16, g3_T128_S256_B4_P1_A16_F0` (timing probes) hung a kernel on 0001:01:00.0.
- At the 900 s timeout, MES failed REMOVE_QUEUE / evict and the SDMA rings timed out (recovery disabled).
- k2safe wrote `GPU_UNHEALTHY` (10 new MES errors). All GPU work stopped there.
- Most likely culprit: the non-fused TDM issue (`F0`: each page block split across the 4 waves) combined with the
  padded layout (`P1`). That combination had never run before (non-fused was only tested unpadded), and a TDM op
  that never completes leaves `s_wait_tensorcnt` waiting forever.
- The patched kernel contains only the fused + padded path, which ran ~10 jobs cleanly (32K-340K full, sliding).

## After the 18:05 UTC reboot (hang analysis + gated runs)

The hang cause and fix are in `HANG_ANALYSIS.md`: the cooperative (non-fused) TDM issue combined with the padded
LDS layout, a form never run before, is removed. Every variant below first passed `dryrun.sh`: separate
`k2safe -t 60` calls at 1K, 4K and 340K, with a host-side bounds replay, sentinel partials and max error vs
served. `bench.py` refuses variants without a pass for the current kernel source. No MES errors since boot.

Full layers, `logs/b_av1.log` (11 trials, us warm / cold, split):

| context | served | attn_v2 T128 (served now) | **attn3 T64 B4 pad** | attn3 + V by async copy (B4 / B8) |
|---|---|---|---|---|
| 32K | 15.01 / 15.33 (10.71) | 13.04 / 12.96 (9.92) | **11.74 / 11.69 (8.59)** | 12.01 / 11.95, 12.50 / 12.53 |
| 128K | 27.72 / 26.66 (23.03) | 21.13 / 20.47 (17.66) | **19.09 / 19.16 (16.07)** | 19.89 / 19.10, 20.53 / 20.07 |
| 340K | 52.99 / 52.94 (48.43) | 38.72 / 38.22 (34.77) | **35.32 / 34.99 (31.73)** | 36.46 / 35.72, 37.56 / 37.16 |

Max error equals served's on every fixture (unchanged).

Stream only (no compute), 340K, `logs/b_avstream.log`: split 24.1 us with K and V on TDM, 24.6 us with V on
async copies, 22.8 us with an 8-deep ring. Splitting the stream across both load engines does not raise its
rate. With 508 programs x 4 waves the stream runs at ~7-8 TB/s whichever engine moves it, matching the
plain-load probe (6-7 TB/s at 512 programs, 13.4 TB/s at 2048). So the 340K floor for this structure is
concurrency, not the TDM engine.

Reaching <= 20 us needs:
- many more independent waves streaming per CU, without more partials: e.g. 8-16 waves per program, each wave an
  independent online-softmax stream over its own keys with all 64 rows, merged once in LDS;
- that same change also removes the 4x-redundant LDS reads / conversions (~11 us);
- compute overlapping the stream.
That is a restructure of the kernel, not another variant.

## Next step: restructured full-attention kernel (designed, not built)

Goal: 340K <= 20-25 us cold. The stream alone runs at 22.8-24.6 us with ~500 four-wave programs; plain loads
reached 13.4 TB/s with 2048 programs. Nothing below has been compiled or run; the API checks were done on CPU
against the container's Triton (`/sgl-workspace/triton-custom`).

### Structure

- **Program and grid:** one program per (segment, kv head) with W = 8 one-wave streams, built with
  `gl.warp_specialize`: a 1-warp default partition plus 7 one-warp worker partitions, all running the same
  stream function with a constexpr stream id. Grid (64 segments, 4 kv heads) = 256 programs, one per CU, 8 waves
  per CU. 64 segments halves the partials and the reduce's reads vs today's 128.
  - API check: gfx1250 supports this together with TDM and LDS mbarriers; see
    `third_party/amd/python/examples/gluon/f16_gemm_cdna5.py: gemm_tdm_warp_specialized_kernel` and
    `test_gluon_cdna5.py`.
- **Key split:** stream w of segment s takes the segment's 64-key units u0 + w, u0 + w + W, ... (interleaved, so
  the causal tail spreads over the streams). Each stream keeps all 64 packed rows (8 verify rows x 8 GQA heads) and
  its own fp32 online softmax (M, L per row, O^T 64 x 64).
- **Loads, per stream:** a private NBUF = 3 ring of 64-key units (K 5 KB + V 5 KB padded, 80 B rows). Each unit is
  4 + 4 whole-page `tdm.async_load` ops issued by that one wave: one 16-row block per op, the form that ran
  cleanly; a 1-warp partition can never produce the cooperative split that hung GPU 0.
  - Destinations go through the page-shaped `reinterpret` view and `.index()` (the `.slice()` offset is dropped
    by the TDM lowering).
  - Page ids are scalar-loaded one unit ahead. Past the segment end the last unit is re-issued, never
    predicated off.
  - Each wave waits only on its own `s_wait_tensorcnt`, so there are no workgroup barriers in the loop. The AOT
    ISA must be checked for none being inserted by the membar pass.
  - Loads in flight per CU: 8 x 2 x 8 KB = 128 KB, vs 48 KB today.
- **Compute, per stream and unit:** two 32-key chunks, on a single-warp layout
  `AMDWMMALayout(3, transposed=True, warp_bases=[], instr_shape=[16, 16, 32])`.
  - QK^T: S [64 rows x 32 keys] = 8 C tiles, 16 WMMA.
  - Softmax: row max mostly in-lane (each lane holds 4 rows) plus one `v_permlanex16`, then exp2 / fma / sum.
  - PV: P stays in registers as the A operand, O += P V, 16 WMMA.
  - Every K / V FP8 byte is read from LDS and converted to BF16 once, by the one wave that owns it. This removes
    today's 4x duplicated LDS reads and conversions, which cost ~11 us at 340K in the ablations.
  - Budget per 64-key unit: ~64 WMMA and ~550 VALU for 64 x 64 scores, vs 4 x 331 instructions today.
- **VGPR budget per wave:** O^T 128 + Q 64 + S 64 + P 32 + K op 32 + V op 32 + misc, about 370. That gives 2 waves
  per SIMD, which fits 8 waves per CU. It is above 256, so watch the `s_set_vgpr_msb` count. Fallbacks if it
  spills or the bank switching costs too much:
  - 16-key score sub-chunks for the softmax;
  - Q re-read from LDS per chunk (adds 8 KB of LDS reads per chunk);
  - 2 rows-halves per key range (2x reads, still half of today).
- **LDS:** 8 rings x 30 KB = 240 KB per program. This must be checked against the gfx1250 workgroup LDS limit;
  otherwise use NBUF 2, 192 KB.
- **Merge, once per program:** after its loop and `tdm.async_wait(0)`, each worker stores (M, L, O^T) into its
  own ring, reinterpreted as fp32 (16.5 KB < 30 KB), then calls `mbarrier.arrive` on an LDS mbarrier initialized
  with count 7. The default stream does `mbarrier.wait(bar, 0)` and merges the 8 states in fp32:
  - m = max_w M_w;
  - L = sum_w L_w 2^(M_w - m);
  - O = sum_w O_w 2^(M_w - m), with the M = -inf guard as in the served kernel.
  It then writes the served partials layout (segm_output / segm_max / segm_expsum), so the served
  `reduce_segments` (TILE_SIZE 64, single-tile path) and the patch wiring stay as they are. Fusing the cross-
  segment combine stays out: it needs a device-scope hand-off, which the decode-cu-utilization skill prices above
  a launch.
- **Numerics:** fp32 softmax and accumulation; P rounded to bf16 per 32-key chunk, as served rounds per tile.
  Same bar: max error equal to served's on every fixture.
- **Sliding-window layers (window 2047):** same kernel with SW > 0. The window is ~33 units per kv head, so
  4 programs x 8 streams cover it in ~1 unit per stream, with one merge and no cross-segment partials. The served
  reduce could then be skipped by writing the bf16 output directly (NSEG = 1 path). That removes the ~3.2 us
  reduce + launch, which is the only way found to reach 4-5 us there.

### Order of work and gates

1. AOT on CPU: compile; check VGPRs <= 512, no scratch, LDS <= limit, no s_barrier in the stream loop, and 8 TDM
   ops per wave per unit with `s_wait_tensorcnt` <= the ops issued.
2. Extend `dry.py: static_check` for the stream key split (every stream's units, clamped re-issues, ring size).
3. `dryrun.sh` per variant (1K, 4K, 340K, one `k2safe -t 60` call each); stop at the first timeout or new MES
   error.
4. Only then time it vs served / attn_v2 / attn3 at 32K, 128K, 340K and sliding, cold and warm.

## Deliverable as of this round

`muse_attn3.patch`, flag `SGLANG_MUSE_ATTN3=1` (full-attention layers; falls back to attn_v2 / served). Its
kernel is the dry-run-gated attn3 T64 B4 padded configuration: 340K 35.32 / 34.99 us warm / cold vs attn_v2
38.72 / 38.22 and served 52.99 / 52.94, at served max error. Sliding layers stay on the served path.
