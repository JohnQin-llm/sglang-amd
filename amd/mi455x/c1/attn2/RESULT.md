# Full-attention verify split kernel on MI455X (gfx1250, TP4 rank, 8 verify rows): profile and a faster loop

Tree: `/data/muse455/attn2` (served trees untouched; the served AITER kernels and the vagm / k1att fixtures are
imported, not modified). Every GPU job ran through `k2safe.sh -w 'python3 .../mpol_exec.py'`, single GPU, HIP
graph replay; all ended "GPUs drained, no new MES errors". Microbenchmark only (no serving, no Arbor gate).

## Result

`kern/attn_v2.py:attn_v2_split` is a split kernel specialized for this case. It writes the same fp32 partials and
uses the same device-side segment split as `kernel_unified_attention_3d`, so the served `reduce_segments` combines
them unchanged (single-tile path, SEGMENT_BLOCK = 128). Final run (`logs/final.log`, 15 alternating trials, 4
fixtures, us per layer, split kernel + reduce):

| context | served warm / cold | best config warm / cold | **v2 T128** warm / cold | v2 T64 warm / cold | v2 T128 vs served |
|---|---|---|---|---|---|
| 32K | 14.85 / 15.34 | 12.95 / 13.03 | 12.94 / 13.33 | **12.64 / 12.65** | -1.9 / -2.0 us (-13%) |
| 128K | 28.19 / 28.47 | 25.25 / 24.31 | 23.00 / 22.40 | **21.48 / 21.14** | -5.2 / -6.1 us (-18%) |
| 340K | 54.57 / 53.84 | 49.52 / 48.65 | **40.69 / 39.06** | 45.39 / 44.84 | -13.9 / -14.8 us (-26%) |

Split kernel alone (warm, no reduce): served 10.6 / 23.5 / 49.3, best 9.6 / 21.1 / 46.4, v2 T128 9.9 / 19.1 /
36.9, v2 T64 9.1 / 18.0 / 41.8 us. At 340K cold, v2 T128 streams 174 MB in 39.1 us, 4.5 TB/s (27% of 16.5
TB/s); served gets 3.2 TB/s (19%). The bandwidth floor is 10.6 us. Cold and warm stay within about 1 us of each
other everywhere: the kernel is still not bandwidth-bound.

One tile size has to serve every context (it is a constexpr fixed at graph capture). T128 is the pick for AgentX
(~340K): it is the best or within ~1.5 us of the best everywhere. Across a second run (`logs/sweep6.log`) T128 and
T64 were equal at 128K (21.5 / 21.6 us). On 16 full-attention layers at 340K, v2 T128 saves ~0.22-0.24 ms per
~8.8 ms step (~2.6%) vs served, and ~0.14-0.15 ms beyond the best config.

Numerics (bf16 output vs an fp64 reference over dequantized FP8 KV; 4 fixtures x 65,536 outputs; seeds 1, 2
N(0,1) and 101, 102 heavy-tailed K):

| context | max abs err: served / v2 T128 / v2 T64 | mean abs err: served / v2 T128 / v2 T64 | outputs != served (T128 / T64) |
|---|---|---|---|
| 32K | 7.778e-3 / 7.778e-3 / 7.778e-3 | 7.8287e-4 / 7.8287e-4 / 7.8252e-4 | 5 / 1157 |
| 128K | 7.783e-3 / 7.783e-3 / 7.783e-3 | 7.2661e-4 / 7.2661e-4 / 7.2662e-4 | 9 / 1200 |
| 340K | 7.805e-3 / 7.805e-3 / 7.805e-3 | 7.3042e-4 / 7.3042e-4 / 7.3040e-4 | 14 / 187 |

Max error is identical to served on every fixture: it is the bf16 rounding of the largest output. v2 T128 keeps the
served split and math, so it differs from served in 5-14 of 262,144 bf16 outputs (fp32 contraction order). v2 T64
uses the best config's split and differs from served in about as many outputs as the best config does (1157 vs
1156, 1200 vs 1201, 187 vs 175). Odd lengths 1000, 4093, 100003, 262147 and 340001
(`logs/odd.log`) also give max error identical to served, mean error within 1.4e-6, and no NaNs.

## 1. Where the served split kernel's time goes

The served kernel: `kernel_unified_attention_3d`, T128, BLOCK_M 64 (8 rows x 8 q heads of one kv head; the live
q block has no wasted rows), 4 waves stacked along M (16 rows each, `amd_wmma` version 3,
`warp = [[1,0],[2,0]]`), num_stages 2. It holds 402 VGPRs, 107 SGPRs with 19 SGPR spills, and 35.8 KB of LDS, for
an occupancy of 2 waves/SIMD. At 340K, 508 working programs (127 live segments x 4 kv heads) on 256 CUs is 2
programs per CU, one round of 21 tiles each: 49.3 us / 21 = 2.35 us per 128-key tile per program (~1.2 us per tile
per CU, ~2700 cycles at 2.3 GHz).

Static issue mix of the loop body, per wave per 128-key tile (`asm/served_t128.amdgcn`, `tools/isa_mix.py`):

| class | served | v2 T128 | what it is |
|---|---:|---:|---|
| total | **1095** | **563** | |
| SALU / branch | 307 | 52 | served: 228 `s_set_vgpr_msb` (VGPR bank switches because 402 > 256 VGPRs) |
| mask compares / selects | 151 | 0 | 68 `v_cmp_lt_i64` + 81 `v_cndmask`: 64-bit causal / query masks on every tile (seq_lens is int64) |
| FP8 -> BF16 conversion | 96 | 96 | 64 `v_cvt_scale_pk8_bf16_fp8` (the whole K and V tile, in every wave: 4x redundant across the 4 waves) + 32 `v_cvt_pk_bf16_f32` (P) |
| fp32 VALU | 115 | 84 | scale, max-subtract, row sums, acc rescale (`v_pk_mul/add/fma_f32`) |
| exp2 | 65 | 65 | 64 P values per lane + alpha (`v_exp_f32`) |
| other VALU (moves) | 94 | 28 | served: 68 `v_dual_mov_b32` (register shuffling at 402 VGPRs) |
| waits / nops | 84 | 74 | `s_wait_dscnt`, `s_delay_alu`, `v_nop`, 3-5 barriers per tile |
| LDS reads | 52 | 52 | 32 `ds_load_tr8_b64` (transposing FP8 operand reads) + 16-20 `ds_load_2addr_b64` |
| max | 33 | 37 | `v_max3_num_f32` |
| WMMA | 32 | 32 | `v_wmma_f32_16x16x32_bf16`: 16 QK^T + 16 PV per wave |
| global -> LDS | 8 | 8 | `global_load_async_to_lds_b128`: the 16 KB K+V tile, 16 B per lane each |
| addressing | 36 | 28 | page-table gather + 64-bit row address |

Ablations of the served kernel (`kern/ua_abl.py`, constexpr switches; timing only, 340K split, 49.0 us):

| removed | split us | delta |
|---|---:|---:|
| causal / query masks (ABL 1) | 39.0 | -10.0 (20%) |
| PV + V LDS read + V conversion (ABL 4) | 37.3 | -11.7 (24%) |
| softmax (max, exp2, sums, rescale; ABL 2 / ABL 1+2 vs 1) | 41.7 / 35.5 | -7.3 / -3.5 |
| masks + PV (ABL 5) | 29.4 | -19.6 |
| QK^T (ABL 8) | 81.8 | invalid: S loses the WMMA layout and the compiler re-lays out everything |

At 128K the same pattern holds: masks -3.9 us, PV -4.5 us. The remainder (QK^T + K read/convert + the loop
skeleton) is ~25-30 us at 340K. Counters: on this rocprofv3 / gfx1250 build every SQ instruction counter
(`SQ_INSTS_*`, `SQ_INST_CYCLES_*`, `SQ_WAIT_*`) reads 0; only `GRBM_GUI_ACTIVE`, `SQ_BUSY_CYCLES`, `SQ_CYCLES` and
`SQ_WAVES` report (served 919K GUI cycles vs v2 T128 641K at 340K, consistent with the timings). So the breakdown
above rests on the ISA and the ablations.

What limits it: not HBM (19% of bandwidth, cold = warm) and not WMMA issue (32 of 1095 instructions per tile). The
loop is a single dependent chain per tile: wait for the async copy, then a barrier, then the page-table load for
the next tile (`s_wait_loadcnt 0` before its K/V copies can issue), then LDS reads, conversion, QK^T, cross-lane
max, exp2, P conversion, PV. There is at most one tile in flight per program: every iteration starts with
`s_wait_asynccnt 0`, even at num_stages 3. With 2 waves per SIMD, issue IPC is ~0.3.

gfx1250 instruction options seen in the compiled code: `v_wmma_f32_16x16x32_bf16`, `v_wmma_f32_16x16x64_fp8_fp8`
(M = 16 is the smallest WMMA M, so 64 packed rows = 4 waves x 16), `v_cvt_scale_pk8_bf16_fp8` (8 FP8 -> BF16 per
instruction), `ds_load_tr8_b64` / `ds_load_tr16_b128` (transposing LDS operand reads), and
`global_load_async_to_lds_b128`. TDM (`tensor_load_to_lds`) and cluster loads exist on gfx1250 but are reachable
only from Gluon / FlyDSL; Triton's AMD backend uses async copies for this kernel.

## 2. Design (v2) and what was tried

`attn_v2_split[(segments, kv heads, query blocks)]`, 4 waves, BLOCK_M 64 = 8 query rows x 8 GQA heads packed, D 64:
- Same partial layout and device-side `segment_range` as the served kernel (T128 x 128 segments, or T64).
- 32-bit positions. The tiles before the first query's horizon (keys < ctx + 1) run an unmasked loop with no
  compares or selects. Only the last one or two tiles of the last segment take the masked path (masked loads, the
  causal mask, and the `m_j = -inf` guard).
- Softmax scale folded into the exp2 argument (`max(S) * c`, `exp2(fma(S, c, -m))`; c > 0).
- Single sequence: no `find_seq_idx` and no upper-bound empty query block (the grid is exactly the live blocks).
- Result: 563 instead of 1095 instructions per wave per 128-key tile, 364 instead of 402 VGPRs (232 / occupancy 4
  at T64), and no SGPR spills.

Tried and rejected (all measured; `logs/sweep*.log`, `logs/smoke.log`):

| idea | result | why |
|---|---|---|
| FP8 WMMA for QK^T with Q split exactly into 2 or 3 E4M3 terms (power-of-two row scale; K used as stored, no K conversion) | 19.1-19.6 vs 17.7 us split at 128K; mean err 7.66e-4 vs 7.27e-4 | 366-424 VGPRs; conversion is not the bottleneck, and the extra accumulator costs occupancy |
| transposed GEMMs (S^T = K Q^T, O^T = V^T P^T: waves split keys and head dim, each K/V element converted once) `kern/attn_v2t.py` | 1.7x slower | Triton's cross-wave row max / sum and the P^T layout conversion through LDS cost more than the saved conversions |
| register prefetch of the next tile, converted once in a coalesced layout, BF16 via LDS (`kern/attn_v2p.py`) | 1.4-1.5x slower | loses the async-to-LDS copies (same lesson as vagm's `ua_pf`) |
| L2 prefetch of tile t+1..t+4 (inline-asm `global_prefetch_b8` does not survive the pipeliner; byte "touch" loads instead) | 10-15% slower | the extra in-order loads lengthen the page-table `s_wait_loadcnt` |
| two independent online-softmax streams per program (even / odd tiles) | T64 st3 at 340K: 41.2 vs 43.8 us; worse at 32K / 128K and at T128 | more VGPRs |
| 256 / 512 segments, 1 or 2 waves per program, 8 waves, waves_per_eu 3 / 4, num_stages 1 / 3 | all slower or equal | more programs add reduce and prologue cost without more per-CU throughput; forcing occupancy spills |

Upstream AITER (`/tmp/aiter-up` origin/main, 554 commits newer):
- The FlyDSL gfx1250 PA decode kernels (`9feff1450`, `df6a53e19`, `004245be3`) handle MTP query lengths and page 16,
  but with an FP8 cache they quantize Q and P to FP8. That is the numerics failure already measured (max err
  0.68-0.88). They also need a different cache layout (`[blocks, kv heads, D/x, page, x]`, V transposed or plain
  `[.., D, page]`). Their BF16 wave-pair D64 pipeline applies only to BF16 caches. So they are not a drop-in.
- Gluon `unified_attention_3d` gfx1250 is gated on `shuffled_kv_cache` (`is_3d_gluon_available`), and the served
  cache is unshuffled. #6999 only makes `num_seqs` / `block_table_stride` runtime arguments of the 2D Gluon kernel.
  #6896 (spec-decode verify on the 3D kernel) and #6888 (two-pass reduce) are about correctness and the reduce; the
  served tree already has the reduce's SEGMENT_BLOCK / BLOCK_D.
- Using either would mean changing the KV cache layout written by every layer (a serving-wide change), plus
  per-row FP8 Q/P scaling numerics that would need their own accuracy gate.

## 3. What it would take to serve

1. Call `attn_v2_split` from AITER `unified_attention` (gfx1250, Triton path) in place of
   `_unified_attention_3d_triton` when all of these hold: not shuffled, FP8 KV, head size 64, no sinks / alibi /
   softcap / qq_bias / output scale, SLIDING_WINDOW 0 (full layers), and every sequence has <= 8 query rows. The
   current kernel reads `cu_seqlens_q[0:2]` and `seq_lens[0]` only (num_seqs == 1, the C1 verify). For bs > 1, add
   the sequence index on grid axis 2 and per-sequence `seq_lens` / block-table rows; the segment split is already
   per sequence and device-side.
2. Keep `reduce_segments` as is, with TILE_SIZE = the split kernel's tile and SEGMENT_BLOCK >= 128 (the single-tile
   path from the vagm proposed DEFAULT.json `reduce["any"]`). The partials buffers are unchanged.
3. Config: T128, 128 segments, 4 warps, num_stages 2 (`v2_T128_S128_Q0_w4_st2_F1`), or T64 if the context
   distribution is mostly <= 128K. Selecting the tile by seq_len at run time would need one kernel with both loops
   plus a reduce that takes the tile at run time.
4. Gates still to run: an in-model accuracy check (the error here is at bf16 rounding of served), the sliding
   layers (not touched; they keep the served kernel and the vagm config), bs > 1, and an end-to-end step-time
   measurement. Expected: ~0.22 ms per step at 340K (16 layers x ~14 us).

Beyond this, the remaining gap to the 10.6 us floor at 340K (v2 is at 36.9 us split) needs what Triton cannot
express here: more than one tile in flight per program (precise `s_wait_asynccnt` / TDM double buffering), one
FP8 -> BF16 conversion per element shared by the 4 waves without losing the async copies, and the page table read
ahead into LDS or SGPRs. That is a Gluon / FlyDSL rewrite with explicit layouts, using the gfx1250 Gluon `tdm` /
`async_copy` / `wmma` APIs present in the container's Triton.

## Files

- `kern/attn_v2.py`: the kernel (QSPLIT, DUAL and PFD options for the rejected experiments; served config =
  QSPLIT 0, FOLD 1, DUAL 0, PFD 0). `kern/attn_v2t.py` (transposed) and `kern/attn_v2p.py` (register prefetch) are
  rejected variants. `kern/ua_abl.py`: the served kernel with ablation switches.
- `bench.py`: timing and numerics harness (variants `served`, `best`, `abl<bits>`,
  `v2_T<t>_S<s>_Q<q>_w<w>_st<n>[_F1][_D1][_P<d>][_e<w>]`, `v2t_*`, `v2p_*`). `iso.py`: one process per variant.
  `prof.py` + `tools/profrun.py` + `tools/prof_parse.py`: rocprofv3 counters. `tools/aot.py`: gfx1250 AOT compile
  (no GPU) for ISA. `tools/isa_mix.py`: loop instruction mix.
- `asm/`: served and v2 ISA / TTGIR. `results/bench_*.json`, `results/prof/`, `logs/`.
