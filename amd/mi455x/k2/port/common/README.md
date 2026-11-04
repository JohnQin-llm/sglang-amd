# K2 common primitives, gfx1250 (MI455X, wave32) port

`ops.py` and `mx.py` here are the gfx1250 versions of
`k2bundle/ops/common/{ops,mx}.py` (gfx950, wave64). They keep every public name and signature, so the
stage files can import them unchanged. Where wave32 forces a different meaning, the call signature stays
and the new contract is below (§3).

Status (2026-11-04): **all 29 imported symbols plus the rest of the public API are ported. All 135
GPU unit checks pass on gfx1250** (FlyDSL 0.3.4.1, container `muse_mi455x`, GPU 0). The test logs are
`test_*.log`.

## 1. What the stage files import

I grepped (AST walk over the `from k2bundle.ops.common.{ops,mx} import` statements) `ops/stages/{moe,gemv,oproj,quarter_seam,seam}.py` and `ops/k2.py`:

| file | from `common.ops` | from `common.mx` |
|---|---|---|
| `stages/moe.py` | `CM_DEV CM_NT ballot bf16_round bf_hi bf_lo block_excl_scan butterfly fresh hw_exp2 hw_rsq lanes_below popcount row_sum rsrc traced uniform wave_incl_scan wave_sum` | `FP4 UNIT_SCALE fp4_tile_load mfma_scaled` |
| `stages/gemv.py` | `CM_DEV CM_NT mfma_bf16 rsrc traced` | |
| `stages/oproj.py` | `CM_NT bf16_round mfma_bf16 row_sum rsrc traced` | |
| `stages/quarter_seam.py` | `CM_DEV bf2_f32 bf16_pair bf16_round bf_hi bf_lo hw_rsq rsrc traced wave_sum` | |
| `stages/seam.py` | `CM_DEV bf_hi bf_lo butterfly hw_rsq rsrc traced xshfl` | |
| `ops/k2.py` | `ld_i32 load_ptr64 rsrc traced uniform` | |

The union is 29 symbols: `CM_DEV CM_NT FP4 UNIT_SCALE ballot bf16_pair bf16_round bf2_f32 bf_hi bf_lo
block_excl_scan butterfly fp4_tile_load fresh hw_exp2 hw_rsq lanes_below ld_i32 load_ptr64 mfma_bf16
mfma_scaled popcount row_sum rsrc traced uniform wave_incl_scan wave_sum xshfl`.

Outside that set, `stages/attn.py` also imports `hw_exp mfma_bf16 ld_i32 butterfly ...` and
`stages/mxq.py` imports `hw_exp2 hw_rcp`. Both are covered too.

Other files read `THREADS` / `WAVES` from `common/plan.py` (not ported here). On gfx1250
**`plan.WAVES` must be `THREADS // 32`**. If the `plan.py` next to it disagrees, `ops.py` refuses to
import.

## 2. Per primitive: status, test, note

"Exact" means bit-exact against a torch/numpy reference.

| primitive | gfx1250 lowering | test | result |
|---|---|---|---|
| `xshfl` 1/2/4/8 | one DPP `row_xmask:k` (gfx950 needed 2 DPPs for 4 and 8) | `test_lane` | exact, no LDS |
| `xshfl` 16 | `v_permlanex16_b32` | `test_lane` | exact |
| `xshfl` 32 | pairs only: swaps the two elements | `test_lane` | exact |
| `xred` | 1-8: op(v, DPP); 16: `v_permlane16_swap` and op(row 0, row 1); 32 (pair): op(lo, hi) | `test_lane` | exact |
| `butterfly`, `wave_sum`, `wave_max` | 32-lane on a scalar; gfx950 64-lane order on a pair | `test_lane` | exact (bits vs numpy lane emulation, inputs chosen so other orders differ) |
| `permlane_swap(16)` | `v_permlane16_swap_b32` | `test_lane` | exact |
| `permlane_swap(32)` | pairs: register renaming | `test_lane` | exact |
| `lane_gather` | `ds_bpermute` (source lane mod 32) | `test_lane` | exact |
| `readlane`, `uniform`, `fresh` | unchanged | `test_lane` | exact |
| `row_shr` | constant k: one DPP `row_shr:k` with bound_ctrl; traced k: bpermute as before | `test_lane` | exact, k = 0..15 |
| `wave_incl_scan` | 5 steps (1..16); a pair does a 64-lane scan | `test_lane` | exact |
| `ballot` | `v_cmp` into a 32-bit mask, returned as Int64 (bits 32-63 zero); a pair gives a 64-lane mask | `test_lane` | exact |
| `popcount`, `lanes_below` | unchanged (work on pair masks too) | `test_lane` | exact |
| `block_max`, `block_excl_scan` | 16 waves of 32 (last lane 31) | `test_block` | exact |
| `batched_rounds`, `rows_to_lds` | unchanged (THREADS strided); b128 loads at `SCOPE_DEV` | `test_block` | exact; ISA shows 4 loads before the first store |
| `hw_rsq`, `hw_rcp`, `hw_exp2`, `hw_exp` | `v_rsq/v_rcp/v_exp_f32` | `test_math` | 0.82 / 0.86 / 0.76 / 0.74 ulp max |
| `div_rn` | unchanged (fma) | `test_math` | exact vs IEEE on 65536 pairs (and plain x·(1/d) differs on 17072 of them) |
| `bf16_round`, `bf16_pair` | native f32→bf16 RNE | `test_math` | exact incl. ties, inf, denormals; NaN stays NaN |
| `bf_lo`, `bf_hi`, `bf2_f32` | bit ops | `test_math` | exact |
| `fp8_pack4` | `v_cvt_pk_fp8_f32` | `test_math` | exact vs torch e4m3fn RNE (every code, every tie) |
| `fp8x8_bf16_pk` | **`v_cvt_scale_pk8_bf16_fp8`** (8 values per instruction; gfx950's `cvt_scalef32_pk_bf16_fp8` is not on gfx1250) | `test_math` | exact on all 254 finite codes |
| `mx.clamp_fp8`, `code_ceil`, `pow2`, `rcp_pow2` | unchanged | `test_math` | exact |
| `mx.scale_index` | unchanged (aiter `shuffle_scale` bytes) | `test_math` (host) | exact |
| **`mx.scale_index_gfx1250`** (new) | gfx1250 WMMA scale order | `test_math` (host) | = `layout_gfx1250.shuffle_scale_gfx1250` and `s_dword` |
| `mfma_bf16` | `v_wmma_f32_16x16x32_bf16` | `test_wmma` | layout confirmed; exact on integer data; 3e-8 relative on random |
| `mfma_fp8` | `v_wmma_f32_16x16x64_fp8_fp8` (a 4-dword operand means K = 32) | `test_wmma` | layout confirmed; exact on small values; **3.7e-5 relative on random** (hardware) |
| **`acc_store`, `acc_slot`** (new), `row_sum` | 8 f32 a lane, `(w·32 + lane)·8` | `test_wmma` | exact (16 waves, wave order and subsets) |
| `mx.mfma_scaled` FP4×FP4 | `v_wmma_scale_f32_16x16x128_f8f6f4` | `test_wmma` | exact (narrow scales), ≤ 2.3e-7 (wide), all 4 opsel pairs |
| `mx.mfma_scaled` FP4×FP8, FP8×FP8 | same | `test_wmma` | 3.6e-8 / **7.2e-5** relative |
| `mx.fp4_tile_load` | 2× `buffer_load_b128 th:TH_LOAD_NT` per step | `test_wmma` | end to end: preshuffled [64×1024] FP4 weight with gfx1250 scales × 16 tokens = float64 reference, **exact** |
| `rsrc` | aiter descriptor (CDNA flags; `is_rdna_arch` excludes gfx1250) | `test_mem` | sized: loads past nbytes read 0, stores past it are dropped |
| `ld_i32`, `ld_dev`, `row_live`, `load_ptr64` | unchanged; `ld_dev` = `scope:SCOPE_DEV` | `test_mem` | exact; another CTA's `CM_DEV` store is seen by 127 polling CTAs |
| `global_load` | `global_load_b*`, `nt` → `th:TH_LOAD_NT` | `test_mem` | exact at offsets past 4 GiB |
| `CM_DEV`, `CM_SYS`, `CM_NT` | **16 / 24 / 1** (gfx12 SCOPE_DEV / SCOPE_SYS / TH_NT) | `test_mem` (ISA) | loads and stores encode as intended |
| `memrealtime` | `s_sendmsg_rtn_b64 GET_REALTIME` (`s_memrealtime` cannot be selected on gfx1250) | `test_mem` | 100.05 MHz vs the host clock |
| **`vm_wait`** (new) | `s_wait_loadcnt 0` + `s_wait_storecnt 0` | `test_mem` (ISA) | replaces gfx9 `s_waitcnt vmcnt(0)` (in `sync.py`) |
| `unlikely`, `traced`, `kernel_symbol`, `lds_bytes` | unchanged | `test_mem` | pass |

## 3. Contract changes (wave32)

1. **Lane counts.** `WAVE = 32`, `WAVES = THREADS // 32` (16 at 512 threads). Each of `block_max` and
   `block_excl_scan` needs `WAVES` words of LDS, double the gfx950 count at the same THREADS.
2. **Scalar reductions cover 32 lanes.** On a plain value, `wave_sum`, `wave_max` and `butterfly` reduce
   over the 32-lane wave. `xshfl`, `xred` and `butterfly` *refuse* offset 32 at trace time
   (`ValueError`), so a gfx950 call site cannot silently drop half its data.
3. **Logical wave64 values: `(lo, hi)` tuples.** Logical lane L lives in physical lane L % 32, element
   L // 32. These functions accept a pair:
   - `xshfl`, `xred`, `butterfly`, `wave_sum` and `wave_max` give the **bit-identical gfx950 64-lane
     result**. Offset 32 is the in-lane `op(lo, hi)`, done in gfx950's operand order. The result
     stays a pair until the 32 step, and is one value after it.
   - `wave_incl_scan` scans the 64 logical lanes.
   - `ballot` gives the 64-lane mask; `popcount` and `lanes_below(bal, logical_lane)` then work
     unchanged.
   - `permlane_swap(32, x, y)` returns `(x_lo, y_lo), (x_hi, y_hi)`.

   This is how a stage keeps a 64-lane numerical contract (e.g. the aiter per-head norm order) with no
   cross-wave step: `wave_sum((a, b))` ≡ `wave_sum(a + b)` ≡ gfx950 `wave_sum`. Two 32-lane sums
   combined afterwards (a cross-wave step) are **not** bit-identical, and `test_lane` shows they differ.
4. **WMMA operands and accumulators.**
   - Every gfx1250 operand: lane l holds row (A) or column (B) l % 16, and its 16-byte chunk j holds
     K run 2j + l // 16. That is 8 bf16, 16 fp8 or 32 fp4 per run, with the low nibble as the even k.
   - `mfma_bf16`: `a`, `b` are 16 bf16 per lane and `c` is 8 f32 (gfx950: 8 bf16 and 4 f32).
   - `mfma_fp8`: K is 64 per call (8 dwords per lane). A 4-dword operand gives K = 32, with the upper
     half zero.
   - D layout: lane l holds column l % 16, rows 8·(l // 16) + v for v = 0..7 (gfx950: 4·(l // 16) + v).
   - `row_sum(red, r, t, waves)` reads the layout `acc_store(red, acc, wave, lane)` writes:
     `red + (w·32 + lane)·8`. That is still `ACC_WORDS` = 256 words per wave, so the K2
     `red: Array[f32, WAVES*256]` sizing is unchanged. **Stages must replace their
     `fx.ptr_store(acc, red + (wave*64 + lane)*4)`** (moe.py, oproj.py) **and gemv's local
     `reduce_tensile_split` index** with `acc_store` / `acc_slot`.
5. **Scaled WMMA.** `mfma_scaled(a, b, c, sa, sb, a_fmt, b_fmt, sa_byte, sb_byte)`:
   - Operands: FP4 is 8 dwords per lane, FP8 is 16, and `c` is 8 f32.
   - `sa` / `sb` are a lane's **scale dword**: byte j is the E8M0 of K block j of row/column l % 16. It is
     no longer one byte.
   - `sa_byte` / `sb_byte` are now the **opsel**: 0 reads lanes 0-15's dwords, 1 reads lanes 16-31's.
     gfx950's byte select 0-3 does not exist here; 2 and 3 are refused.
   - The usual pattern is one dword load per lane for a pair of 128-k steps, with opsel = step parity.
     `scale_index_gfx1250` and `layout_gfx1250.s_dword` give the address.
   - The stage code that extracts a byte (`(sw >> ...) & 0xFF`) has to go.
6. **`fp4_tile_load(r_w, base, rg, k, st, at)`** returns 8 dwords: two b128 loads at `at` and `at + 128`,
   with `at = lane * 4` for lane 0..31 (the same formula as gfx950). The weight bytes are the gfx950
   (16,16) preshuffle, unchanged.
7. **Cache policy constants changed value.** `CM_NT` = 1, `CM_DEV` = 16, `CM_SYS` = 24.
   - On gfx12, `2` (gfx950's NT) means TH_HT, the opposite hint.
   - `17` (gfx950's SYS) means NT + device scope.
   - Use the names, never literal numbers; `CM_NT | CM_DEV` is a valid combination.
8. **`memrealtime`** reads the 100 MHz constant clock via `s_sendmsg_rtn_b64`, so the rate is the same as
   gfx950's. It is an SALU message, not an SMEM load.
9. **`lane_gather`** takes its source lane mod 32. `row_live` and `load_ptr64` are wave-uniform as
   before (they go through `readfirstlane`), so pass them a uniform index.
10. New names: `WAVE`, `ACC_PER_LANE`, `ACC_WORDS`, `acc_slot`, `acc_store`, `vm_wait`,
    `mx.scale_index_gfx1250`.

### Call sites that change meaning (for the stage porters)

| call site | gfx950 meaning | on gfx1250 |
|---|---|---|
| `seam.py:71` `butterfly(v, (1, 2, 32, 16, 8, 4))` | 64 lanes | pass `(lo, hi)`, otherwise it raises |
| `quarter_seam.py:62,110,128,134`, `moe.py:326,389,708,1314,1329,1418,1439` `wave_sum(...)` | 64-lane sum | 32-lane on a scalar; pass a pair (or fold `lo + hi` first) to keep the 64-lane order bit-exact |
| `seam.py:97,141,155`, `moe.py:724,873` butterflies with offsets ≤ 16 | sum within 32-lane groups | same values per 32 lanes, unchanged |
| `moe.py:525-552` router `ballot` / `lanes_below` over `lane + 64 i` | 64 candidates per ballot | 32 per ballot, or pair predicates for 64 logical lanes |
| `moe.py:616` `wave_incl_scan`, `:666` `block_excl_scan` | 64-lane / CTA | 32-lane (pair: 64) / CTA (unchanged) |
| `moe.py:1086-1087` `lane = tid % 64`, `wave = tid // 64` | | `tid % 32`, `tid // 32` (stage code) |
| `moe.py` / `oproj.py` `row_sum` + `red + (wave*64+lane)*4` stores | 4-f32 accumulators | `acc_store`; `row_sum` is unchanged at the call |
| `moe.py:920,948,1110,1253` `mfma_scaled(..., FP4, FP4, 0, 0)` | byte scales, 4-dword operands | dword scales, 8-dword operands, opsel = step parity |
| `gemv.py`, `oproj.py` `mfma_bf16` | 8 bf16 / 4 f32 | 16 bf16 / 8 f32 (operand loads change) |

## 4. Hardware observations

- **FP8 WMMA products are not exact in fp32.** The relative error is 3.7e-5 for unscaled 16x16x64 fp8
  and 7.2e-5 for scaled FP8×FP8, while both are exact on small-integer data, so the layouts are right.
  FP4×FP4 and bf16 are exact or near-exact (≤ 2.3e-7 / 3e-8). On this test's data (scales 2^-5..2^5,
  randn fp8) FP4×FP8 measured only 3.6e-8, below the 5e-6..1e-5 the S2 probe reported. The precision
  loss may depend on the data distribution.
- gfx1250 `make_buffer_rsrc` runs with aiter's CDNA flag word (`0x27000`; `is_rdna_arch` excludes
  gfx1250). Bounds checking works: a sized resource reads 0 and drops stores past `nbytes`.
- Every gfx1250 kernel starts with `s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1`, emitted by
  the toolchain.

## 5. Running

```bash
for t in lane math block wmma mem; do
  flock /data/muse455/locks/gpu0.lock docker exec muse_mi455x bash -c \
    "cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=0 timeout 300 python3 /data/muse455/k2/ep1/port/common/test_$t.py" \
    2>&1 | grep -E '^  (PASS|FAIL)|^==' | tee /data/muse455/k2/ep1/port/common/test_$t.log
done
```

`_gpu.py` is the test harness. It builds one-CTA (or grid) kernels over four raw pointers, captures the
ISA (`FLYDSL_DUMP_IR=1` → `<dump>/<kernel>/NN_final_isa.s`), and reports PASS/FAIL. The tests import
`ops` / `mx` as top-level modules. Inside the `k2bundle` package the relative imports (`.plan`, `.ops`)
are used instead.
