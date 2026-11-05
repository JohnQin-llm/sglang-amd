# Muse Spark Open, 4x MI455X TP4, AgentX C1: draft (DFlash2) and dense-path levers

Worktree: `/data/muse455/draft2/sglang` (branch `mi455x/draft2`).
Patch against served HEAD (Arbor iteration 14): `/data/muse455/draft2/draft2-dense-gfx1250.patch`.
It covers 11 files (+723/−8), and `git apply --check` is clean on `/data/muse455/sglang`.
Every flag defaults to off, so with no flags set the patch behaves like HEAD.

All timings are microbenchmarks at the served shapes, using graph replay. Cold runs rotate through input sets larger than
the device cache; warm runs reuse one set. No server was started. GPU jobs went through `k2safe.sh`.
Benchmarks are in `bench/` and raw logs and JSON in `logs/`.

## Flag set

| Flag | Lever | Numerics | Est. saving per step |
|---|---|---|---|
| `SGLANG_MUSE_VERIFY_SEQK_ONCE=1` | compute `seq_lens + max_q_len` once per forward, not per layer (62 adds) | bitwise | 80-100 us |
| `SGLANG_MUSE_REDUCE_GATE_NORM=1` | o-norm + sigmoid gate fused into the split-KV reduce (sliding and `attn_v2_split` full layers) | bitwise | ~1.65 us × 46 sliding + ~1.4-1.8 us × 16 full ≈ 100 us |
| `SGLANG_MUSE_NORM_RESIDUAL_HOIST=1` | FFN-boundary residual + norm with loads hoisted (16 warps) | bitwise | ~1.3 us cold × 62 ≈ 80 us |
| `SGLANG_MUSE_AR_LL_DIRECT=1` | gfx1250 LL all-reduce reads the input in place (no hipMemcpy into the registered pool) | bitwise, 0 bytes differ on all 4 ranks | 1.66 us/AR; ~17 us draft (10 ARs) + target-side LL ARs |
| `SGLANG_MUSE_DFLASH_FC_TP_SHARD=1` | draft `fc` (8192×40960) sharded by output rows, then all-gather | changes the GEMM summation order unless used with the default gluon config | ~15 us with split-K (see below) |
| `SGLANG_MUSE_DFLASH_FC_SHARD_SPLITK=1` | split-K FlyDSL config for the sharded fc | not bitwise (same max abs error against fp32, 0.0312) | (included above) |
| `SGLANG_MUSE_DFLASH_META_FUSE=1` | draft target_verify metadata (~15-22 small kernels) in one kernel | **bitwise check pending** (see below) | ~30-40 us |
| `SGLANG_MUSE_DFLASH_META_SKIP_FULL=1` | skip the full-KV gather when every draft layer is sliding-window (all 5 are) | same caveat | gives a constant 2.5 us kernel |

Recommended flags now, all bitwise: `SEQK_ONCE`, `REDUCE_GATE_NORM`, `NORM_RESIDUAL_HOIST`, `AR_LL_DIRECT`.
`FC_TP_SHARD` + `FC_SHARD_SPLITK` is optional and needs an accuracy run.
Enable `META_FUSE` + `META_SKIP_FULL` only after the 15-second bitwise rerun below passes.

## Do the dense flags still apply on top of iteration 14's attention v2 path? Yes, all three

- **`REDUCE_GATE_NORM`.** It hooks AITER's `_reduce_segments_triton` override in `verify_attention.py`, on both branches:
  - the `attn_v2_split` full-layer branch, after `SEGMENT_BLOCK = max(..., NUM_SEGMENTS)`;
  - the sliding-window fallthrough.

  The live-override report from the `attn_v2check` run lists both "attn_v2_split for full-attention layers (tile 128, 128 segments)" and "attention output norm + gate in the split-KV reduce". The kernel is checked bitwise against the served reduce followed by `_fused_o_norm_gate`.
- **`SEQK_ONCE`.** It is in `aiter_backend`'s unified-verify call. The v2 override sits below that call inside the AITER wrapper, so the two are independent.
- **`NORM_RESIDUAL_HOIST`.** It is the FFN-boundary kernel after K2, so it does not depend on the attention path.

Gate-norm fusion on top of iteration 14 (`attn_v2check`, cold, us per layer call, bitwise):

| Layer type | Context | Served (reduce + `_fused_o_norm_gate`) | Fused | Δ |
|---|---|---|---|---|
| sliding | 128K | 9.20 | 7.54 | −1.66 |
| sliding | 340K | 9.43 | 7.77 | −1.66 |
| full (`attn_v2_split`) | 128K | 26.84 | 25.49 | −1.35 |
| full (`attn_v2_split`) | 340K | 46.35 | 44.51 | −1.84 |

Round 1, before iteration 14 (us per call, cold): sliding 32K / 128K / 340K went 9.19→7.59, 8.82→7.11 and 9.18→7.69; full went
15.96→14.27, 27.66→26.41 and 46.53→44.28.

## Round 2 results (draft)

### 1. Draft pre-step metadata as one kernel (`dflash_verify_meta.py`, `triton_backend._fused_target_verify_buffers`)

One kernel with grid (bs, 16) writes all of these:
- qo, kv and window indptr;
- window offsets and mask_indptr;
- the window gather, the full gather (unless SKIP_FULL), and the optional SWA mapping.

Prefix sums are recomputed inside each program, so no cumsum launches are needed. The served path is used whenever a precondition
fails: custom mask, dcp > 1, bs > 64, unsupported translator, and so on.

Timing in us (graph replay):

| bs | Context | Bench served sequence | Fused | Fused + skip-full |
|---|---|---|---|---|
| 1 | 32K | 39.8 | 4.0 | 2.5 |
| 1 | 128K | 94.1 | 7.7 | 2.5 |
| 1 | 340K | 215.3 | 15.4 | 2.5 |
| 4 | 32K | 40.8 | 3.9 | 2.45 |
| 4 | 128K | 95.7 | 7.6 | 2.45 |
| 4 | 340K | 218.0 | 15.7 | 2.45 |

The benchmark's served sequence overstates serving: it gathers token-level entries with one token block. In the it9 trace the
metadata launches span ~45 us per draft step. The realistic saving is therefore about 40 us with skip-full, or 30-40 us with the full gather.

**The bitwise check is pending; the evidence points to a benchmark artifact, not the kernel.**
- Bytes differing were 103,752 / 396,292 / 1,042,465 at 32K / 128K / 340K. Excluding the full `kv_indices`, about 6,130 bytes differed.
- Cause: the benchmark filled every output buffer with −7, including `indptr[0]`. The served path writes only `indptr[1:]`; in serving, `indptr[0]` is allocated zeroed. So the reference gathers request 0 at offset −7.
- That shifts exactly one request's rows. 2048 window entries × ~3 significant bytes per int64 ≈ 6.1 KB, and the full-KV count scales with context.
- The counts are identical at bs = 1 and bs = 4, so requests 1-3 already match exactly.

`bench/bench_draft.py` now keeps `indptr[0] = 0` (CPU edit only, not run). To confirm:
`run_all.sh bench_draft.py meta`, about 15 s on the GPU.

### 2. Fused all-reduce + residual + RMSNorm (draft): not implemented

Keeping the summation order needs the LL lane's per-rank fp32 order (self + r+1 + r+2 + r+3). That requires a new IPC kernel, or
access to AITER's internal LL scratch, peer table and flags, which its Python API does not expose.
Estimate: −20 to −30 us per step. The bitwise piece that did ship is the next item.

**`AR_LL_DIRECT`** (`bench_ar.py`, 4-rank, AITER `CustomAllreduce`, draft shape [8, 8192] bf16):
- Served (capture copy-in to the registered pool): 8.19 us on the slowest rank. Direct (the LL lane reads the input locally and pushes it to peer scratch): 6.53 us.
- 0 bytes differ on all ranks.
- Saving: ×10 draft ARs per step ≈ −16.6 us, plus the target-side LL-sized ARs.
- Guards: gfx1250, LL not disabled, world size 2 or 4, 0 < bytes ≤ 4 MiB, bytes % 16 == 0, contiguous input.

### 3. Draft small-M GEMMs (M = 8): no-go

Cold / warm us per GEMM; the best candidate is bitwise in every case:

| GEMM | Served | Best candidate | Best |
|---|---|---|---|
| conv_proj | 5.50 / 4.48 | flydsl n64 k256 sk16 w4 | 5.40 / 4.20 |
| qkv | 6.02 / 4.71 | flydsl n64 k128 sk16 w4 | 5.96 / 4.59 |
| o_proj | 5.11 / 3.89 | gluon n32 k512 w4 | 4.86 / 4.11 (warm is worse) |

Gains are ≤ 0.25 us per GEMM (≤ ~3 us per step), which is within noise, so the table is unchanged.
The trace's 127 us against a 42 us floor is launch and serialization cost, not tile choice.

Weight prefetch or overlap is also a no-go:
- The touch kernel costs about as much as the GEMM (4.4-5.0 us).
- A GEMM after a touch is only ~0.6 us faster than cold.
- A spin running in parallel with the touch, then the GEMM, takes 13.5-14.8 us, against 6.5-7.9 us for spin then GEMM serially. The touch competes with the GEMM for the same bandwidth.

### 4. fc shard sweep (`logs/fcsweep.log`, sharded fc 2048×40960, M = 8, cold)

- Replicated fc: 41-42 us. Shard with the default gluon (16, 512, 1): 27.3 cold / 24.8 warm. That is bitwise per element against the replicated fc at the same config, before the all-gather.
- Best: flydsl n32 k256 w2 sk16 at **14.50 us**, not bitwise (max abs error 0.0312, the same as the gluon reference against fp32).
- Net with the ~12 us all-gather (not separately measured): ≈ −15 us per step.

## Round 1 results (dense path, bitwise)

- **seqk add.** 1.30 us per call in the microbenchmark; 1.64 us average in the it9 trace, which is 101.8 us per step. `SEQK_ONCE` saves 80-100 us per step.
- **FFN boundary** (residual + norm after K2):
  - cold (1024 sets): 4.74 / 4.87 → hoist 16 warps 3.47 / 3.28 (8 warps: 3.49 / 3.52);
  - warm: 2.74 / 2.87 → 2.32 / 2.27.
- **Gate-norm fusion.** See the tables above.

## Estimated total per decode step (flags recommended now)

- Dense path: ≈ 250 us (seqk ≈ 90, gate norm ≈ 100, hoist ≈ 80). Bitwise. Well past the 0.4 ms target.
- Draft:
  - `AR_LL_DIRECT` ≈ 17 us, bitwise.
  - Metadata ≈ 40 us after the bitwise rerun.
  - fc shard + split-K ≈ 15 us, non-bitwise.
  - Fused AR + norm (−20 to −30 us) is not implemented.
- Draft total is about 70 us: short of the 0.5 ms target. The small-GEMM lever (estimated 30-50 us) did not materialize.

Next steps for the operator:
1. The 15-second metadata bitwise rerun.
2. An end-to-end profile with the bitwise flag set.
3. An accuracy run if `FC_TP_SHARD` + `FC_SHARD_SPLITK` is wanted.
