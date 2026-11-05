# K2 mono kernel, TP4/EP1, MI455X (gfx1250)

Workspace for the TP4/EP1 persistent per-layer MoE kernel. Served tree `/data/muse455/sglang` and
`/sgl-workspace/aiter` are **not modified**. Stage-1 (EP4-shaped) work this builds on: `../README.md`.

Status log (newest last). Times are UTC (container clock).

## S0: XGMI sync floor (done, 2026-11-04 02:54-03:13)

Files: `s0/xgmi_bench.hip` (+ binary), `s0/*.txt`, `s0/*.json`, `s0/car_bench.py`, `s0/car_bench.json`.

**Caveat.** A TP4/EP4 SGLang server (not started by me) came up at 03:02:05 on GPUs 0-3 and loaded weights while
the 03:05-03:12 runs below were going. The ping-pong numbers match the clean 02:54 run (2.33 µs/hop) to within 3%,
so they stand. The no-fan-in exchange timeouts (below) and the custom-AR numbers are **confounded** and should be
re-run with the GPUs idle. All later stages run on GPU 3 only.

### Point-to-point and barrier

| measurement (uncached IPC-style memory, 1 thread) | µs |
|---|---:|
| one-way hop, push (store to peer, poll own), sys-scope release/acquire, 6 pairs | **2.33-2.40** (range 2.24-2.50) |
| ring 0→1→2→3→0 and 0→2→1→3→0, per hop (02:54, idle GPUs) | 2.33 / 2.40 |
| one-way hop, push, **relaxed** store/load (no fences; the tagged-mailbox style) | **2.02** |
| one-way hop, pull (store own, poll peer) | 3.56-3.65 |
| same-GPU two-CTA hop, sys-scope release/acquire | 0.98-1.08 |
| 4-rank barrier (push to all, poll own 4 slots), 1 CTA/rank | **3.6** |

- Memory kind (uncached / fine-grained / coarse) makes no difference to any of these (±2%).
- The release/acquire pair costs ~0.4 µs per hop over relaxed, and even a same-GPU hop costs ~1 µs: the gfx12
  system-scope release is an L2 writeback. **K2 hand-offs must use the tagged mailbox (value+tag in one 64-bit
  relaxed store), not flag + fence.**
- Push beats pull by 1.2 µs: **always write into the consumer's memory and poll locally.**

### Tagged-mailbox all-to-all exchange (K2 hand-off shape)

Every rank sends P dwords to each rank (including itself) as 64-bit {value, tag=epoch+1} pairs, double-buffered
by epoch parity, then polls its own inbox until every tag matches. Slowest rank, µs per exchange:

| payload per peer | G=1 CTA | G=16 | G=64 | G=256 |
|---|---:|---:|---:|---:|
| 1 dword | 3.38 | - | - | - |
| 256 B | 3.49 | 3.69 | 5.50 | - |
| 4 KB | 8.78 | 3.47 | 3.57 | 3.82 |
| 32 KB | 67.7 | 5.19 | 3.59 | 3.54 |
| 128 KB (EP1 yr reduce-scatter per peer) | 271 | 17.1 | 7.14 | **3.91** |
| 512 KB | 1084 | 69.1 | 20.0 | 18.5 (with fan-in) |

- **A K2 cross-GPU hand-off costs ~3.5-3.9 µs** for anything up to 128 KB per peer if ≥64-256 CTAs share it.
  128 KB per peer in 3.9 µs is ~100 GB/s per link-direction of payload (200 GB/s with tags).
- **Grid fan-in is expensive:** exchange + a 256-CTA device-scope atomic fan-in = 6.1 µs vs 3.4 µs without, so a
  grid-wide barrier costs ~2.6 µs at 256 CTAs (1.0 µs at 16 CTAs). K2 must use targeted flags, not grid barriers,
  on the critical path.
- **Soak (K2 regime, fan-in between epochs): 3×10⁶ epochs, 0 stale reads, 0 timeouts** (128 KB/peer 9.2 µs,
  32 KB/peer 7.2 µs, 1 dword 6.1 µs per exchange+fan-in).
- **No torn {value, tag} words were ever observed.** But without a fan-in between epochs, ≥128 KB/peer at G=256
  hit 2-second receive timeouts in 7 of 9 repetitions (and one 1 s stall that recovered). These runs overlapped
  the server's weight load, which may have kept some of my 256 CTAs from being resident (that alone deadlocks a
  cross-rank all-CTA protocol). **Open: re-run `xgmi_bench 5000 "" 4` and `... 5` on idle GPUs.** Until then the
  rule is: fan-in (or launch boundary) between epochs that reuse a parity buffer.

### Image custom all-reduce (aiter gfx1250 kernel, IPC), 4 ranks (confounded, see caveat)

| op | graph µs/call | eager µs/call |
|---|---:|---:|
| all-reduce [8,4096] bf16 (64 KB) | **7.5** | 15.3 |
| all-reduce [8,8192] bf16 (128 KB) | 51.1 (suspicious; slow path?) | 89.7 |
| all-reduce [64,4096] bf16 (512 KB) | 55.9 | 22.3 |
| reduce-scatter [64,4096] dim 1 | - | 93.5 |
| all-gather [8,1024] / [8,2048] dim 1 | - | 67.2 / 47.1 |

The [8,4096] all-reduce, the one the served MoE block does today, costs 7.5 µs in a graph, i.e. about two
mailbox hand-offs. The larger shapes are anomalous and need a clean re-run.

### What S0 means for K2 at TP4/EP1

The EP1 layer has 4-5 cross-GPU hand-offs on its critical path (router/pooler gather if TP-split, yr
reduce-scatter to column owners, FIN all-gather, unpool/branch gather, finish consensus): **~15-20 µs of chain
floor** at 3.5-3.9 µs each, plus intra-GPU fan-ins. That is about the same as the byte floor (15-20 µs), so the
40-60 µs estimate stands only if every stage overlaps its weight stream with the preceding hand-off.

## S1: EP1 loader parity (done, 03:20)

Files: `ep1_layout.py` (the slicer, reused by the integration package), `s1/test_ep1_loader.py`,
`s1/test_ep1_loader.{json,txt}`.

Each rank keeps all 256 experts and a quarter (1024) of every expert's intermediate dimension: w13 = [gate
rows r·1024..; up rows r·1024..] (the C1 `_tp_expert_stack_slice` order, which is also the row order the FlyDSL
stage-1 kernel reads: tile rg = gate, rg + tiles = up), w2 = down's packed-K bytes r·512.. and scale blocks
r·32... The slice is taken on the native checkpoint tensors, then the gfx1250 layout is applied (weight bytes in
the (16,16) preshuffle, scales permuted for the WMMA opsel order). 1024 is a multiple of the 32-wide MXFP4
block, so no scale block is split.

| check (layers 1, 31, 60) | result |
|---|---|
| A: the 4 ranks' gfx1250 slices, un-shuffled and un-sliced, reproduce every byte of all 256 experts (w, scales) | **PASS** ×3 |
| B: gfx1250 layout round trip of every rank slice | **PASS** ×3 |
| B: kernel addressing (`w_dword`/`s_dword`) with EP1 shapes (w13 2048×4096, w2 4096×1024) gathers the WMMA lane operands and scale bytes | **PASS** ×3 |
| C (GPU 3): Σ over ranks of the partial FFN with each slice vs the full expert FFN, 8 experts × 8 rows per layer, float64 | rel-L2 ≤ 2.1e-15 (fp32: 1.1e-6) **PASS** ×3 |

Memory: the EP1 MXFP4 copy is **102.7 GB per GPU** (layers 1-60; layers 0/61 stay on the BF16 path, as on
MI355X). Today's TP4/EP4 server uses 154.7 GB of weights + 3.6 GB draft; at mem-fraction 0.85 the KV pool
gets the rest (~215 GB), and the GPUs show 86 GB free with the server idle. So:
- The EP1 copy does **not** fit as a hot add beside the running server (86 < 103 GB).
- It **fits if loaded with the weights** (the KV pool shrinks to ~110 GB). One C1 request at 300K context
  needs ~10 GB of FP8 KV per GPU at TP4 (4 KV heads × 64 × 2 × 62 layers × 1 B ≈ 32 KB/token), so the pool
  is not a constraint at C1. The EP4 copy stays for prefill and non-verify batches.

## S2: expert stage at EP1 (numerics + per-rank time done; 4-GPU exchange blocked by the resident server)

Files: `fly_expert_gemm_ep1.py` (FlyDSL A4W4 / **A8W4** grouped expert GEMM, EP1 shapes), `s2/ep1_moe.py`
(harness), `s2/ep1_moe.json` (A8, default), `s2/ep1_moe_a4.json`, `s2/act_quant_study.{py,json}`,
`s2/wmma_probe_fp8.py` (+ `_map.json`).

**Method.** One GPU (GPU 3; a server holds 0-3) simulates the 4 ranks one after another; the cross-rank
reduction is done in torch with the exact arithmetic the in-kernel owner will do (sum the 4 bf16 partials of each
route in rank order in fp32, round to bf16). Inputs follow the MI355X harness (`eval_moe_layer.py`): 8 draft rows
share a base vector, `corr` is calibrated per layer so the rows touch ~43 distinct experts (got 39-50), 4 input
sets per layer; `ffn_input = pre_feedforward_layernorm(stream)` (attention not simulated). No server dump of real
activations was possible (that needs a hook in the served tree). Arms:
- **stock** = served TP4/EP4 path: fp32 router, the routing rule, bf16 pooler + pre-expert norm, served Triton
  `muse_moe_triton` per EP4 rank (A16), sum over ranks, bf16 unpooler, FFN boundary;
- **ep1** = the K2 dataflow: FlyDSL gate/up+SiLU over each rank's 1024-wide slice of all 256 experts, quantize
  `inter`, FlyDSL down over K = 1024, bf16 partial per route, rank-order reduction, finalize (post-expert norm,
  routed sum), unpooler, boundary;
- **ref** = float64, A16, exact dequantized weights.

### Numerics: A4 activations miss the bar, A8 activations pass

Residual-stream cosine (min over 32 rows) after the FFN boundary, ep1 vs stock:

| layer | A4W4 (MXFP4 xe + inter) | **A8W4 (MXFP8 e4m3 xe + inter)** | next-attn input, A8 | per-route y vs ref, A4 / A8 |
|---|---:|---:|---:|---:|
| 1 | 0.99902 | **0.99993** | 0.99991 | 0.9856 / 0.9988 |
| 31 | 0.99823 | **0.99990** | 0.99988 | 0.9745 / 0.9981 |
| 60 | **0.99697 (below the 0.997 floor)** | **0.99981** | 0.99977 | 0.9698 / 0.9980 |

- Stock vs the float64 reference is 0.999998-0.999999 on the stream, so the whole gap is the expert activation
  quantization. The torch emulation (`act_quant_study`) reproduces the kernel's stream cosine to 1e-6 for both
  formats. It also shows each input matters: A8 on xe only gives 0.9986 on layer 60, A8 on inter only 0.9981.
- **Decision: the gfx1250 K2 runs A8W4** (MXFP8 e4m3 activations, e8m0 per 32, the OCP rule). It costs nothing at
  8 rows: activations are 32 KB/layer, and the measured time is unchanged.
- The FP8 B-operand layout of `V_WMMA_SCALE_F32_16X16X128_F8F6F4` was confirmed on hardware: lane l holds column
  l%16; 16-byte chunk c (dwords 4c..4c+3) holds K = 32c + 16·(l//16) + 0..15. Scales are unchanged.
- Exactness: A4W4 EP1 kernel vs a torch model of the same arithmetic: rel-L2 **0** (stage 1 and 2). A8W4: 1-2e-3
  on bf16 outputs, because the hardware FP8×FP4 scaled dot itself is not exact (random probe: 5e-6 to 1e-5
  relative per instruction vs ~1e-8 for FP4×FP4, so it loses internal precision). That error is ~50x below the
  A8 quantization error and invisible at the stream level.
- Routes are identical between arms by construction here (same torch router); in-kernel routing is checked in S3.
  The smallest 8th-vs-9th biased-score gap seen was 7.5e-5 (layer 60), i.e. near-ties do occur.

### Time per layer (graph replay, weights streamed from HBM)

| per rank, 8 rows | layer 1 | layer 31 | layer 60 |
|---|---:|---:|---:|
| **EP1 FlyDSL A8W4, stage 1 + stage 2** | 23.9 + 11.1 = **35.0 µs** | 24.4 + 11.4 = **35.8 µs** | 24.3 + 11.2 = **35.5 µs** |
| served EP4 Triton (align → finalize), per rank | 96-113 | 95-129 | 89-113 |
| served EP4 Triton, worst rank (the critical path) | **113.3 µs** | **128.5 µs** | **113.1 µs** |

- EP1 removes the rank skew. EP4 ranks hold 6-18 of the ~45 distinct experts, and the slowest one sets the layer
  time; the per-rank EP1 work is the same on every rank.
- EP1 reads ~45 × 6.7 MB = 300 MB per rank in 35.5 µs, i.e. **8.5 TB/s (36% of peak)**; the floor at 23.3 TB/s is
  12.9 µs. Stage 1 (200 MB) runs at 8.3 TB/s. Same tuning headroom as noted in stage 1 of the port: in-CTA
  split-K, launching only live units, TDM.
- Not included: align (Triton, ~7 µs today), activation quant (torch here; fused in K2), finalize, the exchange.

### Cross-rank exchange: blocked

The reduce of [8 rows × 8 routes × 4096] bf16 partials to their owners, first with the image collectives and then
in-kernel via the mailbox, needs GPUs 0-3, and the server has held them since 03:02. What is known:
- **Owner by row, not by column.** Each rank owns 2 of the 8 rows (all 8 routes of each). It receives 16 routes ×
  4096 bf16 = 128 KB from each peer, which is the same bytes as a column split. The owner then has whole route
  rows, so it does the post-expert norm locally, with no extra exchange of Σx² partials.
- Mailbox cost from S0: **3.9 µs** for 128 KB per peer with 256 CTAs (tagged 64-bit words, two bf16 per word).
- Custom-AR reduce-scatter of the same [64, 4096] took 93 µs eager (confounded run). The image collectives are not
  a viable path for this exchange, so the in-kernel mailbox is the plan.

## S3: module port with four helpers (from 04:00; lead = this agent)

The operator switched S3 to a parallel module port of the MI355X K2 (`k2bundle`, the C1 path), keeping its
function signatures. Contract: **`port/INTERFACE.md`**. Fixed constants: **`port/consts.py`**. Numerics harness:
**`port/validation.py`** (all three lead-owned; helpers document their work in `port/<dir>/README.md`).

| dir | owner | modules | status (04:30) |
|---|---|---|---|
| `port/common/` | P1 | `ops.py`, `mx.py` | delivered: 137/137 GPU unit checks pass (lane, math, block, wmma, mem) |
| `port/sync/` | P2 | `sync, residency, epoch, plan, ranks, peer, abi, failstop, tp4` | ported, ISA checked; 4-rank test written, result pending (needs all 4 GPU locks) |
| `port/moe/` | P3 | `moe.py`, `mxq.py` | layer 31 A4: every stage bit-exact vs reference; A8, layers 1/60, timing in progress |
| `port/gemv/` | P4 | `gemv, oproj, seam, seam_b, quarter_seam, moe_dense` | in progress (own copies of common/mailbox until merge) |
| `port/k2.py` | lead | assembly, CTA map | next |

Decisions taken while writing the contract (04:00-04:30):
- **Grid: 256 CTAs × 512 threads = 16 wave32 waves** (the helpers' choice; MI355X loops over THREADS carry over).
- **The target is the build C1 served:** width 8 ran the q8 build `K2Build(tokens=8, experimental_two_cta=True)`
  (512 co-resident CTAs on MI355X). Width 1 ran the default build. The CTA placement is redone at TP4 by the lead.
  The MI355X placement breaks: the pooler needs 80 + 256 CTAs, and the unpooler overlaps the gather.
- **Row owner `t // 2`** (P2 and P3 agree), expert activations MXFP8 (S2), cache policy 16/24/1 (DEV/SYS/NT).
- Incident 04:22: the tree was rolled back to a snapshot from about 04:17. Files created later were deleted
  (`port/INTERFACE.md`, `port/consts.py`, P3's `moe/test_moe.py`) and later edits reverted (this README).
  The lead files were restored at 04:30, and copies are kept outside the tree.

### Lead progress (04:30-04:40)

- **Numerics harness `port/validation.py`** (torch, no kernel). `k2ref` is the TP4/EP1 dataflow with the kernel's
  rounding points; `stock` is the served math. `check()` grades kernel outputs: stream cosine after the native FFN
  boundary, plus routes equal except near-ties (a near-tie row is graded against stock on the kernel's routes).
  The predicted bar, k2ref (A8W4) vs stock, 43 distinct experts, 2 seeds a layer
  (`port/results/validation_k2ref_a8.json`):

  | layer | stream cosine min (seed 0 / 1) | routes |
  |---|---|---|
  | 1 | 0.99986 / 0.99991 | equal |
  | 31 | 0.99588* / 0.99991 | equal except 2 near-tie rows (seed 0) |
  | 60 | 0.99979 / 0.99980 | equal |

  \* Graded against stock's own routes. That run predates the near-tie rule. The flip comes from the rank-ordered
  bf16 o_proj partial sum versus one GEMM. It ran at 04:31 on GPU 1 for 5 s, overlapping the start of the
  campaign server (seen only afterwards).
- **Served baseline for the span K2 replaces** (o_proj .. unpooler, FFN boundary excluded), from the C1 128K
  profile (`/data/muse455/prof/ctx128k`, 300 verify layers a rank, `s3/served_span_layers.py`): **170 µs kernel
  time, 180 µs wall a layer a rank**. Parts: expert GEMMs 50, all-reduces 22-32, align 32, unpooler (Tensile) 19,
  router+pooler `_dense_partial` 10, o_proj 8.5, norm 5.6, finalize 5.4, routing epilogue 5.2, copies 3.5.
  Across 60 MoE layers that is ~10.8 ms of an 80 ms verify step at 128K.
- **`port/k2.py`** (assembly: TP4 constants, peer/scratch layouts, 256-CTA role map, stage order, finish
  consensus), **`port/runner.py`** (peer runtime, per-layer operands, EP1 expert copy read with safetensors slices),
  and **`port/layout.py`** (vendored gfx1250 weight layout) are written. They are not compiled yet: the GPUs have
  been held by the campaign bench since 04:31, and the MoE stages still need in-kernel hand-offs (request to P3
  in INTERFACE.md).
- **`s3/run_k2.py`**: the 4-GPU K2 test (numerics vs the bars + graph-replay timing a layer).
- **S4 draft patch `s4/k2_gfx1250.patch`** (31 files) is built by `s4/make_patch.py` from copies of the served
  tree (the served tree is not touched). It applies with `git apply`, and all files parse. Hooks, all behind
  `SGLANG_MUSE_MONO_GFX1250=1`:
  - attention `core_only` (returns the o_proj input);
  - in the decoder layer, K2 then the native FFN boundary, for T=8 steps on layers 1..60;
  - an epoch advance once a step;
  - the runner, built in `post_load_weights`.

  The served EP4 path stays for every other step.

### Lead progress (04:45-05:00): the full K2 compiles

- **P3 hand-offs merged by the lead.** P3 delivered nothing after the 04:22 rollback, so the lead made the
  04:40 edits in `moe/moe.py`. They are gated on `"poll" in c`, so P3's one-stage launchers are unchanged:
  - xq polls POOL pairs and publishes XQRDY;
  - load_route polls ROUTE pairs;
  - UG publishes UGRDY per (slot, group), and down polls its slot's 32 flags;
  - down writes YR as tagged bf16 pairs to the row owner;
  - the finalize polls YR, FMSQ and the route weight, stores FIN plain into every rank's peer copy, then
    publishes FRDY (row, slice) at system scope.

  Two fixes in P3's code: `_chain` needed a static step list (the K part `st0` is traced in a 16-wave CTA), and
  `from __future__ import annotations` broke the launchers' LDS structs. P4's `stage_up` now polls
  `c["frdy_flags"]` (K2: 256) instead of 32.
- **Compile-only check `s3/compile_k2.py`** needs no GPU (`HIP_VISIBLE_DEVICES=` empty). The K2 kernel
  (`muse_mono_k2g_s8_a8`) compiles in 5.4 s: **194 VGPRs (cap 256), no VGPR spill, 0 scratch, LDS 117 KB** (of
  320), 38 SGPR spills to VGPR lanes (`s3/compile_k2_a8.json`). P3's six standalone launchers compile too.
- **Helper stage timings so far** (separate launches, net of the ~1.3 µs empty launch): o_proj 3.0-3.7 µs, pooler
  3.2-3.6, unpooler 2.9-3.1, q8 router 2.8 (P4, `gemv/results/`); experts A8 per rank: UG 25, down 14, xq 3.7,
  finalize 4.1 µs (P3, `moe/test_moe_a8.txt`; finalize output vs stock A16 cosine ≥ 0.9986). Rough K2 critical
  path: o_proj 3.5 + AR hop 3 + seam/route 4 + xq 3.7 + experts 39 + YR exchange 4 + finalize 4 + FIN all-gather
  3 + unpooler 3 + BR gather 3 ≈ **70-75 µs a layer**, against 170-180 µs served (to be measured).
- **Served baseline moved.** The campaign committed `m-moe-triton` at ~04:45 (served MoE layer 122 → 84 µs). The
  170-180 µs span above predates it, so the served span will be re-measured for the S3 comparison.
- **Not run yet:** `s3/run_k2.py` (4 GPUs). The campaign server and bench have held the GPUs again since 04:46.

### First 4-GPU K2 runs (05:03-05:15, all four locks)

- **Numerics pass on layer 31 (real weights, 43 distinct experts, one launch a layer a rank):** stream cosine
  after the FFN boundary min **0.99992** (bar 0.997 / target 0.999). Routes equal; rows 0 and 2 are near-ties.
  Branch cosine vs stock 0.99921, vs k2ref 0.99994. Raw stream_out cosine 0.999999. Four back-to-back eager
  launches give the same result.
- **Open bug: the second back-to-back launch on a fresh side stream (graph warm-up) stalls.** It reproduces at
  epoch 7 on every run. Rank 0's gather CTAs wait on a few BR pairs from rank 3's unpooler task 103 (tokens 6, 7,
  columns 7804..7807) and still see the epoch-5 tag. Host launch times are < 1 ms, so this is not a late start.
  Diagnosis is in progress: `run_k2.py` now reports every rank's give-ups and re-reads the pairs after the kernel.
- The first timing figure (4.4 µs a layer) was invalid and is discarded: the launch went to the pre-capture
  stream, so the graph held only the epoch increments. Fixed: the launch uses the current stream.
- **Incident 05:11:** a diagnostic run started while a server process (PID 133462, not the harness's own server)
  was already up. The command printed the pre-check but did not gate on it. The run overlapped the server for
  about 4 minutes and ended with no output. GPU commands now refuse to start when `pgrep` finds a server or the
  driver log shows a bench.

- **Operator, 05:08:** back-to-back launches without a host barrier fail the in-kernel wait; with a barrier they pass. See `s3/OPERATOR_FINDING.md`. This blocks both the S3 timing and S4.

## Operator, 05:30-06:20: back-to-back failure root-caused and fixed; first K2 timing

- **Cause:** on gfx1250 a spinning mailbox poll (`buffer_load` with `scope:SCOPE_SYS`, peer regions; `SCOPE_DEV`,
  scratch) can keep returning a stale line from this XCD's L2 for the whole 10 s budget, although the producer wrote
  the new value. Seen as `br` / `yr` / `frdy` waits stuck on the tag from two epochs earlier, producers never giving up.
  It also fails with barrier-separated launches (15 timed singles), so it is not specific to back-to-back launches.
- **Ruled out:** stale epoch read (every CTA's epoch recorded: correct), GPU hardware (fails the same with the rank->GPU
  map reversed), load temporal hints TH=1/3, fine-grained (0x1) or coarse (0x0) peer memory instead of uncached.
  ISA check: the polls are compiled with the intended scopes.
- **Fix (now default, `sync/sync.py`):** `global_inv scope:SCOPE_DEV; s_wait_loadcnt 0` before a re-poll
  (`K2_INV_POLL=1`, `K2_INV_SCOPE=SCOPE_DEV`, `K2_INV_EVERY=N`). SCOPE_SE fails, SCOPE_DEV and the LLVM SYS acquire
  fence pass. A/B: 30 back-to-back launches pass with it, fail without. Every 16th re-poll: passes; every 128th: fails.
- **K2 timing, layer 31, graph of 20 launches, all ranks:** 233 us/layer (invalidate every re-poll), 214 us (every 16th).
  Numerics unchanged (stream2 cosine min 0.99992, routes equal except near-ties).
- **Stage timeline** (`K2_DEBUG_TL=1`, `s3/probe_b2b.py --timeline`, per-CTA clock, rank 0 medians): oproj 7, router 24,
  pool 21, xq 23, route_wait 23, UG 51, down 72, finalize 82, up 46, gather 65 us; CTA span 234 us. Standalone helper
  timings were UG 25, down 14, xq 3.7, finalize 4.1 us: the expert stages run 2-5x slower inside K2.
- Debug switches (off by default, part of the build key): `K2_DEBUG_EPOCH`, `K2_DEBUG_TL`, `K2_POLL_TH`, `K2_PEER_MALLOC`.
- **Update 06:23 (operator):** the root cause is a poll storm, not stale caches. With `s_sleep(1)` between re-polls
  thousands of waves starve the producers; `s_sleep(32)` with no invalidate passes 30 back-to-back launches and graph
  replay (sleep <= 8 fails, 16 nearly stalls). New defaults: `K2_POLL_SLEEP=32`, `K2_INV_POLL=0`. Timing 199-200 us/layer.
  Remaining cost: the first down task of each CTA waits ~46 us for UGRDY flags set 30-75 us earlier (its chain: 3-12 us).
  Next: down tasks on the CTA that produced the UG slice (CTA barrier, no flag), one poller per CTA + LDS broadcast
  for the remaining hand-offs (mega_moe_tp pattern).

## k2v2ho (hand-offs off the tail, from k2v2pre, 2026-11-04 16:57-18:57 UTC)

Switches (build key `_ho_key`, `port/k2.py`); logs `logs_ho/`:
- `K2_UP_ALL` (**default 1**): every rank unpools all 8192 branch columns of all 8 tokens from the FIN all-gather
  (`w_unpool` is already the full replicated [8192, 4096] on every rank), 2 tasks a CTA on all 256 CTAs (task
  g = 2 bid + i: rank r's old task j is g = 128 r + j, same rows, chunks, wave order, row sums), the bf16 words
  stored straight to `branch_out` (`moe_dense.stage_up_all`). The BR all-gather and the gather stage are gone
  (one cross-GPU hop). Cost: +48 MB of unpool weight reads a GPU (64 MB, was 16), 128 VGPRs of prefetched unpool
  weights live across the finalize: VGPRs 213 -> 251, 0 spills, LDS unchanged (145712 B).
  `K2_UP_FINW=post` (finalize CTAs issue their unpool weights after the finalize): 69.2 vs 67.9 us, kept `pre`.
- `K2_FINISH_ASYNC` (**default 1**): no end-of-launch consensus (no finish flags, no done4 exchange). Safety: a
  rank can start launch n+1 only after its unpoolers saw every peer's FIN(n), which implies each peer is past its
  seam / pool / xq / YR reads of launch n; the regions written early in n+1 (AR, POOLP) are read early in n, the
  late ones (YR, FINP) are written in n+1 only after the peer's own n+1 data arrived; stale pairs are rejected by
  the (epoch, slot) tag. Give-ups still raise the output rank's alarm word (the runner reads it after a step);
  CTA 0 wave 1 imports those words into its sticky FAILED before the unpooler (best effort, under the FIN hop),
  instead of the done4 consensus. Without K2_UP_ALL the import runs on CTA 192 after its finalize.
- Tried, not kept: `K2_XQ_GATE` (prototype in `dev/` only, not in this tree's `port`): one wave polls one POOLP
  pair a producer task, then xq reads its 128 KB once. TL: xq sees all parts 21.6-24.3 vs 19.8-22.1 us; graph
  71.0 vs 68.5 us (L31). The serial gate + single read costs more than the re-polls it saves.

Layer 31 attribution (`s3/run_k2.py`, graph, median of ranks, one session): both off (k2v2pre program) 74.9;
UP_ALL 72.4; FINISH_ASYNC 70.3; both 67.9 (68.5 in a re-run). Numerics digit-identical in every config
(stream2 cos min 0.9999225495115383, branch cos 0.9992055330424373, routes equal, near-tie rows 0, 2).
Timeline (L31, rank 0..3 earliest-CTA ABS us; `logs_ho/tl_base.log` vs `tl_ua1_fa1.log`): FIN done 53.6-55.1 ->
unpooler sees FIN 57.6-58.7 -> up done 60.4-61.7 -> gather sees BR 61.5-62.5 -> CTA span max 69.2-70.9 (base);
FIN done 54.4-55.5 -> unpooler sees FIN 58.0-59.8 -> up done (2 tasks) 61.7-63.7 = kernel end, span max 63.2-64.7.
Robustness (defaults): probe_b2b --eager 30 0 give-ups, PROBE ok; lane C test_b2b_varied (`--out
test_b2b_varied_k2v2ho.json`, moved to logs_ho/): every SUMMARY failed false, gave_up 0, bitwise_golden = launches.
k2_ab trial 1 (`logs_ho/ab_*_t1.{log,json}`; checks identical in both arms of every run, routes equal, replay
matches first): vs k2v2pre L1 75.0 -> 68.3 (1.098x), L31 74.5 -> 68.2 (1.093x), L60 75.6 -> 68.0 (1.112x); vs the
kept tree ep1 L1 80.3 -> 68.5 (1.173x), L31 80.3 -> 68.6 (1.171x), L60 80.7 -> 68.1 (1.185x). Trial 2 not run
(time box; `logs_ho/ab2.sh` runs it: remove `logs_ho/STOP` first).
Files changed vs k2v2pre: `port/k2.py` (switches, `_ho_key`, `_tail_split` = the old tail moved into a function,
`_tail_up_all`, `_up_all`, `_fin_task`, `_import_alarms_async`), `port/gemv/moe_dense.py` (`UP_ALL_TASKS`,
`up_loads_all`, `stage_up_all`); the rest are new_candidate path rewrites. `dev/` holds the K2_XQ_GATE prototype.
