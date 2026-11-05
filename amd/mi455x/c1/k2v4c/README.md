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

## k2v4 (from k2v2_land1, 2026-11-05 03:13-05:40 UTC): fresh stage timeline, rf fixes, per-site poll sleep, UG/down balance

Tree = `cp` of `k2v2_land1` (`lane/new_candidate.sh k2v4 .../k2v2_land1`; land1 not modified). Every GPU job through
`k2safe.sh -w '.../k2v2_poll/s3/mpol_exec.py'`. All switches are in the build key (`k2._v4_key`). With
`K2_POLL_CHK=1 K2_DNACQ2=0 K2_RT_POLL1=0 K2_FIN_DEDUP=0 K2_FAST_SITES= K2_VR_UGW=16` the ISA is identical to land1's
(0 diff lines, `logs/cmp/`).

Switches (defaults = the k2v4 program):
- k2v2rf review fixes, ported onto the k2v2ho tail: `K2_POLL_CHK=8` (deadline clock + health word every 8th re-poll),
  `K2_DNACQ2=1` (UG -> down acquire: invalidate before the down prime, wait for older loads only), `K2_RT_POLL1=1`
  (ROUTE polled by waves 0-1 + LDS broadcast), `K2_FIN_DEDUP=1` (finalize waves 8-15 skip duplicate polls / puts).
  `K2_GATHER_DEDUP` does not apply (no gather stage with K2_UP_ALL); YR_KEEP / UP_EARLY / MON_FIX not ported (no gain).
- `K2_FAST_SITES` (new, `sync.site_sleep`): the s_sleep between re-polls per hand-off site, default
  `ar:4,xq:4,rlog:4,xqrdy:8,xrdy:8,route:8,fin:8,yr:8,fmsq:8,finld:8`; all other polls keep `K2_POLL_SLEEP=32`
  (~1 us). Sites: `ar` seam AR#1 (8 CTAs), `xq` POOLP (8 CTAs), `rlog` route RLOGL (8 waves), `xqrdy` / `xrdy` /
  `route` / `fin` (one wave or two a CTA), `yr` / `fmsq` (finalize, 8 waves x 64 CTAs), `finld` (bulk FIN pairs after the
  gate; re-polls only when a pair landed after the gate's), `ugrdy` (down, every wave: tagged, default off), `xg` /
  `ppart` (only with SEAM_DIRECT=0 / POOL_DIRECT=0). The poll storm (README 06:23) needs thousands of waves spinning;
  the 4-us-sleep sites have 8 CTAs.
- `K2_VR_UGW=24` (was 16): a CTA with one more UG task gets 24 fewer down wave-tiles; 6-UG-round CTAs finished ~1 us
  after 5-round ones at 16 (`logs/tl/base_L31.json`).
- `K2_UP_MERGE` (new, default 0): both unpool tasks' WMMAs sharing the B operand, one barrier; no gain (sweep4).

Screens (`s3/sweep.py`, run_k2 graph of 20, rank median / slowest rank, us; digests equal in every row):
L31 sweep2: off 67.7/68.4, rf 67.4/68.2, +fast narrow (ar,xq,rlog:4) 66.3/66.9, +wide sites at 8 65.2/66.1,
narrow + UGW24 65.8/66.0; POOL_DIRECT=0 67.1/68.6, SEAM_DIRECT=0 70.1/70.4 (both bit-identical, slower).
L1/31/60 sweep3: rf 68.0/68.4, 68.9/69.1, 68.8/68.8; config A (sites + UGW24) 66.0/66.2, 65.9/65.9, 65.8/65.9;
all sites at 4 + UGW24 65.8/65.9, 66.0/66.2, 65.7/65.9; wide at 16: 65.8/65.9, 66.3/66.3, 66.7/66.7; VR_YR=tile ~90.

Timeline (`K2_DEBUG_TL=1 K2_TL_PAIR=1 K2_TL_RAW=...`, `s3/probe_b2b_tl.py --timeline`, analysis `s3/tl_an.py`,
`s3/tl_table.py`): XCD = bid % 8 and the XCD clocks are offset by up to ~4 ms, so every CTA is aligned to its XCD's
first start; rank clocks aligned on the FIN all-gather (+-1 us). Stage tables `logs/tl/stage_table.md` (land1 program
= `off_*`, k2v4 = `v4_*`), event chains `logs/tl/events_*.txt`.

A/B (`s3/ab_run.sh`, `s3/ab_spike.py` = k2v2_poll's spike.py sync loop: graph of 20 chained calls, slots 1..20, one
epoch advance a replay; warm = one weight copy, cold = 8 rotated copies; 300 replays; slowest rank's p50 per call;
`s3/ab_sum.py`): see the table appended below.

| layer | mode | trial | k2v2_land1 us | k2v4 us | delta | speedup | outputs bit-identical | replay == golden |
|---|---|---|---|---|---|---|---|---|
| 1 | warm | 1 | 64.59 | 62.05 | -2.54 | 1.041x | yes 363d3c1a6b210d77 | True |
| 1 | warm | 2 | 64.72 | 62.04 | -2.68 | 1.043x | yes 363d3c1a6b210d77 | True |
| 1 | cold | 1 | 67.95 | 65.20 | -2.75 | 1.042x | yes 363d3c1a6b210d77 | True |
| 1 | cold | 2 | 67.88 | 64.88 | -3.00 | 1.046x | yes 363d3c1a6b210d77 | True |
| 31 | warm | 1 | 64.50 | 61.94 | -2.56 | 1.041x | yes 002c3013de4a779a | True |
| 31 | warm | 2 | 64.38 | 61.87 | -2.51 | 1.041x | yes 002c3013de4a779a | True |
| 31 | cold | 1 | 67.38 | 65.28 | -2.10 | 1.032x | yes 002c3013de4a779a | True |
| 31 | cold | 2 | 67.81 | 64.90 | -2.90 | 1.045x | yes 002c3013de4a779a | True |
| 60 | warm | 1 | 64.04 | 61.89 | -2.15 | 1.035x | yes cbf9461d885e8376 | True |
| 60 | warm | 2 | 64.63 | 61.92 | -2.71 | 1.044x | yes cbf9461d885e8376 | True |
| 60 | cold | 1 | 67.42 | 65.14 | -2.28 | 1.035x | yes cbf9461d885e8376 | True |
| 60 | cold | 2 | 68.05 | 65.59 | -2.46 | 1.037x | yes cbf9461d885e8376 | True |
mean L1 warm: land1 64.65 k2v4 62.04 delta -2.61 (1.042x, 2 trials)
mean L1 cold: land1 67.91 k2v4 65.04 delta -2.88 (1.044x, 2 trials)
mean L31 warm: land1 64.44 k2v4 61.91 delta -2.53 (1.041x, 2 trials)
mean L31 cold: land1 67.60 k2v4 65.09 delta -2.50 (1.038x, 2 trials)
mean L60 warm: land1 64.33 k2v4 61.90 delta -2.43 (1.039x, 2 trials)
mean L60 cold: land1 67.73 k2v4 65.37 delta -2.37 (1.036x, 2 trials)

Robustness (k2v4 defaults): `probe_b2b.py --layers 1,31,60 --eager 30`: 0 give-ups, PROBE ok on every layer
(`logs/robust_b2b.log`); lane C `test_b2b_varied.py --operands harness --routed-only` with `K2C_PORT_ROOT=k2v4`
(`logs/test_b2b_varied_k2v4.json`): eager_step / eager_launch / graph_step / graph_launch failed false, gave_up 0,
bitwise_golden = launches (27/27/36/36), stream2 cos min 0.999795. ab_spike: 72,000 chained calls, no give-up.

Where the ~64-68 us go (land1 program, L31, critical path in the global frame, `logs/tl/events_off.txt`;
floors: bytes / 16.5 TB/s, 2.05 us a XGMI hop, ~1 us a device hop), gap = measured - floor:
1. expert phase, UG start -> last YR out: 25.6 vs 16.6 (274 MB): ~9 us. UG runs at ~HBM peak (11.0 median a CTA),
   but the slowest CTA ends UG ~4 us after the median (5 vs 6 UG rounds), the UG -> down acquire costs 3.5 us a
   CTA, and down runs at ~8 TB/s (latency-bound tiles, YR burst at the end of the run).
2. AR#1 hop -> seam: 8.9 vs 2.05: ~6-7 us. 256 KB of tagged bf16 pairs a XGMI link (~4 us of serialization),
   + ~1 us poll quantization (s_sleep 32), + the rank start skew (the latest rank waits 5.3, the others 9).
3. POOLP all-gather -> xq: 5.9 vs 2.05: ~4 us (256 KB fp32 parts a link; 8 xq CTAs each poll 128 KB).
4. YR hop: 5.8 vs 2.05: ~3.7 us (256 KB a link, pushed at the end of each down run).
5. tail after the FIN gate: FIN pairs -> LDS 1.7 + unpool GEMV 1.8 vs ~1: ~2.5 us.
6. XQRDY + acquire -> UG start: 3.0 vs 1.0: ~2 us (partly under the first UG weights).
7. small pre-expert stages (o_proj 2.6/2.0, seam 1.2/0.5, x hop 1.7/1.0, pool 1.5/1.0, xq 1.1/0.3): ~3.3 us.
8. FMSQ + finalize out: 2.1 vs 1.3: ~1 us. Launch boundary: ~1-2 us in the chain harness (~5 us in run_k2's
   graph, which adds an epoch-advance kernel per launch).
What k2v4 removed (same probe, L31 per-CTA medians, worst rank): AR wait 9.2 -> 6.4, RLOG wait 2.5 -> 1.7, POOLP
-> xq 5.0 -> 4.4, UG -> down acquire 3.5 -> 2.7, rank start skew 1.4 -> 0.8 us; kernel span 67.3 -> 63.1 us.
Next (not done): fewer XGMI bytes a hop (denser tagged words: 3 bf16 + 16-bit tag, or 16-byte {6 bf16, tag}
stores -- needs a 128-bit tearing test) for AR / POOLP / YR (~1-1.5 us each); UG/down overlap per slot (down tiles
of slots 0-7 can start after UG round 0) to bring the expert phase toward 17 us.

## k2v4b (from k2v4_land1, 2026-11-05 08:45 UTC ->): the serving give-up of K2 v4, root-caused and fixed

Tree = `lane/new_candidate.sh k2v4b .../k2v4_land1` (land1 trees not modified). Study files in `stress/`.

### What gave up in serving (Arbor iteration 13, 08:25:59, 12,169 K2 steps, flags [1, 1, 1, 1])
Not the fast poll sleeps and not NUMA balancing: a **capacity overflow of the down run at nu = 63** distinct routed
experts, introduced by `K2_VR_UGW=24` (land1 / k2v2: 16). `moe.vr_down_range` gives the r = 32 (nu % 8) heavy CTAs
`ugw` tiles fewer, so a light CTA's run is nu + 3 (nu % 8) tiles at ugw 24 (nu + 2 (nu % 8) at 16). The LDS sums
(`VR_DN_LOCAL = 80`, whose comment still said "<= 77") and the YR units (`VR_YR_ITERS = 5` = 80 tiles) cover 80
tiles; at nu = 63 the 32 light CTAs of every rank get 84. Tiles 80..83 are summed past `vr_dy` (into `vr_cnt` /
the route LDS) and **never stored to YR**, so on every rank the finalize's `yr` poll waits for pairs that never come,
gives up after `WAIT_SECONDS` = 10 s, and `finp` / `fmsq` follow: all four ranks raise their flag. Exhaustive check
(`stress/geom_check.py`, emulates `vr_down_range` + `vr_down_geom` + `vr_yr_out` for nu 1..64): k2v4_land1 at ugw 24
has 128 tiles a rank past the LDS sums and never YR-stored at nu = 63 and nowhere else (ugw 32, screened in k2v5,
would fail at nu 55 / 61 / 62 / 63); k2v2_land1 (ugw 16) and k2v4b: none. The harness never saw it because its cases
are calibrated to ~43 experts (synthetic shared-direction rows top out at nu ~57; `stress/nu_scan.py`); nu = 63
needs eight verify rows that share exactly one expert, which real traffic produces rarely (once in ~730 k layer
launches in iteration 13).

### Fix (default on; ISA = land1's + ~12 scalar instructions once a CTA; 251 VGPRs, 0 spills, LDS 145968 B)
- `moe._down_split`: the heavy CTAs take at least ceil((nu 256 - 80 (256 - r)) / r) tiles, so no light run exceeds
  `VR_DN_LOCAL`. At ugw 24 this changes only nu = 63 (to ugw 16's split, 77-tile runs); every other nu keeps
  k2v4's split (`stress/split_check.py`). Outputs do not depend on the split (a tile's two K halves add into a zeroed
  LDS word), so they stay bit-identical.
- `moe.check_down_split(ugw)` runs at build (`k2._make`): every nu tiles exactly and fits, or the build fails.
- `K2_FAST_POLLS` (new, **default 0 = off**): a fast-site wait falls back to `K2_POLL_SLEEP` after N re-polls
  (insurance against a poll storm during a long peer stall; not needed for this give-up; in the build key).

### After a give-up, in serving (`sglang/.../mono_gfx1250.py`, not modified)
Nothing falls back. The runner launches K2 every verify step with no per-step check (`SGLANG_MUSE_MONO_CHECK=1` is
an eager-only debug sync); the rank-0 reporter logs the flags once and stops. FAILED is sticky (never reset): from
then on every K2 wait of that rank exits at its first deadline check (8th re-poll, `K2_POLL_CHK=8`: ~1-10 us), so
each later step reads whatever its mailboxes hold by then, and tokens keep being returned until restart.

### Measurements (2026-11-05 16:53-17:27 UTC, after the 16:24 reboot; kernel.numa_balancing = 0 on the host from now on)
`stress/serve_stress.py` (4 ranks, 16 layers 1, 5, .., 57, 60 with all 256 EP1 experts, a case pool a layer =
24 synthetic + 12 adversarial cases nu 58..64 from `stress/make_cases.py`; phase 1: one eager launch per case on
layers 1 / 33 / 60; phase 2: CUDA graph a step = epoch + 16 x (attention-like over 120k cached tokens, K2), cases
re-picked every step, nu = 63 only at --inject steps; give-ups decoded and reset). Logs / json: `stress/out/`.

| tree | sweep give-ups | serve K2 calls | serve give-ups | at | bitwise vs k2v4b |
|---|---|---|---|---|---|
| k2v4_land1 | 6 / 108 (every nu = 63 case, none else) | 192,000 | 4 | exactly the 4 injected nu = 63 steps | equal on 11,995 x 16 non-failed outputs; differs only on its nu = 63 give-ups |
| k2v4b | 0 / 108 | 640,000 (8 nu = 63 injected) | 0 | - | - |
| k2v2_land1 | 0 / 108 | 192,000 (4 injected) | 0 | - | equal on all 108 sweep cases (nu = 63 too) and 11,995 x 16 steps |

Every k2v4_land1 give-up: flags [1, 1, 1, 1], regions `yr` (a pair still carrying the previous launch's tag) then
`finp` / `fmsq` on 10-225 CTAs a rank, the launch 10.04 s. Same result on the lost 09:40 run with NUMA balancing on.

Span (`s3/ab_spike.py`, graph of 20 chained calls, slowest rank's p50 per call, us; trial 1 only, no mpol wrapper):

| layer | mode | k2v2_land1 | k2v4_land1 | k2v4b | k2v4b vs land1 | digests (all three equal) |
|---|---|---|---|---|---|---|
| 1 | warm / cold | 64.47 / 67.62 | 61.67 / 64.79 | 61.24 / 63.99 | 1.053x / 1.057x | 363d3c1a6b210d77 |
| 31 | warm / cold | 64.89 / 68.03 | 60.99 / 64.10 | 61.94 / 64.92 | 1.048x / 1.048x | 002c3013de4a779a |
| 60 | warm / cold | 65.22 / 69.07 | 61.09 / 64.25 | 61.29 / 64.89 | 1.064x / 1.064x | cbf9461d885e8376 |

k2v4b - k2v4_land1: -0.43 / -0.80, +0.95 / +0.82, +0.20 / +0.64 us (k2v4_land1 alone moved up to 1.3 us between
trials in k2v5's A/B). Not run (GPUs marked unhealthy at 17:44 by another agent's job): span trial 2,
`K2_FAST_SITES=` on k2v4_land1, the sticky-window measurement, `K2_VR_UGW=16` on k2v4_land1, `K2_FAST_POLLS=64`.

## k2v4c (from k2v4b, 2026-11-05 18:12 UTC): k2v2pf's touch-then-prefetch of the next layer's QKV weight

Tree = `lane/new_candidate.sh k2v4c .../k2v4b`. Only `port/k2.py` changed: k2v2pf's diff vs k2v2_land1 (`diff -u`,
applied with `patch --fuzz=0`; the context-dict hunk by hand): `K2Build` fields `pf_at / pf_cpol / pf_chunk /
pf_rot / pf_bchunk / pf_touch / pf_tat / pf_nj / pf_split / pf_btouch / pf_ctas` (defaults = off), four kernel
arguments `pf_a, pf_an, pf_b, pf_bn` appended to the ABI after `tl`, `k2_launch(..., pf=((ptr, bytes), (ptr, bytes)))`,
hooks at the route wait (`_pf_all(c, "route")`), the FIN hop (`_tail_up_all`) and the end. Default build = the k2v4b
program plus the unused arguments (ISA 7890 vs 7878 lines, 251 VGPRs, 87 SGPRs, 0 spills, LDS 145968 B).

**Knob set:** `K2Build(pf_at="fin", pf_touch=2, pf_tat="route", pf_chunk=8192, pf_cpol=17)` with
`k2_launch(..., pf=((w_qkv_next, 75497472), (0, 0)))` (k2v2pf arm `rt_finx_nt`; every k2v4b knob at its default).
ISA: 251 VGPRs, 91 SGPRs, 0 spills, LDS unchanged. Down-split check (`stress/geom_check.py`, moe.py = k2v4b's):
no tile past the LDS sums or without a YR unit for any nu at ugw 16 / 24 / 32.

**Serving needs a runner change** (not in this tree): the SGLang runner calls `k2_launch` with `K2Build()` and no
`pf`; it has to build the key above and pass layer L+1's QKV weight (this rank's [4608, 8192] bf16) at each launch.

### Chain (`s3/chain_pf.py` from k2v2pf: attention 3D -> reduce -> norm-gate -> K2 -> boundary -> QKV(L+1), ctx 131072,
graph of 16 slots x 20 replays, slowest-rank median us/layer; cold = 8 rotated copies; results `results/chain_*_v4c.json`,
`*_v4b.json`; 2026-11-05 18:29-18:30 UTC)

| layer | mode | k2v4b tree (2 trials) | k2v4c pf off | k2v4c pf on | pf on - off (median of 4) | per trial |
|---|---|---|---|---|---|---|
| 1 full | cold | 122.55 | 122.99 | 120.96 | **-2.03** | -1.98 / -2.56 / -1.77 / -2.45 |
| 1 full | warm | 118.19 | 117.58 | 115.06 | **-2.52** | -2.82 / -2.10 / -2.58 / -2.52 |
| 31 swa | cold | 100.75 | 100.25 | 98.72 | **-1.53** | -1.75 / -1.17 / -1.68 / -1.24 |
| 31 swa | warm | 91.11 | 91.72 | 92.12 | +0.40 | +1.14 / -0.14 / +0.56 / +0.06 |
| 60 swa | cold | 100.91 | 100.59 | 99.30 | **-1.29** | -1.94 / -0.46 / -0.96 / -1.29 |
| 60 swa | warm | 91.30 | 90.26 | 91.00 | +0.74 | +0.74 / +0.63 / +0.60 / +0.68 |

Bitwise: golden digests of all outputs (stream_out, branch_out, routes, new stream, attn_in, q / k / v / gate, both FP8
caches) equal on every rank for k2v4b, k2v4c pf off and pf on, every layer and mode; graph replays == golden.
(k2v2pf on k2v2_land1: L31 cold -1.86 / -2.05, warm +0.35 / +0.23; L60 -0.60 / +0.83; L1 cold -2.10.)

### Serving-like stress (`stress/serve_stress.py --pf`: the knob set above, a 75.5 MB QKV-sized region a layer)
| run | sweep give-ups | serve K2 calls | give-ups | bitwise vs k2v4b (v4b_nb0) |
|---|---|---|---|---|
| k2v4c pf on | 0 / 108 (nu = 63 incl.) | 480,000 (6 nu = 63 injected) | **0** | equal on 29,990 x 16 steps and all 108 sweep cases |
| k2v4c pf off | 0 / 108 | 96,000 (2 injected) | 0 | equal on 5,997 x 16 and all 108 |
Step p50 22.44 ms (pf on) / 22.47 ms (off), no step > +1 ms.
