# K2 small-GEMM stages on MI455X (gfx1250, wave32), TP4 / EP1

This is the gfx1250 port of the MI355X K2's dense stages: o_proj, the AR#1 seam norms, the router GEMV and top-8
routing, the pooler, the unpooler and gather, plus K1's FFN-boundary seam A. Everything is in this directory and
written in FlyDSL 0.3.4.1. It was tested on one GPU, with the four TP ranks simulated one after another, against
float64 references built from layer-31 weights.

**Status:** every stage passes. The GEMV outputs bit-match bf16(fp64 reference) on at least 99.98% of elements,
with max 1 bf16 ulp and cosine equal to the floor set by bf16 rounding itself. Routing is bit-identical to the
served Triton routing kernels (ids and fp32 weights) on 2,496 rows. The isolated time per call at 8 rows is
3.0–3.5 µs net for each of o_proj, the pooler, and the unpooler. These stages run against the shared
`common.py` / `mailbox.py` in this directory; they still have to be wired into the port's `k2.py`
(see "Integration").

## What C1's K2 runs, in order

C1 sets `SGLANG_MUSE_MONO_EXPERT_STACK=1` at width 8, so the runner builds `K2Build(tokens=8,
experimental_two_cta=True)`, the q8 build (`bundles-c1/sglang.patch.diff`, `runner.k2_keys`). In
`ops/k2.py`, that build calls these stages, in this order:

1. `oproj.stage_oproj_deep(c, bid)` on the o_proj CTAs: AR#1 reduce-scattered to the column owners.
2. Early tasks with their weights in flight before x lands: `moe.router_loads` + `moe.router_compute_pair`
   (q8 token pairs) and `moe.pool_loads` + `moe.pool_compute`.
3. `quarter_seam.stage_ar_owner_sum(c, bid)` (CTA `4 t + quarter`: rank-order sum, bf16, all-gather of x and of
   the per-slice sums of squares), then `quarter_seam.stage_seam_token(c, t)` (post_attn_norm, gated residual
   -> `stream_out`, pre_ffn_norm -> x rows, XRDY[t, quarter]).
4. `moe.stage_route(c, t)` (with `rlog_parts`: sums the 4 K-quarter partials in part order).
5. Experts: `stage_xq` (which includes pre_expert_norm), UG/down, fin. These belong to the expert port
   (`../moe`), not this package.
6. `moe.up_loads_masked(c, j, live)` (issued ahead of FRDY), then `moe.stage_up(c, j, ops)`, then
   `moe.stage_gather(c, j)`.
7. In the next layer's K1 (stage 1a): `seam.stage_seam(c, s)` (seam A: post_feedforward norm, gated residual,
   pre-QKV norm).

C1 does not run `oproj.stage_oproj` (push-all AR#1), `moe.router_compute` (RPART) or `seam_b.stage_seam`
(the default build). They are ported anyway: `seam_b.stage_seam` and `stage_oproj` are tested (`seamb`), and
`router_compute` compiles but is not tested.

## Public interfaces (what `k2.py` calls), gfx1250 / TP4 values

| module | functions | constants |
|---|---|---|
| `oproj` | `stage_oproj_deep(c, bid)`, `stage_oproj(c, bid)`, `load_core(c)`, `ar_pairs(s)`, `o_loads(c, bid)`, `owner_of_tile(rg)` | `TP=4`, `H=8192`, `CORE=2048`, `ROWS=16`, `OTASKS=512`, `O_ROUNDS=OTASKS//BLOCKS` (2 at 256 CTAs), `CORE_ROW`, `OWNER_COLS=2048` |
| `moe_dense` (merge into the port's `moe.py`) | `router_loads(c,i,q)`, `router_compute(c,i,q,ops)`, `router_compute_pair(c,i,q,tg,ops)`, `pool_loads(c,i,q)`, `pool_compute(c,i,q,ops)`, `stage_route(c,t)`, `up_loads(c,j)`, `up_loads_masked(c,j,live)`, `stage_up(c,j,ops)`, `stage_gather(c,j)`, `early_x(c,q)`, `peer_pairs(s)`, `scratch_pairs(s)` | `E=256`, `TOPK=8`, `EPR=64`, `MH=4096`, `R_GROUPS=4`, `R_PARTS=4`, `R_K=2048`, `R_TASKS=16`, `R_TASKS_Q8=64`, `POOL_N=1024`, `P_GROUPS=64`, `P_PARTS=4`, `P_TASKS=256`, `EARLY_K=2048`, `UP_N=2048`, `UP_TASKS=128`, `GATHER_TASKS=64`, `FIN_TASKS=32` |
| `quarter_seam` | `stage_ar_owner_sum(c, bid)`, `stage_seam_token(c, token)` | `OWNER_COLS=2048`, `OWNER_PART=512`, `NSL_ROW=64` |
| `seam_b` | `stage_seam(c, j)` | `TP`, `H`, `COLS=128`, `NSL=64` |
| `seam` (K1) | `stage_seam(c, s)` | `HIDDEN`, `NS=64`, `COLS=128` |
| `gemv` | `loads`, `chunk_loads`, `wmmas` (alias `mfmas`), `store_acc`, `rows_to_lds`, `dev_rows_to_lds`, `chunks`, `shuffle_rows` (host) | `ROWS=16`, `KCH=128`, `LDS_PAD=8`, `SHUFFLED`, `INTERLEAVE` |

The signatures and the `c` dict keys are the MI355X ones. These are the differences `k2.py` has to absorb:

- `c["lane"] = tid % 32`, `c["wave"] = tid // 32` (16 waves of 32 at 512 threads; `port/consts.py`).
- LDS: `c["red"]` needs `common.RED_FLOATS` = 4096 floats (a WMMA accumulator is 8 f32 a lane), `c["av"]`
  THREADS floats, and `c["xl"]` 8 × 4104 / 2 words (65.7 KB: the 4096-wide FIN rows plus padding).
- Task counts at TP4: `P_TASKS` is 256 (was 128), `UP_TASKS` 128 (was 64), `R_TASKS_Q8` 64, `OTASKS` 512 (two
  tiles a CTA at 256 CTAs, one at 512). `k2.py`'s CTA map (`R0`, `P0`, `RX0`, `U0_Q8`, and its asserts) must be
  redone. One option at 256 CTAs: every CTA runs o_proj, then pool task `bid`; router tasks go on 64 of them; the
  unpooler goes on 128 CTAs that are idle during fin.
- Peer regions per parity (pairs): `ar` [4][8][8192/2], `rlog_parts` [4 src][4 q][8][64], `pool` [8][4096/2],
  `br` [8][8192/2], `xg` [8][8192/2], `xsq` [8][64]. Scratch: `ppart` [256][8][16], `rpart`, `route` [8][16],
  `xrdy` [8 × 4], `frdy` [32]; plain `xrow` (bf16 [8][8192]) and `fin` (bf16 [8][4096]). The FRDY count (32 =
  4096 / 128) must match the expert port's fin task count.
- `ar_owner[j]` = the peer base of rank `owner_of_tile(bid + BLOCKS j)` = `(tile · 16) // 2048`. `k2.py`'s
  existing expression already computes this for any TP.
- **Weights in load order:** with `gemv.SHUFFLED = 1` (the default), `w_o` (the rank's [8192, 2048] shard),
  `w_pool` and `w_unpool` must be passed through `gemv.shuffle_rows` once at load time. It produces the same bytes
  and shape, permuted inside 16-row × 128-K tiles. The router stays row-major fp32 (its loads are already
  coalesced). Memory cost: K2-only shuffled copies of the o_proj shard plus the rank's pooler and unpooler slices
  are (32 + 16 + 16) MB × 60 layers = 3.8 GB per GPU. That is 3.8 GB if the pooler/unpooler copies are rank
  slices, which needs the base pointer shifted back by `rank · 1024 · 8192 · 2` bytes; full replicated copies
  would be 9.6 GB. `SHUFFLED = 0` reads the served tensors in place and costs 1.1–2.7 µs more per stage
  (table below).
- Cache policy is the gfx12 CPol: NT = 1 for weights, SCOPE_DEV = 16 for intra-GPU hand-offs and activation
  rows, SCOPE_SYS = 24 for peers, and stores add TH_WB = 3. A poll without DEV scope never saw another CTA's store
  (measured: it timed out), so every hand-off load keeps the scope bit. These match `port/common/ops.py`
  (`CM_NT = 1`, `CM_DEV = 16`, `CM_SYS = 24`), and its `acc_slot` / `row_sum` layout matches `common.py`, so
  switching the imports to `port.common.ops` is a rename (`wmma_bf16` -> `mfma_bf16`). Note that
  `port.common.ops.ballot` returns an Int64 with the top 32 bits zero; `stage_route` works with either ballot.

## How the port works

- **WMMA GEMV** (`gemv.py`): 16 weight rows × up to 16 tokens × 32 K per `v_wmma_f32_16x16x32_bf16`.
  Lane l holds weight row l % 16. In each 128-K chunk a lane issues eight b128 loads, and WMMA q consumes loads 2q
  and 2q+1. The token operand is read from LDS at the same K. A and B take identical K sets per lane, which
  `probe.py` checked on hardware (max |err| 9.5e-7 against A·Xᵀ), so a K permutation shared by both operands leaves
  the dot product unchanged. The task's K chunks are split over the CTA's 16 waves; each wave stores its
  accumulator to LDS, and the rows are summed in wave order. o_proj gives 8 waves to each of its two tiles
  (K = 2048 is 2 chunks a wave). The pooler gives one chunk a wave, and the unpooler two.
- MI355X's q8 o_proj kept one wave per tile with a single K chain (`stage_oproj_deep`), to match production
  gfx950 kernels bit for bit. There is no such gfx1250 reference, and the split-K form is what makes the weight
  stream fit one round trip.
- **Router**: fp32 VALU, as on MI355X. A wave is an expert, and a lane reads 4 fp32 every 128 elements (coalesced
  b128). Each lane runs a sequential fma chain, then the wave32 butterfly (16, 8, 4, 2, 1). Logits differ from
  float64 by at most 4.6e-7 of the logits' RMS.
- **Route**: the same algorithm as MI355X with 32-bit ballots and 8 logits a lane. The four K-part partials are
  summed in part order, then multiplied by `router_output_multiplier` (0.1767767). The score is `1 / (1 +
  exp2(-x·log2e))` with an IEEE divide, plus the bias, then a radix-select top-8 (equal keys go to the lower
  expert), slot order by biased score, and weights `s_k · (1 / (Σ_slot-order s + 1e-15))`. This matches the served
  `fused_muse_spark_routing` topk8 kernel, `_joint_epilogue` and `_reduce_route` exactly (all read in
  `fused_muse_spark_routing.py` / `router_pooler_topk.py`: sigmoid mode, bias for selection only, scale 1.0, eps
  1e-15).
- **Seams**: a wave32 owns a 128-column slice, and a lane owns 4 columns (two bf16 pairs per 16 B mailbox load).
  Sums of squares are a lane's `(c0² + c1²) + (c2² + c3²)` followed by the butterfly. Totals over the 64 slices are
  lane l's slices l and l+32 added, then the butterfly. This is the same everywhere, so every rank and slice
  applies the same rstd. MI355X reproduced gfx950 Triton's 16-wave sum order there; with no gfx950 kernel to
  match on this GPU, the order is fixed but new.

## Results

Run: `run_gpu.sh test_stages.py --trials 2000` on GPU 0, with the default configuration (preshuffled weights,
NT, 256 CTAs × 512 threads). The results are in `results/test_stages.json`. Inputs are seeded synthetic
activations: core ~ N(0, 1) bf16, stream ~ N(0, 2²), FIN ~ N(0, 1). Weights are layer 31's from
`/data/Muse-Spark-Open-816B-A42B-Quark-MXFP4`, with layer 32's input_layernorm for seam A. The residual alpha
and beta come from the checkpoint gates (temperature 0.3), as in `layers.py`.

### Correctness (vs float64 torch on the stage's own inputs)

| stage | result |
|---|---|
| o_proj (4 ranks, owner slots) | 99.989% bit-equal to bf16(ref), max 1 ulp, cos 0.99999862 = bf16 floor; tags OK |
| AR owner sum | XG bit-exact vs the rank-order fp32 sum of the 4 bf16 partials, on all ranks; slice Σx² max rel 1.4e-7. Against fp64 full o_proj the cos is 0.9999972: bf16 partials, as in any bf16 all-reduce |
| seam per token | `stream_out` max abs 4.1e-7 (cos 1 − 4e-16); x 99.997% bit-equal, max 1 ulp; 4 ranks identical |
| router (q8 pairs) | logits max abs err 2.6e-6 = 4.6e-7 × RMS; all ranks hold identical partials |
| pooler | 99.98% bit-equal, max 1 ulp, cos = bf16 floor |
| route | pipeline: ids and weights **bit-identical** to the served op. 2,000 random rows (including forced exact ties and quantized logits): **0** id and **0** weight-bit differences. Served 8-row epilogues (`_joint_epilogue`, `_reduce_route`), 496 rows: **0 / 0**. Whole served front (`joint_router_pooler_topk`, the MI455X 8-row path with bf16 hi+lo router) on the same x: 8/8 rows have the same expert sets, weights within 2.0e-7. fp64-router sets: 8/8 (min 8th-vs-9th gap 8.0e-4) |
| unpooler | 99.989% bit-equal, max 1 ulp, cos = bf16 floor |
| gather | bit-exact on all ranks |
| seam A (K1) | `attn_in` 100% bit-equal to bf16(ref); `stream_a` max abs 5.4e-7 |
| seam B (default build) | x 99.997% bit-equal, max 1 ulp |

### Time per call, 8 rows, one stage alone

These are graph replays of 96 calls, with weights rotated over about 1.5 GB of copies so every call streams from
HBM. Net = per-call time minus the empty stage kernel on the same grid (1.28 µs: launch + gap), which a
persistent K2 does not pay. Mailbox inputs were published beforehand, so the times include the stage's
intra-launch hand-offs but no cross-GPU wait. Floors: bytes / 23.3 TB/s, and the copy sweep in
`/data/muse455/gemm/results_hbm_bw.txt` (16 MB: 5.54 TB/s, 3.0 µs; 32 MB: 8.55 TB/s, 3.9 µs; 64 MB:
11.5 TB/s; large: 18.2 TB/s; all including launch).

| stage | bytes | call µs | net µs | floor @23.3 | floor @18.2 | net TB/s |
|---|---:|---:|---:|---:|---:|---:|
| o_proj (256 CTAs) | 33.6 MB | 4.70 | **3.42** | 1.44 | 1.85 | 9.8 |
| o_proj (512 CTAs, one tile each) | 33.6 MB | 4.32–4.38 | **3.04–3.10** | 1.44 | 1.85 | 10.9–11.1 |
| pooler (256 tasks) | 16.9 MB | 4.80 | **3.52** | 0.73 | 0.93 | 4.8 |
| unpooler (128 tasks) | 16.8 MB | 4.20 | **2.92** | 0.72 | 0.93 | 5.8 |
| router q8 (64 tasks) | 2.2 MB (8.4 MB of reads) | 4.18 | **2.90** | 0.10 | 0.12 | — |
| route (8 CTAs) | 64 KB | 5.09 | 3.81 | — | — | — |
| AR owner sum (32 CTAs) | 0.26 MB | 1.59 | 0.30 | | | |
| seam per token (8 CTAs) | 0.87 MB | 3.13 | 1.85 | | | |
| gather (64 CTAs) | 0.39 MB | 1.47 | 0.18 | | | |
| seam A (64 CTAs, two 64-way rendezvous) | 0.88 MB | 5.01 | 3.72 | | | |

What limits them (`bench_stream.py`, `results/bench_stream.json`; pure streaming, launch included):

- **o_proj** is within 0.3 µs of pure streaming of its own bytes on its grid. A flat stream of 32 MB at
  256 × 512 takes 4.43 µs against o_proj's 4.70; the best grid, 1024 × 256, takes 3.61 µs (9.3 TB/s). The
  512-CTA build gains 0.4 µs.
- **Pooler and unpooler**: pure streaming of 16 MB takes 2.37–2.44 µs, so they spend about 2 µs more.
  For the pooler that is the intra-launch K-part hand-off (PPART, one device-scope hop at about 1 µs, as in S0)
  plus staging the x quarter (32 KB a CTA, 8 MB device-scope in total). For the unpooler it is staging FIN
  (64 KB a CTA, 8 MB) after the FRDY poll. Inside K2, `pool_loads` / `up_loads_masked` issue the weights before x
  or FIN lands, so the isolated numbers are upper bounds.
- **Route**: 3.8 µs net alone. Polling its 32 pairs a lane in one round instead of four made no difference, so
  the cost is the 32-step radix select over 8 keys a lane (twice MI355X's 4 a lane) plus the system-scope load
  latency. I did not split those further. Bounding the search by the 8th-largest lane maximum and the global
  maximum would skip the common high bits; not tried.
- **Weight byte order** matters most. Measured net, with plain row-major (RT) vs preshuffled (NT) weights:
  o_proj 4.66 vs 3.42, pooler 4.63 vs 3.52, unpooler 5.62 vs 2.92 µs (`results/test_stages_plain.json` vs
  `results/test_stages.json`; the earlier sweep `results/test_stages_s{0,1}c*b*.json` agrees within 0.3 µs). With lane = row, a plain-layout b128 instruction touches 16 rows ×
  32 B; preshuffled, it reads 512 contiguous bytes.
- On plain layouts, pairing lanes r and r+16 on adjacent 16 B (`INTERLEAVE=1`) beat 64 contiguous bytes a lane
  by 1.1 µs on o_proj. NT (TH=1) helps only on coalesced streams; on the plain row pattern RT (TH=0) is better.

## Reproduce

```bash
cd /data/muse455/k2/ep1/port/gemv
./run_gpu.sh probe.py                                    # primitives on hardware
TMO=1200 ./run_gpu.sh test_stages.py --trials 2000      # all stages + benches -> results/test_stages.json
XENV="K2G_BLOCKS=512 K2G_TAG=_b512" ./run_gpu.sh test_stages.py --stages oproj   # 512-CTA o_proj
XENV="K2G_SHUF=0 K2G_CMW=0 K2G_TAG=_plain" ./run_gpu.sh test_stages.py --stages oproj,pool,up
./run_gpu.sh bench_stream.py
```

`run_gpu.sh` refuses to start while `sglang.launch_server` runs or the campaign driver says "Harness running
bench", and takes `/data/muse455/locks/gpu${GPU:-0}.lock`. The weights are cached in the container's `/tmp`
after the first load.

## Not done / open

- Wiring into the port's `k2.py` (the CTA map, the regions above, the load-time shuffles) and a 4-GPU run. Only
  one GPU was used: cross-GPU hand-off cost is in S0, not here.
- `stage_xq` (pre_expert_norm + MXFP8 quant), the expert UG/down and fin belong to the expert port.
- No real-activation fixtures: inputs are synthetic and the weights are real.
- The pooler's PPART hop could be removed by sending the four K-part fp32 partials straight to the `stage_xq`
  consumer (as `rlog_parts` does for the router). That trades 4× the POOL bytes over XGMI for one fewer hop; not
  tried.
- `router_compute` (non-q8) compiles but is untested.
