# k2v6: K2 redesign study (TP4/EP1 mono MoE kernel, MI455X gfx1250, 8 verify rows)

Tree = copy of `k2v4` (`lane/new_candidate.sh k2v6 .../k2v4`; k2v2_land1, k2v4_land1, k2v4 not modified), **rebased on
k2v4b** at 18:08 UTC: `port/sync/sync.py` and `moe._down_split` / `check_down_split` / `vr_down_range` are k2v4b's
verbatim (the split clamp for the nu = 63 / UGW 24 overflow that was K2 v4's serving give-up, the build-time check,
`K2_FAST_POLLS` default off). Every k2v6 change is a knob in the build key (`k2._v6_key`), **all default off**: with
the defaults the ISA is byte-identical to k2v4b's (`logs/isa_def_k2v4b.s` vs `logs/isa_def_k2v6.s`, 0 diff lines;
251 VGPRs, 0 spills). Geometry: `stress/geom_check.py` (k2v4b's exhaustive nu = 1..64 check pointed at k2v6, plus the
`K2_YR_WAVES` unit map for 16/8/4/2/1 sender waves): every down tile computed once, inside the LDS sums and stored by
exactly one YR unit for every nu and CTA run; k2v6's split == k2v4b's for UGW 16/24/32 (`stress/geom_check.txt`).
`K2_FIN_LOCAL` and `K2_POOL_REP` do not touch the expert split (finalize CTA b = (b // 32, b % 32): all 8 x 32 tasks
once; pool CTA b = 4 tasks (rr, b // 4, b % 4): all 4 x 64 x 4 once).

Session log (UTC): 09:41-09:54 design + code; GPUs lost 09:41 (MES REMOVE_QUEUE hang from a k2v4b stress run, host
rebooted 15:24 / 16:24); 16:42-17:44 GPU screens (L31); 17:44 GPUs lost again (another lane's attention bench,
10 new MES REMOVE_QUEUE errors; host rebooted ~18:05); 18:08-18:28 rebase on k2v4b, geometry check, L1/L60/L31
screens, chain A/B (2 trials), robustness probes. Microbenchmarks only (no end-to-end runs).

## 1. Today's critical path (L31, k2v6 defaults = k2v4 program, rank 0, per-CTA min/median/max, us from launch)

From `logs/tl/events_base.txt` (probe_b2b_tl, `K2_DEBUG_TL=1`, XCD-aligned clocks). Floors: bytes / 16.5 TB/s,
2.0 us an XGMI one-way relaxed push (k2v4 S0), ~1 us a device hop.

| # | segment | from -> to (median) | us | floor | kind |
|---|---|---|---:|---:|---|
| 1 | o_proj (33.5 MB) | 0.4 -> 2.7 | 2.3 | 2.0 | HBM |
| 2 | **AR#1 hop** (256 KB/peer tagged, every rank sums) | 2.7 -> 8.5 (seam sees AR) | **5.8** | 2.0 | XGMI |
| 3 | seam (2 norms) | 8.5 -> 9.7 | 1.2 | 0.5 | compute |
| 4 | x -> pool | 9.7 -> 11.0 | 1.3 | 1.0 | device hop |
| 5 | pool task (WMMA 0.4 + epilogue/push 0.3, x staging 0.7) | 11.0 -> 12.4 | 1.4 | 1.0 | compute |
| 6 | **POOLP all-gather** (256 KB/peer) | 12.4 -> 17.9 (xq sees all parts) | **5.5** | 2.0 | XGMI |
| 7 | xq | 17.9 -> 18.8 | 0.9 | 0.3 | compute |
| 8 | **XQRDY** (+ first UG weights in flight) | 18.8 -> 20.6 | 1.8 | 1.0 | device hop |
| | (route chain, parallel: router 13.6, RLOG 15.5, route 17.3, route seen 18.2, route table + UG w issued 19.8; slack ~1) | | | | |
| 9 | **expert phase** (UG 183 MB + down 91 MB) | 21.1 -> last YR out 44.8 (med 42.2) | **23.7** | 16.6 | HBM + device hops |
| | of which: UG per CTA 11 (HBM peak), heavy/light UG rounds spread 4 us, UG->down acquire 2.6, UGRDY 0.9, down tiles ~5, YR out 2.5 | | | | |
| 10 | **YR hop** (reduce-scatter to row owner, 256 KB/peer) | down done -> fin sees YR: +2.5 (store stall) +2.0 | **4.5** | 2.0 | XGMI |
| 11 | **FMSQ** (32 slices' sums of v^2; = wait for the slowest slice's YR) | 46.9 -> 50.2 | 3.3 | 1.0 | device hop |
| 12 | finalize out | 50.2 -> 50.7 | 0.5 | 0.3 | compute |
| 13 | **FIN all-gather** (16 KB/peer only) | 50.7 -> 54.8 | **4.1** | 2.0 | XGMI |
| 14 | FIN -> LDS (128 KB pairs a CTA) + unpool GEMV | 54.8 -> 58.2 | 3.4 | 1.0 | compute |
| | kernel span ~58.5; chain-harness per-call 61.9 warm / 65.0 cold (k2v4) | | | | |

**The hand-offs (rows 2, 4, 6, 8, 10, 11, 13 plus the UG->down acquire) take ~29 us of the ~58 us span.** An XGMI
hop costs 4.1-5.8 us on the critical path whatever its size (FIN, 16 KB/peer, costs 4.1 us; AR, 256 KB/peer, 5.8);
a device hop costs 1.3-3.4 us. Bytes are not the limit (k2v5 tear test: ~1.4 us extra per 500 KB/peer).

## 2. Alternatives and their lower bounds

Bound = compute/byte floors of the stages on the critical path + hand-offs at their **measured** in-kernel cost
(XGMI ~4.1, device ~1.5), the right currency on this platform (a launch boundary costs 0.8 us; the S0 exchange
microbenchmark 3.4-3.9 us).

| design | XGMI hops | device hops | expert floor | lower bound | verdict |
|---|---:|---:|---:|---:|---|
| today (TP4/EP1: AR, POOLP, YR, FIN) | 4 | 4 | 16.6 (274 MB, even) | ~16.4 + ~6 + 16.6 + ~6 compute ≈ **45** (39 at S0-microbench hop costs) | baseline: 62-65 measured |
| (a) EP4 dispatch/combine (64 experts' full columns a rank) | 3-4 (AR, dispatch or replicated xq, combine, gather) | 3 | **29.1** (worst rank 18 of ~45 experts x 26.7 MB = 481 MB, S2 skew) | ≈ **57** | rejected: the routed load skew (6-18 experts a rank at 8 rows) costs +12.5 us of HBM on the slowest rank; static EP cannot rebalance (all-expert replication = 4x the 103 GB EP1 copy); not bit-identical (one full-K down vs rank-ordered bf16 partial sums). AITER `mega_moe_gfx1250` is this design for large M (TDM dispatch, gemm2-epilogue P2P scatter with an optional MXFP8/FP4 combine wire, a 1-block cross-device barrier kernel, then a reduce kernel: launch boundaries as the sync); at M = 8 its skew and its 3 launches lose to the mono kernel |
| (b) o_proj reduce-scatter + seam on owners + all-gather of x | 5 | 4 | 16.6 | ≈ 49 | rejected: today's AR#1 is already one hop (every rank pushes its partial to every rank and sums all 4 in rank order); RS + AG is two hops and the hop is latency-, not byte-bound |
| (c) replicate the pool (drop POOLP): every rank computes all 4096 pooled columns (`w_pool` is already the full replicated matrix) | 3 | 4 | 16.6 | ≈ 42 + 3x pool compute (+1.2 to +2.6) | **built** (`K2_POOL_REP`): bit-identical, neutral to slower (sect. 3) |
| (c') move router / pool earlier | - | - | - | - | router is already replicated on every CTA (K2_RLOCAL) with ~1 us slack; pool and router need x (after AR#1 + seam norms, nonlinear): cannot start earlier |
| (d) next layer's routing-independent weights (o_proj, router, pool: 58.7 MB, 7.3 MB an XCD) prefetched into L2 during the tail hops | 4 | 4 | 16.6 | today - ~1 (o_proj from L2) in cold | not rebuilt: k2v2pf measured it (`RESULTS_PF.md`): prefetching the next K2's o_proj slice from inside K2 = **+1.25 us cold, +3.9 us warm** (L60); any prefetch at K2's end = K2 +3.1 us. Only the QKV prefetch under the FIN hop paid (-2.05 us cold, served separately) |
| (e) remote-store address translation (2 MB pre-mapped peer pages, persistent mailboxes, dedicated sender waves) | 4 | 4 | 16.6 | today - ~2.5 (the YR store stall) - ~1 a push hop | **built/screened** (`K2_PEER_VMM=1`, `K2_YR_WAVES`): no gain (sect. 3); the stall is in the data path |
| (f) YR all-gather + every rank finalizes all 8 rows (drop FIN) | 3 | 4 | 16.6 | today - 4.1 (FIN) + YR x4 bytes (+~2) + 4x YR stores | **built** (`K2_FIN_LOCAL`): bit-identical, +1 to +2 us slower (sect. 3) |
| (c)+(f) both | 2 (AR, YR-AG) | 5 | 16.6 | ≈ 41 at measured hop costs (2 x 4.1 + 5 x 1.5 + 5.1 compute + 16.6 + ~1.2 pool x4 + ~2 YR x4); ≈ 37 at S0 microbenchmark costs | parts measured, each lost; see sect. 4 |

## 3. What was built and measured (all bit-identical; L31 screens)

Screens: `s3/sweep.py` + `run_k2.py` (graph of 20 chained launches, all 4 ranks, us/layer, rank median / slowest rank;
digest = sha of stream_out + branch_out + route ids). Every arm below: digest **002c3013de4a779a** (= k2v4 = land1),
routes equal (near-tie rows 0, 2 as always), stream2 cos min 0.9999225495, **0 give-ups**. Baseline (k2v6 defaults)
across the session: median 64.5-65.3, slowest 64.9-66.0 (7 runs: base, base2, base3, b4..b7, c0, c1).

| knob (default off) | what | L31 med / slow us | vs baseline in the same hold | ISA |
|---|---|---|---|---|
| `K2_FIN_LOCAL=1` | every down CTA puts YR on all 4 ranks; CTA b finalizes row b // 32, slice b % 32 into its own FINP; no FIN all-gather | 67.2 / 68.1 | base 65.0 / 65.7: **+2.1** | 251 VGPR, 0 spill |
| `K2_FIN_LOCAL=1 K2_PEER_VMM=1` | + 2 MB VMM uncached peer pages | 66.5 / 66.6 | vmm 64.7-65.0: +1.4 | |
| `K2_YR_WAVES=8 / 4 / 2` | YR stores issued by the last 8/4/2 waves only | 66.0/66.2, 66.4/66.7, 69.7/71.2 | base2 65.1/66.0: +0.2, +0.7, +5 | 250 VGPR |
| `K2_YR_WAVES=8 / 4` + `K2_FIN_LOCAL=1` | | 66.1/66.9, 67.2/67.3 | +0.9 .. +1.3 | |
| `K2_PEER_VMM=1` (existing knob) | 2 MB-granular VMM peer buffer | 64.7/65.2, 65.0/65.0 | c0 64.6/64.9: noise | unchanged |
| `K2_POOL_REP=1` | every CTA: router task, then pool task (pi, pq) of all 4 ranks' column blocks (one x staging), parts into its own POOLP; xq unchanged | 66.3/67.3, 66.1/66.2 | b4/b5 65.1-65.3: **+1.0** | 251, 0 spill |
| `... K2_POOL_REP_ORDER=pr` | pools before router | 67.6/69.2 | +2.5 | |
| `... K2_POOL_REP_MERGE=1` | the 4 tiles in one `wmmas_multi` pass (shared B operand), one barrier, red4 LDS (+64 KB, 211.5 KB total) | 65.7/65.8, 65.0/66.0 | b6/b7 64.5-64.6 / 64.9-65.0: **+0.5 .. +1.0** | 251, 0 spill |
| `... K2_POOL_REP_EARLY=1 / p` | the router + pool (or pool only) weights right after o_proj | not run | | **spills** 128 / 59-61 VGPRs |
| `K2_XQ_WT=1` (existing) | XQ write-through, XQRDY without L2 write-back | 64.8/65.0 | c0 64.6/64.9: noise | |
| `K2_FAST_SITES=...xqrdy:2...` | faster XQRDY re-poll | 66.5/67.4 | +2 (poll storm) | |

Why they do not pay (timelines `logs/tl/events_{base,fl1,vmm,rep,repm}.txt`, per-CTA intervals rank 0..3):
- **FIN_LOCAL.** The FIN hop does shrink (fin done -> FIN seen 4.1 -> 2.0 us), but the 4x remote YR stores cost more:
  wave 0's "YR stores issued -> next stage" grows 2.5 -> 4.3 us (median; max 5.0 -> 7.4), and the FMSQ wait (the slowest
  slice's YR) grows 3.3 -> 3.4-3.7 (max 4.7 -> 7.8). YR goes 256 KB -> 1 MB a link (tagged): ~+2 us of link time.
- **Sender waves / VMM.** Moving the stores to 8 waves (the others go straight on) and 2 MB pages change nothing: the
  ~2.5 us after a remote-store burst is the stores' own latency to the peer (translation + XGMI), not a per-wave issue
  stall that other work could hide. With 2 or 4 senders the per-wave store count grows and it gets worse.
- **POOL_REP.** The POOLP XGMI hop disappears (pool done -> xq sees all parts 5.5 -> 1.6 us when the producers are
  even), but: 4 sequential pool tasks take x seen -> pool done 1.4 -> 4.4 us (2.8 merged); the 8 seam CTAs see x
  ~2.5 us late because their x poll returns behind the 192 KB of weights they issue after the seam (issuing them
  before the seam spills); and the next gate moves to the route chain / the XQRDY device hop (xq done -> XQ seen
  2.4 us), so UG start moves by < 0.5 us while the extra 50 MB of pool reads and compute cost ~1 us.

## 4. Conclusion and chosen design

Chosen design for the next round: **keep TP4/EP1 and the 4 XGMI hops; spend the effort on the expert phase and on
hand-off cost itself, not on removing hops by replication.** Measured here: every hop removal by replication
(pool: +3 tasks a CTA; finalize: 4x YR) costs about as much as the ~2-4 us it saves, because in this kernel a
hand-off is latency-bound (~4-6 us XGMI, ~1.5-3.5 us device) and the replicated work lands on the same critical
CTAs. The bounds in sect. 2 show where the remaining ~25 us over the 35.4 us roofline sits:

| bucket | measured | floor | gap | lever (not built here) |
|---|---:|---:|---:|---|
| expert phase (UG start -> last YR out) | 23.7 | 16.6 | **7.1** | slot-pipelined UG/down: down tiles of slot s start when its 32 UG groups land (no per-CTA UG->down acquire 2.6 + UGRDY 0.9); heavy/light UG round imbalance (4 us spread) absorbed by down work of early slots |
| 4 XGMI hops | 19.9 | 8.2 (2.05 each) | **11.7** | per-hop: producer skew (o_proj done spread 1.9-10.6 incl. seam CTAs), single-CTA consumers reading 128 KB of tagged pairs (seam, xq, FIN -> LDS 1.8); fewer consumers' bytes, not fewer producers' bytes |
| device hops (x, XQRDY, FMSQ) | 6.4 | 3 | 3.4 | FMSQ is slice skew: finalize CTAs that wait on fewer slices |
| small compute (seam, pool, xq, finalize, FIN -> LDS + unpool) | 7.4 | 3.1 | 4.3 | |
| o_proj | 2.3 | 2.0 | 0.3 | |
| launch boundary (chain) | ~3.5 | 0.8 | 2.7 | |

**Remaining gap to the 35.4 us roofline: ~27-30 us** (61.9 warm / 65.0 cold per call, k2v4 = k2v6 defaults). The
38-42 us goal is not reachable by hop restructuring alone: the 2-XGMI-hop design ((c) + (f)) bounds at ~41 us only
with the expert phase at its HBM floor and zero producer skew, and both of its halves measured slower than today
(+1 and +2 us) because the replicated work and the 4x YR stores land on the critical CTAs. The order of work is
expert phase first (7 us), then per-hop skew and consumer read cost (up to ~12 us over the 2.05 us floors).

## 5. Chain A/B, all layers (18:15-18:25 UTC, after the rebase)

L1 / L60 screens (`logs/sweep6_L*.log`, run_k2 graph of 20, rank median, delta vs the two baselines of the same hold):
`K2_FIN_LOCAL` +1.7 / +2.0, `K2_POOL_REP + MERGE` +0.2 / +0.3, `K2_YR_WAVES=8` +0.7 / +0.6, `K2_PEER_VMM` +0.1 / -0.2
(L31 in the same session: +1.9, +0.7, +1.0, -0.1). Digests per layer 363d3c1a / 002c3013 / cbf9461d in every arm.

Chain harness (`s3/ab_run6.sh` + `s3/ab_spike.py`: graph of 20 chained calls, warm = 1 weight copy, cold = 8 rotated
copies, 300 replays, slowest rank's p50 per call; 2 trials, arm order reversed in trial 2; no NUMA wrapper, host
balancing off). Full table `logs/ab/ab6_summary.md`; means of the 2 trials, us (delta vs k2v4b):

| layer | mode | k2v4b | k2v6 defaults | + POOL_REP + MERGE | + PEER_VMM | + FIN_LOCAL |
|---|---|---:|---:|---:|---:|---:|
| 1 | warm | 61.65 | 61.54 (-0.11) | 62.44 (+0.80) | 61.82 (+0.17) | 63.13 (+1.48) |
| 1 | cold | 64.72 | 64.68 (-0.04) | 66.02 (+1.30) | 64.98 (+0.26) | 66.33 (+1.61) |
| 31 | warm | 61.95 | 61.60 (-0.36) | 62.50 (+0.55) | 61.76 (-0.20) | 63.31 (+1.35) |
| 31 | cold | 64.76 | 64.52 (-0.24) | 65.59 (+0.83) | 64.54 (-0.22) | 66.73 (+1.97) |
| 60 | warm | 61.76 | 61.60 (-0.17) | 62.18 (+0.42) | 61.70 (-0.07) | 63.63 (+1.86) |
| 60 | cold | 65.21 | 65.36 (+0.15) | 66.54 (+1.33) | 64.82 (-0.39) | 66.72 (+1.51) |

k2v6 defaults = k2v4b's ISA, so -0.36 .. +0.15 is the A/B noise. Every arm, layer, mode and trial: golden digests
equal k2v4b's on all ranks, replay == golden on all ranks. POOL_REP costs more cold (+0.8 .. +1.3) than warm: its
extra 50 MB of pool reads.

Robustness (every arm above, 18:27 UTC): `probe_b2b.py --layers 1,31,60 --eager 30` (30 eager back-to-back launches a
layer): PROBE ok and same_out on every layer, 0 CTAs gave up on every rank (`logs/sweep/pb*_L1,31,60.log`). Long run
without the NUMA wrapper: the chain A/B itself, 2 trials x 3 layers x 2 modes x 320 replays x 20 calls = **76,800
chained calls an arm, 0 give-ups** (ab_spike checks health after each loop and raises on a give-up). Lane C varied
inputs not run (no knob is a serving candidate).

## 6. Knob set / tree to serve

Serve **k2v4b** (or k2v6 with every knob off: the same ISA). No k2v6 knob is recommended; all stay off. Per-layer
span stays 61.5-62.0 warm / 64.5-65.4 cold per call; remaining gap to the 35.4 us roofline ~26-30 us, with the
levers in sect. 4 (expert phase first, then per-hop skew and consumer read cost), not hop removal by replication.

## 7. Targeted round 2 (18:36-18:51 UTC): consumer reads and the UG -> down switch

Two knobs (default off), both bit-identical by construction, compiled and screened on L31 only (the operator closed
the round before a chain A/B; none won the screen, so none was taken further):

- `K2_DN_DEV=1` (expert phase): no per-CTA UG -> down acquire (no `global_inv scope:SCOPE_DEV`, the 2.6 us switch
  cost); the down's INTER / ISC reads become device-scope loads past the per-XCD L2 instead. 251 VGPRs, 0 spills.
- `K2_UP_WAVEPOLL=gate|spin` (+ `K2_UP_WP_QS`, B operands a poll batch, default 2) (FIN hop consumer): each wave reads
  only its 2 K-chunks of FIN (8 KB of tagged pairs) straight into its WMMA B operands (the values
  `gemv.x_chunk_operand` reads from LDS; same WMMA sequence per task), no 128 KB FIN -> LDS staging, no barrier; `spin`
  also drops wave 0's FIN gate so a wave starts when its own slices land. QS 4 spills (28 VGPRs: a chunk's raw tagged
  loads next to the 128 VGPRs of prefetched unpool weights); QS 2: 225 VGPRs, QS 1: 214, 0 spills.

Geometry: neither knob changes the expert task split (`stress/geom_check.py` still passes nu 1..64).

| arm (L31, run_k2 graph of 20) | rank median / slowest us | vs baselines f0 64.9 / 65.6, f1 65.2 / 65.5 | digest |
|---|---|---|---|
| `K2_DN_DEV=1` | 65.4 / 66.1 | +0.5 | 002c3013de4a779a |
| `K2_UP_WAVEPOLL=spin` (QS 2) | 66.2 / 66.5 | +1.2 | 002c3013de4a779a |
| `K2_UP_WAVEPOLL=gate` (QS 2) | 67.7 / 68.1 | +2.7 | 002c3013de4a779a |
| both (`DN_DEV` + `spin`) | 66.1 / 66.6 | +1.1 | 002c3013de4a779a |

0 give-ups in every arm. Reading: the device-scope INTER reads (each of 4096 waves re-reads its slot's picks past the
L2) cost about what the dropped invalidate saved; the wave-level FIN reads replace one bulk round trip a CTA by 2-4
sequential per-wave round trips (QS 2: 4 poll batches a wave), which costs more than the LDS staging and barrier
they remove. Not built this round (out of time): dedicated down CTAs fed through LDS-free hand-offs, UG rounds
evened by a finer UG task split, producer-skew staging of the remote stores.

**Final: no k2v6 knob wins; serve k2v4b** (k2v6 with every knob off = k2v4b's ISA). All k2v6 knobs default off:
`K2_FIN_LOCAL=0`, `K2_YR_WAVES=16`, `K2_POOL_REP=0` (`_ORDER=rp`, `_EARLY=0`, `_MERGE=0`), `K2_DN_DEV=0`,
`K2_UP_WAVEPOLL=0` (`K2_UP_WP_QS=2` only read when on), `K2_FAST_POLLS=0` (k2v4b's).

## Files

`port/k2.py` (`_tail_fin_local`, `_fin_task_local`, `_router_pool_rep`, `_rep_loads`, `_v6_key`, red4 LDS),
`port/moe/moe.py` (`_FIN_LOCAL`, `_YR_WAVES`, `yr_out`, `_yr_out_senders`, `stage_fin` / `_fin_sum_put` / `_fin_out`
fin_local, `_down_split` / `check_down_split`), `port/gemv/moe_dense.py` (`pool_loads_rep`, `pool_partial_rep`,
`pool_partial_rep4`), `s3/tl_an.py` (`TL_FIN=all`, `TL_ALIGN=ar`). Logs: `logs/sweep*.log`, `logs/sweep/*_L31.log`,
`logs/tl/*.json` (raw), `logs/tl/events_*.txt`, `logs/compile_*`, `logs/isa_*.s`, `logs/ab/` (chain A/B),
`s3/ab_run6.sh`, `s3/ab_sum6.py`, `stress/geom_check.py`; round 2: `port/k2.py` (`_DN_DEV`), `port/moe/moe.py`
(`_DN_DEV` in `_vr_dn_acts`), `port/gemv/moe_dense.py` (`UP_WAVEPOLL`, `_fin_chunk_operands`, `stage_up_all_wp`),
`logs/sweep7.log`, `logs/sweep/{f0,dn,wps,wpg,dnwps,f1}_L31.log`.
