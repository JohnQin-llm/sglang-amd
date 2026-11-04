# Lane F: K2 non-expert phases (pre-expert chain and tail), layer 31 / 30, a8, TP4/EP1

**Result: about 128 -> 107 us/layer.** Graph replay, 20 x 20, all four ranks within ~1 us.
- Same-session baseline: 127.1-129.5 (layer 31) and 128.0-128.7 (layer 30).
- Final default build: 106.5-107.9 (layer 31) and 107.3-109.6 (layer 30).
- Run-to-run noise is about +-1.5 us, so every switch was A/B'd in the same session.

Numerics are unchanged:
- The routes are identical, with the same near-tie rows.
- stream2 cosine min is 0.99992 / 0.99991, the same digits as the baseline.
- In lane C's varied test, every launch is bitwise equal to the golden output.

Tree: `/data/muse455/k2/lanes/f` (copy of `ep1`, paths rewritten). The expert phase (`stage_ug` / `stage_down`, the
UG / down loops in `k2.py`) is untouched.

## Kept switches (all in the k2.py build key, default on in this tree)

| switch | what | us/layer L31 / L30, before -> after (same session) |
|---|---|---|
| `K2_FINISH2` | Lane B's one-level finish. CTA 0's wave 0 polls the 256 local finish flags. Thread 0 pushes its sticky FAILED word as a `done4` pair to every rank and polls its own four, so no remote pulls. Before this, ranks 1-3 polled `done` words in rank 0's memory across XGMI, so their kernel ended ~10 us after their last CTA and the next launch started late (the launch skew) | 127.1 / 128.0 -> **119.7 / 121.0** |
| `K2_ROUTE_CTAS` | Route token t runs on CTA 96 + t, after that CTA's pooler task. On CTAs 0-7 it ran *before* their pooler tasks, so 8 pool tasks waited for router -> RLOG -> route | 119.7 / 121.0 -> 118.1 / 119.4 |
| `K2_FIN_PAIRS` | Lane B's FIN as tagged bf16 pairs into every rank's `finp`: no store drain, no `global_wb scope:SCOPE_SYS`, no FRDY. The unpooler gates on one wave (lane l = slice l of every token, default sleep), then every thread polls its pairs into LDS | 117.6 / 119.0 -> 116.1 / 116.7 |
| `K2_LDS_BARRIER` | `cta_barrier` (`s_wait_dscnt 0; s_barrier_signal -1; s_barrier_wait -1`, the same barrier `gpu.barrier` emits) where the barrier only syncs LDS. `gpu.barrier` also waits `s_wait_storecnt 0`, so after a peer put every wave sat out a whole XGMI write round trip (the router took 5 us from x-seen to done, mostly this). Applied after the AR / RLOG / PPART / POOL / route / BR / FIN puts and to the LDS-only barriers of the pool / x staging. Kept as `gpu.barrier`: every `publish` and the seam's drain before XRDY | 116.1 / 116.7 -> **112.3 / 113.4** |
| `K2_SEAM_DIRECT` | New. o_proj pushes its AR#1 partials to every rank (`stage_oproj`, push-all, 4x the AR bytes, ~1 MB a rank). Seam CTA t sums the four ranks' partials of its whole row itself, with the owner sum's lane -> column map, rank order and bf16 rounding, and the slice sums of squares go through LDS. So x, XSQ and rstd1 are bit-identical. The owner-sum stage and the XG / XSQ all-gather (one cross-GPU hop) are gone: the seam sees its input at 10.8 us instead of 13.6-14.6 | 111.4 / 113.4 -> **108.4 / 111.3** |
| `K2_ROUTE_ARGMAX` | Lane B's top-8 as 8 wave argmax rounds (key desc, expert asc: the radix path's order; weights summed in slot order). Route compute 3.0 -> 1.8 us; the route chain is ~1-2 us behind the POOL / xq chain (see below), so it counts | final stack: off 108.6 / 108.1 and 108.1 / 110.3, on **106.5 / 107.3** |
| `K2_FINISH_LITE` | Finish flags published with drain + flag (`publish_wt`), not `publish`'s L2-wide `global_wb`. Nothing reads plain data on the strength of a finish flag (the host reads after the kernel ends). The drain still orders a give-up's FAILED store before the flag | 107.1 / 109.2 -> **106.0 / 107.8** |

Final default build: 107.9 / 109.6 (acceptance run) and 106.5 / 107.3 (A/B run). VGPRs 203 (TL build 192), no VGPR
spills, LDS 117 KB.

## Tried and rejected (measured, default off, still in the key)

| switch | what | result |
|---|---|---|
| `K2_ROUTER_X8` (+ `K2_POOL_EARLY`) | Lane B's router x8 staging (router CTAs stage all 8 rows of their K quarter, their pooler task reuses the LDS rows); pool weights issued right after o_proj | Neutral every time: 127.2 / 130.2 on the base (base 129.5 / 128.7); 111.3 / 113.2 vs 112.3 / 113.4; 108.5 / 111.0 vs 107.8 / 109.9. POOL_EARLY on all CTAs hurts: the 128 KB / CTA of weights sit in front of every later poll in the wave's in-order load return (~2 us at a CU's ~60 GB/s share) |
| `K2_POOL_FIRST` | Router CTAs run their pooler task first, then the router on the same LDS x (router weights prefetched with a mask) | POOL lands ~6 us earlier, but the route chain becomes critical: 125.1 / 123.5 vs 118.5 / 120.0; 116.6 / 117.2 vs 113.1 / 114.0 on the LDS-barrier stack |
| `K2_POOL_MOVE` | New. Router CTAs drop their pooler task; CTAs 192-255 run it as a second same-quarter task (x already staged) | Neutral: 108.3 / 110.6 vs 107.8 / 109.9; 108.4 / 109.0 vs 108.1 / 109.3 after moving the 2nd task's loads into its branch and making its first barrier LDS-only |
| `K2_XQ_WT` | Lane B: XQ / XSC stored write-through (SCOPE_DEV, TH_WB), XQRDY via `publish_wt` | Neutral: 112.3 / 112.4 vs 111.7 / 113.8 (xq's publish is ~0.3-0.8 us) |
| `K2_POLL_MONITOR=1` (existing) | `global_load_monitor` + `s_monitor_sleep 32` (timeout = the default sleep) | 113.2 / 113.8 vs ~111.8 / 113.3 |

Critical-path probe `K2_DEBUG_XQ_DELAY=<10 ns>`: XQRDY published 3 / 6 us late -> +2.0 / +3.8 us per layer. The pool ->
xq chain is critical, but only ~1-2 us ahead of the parallel router -> route -> route table -> UG round-0 weights
chain. That is why shortening only the POOL branch (X8 / POOL_FIRST / POOL_MOVE / XQ_WT) never showed, and why
SEAM_DIRECT (the common prefix) and ROUTE_ARGMAX (the route branch) did.

## Step 1 (L2-cached reads + one acquire per CTA): why it was not applied
- **POOL, XG, FIN, BR, RLOG live in the peer buffer, which is `hipExtMallocWithFlags(hipDeviceMallocUncached)`
  (MTYPE UC, `sync/peer.py`).** A plain load of UC memory is not cached in the L2, so lane A's treatment cannot apply.
  The reads are already staged once per CTA into LDS (FIN in the unpooler), or one CTA reads each row (POOL by xq,
  BR by gather). FIN_PAIRS / SEAM_DIRECT removed the flag + data double reads instead.
- **x rows (scratch, cacheable), read by all 256 pool CTAs and 64 router CTAs.** The timeline shows x-seen -> x in LDS
  at 0.3-0.4 us. The channel hot spot lane A found (45 us loads) does not happen here, so a per-CTA acquire would
  only add a `global_inv`. I did not build it.

## Timeline (rank 0 unless noted; absolute us from the launch's first stamp; `K2_DEBUG_TL=1`, graph replay)

| point | baseline (eager TL) | final (graph TL, 4 ranks) |
|---|---:|---:|
| owner sees AR / seam sees its input | 8.7 / 14.1 | - / 10.8-11.0 |
| seam done -> pool sees x | 16.2 -> 18.1 | 12.0 -> 13.7-14.1 |
| pool done (normal / router CTAs, max) | 20.4 / 33.4 | 15.3-15.9 / 22-25 |
| route done | 25.9 | 20.4-22.5 |
| xq sees all POOL -> XQRDY seen -> UG start | 31.0 -> 37.4 -> 37.9 | 26.5-27.7 -> 29.4-30.4 -> 29.9-30.9 |
| finalize: YR seen / FMSQ seen / FIN done (median) | 92.1 / 97.6 / 106.5 | 80-85 / 83-92 / 90-92 |
| unpooler sees FIN -> gather sees BR | 110.0 -> 115.0 | 94.8-97.0 -> 101.3-102.3 |
| last CTA end (CTA 0) | 124.2 | 108.3-109.0 |

Pre-expert (start -> UG) went from ~38 to ~30 us. The tail (FIN done -> end) went from ~18 to ~17 us. The tail's
end also comes in ~10 us earlier on ranks 1-3 (FINISH2), which is where most of the per-layer gain came from.
Files: `s3/lf_tl_base2.txt` (baseline, eager), `s3/lf_tl_final.txt` (final, graph).

## Acceptance (final default build)
- `s3/run_k2.py --layers 31,30`: pass_target true on both layers (107.9 / 109.6 us), routes ok.
- `s3/probe_b2b.py --layers 31 --eager 30`: 0 give-ups on every rank, `PROBE ok=True`. This was also run for the
  FINISH2, LDS-barrier and SEAM_DIRECT stacks on the way.
- Lane C `test_b2b_varied.py` (layers 1, 31, 60 x seeds 0-2, served operands, runner): every SUMMARY has
  failed false, gave_up 0, pass_target = launches (27 / 36) and bitwise_golden = launches.
- Every GPU job went through `k2safe.sh`: no UNHEALTHY, no new MES errors. No banned features; poll sleep stays at
  the default 32.

## Where the remaining non-expert time goes (next steps)
1. **Each hop costs 3-5 us inside K2, vs 2.0-2.5 us in lane D's microbenchmark.** A poll of UC peer memory is a
   ~1 us round trip plus the 32-unit sleep, so every wait is quantized at ~2 us. Two polls in flight on one
   gate wave per CTA (lane B's `_spin2`, at the default sleep) would roughly halve that. I did not do it: it raises
   a poller's re-poll rate, which the safety rules forbid in spirit. The pre-expert chain is 2 cross-GPU hops
   (AR, POOL) + 3 device hops; the tail is 4 cross-GPU hops (YR, FIN, BR, done4) + 1 device hop (FMSQ).
2. **The POOL straggler is the router CTAs' pool tasks (done 22-25 vs ~16 us).** Moving them is useless while the
   route chain is 1-2 us behind. A fix has to shorten both chains, e.g. 16 router CTAs computing all 8 tokens from
   x8 staging (a quarter of today's 4x-redundant router weight traffic) plus their pool weights prefetched.
3. **POOL partials all-gathered directly.** Each pool task pushes its fp32 partial to every rank and xq sums the 4
   parts in part order (same numerics). That removes the PPART device hop (~1.5-2 us).
4. **Notes for lane E:**
   - The `gpu.barrier()` after the down loop waits for the YR put acks (`s_wait_storecnt 0`) before finalize;
     `cta_barrier` would not.
   - `_ug_first_weights` (68 VGPRs of loads) is issued right before the XQRDY poll, so the poll's result returns only
     after those weights (in-order return), adding ~1.5 us to the XQRDY hop.

## Incident
At 08:11:15 UTC `lanes/f/port/k2.py` was overwritten by an older copy of itself (an editor save of a stale buffer;
`lanes/e/port/k2.py` was saved 0.3 s later). One edit (POOL_MOVE) was lost and re-applied, and the affected TL run
was discarded. `s3/lf.sh` now refuses to run if the tree differs from my snapshot (`/tmp/lf_snap`) and flags a change
during a run.

## Files changed (vs ep1)
- `port/k2.py`: the 12 lane-F switches and `_lane_f_key` in the build key; `finish_consensus2`; route CTAs;
  seam-direct wiring; pool variants; `finp` / `done4` peer regions; TL stamps 22-47 (`TL_WORDS` 48, `_tlx`).
- `port/gemv/moe_dense.py`: router x8 / masked router loads; `pool_compute(x_ready)`, masked `pool_loads`; route
  split into `_route_argmax` / `_route_radix`; FIN-pairs unpooler (`_fin_pairs_to_lds`, gate); `cta_barrier`s;
  stamps.
- `port/gemv/quarter_seam.py`: `_direct_rows` / `stage_seam_token(direct)`; `cta_barrier`s; stamps.
- `port/gemv/oproj.py`, `port/gemv/gemv.py`: `cta_barrier` after the AR puts / in `rows_to_lds`.
- `port/moe/moe.py`: FIN pairs in `_fin_out`; `K2_XQ_WT`; `K2_DEBUG_XQ_DELAY`; stamps. `stage_ug` / `stage_down`
  are unchanged.
- `port/sync/sync.py`: `cta_barrier` (`K2_LDS_BARRIER`), `gate`, `publish_wt`, `stamp`.
- `s3/probe_b2b.py`: the timeline replays a CUDA graph of 20 launches (`K2_TL_GRAPH=1`, the default) and prints
  absolute stage times (`ABS`), a CTA group (`K2_TL_GROUP=a:b`) and the latest CTAs of chosen stamps (`K2_TL_TOP=..`).
- New: `s3/lf.sh` (k2safe A/B wrapper: `time|tl|b2b|varied "ENV=.."`), `s3/tlsum.sh` (stamps of all ranks side by
  side).
