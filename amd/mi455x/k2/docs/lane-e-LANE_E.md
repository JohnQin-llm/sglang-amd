# Lane E: the K2 expert phase on the vres2 pattern (K split over waves, activations resident, LDS sums)

**Result: 127.0 -> 101.6 us/layer** (layer 31; layer 30: 129.3 -> 102.7), graph replay 20 x 20, all four ranks
within 0.5 us. Switch **`K2_VRES2=1`** (default 0 in this tree; in the build key). Numerics are **bit-identical to
the baseline** (same stream2 cosine to 16 digits, same routes / near-ties); all three acceptance tests pass.
Expert phase (UG + down stage medians, rank 0): **46.3 -> 23.3 us**. The varied-input test's own graph timing:
130.4-131.5 -> 103.0-103.9 us per K2 launch (graph_step).

Tree: `/data/muse455/k2/lanes/e` (copy of `ep1` at 07:47 UTC). The operator merged the 08:36 state into
`ep1/port` with `K2_VRES2` default "1" and `K2_VR_EPI` default "rot"; the final state here differs from that
merge only in the `K2_VR_EPI` default ("last", same speed, no spin loop) and a cosmetic loop refactor in `k2.py`
(`moe.py` is identical).

## Design (`port/moe/moe.py` `vr_*`, `port/k2.py` `_expert_vres2`)

**UG.** Same task map as before (task u = (slot, 32-column group) on CTA u % 256), but inside a task the K split
is over all 16 waves: wave w owns K steps 2w, 2w+1 of all four chains (gate / up x 2 tiles). Its 2 steps of the
8 tokens' XQ are loaded once after the XQ acquire and stay in VGPRs (33 VGPRs) for every task. Weights are
`buffer_load`ed (NT) one task ahead (68 VGPRs in flight, issued before the current task's waits, pinned with
`sched_barrier(0)`). After its 8 WMMAs a wave `ds_add_f32`s its partials (token lanes only) into the task's LDS
sums [gate/up][token][32 cols] (row stride 33: conflict-free), drains its LDS ops (`s_wait_dscnt 0` as an asm
barrier) and counts itself in on the task's LDS counter (one lane, `ds_add_rtn_u32`). The wave that arrives
16th runs the epilogue alone: lane = (token, column octet), silu(g) u -> bf16, the 32-column amax over 4 lanes,
MXFP8 requant (`Q.code(..., inter=True)`, `Q.pack8`, unchanged), INTER / ISC write-through stores (lane A's
`K2_UG_WT` semantics, trash pick row and 256-B regions unchanged). No CTA barrier anywhere in the UG stream.
- **Lazy UGRDY** (`K2_VR_LAZY=1`): the epilogue wave does not wait for its write-through stores. It drains
  them (`s_wait_storecnt 0`) and puts UGRDY at its next task, by which time they have landed. A final
  `vr_flush` runs after the UG loop. The data is still drained before its flag (same ordering as `_publish_wt`).
- **XQ wait** (`K2_VR_LATE=1`): CTAs 8..15 (the xq CTAs, which just published XQ) poll XQRDY and acquire
  *before* issuing their first weights. A poll behind 20 weight loads returns only after them (in-order
  loads), which cost these CTAs ~3.6 us.
- **UG CTA order reversed** (`K2_VR_REV=1`): the r = 32 (nu % 8) CTAs with an extra UG task are the high ones
  (160..255), not the seam / route / xq CTAs that enter the expert phase late.

**Down.** A CTA owns a **contiguous run** [d0, d1) of the slot-major (slot, 16-row MH tile) list, so it touches
at most 2 slots. The runs are balanced against the UG tasks: one UG task = 16 tiles of bytes, so the extra-task
CTAs get 16 tiles fewer and the run's tail (the late slots they themselves produce last). nu = 43 gives 33 vs 49
tiles, 129 tile-units on every CTA (`vr_down_range`, exact for nu + 2(nu % 8) >= 16, a floor split otherwise).
- Waves: K half h = w % 2 (4 of the 8 K steps), tile stream j = w // 2. Each stream's tiles lie in **one** slot:
  the run's b first-slot tiles go to round(8 b / n) streams (`vr_down_geom`). A wave polls UGRDY of its slot once
  (32 flags, device scope), loads its resident INTER operands once (66 VGPRs; unpicked lanes and dead waves read
  the trash pick row), then streams w2 tiles 2 ahead (`sched_barrier` again) and `ds_add_f32`s 8 sums a tile into
  LDS [tile][row][token]. The two K halves add into a zeroed sum: exact and order-free.
- **YR after the run** (`K2_VR_YR=end`, `vr_yr_out`): after the run's CTA barrier, the CTA writes its tiles' sums
  from LDS in one pass. A wave unit = (token t, 8 consecutive tiles), lane = (tile, row quad), 2 tagged pairs a
  lane: 512 contiguous bytes of route (t, k), with row t's owner rank uniform across the wave (one resource, no
  owner branches). Unpicked lanes are masked by an out-of-range offset of a resource bounded by the YR extent;
  all words are built first and the 5 stores issued back to back.
- One `acquire_cta` between UG and down is kept (`K2_VR_DNACQ=1`). Removing it is safe by construction, since the
  UG-start acquire is in the same launch and INTER is read only after its flags, but it measured no gain.

LDS: +57 KB (UG sums 16.9 KB, down sums 40 KB, 88 counters; zeroed before the XQ wait's barrier); the old
`kred` / `amx` are then unused and dropped by the compiler. Total 145,168 B. VGPRs **213** (was 249), no VGPR
spills, scratch 0. SGPR spills to VGPR lanes: 42, was 32.

## Per-stage timeline (rank 0 medians, us; `K2_DEBUG_TL=1`, `s3/probe_b2b.py --layers 31 --timeline`)

| stage | before (127 us build) | after (101.6 us build) |
|---|---:|---:|
| oproj / router / route / pool | 7.3 / 14.5 / 9.4 / 10.9 | 7.1 / 14.5 / 9.8 / 10.9 |
| xq / route_wait | 14.5 / 16.0 | 15.2 / 16.3 |
| **UG** (wave 0) | **22.6** (6 rounds, ~4.5 each) | **11.1** (~2 us a round = 17 TB/s) |
| **down** | **23.7** | **12.2** |
| down: UG->down acquire / first poll / run | 0.3 / 16.0 / (chain 3.9, YR drain 4.2) | 3.3 / 0.6 / 8.3 |
| finalize | 16.9 | 16.5 |
| up / gather | 28.3 / 29.3 | 23.7 / 23.3 |
| CTA span median (max) | 111.7 (121.1) | 83.3 (90.9) |

Per CTA (rank 0, final build): XQ ready at 36.3 us, down run done at 59.6 us (max 62.4): 23.3 us for the
rank's 281 MB of expert weights (12 TB/s average; UG runs at ~15-17 TB/s, down at ~10.6 TB/s).

## How it got there (graph replay, layer 31 / 30, us/layer)

| step | L31 | L30 | note |
|---|---:|---:|---|
| baseline (switch off; this tree = ep1) | 127.0 | 129.3 | |
| vres2 UG + down, YR put a tile by the second K half | 124.3 | 123.9 | UG 22.6 -> 10.4, but down 23.7 -> 28.6 |
| + YR after the run (`vr_yr_out`) | 103.0 | 104.6 | down 28.6 -> 14.5 |
| (no second acquire) | 104.5 | 105.6 | rejected (no gain) |
| + single-slot waves | 103.2 | 104.3 | |
| + reversed UG order, xq CTAs poll first | 102.3 | 104.1 | |
| + lazy UGRDY, late CTAs acquire before their weights | 101.6 | 103.3 | UG->down gap 5.6 -> 4.4 |
| (`K2_VR_UGW` 24 / 32) | 101.9 / 101.5 | 102.1 / 102.9 | noise; kept 16 (bytes) |
| (epilogue rotated over waves, `K2_VR_EPI=rot`) | 101.6 | 103.3 | gap 4.4 -> 3.3 but no end-to-end change |
| (down weights issued under the last UG task) | 104.6 | 105.4 | rejected: the peeled loop made the allocator copy the in-flight weights at loop entry (`s_wait_loadcnt 0`) |
| **final default** (`last`) | **101.6** | **102.7** | |

## Findings (gfx1250 / FlyDSL)

1. **Remote stores in a streaming loop stall the wave.** On gfx1250, restoring EXEC after a divergent branch
   that issued memory ops waits for their address translation (`s_wait_xcnt 0`), and so does overwriting their
   source VGPRs. The per-tile YR puts (system-scope tagged stores to the peer GPUs, 4 owner branches) cost
   ~4 us a tile per wave. Moving all YR stores to one branch-free pass after the run took down from 28.6 to
   ~12 us. This is the likely cause of lane A's unexplained "16 us first down wait" too: its down chain
   puts YR per task.
2. LLVM sank the next task's prefetch loads below the current task's `s_wait_loadcnt 0` (the expert id comes
   from an LDS load); `rocdl.sched_barrier(0)` right after issuing them keeps the pipeline.
3. **Hang (caught by k2safe, GPUs healthy).** A spin `while count < 16` around a lane-0-only LDS atomic compiled
   into a divergent loop: once lane 0 left EXEC, `readfirstlane` read a lane that never loaded (0 forever). Fix:
   every lane does the same volatile atomic load (a broadcast, uniform loop), plus an iteration bound
   (`VR_SPIN_MAX`). Rule: no lane-predicated loads inside a traced `while`.
4. The FP4 x FP8 partial sums are exact in fp32 at this size, so reordering the K reduction (16 `ds_add_f32`
   contributors) leaves the outputs bit-identical.
5. Peeling a software-pipelined `scf.for` (to hand its last prefetch slot to another stage) makes the register
   allocator copy the carried in-flight loads at the loop entry, which forces their completion.
6. Tooling: edits made through the IDE editor and through shell scripts on the same file clobbered each other
   twice (an older buffer was written back). Use one mechanism per file. The operator's rule: shell / python
   only, and re-read the file right before each edit.

## Acceptance (final default build, `K2_VRES2=1`)

1. `s3/run_k2.py --layers 31,30 --reps 20 --iters 20`: L31 **101.6 us** (ranks 101.6 / 101.1 / 101.4 / 101.5),
   L30 **102.7 us**. pass_target true on both. L31: stream2_cos_min 0.9999225, routes ok, near-ties [0, 2].
   L30: 0.9999088, routes ok. Identical to the baseline.
2. `s3/probe_b2b.py --layers 31 --eager 30`: 0 CTAs gave up on every rank, PROBE ok.
3. `lanes/c/s3/test_b2b_varied.py` (K2C_PORT_ROOT = this lane): every SUMMARY line failed false,
   gave_up_ctas [0, 0, 0, 0], pass_target = launches (eager_step 27/27, eager_launch 27/27, graph_step 36/36,
   graph_launch 36/36, runner_eager 27/27, runner_graph 36/36), bitwise_golden = launches, stream2_cos_min
   0.999795 (same as the switch-off run).

Logs: `logs/final_time.log`, `logs/final_b2b.log`, `logs/final_varied.log`, `logs/final_tl.log` (+ the
per-CTA dump `logs/tl_final_rank*.json`, `python3 logs/tlgroups.py logs/tl_final 4`); baselines `logs/base_*`.

## Not done / next

- **tdm2 (TDM loader wave) not attempted.** Standalone, Lane D measured it only 4% faster than vres2 (20.2 vs
  21.0 us). The expert phase is now 23 us of the layer, ~6 us above the 281 MB / 16.5 TB/s floor. Of that, ~3 us
  is the UG -> down transition (wave skew at the end of UG plus the acquire), ~1.5 us is the YR pass plus its
  barrier, and the rest is down running at ~10.6 TB/s. A TDM loader streaming UG and down weights continuously
  into an LDS ring would attack exactly the transition: estimate -2..-3 us a layer (ring 4 x 34 KB fits: 145 +
  136 KB < 320 KB). Plain `tensor_load_to_lds` from one wave, no clusters / masks.
- **Owner-computes down** does not fall out naturally. A down tile needs all 32 UG groups of its slot (K = 1024
  spans 32 CTAs), so owner-computes means cross-CTA fp32 split-K atomics on y (~33 MB a rank), or 4x WMMA
  waste at K = 32. The UGRDY poll now costs 0.6 us median, so there is little to win.
- The layer is now dominated outside the expert phase: ~36 us to XQ ready (router / pool / xq hand-offs) and
  ~24 us after down (finalize 16.5, up / gather ~23). That is lane B's L2-cached-read / acquire treatment for
  FIN / BR / POOL.

## Switches (all in the `k2.py` build key)

| switch | default | meaning |
|---|---|---|
| `K2_VRES2` | 0 | the vres2 expert phase (needs `K2_ACT_L2=1`, `K2_UG_WT=1`) |
| `K2_VR_YR` | end | YR puts after the run (end) or by a tile's second K half (tile) |
| `K2_VR_DNACQ` | 1 | `acquire_cta` again between UG and down |
| `K2_VR_REV` | 1 | UG CTA order reversed |
| `K2_VR_LATE` | 1 | xq CTAs wait for XQ (and acquire) before their first weights |
| `K2_VR_LAZY` | 1 | UGRDY published at the epilogue wave's next task |
| `K2_VR_UGW` | 16 | down tiles an extra UG task counts for |
| `K2_VR_EPI` | last | UG epilogue on the last wave to add (last) or wave i % 16 (rot, bounded LDS spin) |

## Files changed

- `port/moe/moe.py`: the `vr_*` section (LDS sums / counters, UG weights / activations / task / epilogue /
  lazy flush, down range / geometry / weights / activations / loop, `vr_yr_out`). Imports `bf16_pair`, `rsrc`,
  `CM_SYS` from `port/sync/memmodel` and `_noted` from `port/sync/sync`. Existing stages unchanged.
- `port/k2.py`: the switches and build key, the LDS arrays, the context entries (`vr_*`, `tag`, `tl_on`),
  `_expert_vres2` / `_vr_ug_loop` / `_vr_first` / `_vr_ub`, and the expert block under
  `if const_expr(_VRES2) ... else` (the else branch is the old code; the switch-off build matches the baseline
  build's stats: 249 VGPRs, 119,856 B LDS, 9,493 ISA lines).
- `port/sync/sync.py`: refreshed from ep1 (the operator's monitor-wait fix, which landed mid-copy); no change of
  mine.
- `s3/probe_b2b.py`: extra timeline gaps (`vr_seg0`, `vr_dn_all`, `ugend_lo/hi`, `dnall_lo/hi`;
  `K2_TL_SPLIT`) and `K2_TL_DUMP=<prefix>` (per-CTA stamps to JSON). `logs/tlgroups.py`: per-CTA-group view of a
  dump.
