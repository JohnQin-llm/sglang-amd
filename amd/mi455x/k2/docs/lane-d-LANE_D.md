# Lane D: gfx1250-specific features for K2 (monitor wait, TDM, clusters, named barriers)

Workspace: `/data/muse455/k2/lanes/d` (`port/` = a copy of `ep1/port`, `bench/` = microbenchmarks,
`results/` = JSON). GPU 3 only for single-GPU runs (`flock gpu3.lock`), all four locks for the cross-GPU run.

**Status (07:25 UTC): GPU work stopped. The cluster-16 TDM multicast run hung GPU 3** (details at the end).
Step 1 a/b/c are measured; Step 2 (expert-phase prototype) was not run.

## 1b. Weight streaming (TDM / async-to-LDS / VGPR), one persistent CTA a CU (256 CTAs)

### Raw HBM read bandwidth (`bench/stream_bw.py`, `results/stream_bw_*.json`)

287 MB a launch (the K2 expert span), rotating over 4 cold regions; graph replay.

| path | config | us / 288 MB | TB/s |
|---|---|---:|---:|
| buffer_load b128 to VGPR (today's K2 access shape) | 16 waves, 8 x 1 KB steps in flight a wave | 17.5 | 15.3 |
| same | 16 waves, depth 4 / 16 | 18.1 / 28.1 | 14.8 / 9.6 |
| same | 8 waves depth 8 / 4 waves depth 16 | 25.4 / 84.6 | 10.6 / 3.2 |
| **tensor_load_to_lds (TDM)** | **1 loader wave**, 4 x 32 KB ring | 16.7 | 16.1 |
| TDM | 1 wave 4 x 64 KB; 4 waves 2 x 32 KB each | 16.5 / 16.5 | 16.2 / 16.3 |
| global_load_async_to_lds_b128 | 16 waves x 8 in flight; 8 waves x 16 | 16.2 / 15.7 | 16.5 / 17.1 |
| TDM, 1 GB a launch (asymptote) | 1-4 loader waves | 59.0-60.3 | 17.8-18.2 |

- Every path reaches the same ~16-17 TB/s at the K2 size (~18 TB/s asymptotically, the practical HBM peak
  here). TDM's advantage is not more bandwidth but that **one wave** saturates the CU's share of HBM (VGPR
  loads need all 16 waves with 8 steps in flight each, and collapse with fewer waves).
- NT (TH=1) vs the default cache policy: within noise for TDM and VGPR.
- FlyDSL: raw-address TDM descriptors from `aiter...mega_moe_gfx1250.tdm_prims` (`tdm_group0/1`,
  `tdm_load`), `rocdl.s_wait_tensorcnt`; LDS from `fx.SharedAllocator(static=False).allocate(n, 128)`.
  A launch with more than 320 KB of LDS fails with hipErrorInvalidValue (no compile error).

### Streaming that feeds the scaled WMMA (`bench/gemv_stream.py`, `results/gemv_stream_*.json`)

The w13-like GEMV: 256 CTAs x 33 tiles of 16 rows x K = 4096 (MXFP4 in the gfx1250 preshuffle, 32 KB + 2 KB
scales a tile; 287 MB), 8 MXFP8 activation rows (B operand), fp32 out. Every variant agrees with the chain to
rel-L2 1.2e-7.

| variant | us | TB/s |
|---|---:|---:|
| **chain** = today's K2 `_chain` (16 waves, a wave a whole tile, weights + activations buffer_load'ed every step) | **38.5** | 7.6 |
| vres: K split over 8 waves, activations resident in VGPRs, weights buffer_load'ed, LDS reduction + CTA barrier a tile | 33.4 | 8.8 |
| tdm: 1 TDM loader wave + 4-16 compute waves (K split, weights ds_load'ed from the ring), CTA barrier a stage of 1 tile | 31.4-33.0 | 8.9-9.4 |
| tdm, 3 tiles a stage, 2 slots | 21.7-22.4 | 13.1-13.5 |
| tdm isolation (1 tile a stage): loads only / compute only | 17.5 / 18.9 (sum = 32: no overlap) | |
| **vres2**: vres with no barrier (partials ds_add_f32 into an LDS y, one barrier at the end) | **21.0-21.7** | 13.6-14.0 |
| **tdm2**: TDM loader wave + 8 compute waves, LDS full/empty counters (warp specialization), ds_add_f32 y | **20.2-20.5** | **14.3-14.6** |

- **The 1.9x is structural, not TDM.** Today's chain re-reads the activation operand (2 KB a WMMA, from L2)
  for every 1 KB weight step, and a wave owns a whole tile. Splitting K across the waves, with each wave's
  activations resident in VGPRs (64 VGPRs at 8 waves), and never barrier-syncing the CTA inside the stream,
  takes the same GEMV from 38.5 to 21 us. TDM plus warp specialization adds ~5% (20.2 us, 88% of the raw stream).
- Any per-tile CTA barrier serializes load and compute (isolation run: load 17.5 + compute 18.9 = 32 us).
  The compute side is latency-bound (LDS load -> 4 dependent WMMAs -> park -> barrier -> reduce), so it must
  run decoupled from the loader: LDS counters (here) or named barriers.
- Waves needed: 1 loader + 8 compute waves is the sweet spot (4 compute waves: 22.5-23.2 us; 16: 21.3-21.7).
- What TDM buys for K2 beyond the 5%: the stream needs one wave and no VGPRs, so the other 15 waves are free
  for compute, polling and the next stage, and a CTA can **prefetch its next stage's weights into LDS (up to
  ~300 KB a CU = 77 MB a GPU, more than the 70 MB dense span of a layer) while it waits on a hand-off**.

## 1a. global_load_monitor + s_monitor_sleep (`bench/monitor.py`, `bench/monitor_xgpu.py`)

FlyDSL has no op for it; inline asm works: `global_load_monitor_b32 $0, $1, off scope:SCOPE_SE` +
`s_wait_loadcnt 0`, then `s_monitor_sleep N` (`bench/common.py` `monitor_load` / `monitor_sleep`). LLVM:
`llvm.amdgcn.global.load.monitor.b32(ptr, ordering, scope-metadata)`, `llvm.amdgcn.s.monitor.sleep(i16)`.

- **`s_monitor_sleep N` is a timeout**, same scale as `s_sleep N` (~29 ns a unit): with nobody writing, a
  load + sleep iteration costs 145 / 346 / 1050 / 3787 ns for N = 1 / 8 / 32 / 127. So a missed wake costs at
  most N units and nothing hangs.

One-way hand-off latency, ping-pong of a counter, one wave a side (stores `scope:SCOPE_DEV`; cross-GPU
`SCOPE_SYS` into the peer's memory, waiting on local memory), median of 3 x 1000-2000 rounds, 0 timeouts:

| wait | same XCD (CTA 0 <-> 8, 16) | other XCD (CTA 0 <-> 1..4) | other GPU (2 <-> 3), uncached / fine / coarse |
|---|---:|---:|---:|
| poll + s_sleep 1 | 280-290 ns | 650-680 ns | 2091 / 2027 / 2171 ns |
| poll + s_sleep 32 (K2 today) | 715 ns | 1106-1115 ns | 2509 / 2444 / 2738 ns |
| monitor + s_monitor_sleep 32 | 325 ns | 650-670 ns | 2081 / 2025 / 2168 ns |
| monitor + s_monitor_sleep 127 | 325-335 ns | 650-673 ns | 2068 / 2044 / 2167 ns |

- **A store from another XCD and from another GPU wakes the monitor-sleeping wave** (with the 3.8 us timeout
  the latency equals the 1-unit poll; without a wake it would be ~2 us higher). All three peer memory kinds,
  monitor scope SE or DEV, behave the same.

Poll storm (`storm`): 256 CTAs, waves 0-7 stream 287 MB, waves 8-15 (2048 waves) wait on their own flag line
until their CTA's streamers finish:

| waiters | none | poll s_sleep 0 / 1 / 8 / 32 | poll SYS scope, s_sleep 1 | monitor N = 1 / 32 / 127 |
|---|---:|---:|---:|---:|
| stream us | 31.3 | 50.7 / 50.6 / 47.6 / 31.2 | 50.8 | 50.7 / 30.7 / 31.3 |

- Fast polling slows a concurrent bandwidth-bound stream by 1.6x (the K2 poll storm reproduced: it is the
  load traffic, since monitor with a 1-unit timeout is as bad). **Monitor with N >= 32 costs nothing and wakes
  at the 1-unit poll latency.** Against today's `s_sleep(32)`: -0.45 us a hand-off on the GPU (1.11 -> 0.66 us
  cross-XCD) and -0.4..0.6 us a cross-GPU hop (2.5 -> 2.05 us). Over ~7 cross-GPU hops and ~6 intra-GPU hand-offs
  on the layer's critical path: ~5-6 us a layer.

## 1c. Clusters (`bench/cluster.py`, `results/cluster.json` not written: the run hung)

- Launch in FlyDSL: `kern(..., value_attrs={"rocdl.cluster_dims": "n,1,1"}).launch(grid, block,
  cluster=(n, 1, 1))` (both are needed, as in AITER's `mxfp4_preshuffle_gfx1250_tdm.py`). Sizes **2, 4, 8, 16
  launch** (256 CTAs); `rocdl.cluster_workgroup_id_x()` = 0..n-1; the CTAs of a cluster report one SE field
  (HW_ID1[20:18]) and one HW_ID2, i.e. the same XCD.
- Cluster barrier (`flydsl.expr.rocdl.cluster.cluster_barrier`: CTA barrier + one `s_barrier_signal -3` a CTA +
  `s_barrier_wait -3`): **35 / 41 / 43 / 44 ns a round** for n = 2 / 4 / 8 / 16 (CTA barrier: 6 ns), vs ~0.3-1.1
  us for a flag hand-off between CTAs.
- **But the hold check failed for n = 2 and 8**: CTA i waits i x 3 us, then the barrier; in some clusters a CTA
  left the barrier before the last one arrived (n = 4 and 16 held). Not understood; do not rely on the cluster
  barrier without a follow-up (e.g. signal from every wave, `s_barrier_signal_isfirst`, check the barrier state).
- TDM multicast (every CTA of a cluster issues the same load with workgroup_mask = the cluster): n = 2: 21.4 us
  vs 22.9 unmasked for the same 287 MB a cluster group (+7%), n = 4: 35.3 vs 39.5 (+12%), n = 8: 73.8 vs 73.9
  (no merge visible). **n = 16 with mask 0xFFFF hung the GPU.** For K2 multicast is of little use anyway: each
  expert weight byte is read by one CTA; only the tiny activation broadcasts would share.

Not measured (time / the hang): named barriers (FlyDSL needs `addrspace(15)` named-barrier globals for
`s_barrier_signal_var` / `s_barrier_join`; LDS counters did the warp specialization instead), TDM
`atomic_barrier_enable` with `ds_atomic_async_barrier_arrive_b64`, `global_prefetch`.

## Step 2 (not run) and the K2 v2 recommendation

Planned prototype (code not written yet): each CTA owns a contiguous run of (expert, 32-column group) units of
the rank's 43 x 1024 intermediate columns (1376 groups / 256 CTAs = 5.4 a CTA, max/avg 1.12):
1 TDM loader wave streams its w13 rows (gate + up, 64 KB + 4 KB a group) then its w2 K slice (16 B a row x 4096
rows a group = 64 KB + 4 KB), 8 compute waves split K as in `tdm2` (xe resident in VGPRs), silu(g) u -> MXFP8
requant in LDS (a 32-group is CTA-local), then w2 over the CTA's own K columns with the requantized inter as
the B operand, partials fp32 `global_atomic_add` into the rank's y[pick][4096], a per-expert arrival counter
(the last arriving CTA converts to bf16 and does the YR put): no flag polling inside the expert phase.

Projection from the measured pieces: 281 MB at the tdm2 efficiency (14.3 TB/s) = 19.7 us, x 1.12 imbalance,
+1-2 us ramp, +2-3 us split-K atomics / last-arriver tail = **~24-27 us for the expert phase per rank** vs ~100
us inside K2 today (39 us standalone), floor 16 us.

**K2 v2 structure:**
1. Expert phase as above (owner-computes, no UG -> down hand-off, K split by wave, TDM loader + 8 compute
   waves, LDS counters): -70 us.
2. Every wait = `global_load_monitor` + `s_monitor_sleep 127` instead of `s_sleep 32` polls: no poll storm,
   1-unit-poll latency (-5-6 us a layer), and the sleep default no longer trades latency for safety.
3. While a CTA waits on a hand-off, its loader wave TDM-prefetches the next stage's dense weights into LDS
   (o_proj / router / pooler / unpooler tiles), so the dense 70 MB streams under the hand-off chain.
4. Clusters only after the barrier hold issue is understood (would replace the in-XCD fan-ins: ~40 ns vs ~1 us).

Projected layer: hand-off chain ~7 cross-GPU hops x 2.05 us + ~6 on-GPU x 0.5 us = ~17 us, expert phase ~25 us
(it cannot overlap the hops that feed it), dense stages ~6-10 us not hidden by the prefetch -> **~45-55 us a
layer** vs 200 us today.

## Incident: GPU 3 hang (07:18 UTC)

`bench/cluster.py --tests ids,cbar,check,mcast --sizes 2,4,8,16`: after the n = 16 barrier check, the
multicast kernel `cl_mcast_n16_r0_288_m1` (256 CTAs x 64 threads, cluster (16,1,1), wave 0 of every CTA issues
the same `tensor_load_to_lds` of its cluster's region with workgroup_mask 0xFFFF, 4 x 32 KB ring, NT, then
`cluster_barrier`) never finished. The python process is `<defunct>` under `timeout 120` (container pids
184810 / 184817), GPU 3 (0004:01:00.0) shows 100% GFX at ~1.19 kW, and dmesg repeats
`amdgpu 0004:01:00.0: MES(n, 0) failed to respond to msg=MISC (WAIT_REG_MEM)`. By 07:26 the process was
reaped, but GPU 3 still showed 100% GFX and dmesg moved on to `MES(0, 0) ring buffer is full`. Nothing was
killed; GPU work stopped. GPU 3 needs the operator (queue teardown / reset) before anyone uses it.
