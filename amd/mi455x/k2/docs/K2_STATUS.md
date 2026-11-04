# K2 (persistent MoE mono kernel, MI455X gfx1250, TP4/EP1): status 2026-11-04 07:30 UTC

## Best measured (updated 08:58 UTC)
- **80.1 us/layer** in the main tree (layer 31; layer 30: 81.3): lane E's expert chain (K2_VRES2) + lane F's
  pre-expert/tail switches (FINISH2, ROUTE_CTAS, FIN_PAIRS, LDS_BARRIER, SEAM_DIRECT, ROUTE_ARGMAX), merged by hand.
  Acceptance: numerics pass_target, 30 back-to-back launches 0 give-ups, 189 varied-input launches bit-for-bit
  golden. Backup of the 101.4 us tree: ep1/backup_pre_ef_0858.

## Before (08:37 UTC)
- **101.4 us/layer** in the main tree `/data/muse455/k2/ep1` (layer 31; layer 30: 102.4), lane E's restructured
  expert chain merged (`K2_VRES2=1`, default on; backup of the previous tree in `ep1/backup_pre_vres2_0837`).
  Acceptance on this build: numerics pass_target, 30 back-to-back launches with 0 give-ups, 189 varied-input
  launches bit-for-bit golden.

## Earlier
- **126.7 us/layer** (layer 31; layer 30: 128.6), graph replay 20 x 20, all four ranks within 0.3 us.
  Tree: `/data/muse455/k2/lanes/a` (report `lanes/a/LANE_A.md`). Numerics: stream2 cosine min 0.99992,
  routes equal except near-tie rows. 30 back-to-back launches: 0 give-ups. Served span it replaces: ~130-140 us.
- Main tree `/data/muse455/k2/ep1`: ~200 us (poll storm fix only: `K2_POLL_SLEEP=32`, `K2_INV_POLL=0`).

## Root causes found today
1. Hang with back-to-back launches: a poll storm (thousands of waves re-polling with `s_sleep 1` starve the producers).
   Fix: `s_sleep 32` between re-polls (AITER mega_moe_tp uses 16). Not a stale-cache problem.
2. The "46 us UG->down wait": device-scope reads of data every CTA reads (XQ, INTER) hot-spotting a few memory
   channels. Fix (lane A): L2-cached reads + one acquire per CTA, 256-B aligned regions. Expert phase 145 -> 62 us.
3. `s_wait_xcnt` stalls = address translation (LLVM GFX125x memory model).

## Measured building blocks (lane D, `lanes/d/LANE_D.md`)
- HBM streaming at the K2 expert size: 16-17 TB/s every path; TDM saturates with ONE loader wave.
- Expert GEMV (w13 shape, 287 MB): today's chain 38.5 us -> split-K over waves, activations resident, no CTA barrier
  in the stream 21.0 us -> + TDM loader wave + 8 compute waves (LDS counters) 20.2 us (14.5 TB/s).
- `global_load_monitor` + `s_monitor_sleep`: wake 325 ns (same XCD), 650-670 ns (other XCD), ~2.1 us (other GPU).
- Clusters: barrier works up to 16; the cluster-16 TDM multicast test HUNG GPU 3 (07:18) -> keep clusters off.

## Projection
| version | us/layer |
|---|---:|
| today (lane A) | 127 |
| + lane D expert chain | ~95 |
| + L2-cached broadcast reads and fewer cross-GPU hops (pre/tail) | ~70-80 |
| K2 v2 (TDM expert phase, monitor waits, dense prefetch under hand-offs) | ~45-55 |
End to end after the 5-patch stack (~11.3 ms cycle, P90 ~350): K2 95 us -> ~450-470; 75 us -> ~540; 50 us -> 600+.

## Serving (lane C, `lanes/c/LANE_C.md`)
- S4 patch refreshed (K2 imported from `SGLANG_MUSE_MONO_PORT`), varied-input b2b test, locked server start script.
- Memory: the two-copy plan costs 105.5 GiB/GPU (EP4 served experts + EP1 K2 copy). Target instead: TP4/EP1 single
  copy, as MI355X shipped (`bundles-c1/sglang.patch.diff`, `original_mono_tp`, `_tp_expert_stack_slice`).

## Next steps after the reboot (GPU 3 MES hung; operator reboots)
1. Merge lane D's expert chain (vres2 / tdm2) into lane A's tree; measure.
2. L2-cached reads + one acquire per CTA for POOL / XG / FIN / BR hand-offs (lane B's area; lane B's own tree is at
   174.7 us on the old base, rebase onto `lanes/ab` = copy of lane A).
3. Port the EP1 single-copy serving mode; first end-to-end K2-on run with `lanes/c/bench/start_probe_server_locked.sh`
   (holds all four GPU locks for the server's lifetime).
4. Arbor: campaign paused (`campaign/c1/session/STOP`); best kept iteration 6 = 245.1 tok/s/user. The served tree has
   the 5 exact-numerics patches applied, uncommitted and unmeasured (iteration 7 pending in the inbox).

## Serving A/B, c=1 (09:28-09:36 UTC), served tree = iteration 6 + five staged patches, simulated acceptance 4.05
- K2 on (SGLANG_MUSE_MONO_GFX1250=1, port ep1 at 80 us/layer): batch-1 decode 475-508 tok/s/user (median ~489,
  ~2.04 ms/token) over 1K/8K/32K; 1,305 K2 steps, no rank failed.
- K2 off, same tree: 367-397 (median ~377, ~2.65 ms/token).
- K2 = +30% decode at c=1 (-0.6 ms/token). Loader: single-threaded shard loading + warm page cache (host MCE avoidance).
- The first K2-on run (09:17) reported a give-up after 892 steps (not reproduced at c=1 over 1,305 steps); the reporter
  now logs the give-up records (K2_REPORT_DELAY).
