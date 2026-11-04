# Lane A: K2 expert phase (UG -> down hand-off), layer 31, a8, TP4/EP1

**Result: 200 -> 128 us/layer** (graph replay, 20 x 20, all four ranks within 0.3 us; layer 30: 128.6 us).
Numerics unchanged: stream2 cosine min 0.99992 (target 0.999), routes equal except near-ties rows 0 and 2
(identical JSON to before). Layer 30, run right after layer 31 in the same process, also passes (0.99991,
routes equal); that is the cross-launch stale-cache check. 30 back-to-back eager launches: 0 CTAs gave up on every rank.

## What the hand-off wait really was

The "~46 us UGRDY wait" was **not** a sync problem. Instrumenting the first down poll showed that, at the
first look, **all 32 flags were already set on every CTA**, yet that single 8-byte load took ~45 us to return.
Rotating the UGRDY copies by epoch (to rule out stale lines left by earlier launches) changed nothing.
The real cost: every expert activation read was a **device-scope load** (`xcm = CM_DEV`, i.e. past the per-XCD L2).
Every UG wave of the grid reads the same 32 KB of XQ, and every down wave reads the same 64 KB of INTER:
roughly 32-64 MB a round, all aimed at a few KB of addresses. These reads queue on one or two memory channels,
and every other device-scope access waits behind them, including the UGRDY poll. So the split-K / slot-group
redesign was not needed. The fix is to make the hand-off data L2-cacheable and keep it correct by construction.

## Changes (all behind env switches in the build key, defaults on)

| switch | what | us/layer |
|---|---|---:|
| (baseline) | | 201.2 |
| `K2_ACT_L2=1` | XQ / INTER / ISC read with plain (L2-cached) loads. One wave per CTA runs `global_inv scope:SCOPE_DEV` + barrier (`acquire_cta`) after the XQ wait and before the down loop. Correctness by construction: scratch regions are 256 B aligned, the ISC pick stride is padded to 256 B, and unpicked down lanes read a trash pick row. So no cache line mixes data that becomes ready at different times, and no read ever pulls in a not-yet-written line | 146.1 |
| `K2_UG_WT=1` | INTER / ISC stored write-through (SCOPE_DEV); UGRDY published without the L2-wide `global_wb` (`_publish_wt`) | 145.9 |
| `K2_UG_PIPE=1` | UG loop software-pipelined. Task u's activation loads go out, then task u+256's weights (68 VGPRs carried through the `scf.for`), then u's WMMAs and epilogue, so the epilogue overlaps the next weight stream. Round 0's weights are issued before the XQ wait. Dead prefetches use out-of-bounds buffer offsets (no traffic) | **130.0** |
| `K2_DOWN_WFIRST=1` | a down task's w2 loads issued before its UGRDY poll | ~neutral |
| `K2_DOWN_REV=1` | down wave-tasks in reversed CTA order: the partial third round lands on CTAs with 5 UG rounds | **128.0** |

VGPRs 249 (cap 256), no spills, LDS 117 KB.

## Per-stage timeline (rank 0 medians, us; K2_DEBUG_TL=1)

| stage | before (200 us build) | after (128 us build) |
|---|---:|---:|
| oproj / router / seam / route / pool | 7 / 22 / 8 / 11 / 18 | 7.6 / 18 / 4 / 11 / 14 |
| xq / route_wait | 16 / 17 | 17.5 / 19 |
| **UG** (6 rounds) | 48-51 (round ~7.5) | **22.6** (round ~4.5) |
| **down** | 53-54 | **23.6** |
| first down task: wait / chain | 45 / 4 | 16.5 / 3.8 |
| **finalize** | 43 | **15.8** |
| up / gather | 30 / 31-38 | 30 / 29 |
| CTA span | 175-181 | 120 |

Expert phase (UG start -> finalize done), medians: ~145 -> **~62 us**.

## Tried and rejected (measured, reverted)

- **UGRDY polled through L2** (invalidate + plain load each try): 156-160 us. The per-poll `global_inv`s hurt
  everyone.
- **One UGRDY poller per CTA + barrier**: worse by about 10 us.
- **Instruction prefetch** at stage entries (`s_prefetch_inst`): no gain. Note that prefetching past the end of the
  code object faults ("page not present").
- **Clock read only every 16th re-poll in `_spin`**: no gain.
- **Plain polls without an invalidate**: hang/give-up on the eager run. That confirms plain loads of cross-CTA
  data really need the acquire.

## Still open (where the remaining expert time goes)

- The first down task's wait is still about 16 us, even though its flags are set. Note that in the TL build the TL
  stamp stores themselves stall about 15 us at the start of the down phase (an `s_wait_xcnt` after a SCOPE_DEV
  store, with no load in between), so the TL clock perturbs this phase. The stall is local to the CU. It does not
  depend on the poll scope, the invalidate, the instruction cache or the realtime clock. The best remaining
  suspect is the device / system-scope data-fabric path being saturated by the YR exchange. YR is about 1 MB of
  32-B tagged system-scope stores per rank, and the first task's puts take 4.2 us to drain.
- Next steps, in order:
  1. Pack YR as larger contiguous stores: gather a CTA's 16 tiles in LDS and write 512 B per pick, or a 64-bit
     pair with 3 bf16 values and a 16-bit tag, to cut fabric transactions 2-4x.
  2. Issue round-0 down weights during the last UG round.
  3. Apply the same L2-cached-read and acquire treatment to the other broadcast hand-off reads: the BR, FIN and
     POOL rows that every CTA reads. That is lane B's area.

## Files

- `port/k2.py`: the switches; `acquire_cta`; the pipelined UG loop; the down order; the scratch alignment; the
  build key.
- `port/moe/moe.py`: `_chain` split into `_wload` / `_xload` / `_mma_steps`; `ug_weights`, `stage_ug_pipe`,
  `_ug_tail`; `_publish_wt`; the ISC stride and trash pick row (`isc_ps`, `picks`, `scratch_plain(l2=)`); down
  weights before the poll; TL stamp 26 (first YR drain). The one-stage launchers are unchanged (the new behaviour
  is keyed on `c` entries only K2 sets).
- `port/sync/sync.py`: no net change.
- `s3/probe_b2b.py`: more timeline gaps (`acq_down`, `acq_ug`, `poll_down`, `yr_drain`).
- `s3/ab.sh`: A/B driver (run inside the container under the four locks): `ab.sh time|tl|b2b "ENV=.." ...`.
