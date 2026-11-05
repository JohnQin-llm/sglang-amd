# K2 sync + runtime, gfx1250 (MI455X) TP4 / EP1 port

Port of the MI355X K2 cross-GPU synchronization layer and host runtime
(`k2bundle/ops/common/{sync,residency,epoch,plan,ranks}.py`,
`k2bundle/runtime/{peer,abi,failstop}.py`) to MI455X (gfx1250, wave32) at
**4 ranks** (TP4, EP1). The public names and signatures that `ops/k2.py` and the
stage files call are kept (listed below). The design is unchanged: tagged mailbox
pairs in uncached IPC peer memory, push style (store into the consumer's memory,
poll your own), epoch-tagged, and fail-stop on every bounded wait.

Import it as the package `sync` with `/data/muse455/k2/ep1/port` on `sys.path`
(`from sync.sync import Mailbox, ...`). Inside the package, imports are relative.

## Files

| file | ports | status |
|---|---|---|
| `sync.py` | `ops/common/sync.py` | gfx1250 encodings; signatures kept (see below) |
| `memmodel.py` | the `ops.py` subset `sync` / `ranks` need | new: GFX12 CPol, drains, fences, clock |
| `plan.py` | `ops/common/plan.py` | wave32: `WAVES` = 16, `LDS_BYTES` = 320 KB |
| `epoch.py` | `ops/common/epoch.py` | unchanged |
| `ranks.py` | `ops/common/ranks.py` | unchanged (tp is a parameter) |
| `residency.py` | `ops/common/residency.py` | unchanged; checked on gfx1250 (see results) |
| `abi.py` | `runtime/abi.py` | `TP` = 4, `K2_DONE_RANKS` = 4 |
| `peer.py` | `runtime/peer.py` | unchanged HIP calls; adds `PeerRuntime` (runner wiring) |
| `failstop.py`, `spec.py`, `debug.py` | `runtime/failstop.py`, `runtime/spec.py`, `ops/common/debug.py` | imports only |
| `tp4.py` | the TP-dependent constants of `moe.py` / `k2.py` / `oproj.py` / `quarter_seam.py` | new: rescaled 8 -> 4, owner maps |
| `test_sync_4rank.py` | - | 4-rank persistent-kernel test (mailbox, barrier, epoch, graphs, fail-stop) |
| `probe/memmodel_probe.hip`, `.s` | - | what hipcc emits on gfx1250 for each hand-off primitive |
| `probe/compile_probe.py` | - | compile-only ISA dump of every sync primitive (`probe/asm/`) |

## Public names the K2 kernel and stages call (grep of `k2bundle/ops`)

Signatures are identical to MI355X unless a column says otherwise. Keyword-only
additions have defaults, so every MI355X call form still works.

| module | name | signature | called from |
|---|---|---|---|
| `sync` | `Mailbox` | `Mailbox(layer, health, alarm, region_ids, *, wait_ticks=WAIT_TICKS)` (new kw) | `k2.py`, `k2_wide.py`, `ep8_k2.py`, `k1.py` |
| `sync` | `Mailbox.put` | `put(a, i, v)` | all stages via `c["put"]` |
| `sync` | `Mailbox.put_bf` | `put_bf(a, i, vs)` (2 or 4 bf16, `i` an element index) | `moe`, `oproj`, `wide_*` via `c["put_bf"]` |
| `sync` | `Mailbox.put_words` | `put_words(a, i, words)` | `attn`, `moe`, `quarter_seam` via `c["put_words"]` |
| `sync` | `Mailbox.poll` | `poll(specs, batch=POLL_MAX)` -> value words | all stages via `c["poll"]` |
| `sync` | `Mailbox.poll_terminal` | `poll_terminal(specs, batch=POLL_MAX)` | `k2.py` `finish_consensus` |
| `sync` | `publish` | `publish(put, a, i, v, who)` | `seam`, `seam_b`, `moe`, `wide_*`, `quarter_seam`, `k2.py`, `ep8_*` |
| `sync` | `import_alarms` | `import_alarms(health, alarm, rank, tp)` | `k2.py`, `k2_wide.py`, `ep8_k2.py` |
| `sync` | `stall` | `stall(fault_rank, fault_epoch, fault_layer, ticks, rank, epoch, layer)` | `k2.py`, `k1.py`, ... |
| `sync` | `sreg`, `preg`, `shift`, `Addr` | `sreg(base, offset, region, extent)`, `preg(...)`, `shift(a, delta)` | `k2.py`, `k1.py`, `ep8_stages.py` |
| `sync` | `_spin` | `_spin(load_all, pending, health, alarm, region_id, index, tag, ticks=WAIT_TICKS)` (new kw) | `stages/attn.py` |
| `sync` | `ld_health`, `POLL_MAX`, `WAIT_TICKS` | unchanged | internal, `attn.py` |
| `ranks` | `peer_bases`, `sum_partials` | `peer_bases(peers, tp)`, `sum_partials(poll, own, pair_of, tp)` | `k2.py`, `seam_b.py`, `ep8_frontend.py` |
| `epoch` | `EPOCH_TAG_PERIOD`, `TAG_SLOTS`, `mailbox_tag`, `advance_epoch_`, `LAST_VALID_EPOCH` | unchanged | `k1.py`, `k2.py`, runner |
| `plan` | `BLOCKS`, `THREADS`, `WAVES`, `LDS_BYTES`, `KernelAbi`, `enter_stage`, `key_tuple`, `pair_layout`, `recording`, `source_digest`, `first_task`, `note`, `Space`, `PAIR_BYTES`, `HEALTH_WORDS`, `Record`, `symbol_params`, `note_unnamed` | unchanged signatures; adds `WAVE` = 32 | `k1.py`, `k2.py`, stages, `validation.py` |
| `residency` | `residency`, `code_objects`, `Residency` | `residency(launch, block)` | `k2.py resources`, `k1.py` |
| `abi` | `FAILED`, `RECORD_WORDS`, `PEER_HEAD`, `K2_DONE_OFFSET`, `K2_DONE_BYTES`, `K2_DONE_RANKS`, `WAIT_SECONDS`, `health_words` | values below | `sync`, `k2.py`, `debug.py` |
| `peer` | `PeerBuffer`, `tp_agree`, `bind_agreed` | `PeerBuffer(nbytes, group, rank, npes, device)` | runner |
| `peer` | `PeerRuntime` (new) | `PeerRuntime(nbytes, group, rank, tp, device, *, blocks, output_rank=0)` | runner, test |
| `failstop` | `check`, `count_step`, `drain` | unchanged | runner |
| `debug` | `region_ids`, `region_id`, `given_up_waits`, `stamp*` | unchanged | `k2.py`, runner |

## What changed for gfx1250, and why

| item | MI355X (gfx950) | MI455X (gfx1250) | evidence |
|---|---|---|---|
| device-scope cache policy `CM_DEV` | 16 (SC1) | **16** (SCOPE_DEV) | `probe/memmodel_probe.s`: relaxed agent atomics = `scope:SCOPE_DEV` |
| system-scope cache policy `CM_SYS` | 17 (SC0\|SC1) | **24** (SCOPE_SYS). The value 17 here would mean SCOPE_DEV\|TH_NT | relaxed system atomics = `scope:SCOPE_SYS` |
| nontemporal `CM_NT` (`memmodel` only; the weights' own) | 2 (NT) | **1** (TH_NT). The value 2 here would be TH_HT | LLVM GFX12 CPol |
| store drain | `s_waitcnt vmcnt(0)` | `s_wait_storecnt 0` (`memmodel.wait_stores`); FlyDSL's `rocdl.s_waitcnt` raises on gfx1250 | ISA |
| `publish` | drain, barrier, flag | drain, barrier, **release fence at the pair's scope** (`global_wb scope:SCOPE_DEV` or `SCOPE_SYS`), flag | hipcc: an agent release fence = `global_wb scope:SCOPE_DEV`. The per-XCD L2 is not device-coherent for plain stores |
| clock | `s_memrealtime` | `llvm.readsteadycounter` = `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`, 100 MHz | ISA; S0 `wall clock 100000 kHz` |
| `import_alarms` | two 16 B loads (8 flag words) | `ceil(tp/4)` 16 B loads (TP4: one) | |
| wait budget | module constant | same default (`WAIT_SECONDS` = 10); `Mailbox(..., wait_ticks=)` / `_spin(..., ticks=)` let tests fail in 0.25 s | |
| waves | 8 × wave64 | 16 × wave32 (`plan.WAVES`). Nothing in `sync` depends on the wave size | |
| LDS a CTA | 160 KB | 320 KB (rocminfo GROUP segment) | |
| CUs | 256 | 256 (`BLOCKS` unchanged) | rocminfo, `multi_processor_count` |

ISA of the ported primitives (`probe/asm/sync_probe/22_final_isa.s`):

- `put` / `put_bf` produce `buffer_store_b64` / `_b128 ... scope:SCOPE_SYS` (PEER) or `scope:SCOPE_DEV` (SCRATCH);
- the `poll` loop is `s_sleep 1` plus a `buffer_load_b128 ... scope:SCOPE_SYS` re-issued every iteration (not hoisted),
  with the health load and a clock read;
- `publish` produces `s_wait_storecnt 0; s_barrier_signal/wait; global_wb scope:SCOPE_DEV; s_wait_storecnt 0;
  buffer_store_b64 ... scope:SCOPE_DEV`;
- `import_alarms` produces one `buffer_load_b128 ... scope:SCOPE_SYS`.
- LLVM narrows a polled pair whose value is unused (the terminal `done` pairs) to a 32-bit load of the tag.
  That is harmless.

## Constants rescaled from 8 ranks to 4 (`tp4.py`, `abi.py`)

| constant | MI355X TP8 | MI455X TP4 | note |
|---|---:|---:|---|
| `TP`, rank count, `K2_DONE_RANKS` | 8 | **4** | `K2_DONE_BYTES` 64 -> 32. Flag words are 0..15; done pairs stay at offset 64 |
| `EPR` (router rows a rank) | 32 | **64** | `E // TP` |
| `RI` (routed intermediate a rank) | 512 | **1024** | `INTER // TP` (S1 loader slices 1024) |
| `POOL_N` (pooled columns a rank) | 512 | **1024** | `MH // TP` |
| `UP_N` (branch columns a rank) | 1024 | **2048** | `H // TP` |
| `AR_OWNER_COLS` (`quarter_seam.OWNER_COLS`) | 1024 | **2048** | AR#1 reduce-scatter owner width |
| `OWNED_FIN_TASKS` / `OWNER_FIN_CTAS` | 4 / 32 | **8 / 64** | `FIN_TASKS // TP` |
| `DOWN_TASKS // TP` (W2 tiles an owner) | 32 | **64** | yr owner map `down_owner(rgd) = rgd // 64` |
| two-CTA down owner | `(bid % 16) // 2` | `(bid % 16) // 4` | `two_cta_down_owner` |
| AR owner | `col // 1024` | `col // 2048` | `ar_owner` |
| fin slice | `rank * 4 + owner // s` | `rank * 8 + owner // s` | `fin_slice`, `fin_slice_owner(j) = j // 8` |
| EP1 row owner (S2 decision) | - | `t // 2` | `row_owner`: 2 of 8 rows a rank, 128 KB a peer |
| derived: `R_GROUPS` (wave32) | 4 | 4 | `EPR // WAVES` = 64 / 16 |
| derived: `P_GROUPS` / `P_TASKS` | 32 / 128 | 64 / **256** | **does not fit** the MI355X placement (see below) |
| derived: `UP_TASKS` | 64 | **128** | **does not fit** the MI355X placement |
| derived: `INTER_W`, `UG_GROUPS`, `W2_STEPS`, `W2_SCALE_G` | 64, 16, 4, 16 | 128, 32, 8, 32 | |
| peer pairs per parity: `yr` | `8·s·8·4096/2` | `4·s·8·4096/2` (half) | `tp4.moe_peer_pairs` |
| peer pairs per parity: `rlog`, `RLOG_PARTS_PAIRS` | `TP·s·EPR`, 8192 | same totals (`TP·EPR = E`) | |
| `ar_pairs(s)` | `8·s·H/2` | `4·s·H/2` | |
| unchanged | | | `BLOCKS` 256, `THREADS` 512, `PEER_HEAD` 256, `RECORD_WORDS` 8, epoch tag (22 bits × 256 slots), `POLL_MAX` 12, `WAIT_SECONDS` 10 |

`tp4.PLACEMENT_CONFLICTS` records the two MI355X CTA placements that TP4 breaks, for the stage porters. The pooler
needs `80 + 256 > 256` CTAs. The unpooler (`U0 = 256 - 128 = 128`) overlaps the gather CTAs (128..191).

## Test: `test_sync_4rank.py` (4 GPUs, one process a rank)

A persistent 256-CTA × 512-thread kernel runs on each of the 4 ranks. The runtime is `PeerRuntime`: uncached IPC
peer buffers exchanged over gloo, a peer table past `PEER_HEAD`, and the alarm word in rank 0's header. Each launch
does the following:

1. a 4-rank barrier: 32 rounds, CTA 0, `put` to every peer and one 4-pair `poll`;
2. a 0↔1 ping-pong: 32 round trips;
3. a device-scope `go` fan-out to all 256 CTAs;
4. an **all-to-all of 128 KB per peer** (32768 words of bf16 pairs per peer, all 256 CTAs). This is the EP1 yr
   reduce-scatter's shape, using `put_bf` 16 B stores, one 2-pair `poll` per thread, and a bit-exact check of every
   bf16;
5. `publish`: each CTA writes 512 plain words, then publishes a flag. The next CTA, probably on another XCD, polls
   the flag and reads the words at device scope, checking every word;
6. the K2 launch tail: a local fan-in over `finish` pairs, then this rank's `done` pair in rank 0's header,
   `poll_terminal` over the 4 `done` pairs, and `import_alarms`.

Every payload depends on the epoch, so a stale pair or stale data would be counted as an error.

Run (about 35 s once the locks are free):

```
flock /data/muse455/locks/gpu0.lock flock /data/muse455/locks/gpu1.lock flock /data/muse455/locks/gpu2.lock \
  flock /data/muse455/locks/gpu3.lock docker exec muse_mi455x bash -c \
  'cd /tmp && PYTHONSAFEPATH=1 timeout 300 python3 /data/muse455/k2/ep1/port/sync/test_sync_4rank.py'
```

`--compile` traces and compiles every build without a GPU launch (no locks needed).

RESULTS_PLACEHOLDER

## Pitfalls found (for the K2 / stage porters)

- **`from __future__ import annotations` breaks FlyDSL kernel arguments** when `Int64` is not a module-level name.
  FlyDSL resolves string annotations in module globals only, and an unresolved annotation falls back to the
  Python value's type, so a pointer becomes `i32`. The first run of this test faulted on all 4 GPUs because of
  that (truncated addresses). Import `Int64` / `Int32` at module level, or do not use the future import.
- **The realtime counter is not common to the CTAs of one GPU**: absolute stamps from different CTAs differ by up
  to tens of ms (`clock_spread_us`). Only CTA-local differences mean anything. On MI355X, `debug.stamp_flush`
  timelines compared CTAs; on gfx1250, compare only within a CTA, or subtract a per-CTA offset taken at a common
  event.
- Kernel names must not contain `-` (the assembler rejects `.amdhsa_kernel` names with a minus sign).
- `rocdl.s_waitcnt` raises on gfx1250; use `memmodel.wait_stores()` (`s_wait_storecnt 0`) / `wait_loads()`.
- The persistent kernel needs one 512-thread CTA per CU on all 256 CUs. The residency check passes (1 CTA a CU).
  With 512 threads, LLVM caps a lane at 256 VGPRs, and the test's full build (64 unrolled spins) spills 52 B a lane.
  A K2 build that spills is refused by `residency.problems`.
