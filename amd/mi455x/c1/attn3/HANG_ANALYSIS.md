# GPU 0 hang at 17:28-17:44 UTC: cause and fix (CPU analysis, nothing re-run)

Job: `bench.py --kind full --ctx 340000 --variants g3_T128_S128_B4_P1_A16, g3_T128_S128_B4_P1_A16_F0,
g3_T128_S256_B4_P1_A16, g3_T64_S512_B4_P1_A16, g3_T128_S256_B4_P1_A16_F0` (timing probes). It ended with k2safe
rc 124, 10 MES errors, SDMA ring timeouts and a devcoredump on every GPU.

## Which variant

The bench runs the variants strictly in order; each first launch JIT-compiles into `tcache/`. Exactly two
`attn3_split` binaries were written during the job:

| time (UTC) | cache dir | TDM instructions | `s_wait_tensorcnt` | VGPRs | config (matches the AOT build) |
|---|---|---|---|---|---|
| 17:28:10 | `B6KI4PYN...` | 10 | 0x4 | 22 | variant 1: fused, padded, stream only (ran cleanly twice before) |
| 17:28:11 | `36WDGDHT...` | 40 | 0x10 | 27 | **variant 2: `_F0`, non-fused (cooperative) TDM + padded LDS, stream only** |

No third binary exists, so variants 3-5 (S256 / S512) never launched. The hang is variant 2:
`TILE 128, NSEG 128, NBUF 4, PAD 1, FUSED 0, ABL 16`.

## What was checked (all clean, so not the cause)

- **Global extents at 340K:** seq 340008, 21251 live pages, pool 21315 pages, descriptor 341040 rows x 256 cols
  (i32 OK). Page indices are clamped to `n_log - 1` = 21250 < pool. Max row 341039 < 341040 and columns
  0..255, so no out-of-bounds row or column for any program, including the clamped tail re-issues and the last
  partial tile (`dry.py: static_check` replays this for every program).
- **Alignment:** global row addresses are 64-B multiples. LDS: K ring 20464 B (padded: 16 B per 64 B row, none
  after the last row), V ring at 20464, Q at 40928, total 50128 B, matching the compiled `shared`. No overlap,
  and well under the LDS limit.
- **Wait counts:** non-fused issues 8 TDM instructions per wave per unit and waits for <= 16 outstanding with
  24 in flight; fused issues 2 per unit and waits for <= 4 with 6 in flight. `s_wait_tensorcnt N` means
  "until <= N outstanding" and is never more than was issued, so it cannot deadlock unless an op never
  completes.
- **Barriers:** one per unit, at uniform trip counts; the early `return` for dead segments precedes every barrier.
- **Grids:** 512 / 1024 / 2048 programs (S128 / S256 / S512 x 4 kv heads); live segments 127 / 242 / 483. No
  limit issue, and those variants never ran anyway.

## Cause

Variant 2 is the only configuration that combines the cooperative TDM form (each `[16, 64]` page block split
across the 4 waves, 4 rows each) with the padded LDS layout. Its descriptor (group1 dword0 `0x06D00000`: pad
enable, interval 64 B, amount 16 B, tile 4 rows, stride 256 B) issues a 4-row padded tile at LDS offsets 320 x
wave. The configurations that worked were fused + padded (16-row pages at 1280 B steps, ~10 jobs) and
cooperative + unpadded (debug, 4K). The cooperative + padded one had never executed. Its IR is
self-consistent, so the hardware rejects or never completes this tile form.

That TDM op never completed. Every wave then waits in `s_wait_tensorcnt 0x10` forever. At the timeout the
killed process's queue could not be removed (MES `REMOVE_QUEUE` failed), which wedged the device. Variant 2
was also a pure stream (no compute between issues), so 8 outstanding padded partial-tile ops per wave were in
flight at all times.

## Fix (in the tree, not run)

- `kern/attn3.py`: the cooperative (non-fused) TDM path and the wide-block probe are removed. Only whole-page
  fused loads remain (one 16-row block per wave per instruction, the measured path). Static asserts require 4
  warps, NBUF 2..8 and TILE % 64 == 0. Its ISA is identical to the patch kernel
  (`patch/b/.../muse_attn3.py`).
- `bench.py`: refuses `_F` / `_X` variants. It also refuses any `g3_*` variant without a passing dry run, at
  1K, 4K and 340K, for the current kernel source hash.
- `dry.py` + `dryrun.sh`: the bounds-checking dry run, one variant and one context per `k2safe -t 60` call:
  - a host replay of every program's TDM rows, columns and block-table indices against the real buffers;
  - LDS size and wait-vs-issued checks;
  - then one launch with sentinel partials (all live partials written and finite, max error <= 1.5 x
    served's).
  Any failure, timeout or new MES error stops GPU work.
