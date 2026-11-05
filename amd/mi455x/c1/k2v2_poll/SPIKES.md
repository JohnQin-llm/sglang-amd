# Slow K2 graph replays: NUMA balancing -> SVM MMU notifier -> KFD queue quiesce (2026-11-05 00:40-01:45 UTC)

Tree: `k2v2_poll` = copy of `k2v2_land1` with `port/` unchanged (byte-identical; no kernel change was needed).
New files only: `s3/spike.py` (harness), `s3/spike_an.py`, `s3/spike_mon.py`, `s3/trace/*.bt` + `s3/trace_an.py`
(KFD tracing), `s3/mpol_exec.py` (the fix), `s3/serve_ab.sh`, `s3/serve_client.py`, `s3/serve_an.py` (serving A/B).
Data: `s3/spike/<tag>_L<layer>_<warm|cold>.json` (every iteration, every rank, host stamps), `s3/serve/`, `logs/`.

## Cause
The served K2 is not the cause. The occasional 2-7 ms replay stalls (tail up to 0.1-0.5 s) are one rank's GPU queues
being **unmapped by KFD** while the other three ranks spin in K2's mailbox waits:

`task_numa_work` (automatic NUMA balancing, `kernel.numa_balancing=1`, runs on the rank's main thread at a timer
interrupt while it busy-waits in `rocr::core::BusyWaitSignal::WaitRelaxed`) -> `change_prot_numa` ->
`mmu_notifier_invalidate_range_start` -> `svm_range_cpu_invalidate_pagetables` (the process has ROCr SVM ranges)
-> `kgd2kfd_quiesce_mm` -> all of that process's user queues evicted -> restore worker (`kgd2kfd_resume_mm` on a
kworker) 2-6 ms later (p50 5.5 ms; tail 61 ms p90 / 416 ms max in the harness, when restores retry under
continued scanning).

Evidence (`s3/trace/trace3.txt`, `kfd_evict3.bt`; stack in `trace4.txt`):
* 276 of 276 slow replays (> 1 ms extra) overlap a quiesce -> resume window of one rank; 291 of 298 windows inside
  the timed loops coincide with a slow replay; window length = stall length (76.1 ms window / 76.2 ms replay, ...).
* Queued loop (1000 graph replays enqueued in 9-10 ms, GPU needs 73-80 ms, so host launch timing cannot matter):
  the stalls persist; the origin rank's call is +X ms, the others absorb X in the next call; the origin moves
  over all four GPUs.
* A/B with the ranks opted out of NUMA balancing (task policy `MPOL_LOCAL`, no `MPOL_F_NUMA_BALANCING`, so
  `task_numa_work` skips their VMAs): 0 quiesce windows, 0 calls > 150 us in 6 x 20,000 queued calls a rank.
* NUMA balancing ran 60x harder during the harness (153 k PTE updates/s vs 2.6 k/s idle; 0.9 M pages migrated
  in 2 min): the harness ranks are unpinned, their memory keeps migrating, the scan period stays at its minimum.
* KFD userptr eviction (`amdgpu_hmm_invalidate_hsa`), TTM eviction tracepoints: none during the loops.

Ruled out: (b) poll back-off (s_sleep(32) = ~1 us granularity, 10 s deadline never hit; stalls vanish with no kernel
change), (c) clocks (not sampled during the runs; ruled out because every stall is a quiesce window and none remain
with MPOL_LOCAL), (d) gpuagent / alloy (no
residual stalls once NUMA scanning is off), (e) MES / firmware on its own: MES only executes the unmap / map that
KFD requests.

Secondary, host side, harness only: with wide CPU affinity (unpinned, or SGLang's 48-CPU sets) ~0.5 % of gloo
barriers release one rank 0.7-7 ms late (all ranks enter together, `nivcsw` 0: a late wake-up). This hits the
1-call loop (`sync1`, a barrier every ~80 us), rarely the 20-call chain. Pinning each rank to 4 cores removes it
(GPU-local or not). It was not visible in serving (max step 8.9 ms with the fix).

## Distribution (served K2 v2, per layer call; chain = layer_ab shape: barrier + graph of 20 calls, us = replay / 20)
Baseline (as the harnesses ran: unpinned, NUMA balancing on), 3000 replays x 20 calls a rank, tag `base`:

| case | rank | iters | p50 | p99 | p99.9 | max | > 150 us | > 1 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 warm | 0/1/2/3 | 3000 | 65.5/65.6/65.6/65.4 | 304/282/283/279 | 2628/2628/2506/2628 | 7044/7044/2721/6790 | 48/43/45/41 | 6/6/5/6 |
| L1 cold | 0/1/2/3 | 3000 | 68.2/68.9/68.7/68.9 | 276/276/277/266 | 397/528/397/399 | 3550/3551/3551/3546 | 50/50/54/46 | 2/2/2/2 |
| L31 warm | 0/1/2/3 | 3000 | 65.0/65.3/64.9/65.1 | 70/214/70/72 | 2638/2638/2638/1855 | 6654/6656/6653/6656 | 25/30/26/27 | 8/8/6/7 |
| L31 cold | 0/1/2/3 | 3000 | 68.5/68.3/68.5/68.4 | 408/606/376/597 | 5142/5142/4914/5142 | 6148/6148/6148/6148 | 55/70/61/69 | 21/24/16/24 |
| L60 warm | 0/1/2/3 | 3000 | 64.8/65.0/64.8/64.7 | 70/71/75/72 | 537/3045/3045/3045 | 10461/10462/10461/10461 | 21/25/29/26 | 2/3/3/3 |
| L60 cold | 0/1/2/3 | 3000 | 68.1/68.0/68.1/67.9 | 316/323/264/326 | 2308/2307/1927/2308 | 4576/4574/2306/4574 | 41/43/37/45 | 11/12/7/12 |

A chain value > 150 us = a replay stalled >= ~1.7 ms; > 1 ms = >= ~19 ms. Per call (queued loop, 19,980 calls a rank,
first call of each chunk excluded): p50 72.7-77.0 us, 5-37 calls > 1 ms per case, max 5.2-141.8 ms, identical counts
on all ranks (one stall, four victims). Full tables: `logs/table_base.md` (also `sync1`, `queued`), `logs/table_*.md`.

Fix (`--mpol local` + each rank pinned to 4 GPU-local cores, tag `pin`), same loops:

| loop | cases | iters a rank | p50 | p99.9 (worst case) | max (worst case) | > 150 us (all ranks) | > 1 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| chain | L1/31/60 warm+cold | 3000 x 6 | 64.0-68.3 | 70.4 | 76 | 0 | 0 |
| queued | L1/31/60 warm+cold | 19,980 x 6 | 72.3-76.9 | 84.0 | 96 | 0 | 0 |
| 1-call | L1/31/60 warm+cold | 3000 x 6 | 76.8-78.4 | 101.1 | 3344 | 4 | 2 |

`MPOL_LOCAL` alone (tag `mpol`, ranks unpinned): chain 1 iteration > 150 us (232 us, host skew) in 6 x 3000;
queued 0 in 6 x 19,980; 1-call loop still 4-25 per 3000 (the host-skew mechanism). Pinning alone (`pinonly`, L31):
cold still 3 queued calls up to 11 ms (fewer scans, not none). SGLang's affinity alone (`sgaff`, L31): both kinds.

## Serving (C1 env / argv, K2 v2 land1 on layers 1-60, single stream, 180 s decode, 22,700 steps, 4.05 tok/step)

| arm | p50 | p90 | p99 | p99.9 | max | steps > 12 ms | quiesce windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| as served | 7.75 ms | 7.89 | 7.97 | 8.35 | 13.0 | 6 | 9 in 181 s, all TP0, 2.9-4.4 ms |
| `mpol_exec.py` (MPOL_LOCAL) | 7.76 ms | 7.90 | 7.98 | 8.04 | 8.9 | 0 | 0 |

Every step > 11 ms in the served arm overlaps a TP0 quiesce window (step = 7.75 + window). Serving exposure today:
~0.05 % of steps (one +3-5 ms hit every ~20 s); P90 interactivity is unaffected, p99.9 by ~0.3 ms. The serving
schedulers are pinned (`SGLANG_SET_CPU_AFFINITY=1`), so their memory stays put and NUMA balancing backs its scan
period off; the harnesses (unpinned) see ~50x the rate (2.4 windows/s a rank, 20 % of wall time stalled).
SGLang's own NUMA bind does not engage here: `numactl` is missing in the container (`SGLANG_NUMA_BIND_V2` path
skipped) and the CPU affinity maps GPU 0/1 (NUMA node 1) to node-0 cores and GPU 2/3 to node-1 cores.

## Fix / recommendations
* Harnesses (no code change): `k2safe.sh -w 'python3 /data/muse455/k2/cand/k2v2_poll/s3/mpol_exec.py' ...`
  (verified: all ranks `local`, no GPU-side stalls); for 1-call / tight-barrier loops also pin each rank to a few
  cores. Medians move by ~1 us at most (slightly lower with the fix: chain 64.0-68.3 vs 64.8-68.9 us).
* Serving: host `sysctl kernel.numa_balancing=0` (AMD guidance, also aiter's warning) removes the mechanism for every
  process; without a host change, prefix the launch with `python3 .../mpol_exec.py` (measured above: 0 windows,
  max step 13.0 -> 8.9 ms, mean +0.01 ms, within noise) or install `numactl` in the image and launch under
  `numactl --localalloc`. Optional: GPU-local CPU affinity (GPU 0/1 -> CPUs 48-95, 144-191; GPU 2/3 -> 0-47, 96-143).
* Nothing to change in K2 (poll constants, deadlines, launch order): a quiesced queue cannot be helped from inside
  the kernel, and every TP collective would absorb the stall the same way.
