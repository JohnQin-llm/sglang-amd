# PDL-style early launch on MI455X (gfx1250): release latency, graph settings, co-residency (2026-11-05 04:37-05:45 UTC)

GPU 0 only, every run through `k2safe.sh` (all runs: rc=0, GPUs drained, no new MES errors). Sources in this
directory: `lat.hip` (release / waitcost / chain / 62-layer graph), `cores.hip` (co-residency), `k2shape.hip` +
`real_qkv.py` (real Triton `_dense_partial` behind a K2-shaped kernel), `tri_meta.py`; drivers `drv_*.py`; logs
`*.log`. Served trees untouched (the QKV kernel is imported read-only).

## Device facts
256 CUs, 320 KB LDS/CU, 131072 regs/CU = 1024 VGPRs per SIMD lane (wave32, 4 SIMDs). Wall clock 100 MHz.
Per-XCD wall clocks are offset (stable within a process): relative to XCD of block 0, block classes b%8 =
+0, +4.35, +8.45, +3934.9, +3939.1, +3943.5, +3947.5, -4.36 us. No XCC_ID hwreg on gfx1250 (HW_ID1/2 only).
Real QKV partials (`_dense_partial`, TP4 M=8: grid 72x1x8 = 576 CTAs x 128 threads): **104 VGPRs, 43.5 KB LDS**,
0 spills; 9.56 us/call cold (62 distinct weights), 4.80 us warm (partials only, no epilogue).

## 1. Release latency (flag write -> consumer's first instruction)
| path | latency | notes |
|---|---:|---|
| eager, sys-release store | 1.44 us | progress counter; offset-corrected clock agrees (1.46) |
| eager, agent-release / relaxed to uncached | 1.40 / 1.32 us | store kind does not matter |
| graph, CLASSIC_PATH=1 | 1.40 us | 0/10 replays started before the write |
| graph, SEG_SCHEDULING=1 + GRAPH_QUEUES=2 | 1.12-1.28 us | 0/10 early |
| graph, default, 1-layer graph | serialized | consumer starts at producer end |
| graph, default, 62-layer graph | **early in 62/62 layers** | default scheduling is topology-dependent |
Earlier `pdl_test` graph runs could not tell "wait honored" from "flag stale": the flag must be reset inside the
graph (before the fork), else the wait passes on the previous replay's value.

Stream-wait cost (satisfied `hipStreamWaitValue32` before each kernel of a chain): graph default/seg +1.47 us
(2.70 vs 1.23 us/kernel), graph classic +2.8 us, eager +2.1-2.6 us.

## 2. Graph-setting cost
| | default | CLASSIC=1 | SEG=1+Q=2 |
|---|---:|---:|---:|
| empty-kernel chain, us/kernel (1-1024 CTAs) | 1.23 | 2.24-2.43 | 1.23 |
| ~1.7 us ALU kernels, us/kernel (64-256 CTAs) | 2.97-3.08 | 3.31-3.39 | 2.97-3.10 |
| 62 layers x 10 small kernels, us/layer | 30.30 | 33.10 (+2.8) | 30.29 |
| 62 layers x (10 small + 44 us persistent + QKV-shaped), linear, us/layer | 112.1 | 112.0-112.5 | 112.1-112.2 |
| 62 layers x (reset + K2-shaped + real Triton QKV), linear, us/layer | 57.7 | 57.7 | 57.7 |
CLASSIC costs ~1 us per launch on small kernels (+174 us per 62-layer step); SEG+Q2 costs nothing in a linear
graph. The real K2 decode graph was not captured (4-rank harness; out of time box).

## 3. Co-residency (`cores.hip`; producer 256 x 512, 145 KB LDS, all 16 waves alive; consumer = real QKV footprint)
| producer VGPRs | consumer CTAs run while all producer CTAs alive | max concurrent | producer slowdown |
|---:|---:|---:|---:|
| 251 / 240 / 232 | **0 %** (0/2880), also for a 6-VGPR 1 KB consumer | - (only fill CUs as producer CTAs retire) | - |
| 224 / 216 / 200 | 100 % (2880/2880) | 256 (1 CTA/CU, VGPR-bound) | 0 (ALU/LDS), +0.05 % with a 72 MB streaming consumer |
K2 at 251 VGPRs x 16 waves = 4 waves/SIMD x 256 = 1024: the whole VGPR file. A QKV CTA fits only if K2 <= 224.

**Deadlock hazard (observed).** If the consumer is released before every producer CTA is resident, it takes the
VGPRs/LDS the late producer CTAs need; the producer's all-CTA barrier then cannot complete (my K2 shapes carry a
1 s guard; real K2 would hang all four ranks). Seen as 1020-2295 guard timeouts per config when flags were reset
by plain stores at the consumer's end, and still 510-1275 with the 251-VGPR all-waves K2 shape even with atomic
resets ordered before the fork, in all three settings (root cause not isolated: stale trigger seen by the CP or
cross-queue ordering between replays). Resetting in a kernel before the fork costs ~1.6 us/layer by itself.

## 4. Per-layer effect (no K2 prototype built: 1-3 not viable)
Real Triton QKV behind a K2-shaped kernel (all waves alive), PDL arm = early launch WITHOUT waiting on K2's output
(an upper bound), per layer vs linear:
| K2 VGPRs | hand-off window | default | CLASSIC | SEG+Q2 |
|---:|---:|---:|---:|---:|
| 251 | 4 us | -5.1 | -5.3 | -5.4 |
| 251 | 12 us | -5.1 | -4.7 | -5.6 |
| 224 | 4 us | -3.0 | -3.0 | -3.7 |
| 224 | 12 us | -1.8 | -2.3 | -2.4 |
Only when K2's non-hand-off waves exit (freeing their VGPRs) does it pay: -0.7 us at a 4 us window, +7.4 us at
12 us, +8.3 us at 24 us (fixed cost ~4.6 us; break-even window ~4.6 us > K2's 3.5-3.9 us last hand-off).
In the emulated 62-layer graph the device-side wait on K2's output adds +1.9 us; prefetching the weights and
re-reading them adds +4 us (1/4) to +18 us (all).

## Verdict: NO-GO
