# MI455X C1 work after iteration 12 (sources and write-ups)

Source code and write-ups only (logs, results, caches and ISA dumps stay on the host under `/data/muse455`). Served
SGLang changes are on branch `mi455x/muse-rocm-port` (kept iterations 13 reverted, 14 kept: P90 490.2 tok/s/user,
7,089 tok/s/chip); the integrated bundle branch is `mi455x/c1-bundle1`, the dense/draft branch `mi455x/draft2`.

| dir | what | status |
|---|---|---|
| `k2v2_land1/` | K2 v2, the served MoE mono kernel tree (iteration 12) | served |
| `k2v4b/` | K2 v4 with its serving give-up fixed (down-split clamp at nu = 63 + build-time `check_down_split`); 1.05-1.06x over k2v2_land1, bit-identical, 0 give-ups in 640,000 serving-like calls (`stress/`) | candidate |
| `k2v4c/` | k2v4b + touch-then-prefetch of the next layer's QKV weight | in measurement |
| `k2v6/` | K2 redesign round (replicated pool, local finalize, VMM peer pages): no gain; `DESIGN.md` has the gap analysis | knobs off |
| `k2v2pf/` | the prefetch study (`RESULTS_PF.md`: per-XCD cache locality, touch-then-prefetch) | merged into k2v4c |
| `k2v2_poll/` | slow-replay root cause: NUMA balancing queue evictions (`SPIKES.md`), `s3/mpol_exec.py` wrapper | done |
| `vagm/` | QKV TDM split-K + split epilogue and verify-attention config study (`RESULT.md`) | QKV + sliding config served (iteration 14) |
| `attn2/` | full-attention split kernel v2 (`RESULT.md`) | served (iteration 14) |
| `attn3/` | FlyDSL full-attention kernel (-3.2 us at 340K vs attn2; `muse_attn3.patch`, `HANG_ANALYSIS.md`, mandatory `dryrun.sh`) | candidate |
| `draft2/` | dense-path fusions (bitwise, ~255-310 us per step) + draft fc shard (`draft2-dense-gfx1250.patch`, `RESULT.md`) | candidate |
| `integ/` | bundle integration (`REPORT.md`, `bundle1.patch`, `bundle1.env`) | iteration 14 |
| `pdltest/` | PDL-style early launch on gfx1250 (`RESULT.md`, `bench/RESULT.md`): works via `hipStreamWaitValue32`, no-go behind K2 | done |

Host notes: `../tools/k2safe.sh` (per-GPU locks, refuses on an SGLang server, MES errors, or recent `SMU: No response`),
`../tools/99-numa-balancing.conf` (NUMA balancing off: its queue evictions hung the GPU driver on 2026-11-05).
The measured results and rules are also in Hyperloom: `packs/muse-spark-open-c1-mi455x/knowledge/microbench-2026-11-04.md`
and `.claude/skills/decode-cu-utilization/SKILL.md` (branch `mi455x-campaign` of JohnQin-llm/Arbor).
