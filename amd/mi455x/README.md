# Muse Spark Open 816B on 4x MI455X (gfx1250): K2 mono kernel and tools

This branch is `mi455x/muse-rocm-port` (the served tree: the ROCm Muse Spark port plus every kept Hyperloom
iteration, one commit each) plus this directory, which holds the code the served tree loads from outside it and the
tools used to measure it.

## `k2/`: the K2 persistent MoE mono kernel (FlyDSL, gfx1250, TP4 / EP1)

One persistent kernel per MoE layer for the 8-row verify step: o_proj, both all-reduces, router, pooler, the routed
experts and the unpooler, with all four GPUs coordinating inside the kernel over XGMI. 80 us per layer on real
layer-31 weights (the served kernels it replaces take ~130-140 us).

- `port/`: the kernel package (`k2.py` build and launch, `moe/` expert phase, `gemv/` dense stages, `sync/`
  cross-GPU waits, `layout.py` gfx1250 weight layout, `serving.py` the operand adapter). `port/INTERFACE.md` is the
  ABI the served tree calls.
- `s0`-`s4/`: the build-up stages and their tests; `s3/run_k2.py` (per-layer timing and numerics),
  `s3/probe_b2b.py` (back-to-back launches, `--timeline` per-stage stamps), `s4/` serving integration.
- `docs/`: `K2_STATUS.md` (results and the switch defaults) and one write-up per optimization lane.

The served tree loads it with `SGLANG_MUSE_MONO_GFX1250=1 SGLANG_MUSE_MONO_PORT=<path to this k2 directory>`
(`python/sglang/srt/rocm_muse_spark/mono_gfx1250.py`). Scripts still carry the absolute paths of the machine they
ran on (`/data/muse455/k2/ep1`); set the port path accordingly.

Rules every GPU job followed: one GPU user at a time through `tools/k2safe.sh` (locks, timeout, drain and MES
checks); no clusters or multicast; waits use `s_sleep(32)` polls (`K2_POLL_SLEEP`).

## `tools/`

- `k2safe.sh`: run one GPU job safely (refuses when a server runs or the GPUs are busy, holds the four GPU locks,
  hard timeout, checks the GPUs drained and that no new MES errors appeared).
- `bench/`: AgentX and correctness validators, the context sweep, profiler drivers and trace analysis
  (`trace_breakdown.py`: GPU time per draft + verify cycle by kernel category, assigned by the step annotations).
- `decode_stack/`: microbenchmarks of the per-layer decode kernels at the served shapes (fused QKV projection on
  Triton and on AITER's gfx1250 TDM GEMM, the whole QKV op bit-compared, the fused norm launch sweep) with their
  JSON results.

## `aiter/`

`served-aiter-on-2bb549102.patch`: the AITER the served tree runs (`/sgl-workspace/aiter` in the image, upstream
`2bb549102`) carries the kept attention iterations as working-tree edits: the gfx1250 spec-decode verify backport
(AITER #6896 with #6336 / #6661 / #6581 / #6888) and the gfx1250 attention series (window-aware split-KV for the
sliding-window layers, device-side segment ranges for long context, tuned configs). Apply with
`git -C <aiter> apply served-aiter-on-2bb549102.patch` on that commit.
