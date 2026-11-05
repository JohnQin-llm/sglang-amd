# Draft2 + dense non-MoE path, Muse Spark Open C1, 4x MI455X (gfx1250, TP4)

Patch: `/data/muse455/draft2/draft2-dense-gfx1250.patch` (against served HEAD 9107618f1d, `git apply --check` passes).
Worktree `/data/muse455/draft2/sglang`, branch `mi455x/draft2`. All flags default off; each falls back to the served
path when its preconditions fail. Benchmarks `/data/muse455/draft2/bench`, logs and JSON `/data/muse455/draft2/logs`.
No server was started; the step baseline is the it9 trace (`prof/it9_tp0.{verify,gap}.txt`, `summary.json`).

## Flag set (all bitwise on the microbenchmarks)

    SGLANG_MUSE_VERIFY_SEQK_ONCE=1
    SGLANG_MUSE_REDUCE_GATE_NORM=1
    SGLANG_MUSE_NORM_RESIDUAL_HOIST=1          # 16 warps (SGLANG_MUSE_NORM_RESIDUAL_HOIST_WARPS)
    SGLANG_MUSE_DFLASH_FC_TP_SHARD=1           # optional: small net gain, frees ~0.5 GB/GPU

Log lines to confirm in serving: `Muse verify attention: attention output norm + gate in the split-KV reduce live`,
`DFlash draft: fc sharded by output rows across TP4 (2048 rows per rank)`.

## Implemented (graph replay, one GPU, alternating trials, us per call)

| Change | Served | New | Per step | Numerics |
|---|---|---|---|---|
| `seqused_k = seq_lens + max_q_len` once per forward instead of per attention layer (aiter_backend) | 1.30 us x 62 (it9 trace 1.64 avg, 101.8 us/step) | 1 add | -80 to -100 us | identical values |
| Gated norm (weightless per-head RMS norm + sigmoid gate) as the epilogue of the split-KV reduce (both the full-attention v2 and the sliding-window path) | sliding cold 8.82 / 9.19 / 9.18 (128K / 32K / 340K); full cold 27.66 / 15.96 / 46.53 | sliding 7.11 / 7.59 / 7.69; full 26.41 / 14.27 / 44.28 | 46 x ~1.6 + 16 x ~1.3-2.3 = -95 to -110 us | 0 bytes differing (3 input sets x 6 cases) |
| FFN boundary (`_fused_norm_residual_triton_kernel`) with every load issued before the first reduction, 16 warps | cold (1024 rotating sets, reproduces the 4.7-5.6 us served) 4.74 / 4.87; warm 2.74 / 2.87 | cold 3.47 / 3.28; warm 2.32 / 2.27 | -80 to -100 us (62 calls) | 0 bytes differing (out and new_stream; both post-norm variants) |
| Draft `fc` [8192, 40960] sharded by output rows + all-gather | 41.0-42.2 us (M 4/8, warm and cold) | shard GEMM 24.8 warm / 27.3 cold + all-gather (~12 us, not measured at this shape; it9 custom AG 12-16 us) | ~-3 us net; frees ~0.5 GB/GPU | shard columns 0 bytes differing at decode widths (<= 16 rows); prefill rows go through hipBLASLt, whose tile may change with N |

Dense total from the three dense flags: about -255 to -310 us per verify step (target was ~-400).
Draft: only the fc shard was implemented; a split-K kernel for the shard (not bitwise) was queued
(`logs/fcsweep.log`, `bench_draft.py fcsweep`) to see whether it brings the shard to ~12-15 us (net ~-15 us).

## Ranked remaining items (estimates; not implemented)

| Item | Estimate per step | Numerics | Notes |
|---|---|---|---|
| Draft pre-step metadata (~22 small kernels, ~55 us span: two flashinfer kv-index builds, cumsums, fills) | -20 to -30 us | bitwise (integer) | one Triton metadata kernel |
| Draft fused AR + residual + RMSNorm (10 per step) | -20 to -30 us | bitwise only if the AR keeps its summation order | medium-high effort |
| Draft AR staging copies (10 x 1.7 us + gap) | -20 to -25 us | bitwise | risky: stale IPC reads seen on MI355X with registered buffers |
| Verify start: embed_norm + fp32 cast + unfused first input norm (~9 torch kernels, ~25 us) | -20 us | not bitwise (torch mean order) | one Triton kernel |
| Draft attention split-KV (partials + reduce 10.5 us/layer, 1 MB window) | -15 to -25 us | likely not bitwise | latency-bound; segment / tile sweep |
| Draft next-step embedding AR (25 us in the VERIFY->DRAFT gap, mostly rank skew) | -15 to -20 us | bitwise | needs a replicated embedding (+2.5 GB/GPU) |
| Draft silu_and_mul into gate_up epilogue | -15 us | not bitwise | |
| Draft small-M GEMMs (conv projection x10, QKV x5, o_proj x5: ~127 us vs 42 us floor) | -30 to -50 us | depends | already tuned per shape; 33-42 MB GEMMs ramp-limited at 5-6 TB/s, needs prefetch / overlap |

## Notes

- Gate-norm fusion arms per attention call (`verify_attention.arm_gate_norm` / `take_gate_norm`): when the call takes
  no split-KV reduce (2D path, gluon path, shape mismatch) the model's own norm-gate kernel runs as before.
- The bitwise checks are on random inputs at the served shapes. An in-model check under real acceptance
  (and the AgentX gate) is still needed for each flag.
- GPU history: the 09:54 UTC run was refused by a `GPU_UNHEALTHY` marker from another job (K2 stress, MES error);
  every number above comes from runs after the 16:24 UTC reboot via k2safe (0 new MES errors).
