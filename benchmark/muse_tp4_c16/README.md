# Muse TP4/EP4 C16 measured preset

This package preserves the best measured 180-second C16 candidate: **149.925 tok/s/user P90 interactivity**, **5,877.937 total tok/s/chip** on four MI355X GPUs. A same-source 3600-second follow-up produced **90.580 tok/s/user** and **19,555.917 total tok/s/chip** from its completed requests. It cancelled one request at the original 30-second drain deadline, so it fails the required zero-cancellation hour and the 150 tok/s/user target. The short result did not hold over the longer workload.

The PR is an aggregate against `optimize/muse-official1020-base` (`93406f598d10`). It includes the shared-layout work from #9, the KV-index integration associated with #10, and the five-line sync-free sequence-length change from #14. Those PRs are overlapping components, not extra patches to apply on top of this branch. The measured preset adds native-order AITER MoE, bounded attention launch tuning, QK/cache fusion, and the two prefill optimizations below.

## What is enabled

| Path | Selected implementation and boundary | Other inputs |
| --- | --- | --- |
| Expert weights | Single resident MXFP4 layout for Muse; original weight export restrictions retained | Existing layout unless explicitly enabled |
| A16 expert G1/G2 | TP4/EP4 gfx950, top-k 8, 64 local experts, hidden/intermediate 4096, contiguous BF16 activations and packed FP4 weights; native K32 math, sorted G1 to slot G2; G1 N64 for M <= 48, N128 otherwise; admitted M 1..512 | Original MoE path if the guard rejects; admission is not full numerical qualification |
| Full-attention verify | Existing native arithmetic; one warp at B8, two warps at B10/12/14/16 under the exact Q32/KV4/D64 FP8-cache ABI and TP4/EP4 gfx950 target-model guard | Original launch configuration |
| SWA verify | Two warps at B10/12/14/16; pinned AITER wrapper uses stage 2 at B1..8 under its Q32/KV4/D64/window-2047 guard | Original launch configuration |
| QK norm/RoPE/cache write | TP4 Q32/KV4, up to 128 rows, original fused reduction order | Existing unfused path |
| Ulysses QKV projection/pack | Default-off `SGLANG_MUSE_TP4_PREFILL_QKV_LAYOUT`; TP4/EP4 gfx950, local M4096/K8192/N18432, contiguous BF16; writes directly to rank-major `[4,4096,4608]` | Original `F.linear` and packing, including other prefill lengths |
| Full-attention prefill | Default-off `SGLANG_MUSE_TP4_PREFILL_MAXSCALE`; Q4096..16384, H32/KV4/D64, FP8 Q/K/V, page16; native-order max-before-scale with pinned AITER device source and Triton 3.7.0 | Original unified attention |

The MoE pair retains the original sorting producer, routing weights, post-normalization and collectives. Its workspace is per call; both stages use the current device stream. There is no global GPU tensor cache. The QKV path removes the transpose-contiguous copy; it does not change K accumulation order. The prefill attention path preserves the native contraction pattern and KV accumulation order; its compiler/source guards are intentional.

`best.env` is the selected preset. Standalone AITER G1, TP4 fused projection/cache, compact A4 verify, full-v2/split-KV experiments and high-batch router/pooler experiments remain disabled. Their pre-existing opt-in definitions are retained to keep the measured module bytes unchanged. Later direct-sort, BF16-prefill, router-cast and draft-append experiments are not included. A4W4 prefill, sequence parallelism and Ulysses remain enabled.

Serving batch sizes outside these optimized boundaries keep their existing dispatch paths. This PR does **not** claim measured performance or numerical qualification for every concurrency from 2 through 64.

## Existing evidence

| Metric | Previous N64 short run | Selected QKV + maxscale short run |
| --- | ---: | ---: |
| P90 interactivity, tok/s/user | 149.254 | 149.925 |
| Total throughput, tok/s/chip | 5,745.044 | 5,877.937 |
| Input throughput, tok/s/chip | — | 5,766.317 |
| Output throughput, tok/s/chip | — | 111.620 |

These are separate runs, not a controlled A/B experiment. The observed changes are +0.45% interactivity and +2.31% total throughput. Component measurements were QKV projection plus packing 1052.211 -> 966.795 us (8.23% paired reduction) and full-attention Q16384/prefix69856 9774.185 -> 9157.307 us (6.29% paired reduction), both byte-exact in the recorded cases. These percentages are component results, not end-to-end speedup claims.

The selected AgentX run completed 62 profiling requests with zero cancellations and zero request errors. The 180-second sending window retained 204.169 seconds of records after drain; throughput uses that retained span, divided by four chips. The official-style P90 aggregation uses full-response ITL and `1 / round(P90_seconds, 5)`; raw P90 ITL was 6.667382 ms. Warmup discarded 177 records. Request, record, source identity, execution and cleanup checks passed. This was a diagnostic run with `--unsafe-override`, not an official InferenceX submission.

Full-model evidence covers B1 and B6 at prefix32768: all 56 captured logit rows byte-exact, per-row relative L2 zero, identical top-1 and identical 128-token natural-acceptance continuations. Kernel witnesses on all four ranks recorded 1,736 QKV and 448 maxscale calls per rank. The baseline outputs were reused from a previously completed baseline arm; the baseline GPU arm was not rerun. This does not establish all-shape correctness or long-duration stability.

### 3600-second follow-up

| Metric | MI355X completed requests, unqualified run | Official B200 C16 TP4/EP4 |
| --- | ---: | ---: |
| P90 interactivity, tok/s/user | 90.580 | 151.286 |
| Input throughput, tok/s/chip | 19,413.642 | 23,986.816 |
| Output throughput, tok/s/chip | 142.275 | 160.905 |
| Total throughput, tok/s/chip | 19,555.917 | 24,147.721 |
| Completed / cancelled / errored requests | 2,159 / 1 / 0 | 2,383 / 0 / 0 |

The MI355X run sent requests for 3600.001 seconds without `--unsafe-override`. All 2,159 successful records exported normally; the retained successful-record span is 3619.893 seconds. Raw P90 full-response ITL is 11.038976 ms. The client exited normally, source identity and cleanup checks passed, and the original cancellation ledger retained the one cancelled request separately. It was not converted into a successful record. Thus these numbers describe completed requests from an **unqualified** hour, not a valid target result.

Against [the B200 source artifact](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39449248229/artifacts/12284243734), observed interactivity is 40.13% lower and total throughput is 19.02% lower. This comparison uses different hardware, FP4 implementations, backends and engine recipes; it is not an isolated hardware or kernel A/B. The unchanged MI355X recipe and zero-cancellation requirement remain the optimization target. C4/C8 results and the separate heavy-load profile are pending.

`results.json` contains the short-run measurements; `hour-results.json` records the follow-up failure and completed-record metrics. Both include original evidence hashes. `source-binding.json` binds all 30 production module files to the measured overlay. The PR packages content-equivalent source; the new Git commit itself has not been benchmarked. The AITER wrapper patch is part of the measured stack and must not be omitted. Raw workload data and internal machine paths are not published here.

## Reproduce the selected configuration

Use `lmsysorg/sglang-rocm:muse-spark-open-1020`, four MI355X GPUs, and the pinned dependency hashes in `source-binding.json`. `runtime.json` records the image identity, AgentX 1.0.12 wheel, dataset/tokenizer revisions and serving configuration. Client dependencies were reused from the local harness environment; they were not established to be byte-identical to the official client environment.

Inside a disposable container with this checkout and the model checkpoints mounted:

```bash
python benchmark/muse_tp4_c16/check_sources.py
python benchmark/muse_tp4_c16/check_sources.py --aiter-root /sgl-workspace/aiter --apply-aiter-patch
export MODEL_PATH=/models/Muse-Spark-Open-816B-A42B-Quark-MXFP4
export DRAFT_MODEL_PATH=/models/Muse-Spark-Open-DFlash2
bash benchmark/muse_tp4_c16/serve.sh
```

The AITER patch installer refuses unknown dependency or wrapper bytes. It is idempotent on the measured wrapper. The launcher uses TP4/EP4, HiCache 253 decimal GB per rank for the target model (draft host cache is additional), SWA/full ratio 0.2, draft window 2048, eight draft tokens, prefill/decode and scheduler receive intervals 20, and seed42. Keep the image's other launch defaults, including its decode graph buckets. Check GPU availability before starting a separate run.

For performance comparison the preset simulates acceptance length 4.05 with `match-expected` and `real-draft-token`. This matches the recorded diagnostic setup. Quality comparisons used natural acceptance. Do not treat simulated acceptance as model accuracy evidence.

With AgentX 1.0.12 and the pinned trace/tokenizer available, run the client separately:

```bash
export OUTPUT_DIR=/results/muse-tp4-c16-fresh-run
bash benchmark/muse_tp4_c16/client.sh
```

The client defaults to the 180-second diagnostic. Set `DURATION=3600` for the hour protocol; this removes the diagnostic unsafe override and retains the original 30-second drain behavior.

There is no new GPU run required to inspect or reproduce the source binding. The existing CPU guard/layout tests are under `test/registered/kernels/ops/muse_spark/`. Repository CI and broader numerical coverage remain separate from the saved two-case evidence.
