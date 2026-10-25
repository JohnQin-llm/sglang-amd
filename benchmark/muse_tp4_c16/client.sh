#!/usr/bin/env bash
set -euo pipefail
exec agentx profile \
  --url "http://127.0.0.1:${PORT:-31101}" \
  --endpoint /v1/chat/completions \
  --model amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 \
  --tokenizer amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 \
  --tokenizer-trust-remote-code \
  --tokenizer-revision c7bc74fd64371315a091bc15f9723882eead0dc1 \
  --concurrency 16 --benchmark-duration 180 --warmup-requests-per-lane 10 \
  --warmup-grace-period 1800 --record-processors 1 \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --server-metrics "http://127.0.0.1:${PORT:-31101}/metrics" \
  --output-artifact-dir "${OUTPUT_DIR:?Set a fresh OUTPUT_DIR}" \
  --unsafe-override
