#!/usr/bin/env bash
set -euo pipefail
duration="${DURATION:-180}"
concurrency="${CONCURRENCY:-16}"
if [[ ! "$concurrency" =~ ^([2-9]|[1-5][0-9]|6[0-4])$ ]]; then
  echo 'CONCURRENCY must be an integer from 2 through 64.' >&2
  exit 2
fi
extra_args=()
case "$duration" in
  180) extra_args+=(--unsafe-override) ;;
  3600) ;;
  *) echo 'DURATION must be 180 (diagnostic) or 3600 (hour protocol).' >&2; exit 2 ;;
esac
exec agentx profile \
  --url "http://127.0.0.1:${PORT:-31101}" \
  --endpoint /v1/chat/completions \
  --model amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 \
  --tokenizer amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 \
  --tokenizer-trust-remote-code \
  --tokenizer-revision c7bc74fd64371315a091bc15f9723882eead0dc1 \
  --concurrency "$concurrency" --benchmark-duration "$duration" --warmup-requests-per-lane 10 \
  --warmup-grace-period 1800 --record-processors 1 \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --server-metrics "http://127.0.0.1:${PORT:-31101}/metrics" \
  --output-artifact-dir "${OUTPUT_DIR:?Set a fresh OUTPUT_DIR}" \
  "${extra_args[@]}"
