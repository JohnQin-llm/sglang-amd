#!/usr/bin/env bash
set -euo pipefail
bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$bench_dir/../.." && pwd)"
source "$bench_dir/best.env"
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
exec python -m sglang.launch_server \
  --model-path \
  "${MODEL_PATH:?Set MODEL_PATH}" \
  --tensor-parallel-size \
  4 \
  --expert-parallel-size \
  4 \
  --served-model-name \
  amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 \
  --attention-backend \
  aiter \
  --page-size \
  16 \
  --kv-cache-dtype \
  fp8_e4m3 \
  --mem-fraction-static \
  0.85 \
  --max-running-requests \
  128 \
  --speculative-algorithm \
  DFLASH \
  --speculative-draft-model-path \
  "${DRAFT_MODEL_PATH:?Set DRAFT_MODEL_PATH}" \
  --speculative-draft-attention-backend \
  triton \
  --speculative-num-draft-tokens \
  8 \
  --reasoning-parser \
  muse \
  --tool-call-parser \
  muse \
  --watchdog-timeout \
  1800 \
  --enable-metrics \
  --prefill-decode-interval \
  20 \
  --scheduler-recv-interval \
  20 \
  --tokenizer-worker-num \
  4 \
  --swa-full-tokens-ratio \
  0.2 \
  --speculative-draft-window-size \
  2048 \
  --random-seed \
  42 \
  --enable-hierarchical-cache \
  --hicache-size \
  253 \
  --hicache-write-policy \
  write_through \
  --hicache-io-backend \
  kernel \
  --hicache-mem-layout \
  page_first \
  --host \
  127.0.0.1 \
  --port \
  "${PORT:-31101}"
