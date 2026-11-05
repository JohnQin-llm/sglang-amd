#!/bin/bash
# k2v6 per-layer span A/B (microbenchmark: ab_spike.py, graph of 20 chained K2 calls, warm = 1 weight copy, cold = 8
# rotated copies, layers 1/31/60, slowest rank's p50 per call). Arms: k2v4b, k2v6 defaults, k2v6 + one knob.
# Usage: ab_run6.sh <trial> [iters]   (odd trial: arm order as listed; even: reversed). No NUMA wrapper (balancing off).
set -uo pipefail
TR=$1
IT=${2:-300}
K2SAFE=/data/muse455/k2/k2safe.sh
V6=/data/muse455/k2/cand/k2v6
V4B=/data/muse455/k2/cand/k2v4b
LOG=$V6/logs/ab
mkdir -p $LOG
run() {  # name tree env...
  local name=$1 tree=$2; shift 2
  local envs=(-e PYTHONDONTWRITEBYTECODE=1)
  for kv in "$@"; do envs+=(-e "$kv"); done
  $K2SAFE -t 1500 "${envs[@]}" $V6 s3/ab_spike.py --tree $tree --tag ab_${name}_t$TR \
    --layers 1,31,60 --modes warm,cold --loops sync --iters $IT --warmup 20 > $LOG/ab_${name}_t$TR.log 2>&1
  echo "[ab_run6] $name t$TR rc=$?"
}
ARMS=("v4b $V4B" "v6 $V6" "v6repm $V6 K2_POOL_REP=1 K2_POOL_REP_MERGE=1" "v6vmm $V6 K2_PEER_VMM=1" "v6fl $V6 K2_FIN_LOCAL=1")
if [ $((TR % 2)) -eq 0 ]; then ARMS=("${ARMS[4]}" "${ARMS[3]}" "${ARMS[2]}" "${ARMS[1]}" "${ARMS[0]}"); fi
for a in "${ARMS[@]}"; do run $a; done
