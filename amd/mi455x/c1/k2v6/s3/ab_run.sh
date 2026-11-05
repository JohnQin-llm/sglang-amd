#!/bin/bash
# k2v4 vs k2v2_land1 per-layer span A/B: ab_spike.py (chain of 20 calls a graph, warm = one weight copy, cold = 8
# rotated copies), layers 1/31/60, 2 trials in alternating order, every arm through k2safe + the NUMA wrapper.
# Usage: ab_run.sh <trial> [iters]   (odd trial: land1 first; even: k2v4 first)
set -uo pipefail
TR=$1
IT=${2:-300}
K2SAFE=/data/muse455/k2/k2safe.sh
W='python3 /data/muse455/k2/cand/k2v2_poll/s3/mpol_exec.py'
V4=/data/muse455/k2/cand/k2v6
LAND1=/data/muse455/k2/cand/k2v2_land1
LOG=$V4/logs/ab
mkdir -p $LOG
run() {  # name tree
  $K2SAFE -t 1500 -e PYTHONDONTWRITEBYTECODE=1 -w "$W" $V4 s3/ab_spike.py --tree $2 --tag ab_${1}_t$TR \
    --layers 1,31,60 --modes warm,cold --loops sync --iters $IT --warmup 20 > $LOG/ab_${1}_t$TR.log 2>&1
  echo "[ab_run] $1 t$TR rc=$?"
}
if [ $((TR % 2)) -eq 1 ]; then run land1 $LAND1; run k2v4 $V4; else run k2v4 $V4; run land1 $LAND1; fi
