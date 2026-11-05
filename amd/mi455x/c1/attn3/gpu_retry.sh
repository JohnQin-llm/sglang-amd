#!/bin/bash
# Retry one bench job through k2safe until the GPUs are free. usage: gpu_retry.sh <log> <timeout_s> <script> [args...]
# Exit codes from k2safe: 91 server running / 92 locks / 93 busy -> retry every 60 s; 95 SMU, 94 unhealthy, 90 -> stop.
LOG=$1; T=$2; shift 2
while true; do
  /data/muse455/k2/k2safe.sh -t "$T" -w 'python3 /data/muse455/k2/cand/k2v2_poll/s3/mpol_exec.py' /data/muse455/attn3 "$@" > "$LOG" 2>&1
  rc=$?
  case $rc in
    91|92|93) sleep 60 ;;
    *) echo "[gpu_retry] rc=$rc" >> "$LOG"; exit $rc ;;
  esac
done
