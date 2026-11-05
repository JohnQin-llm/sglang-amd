#!/bin/bash
# Run the k2v4b study's GPU jobs in order, each through k2safe (retried every 60 s while refused: SGLang server up /
# GPUs busy). Stops at the first job that fails (rc != 0) or if the GPUs are marked unhealthy / SMU refuses (rc 95).
# Usage: queue.sh <jobfile>   (one job a line: name|k2safe args ...; '#' lines skipped; a DONE marker a job)
set -uo pipefail
Q=$1
D=/data/muse455/k2/cand/k2v4c/stress/out
mkdir -p $D
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac
  name=${line%%|*}
  args=${line#*|}
  [ -e $D/$name.DONE ] && continue
  for i in $(seq 1 400); do
    # locks held by someone else (e.g. a GSM8K run that waits for every k2safe to exit): do not queue on them
    if ! flock -n /data/muse455/locks/gpu0.lock true || ! flock -n /data/muse455/locks/gpu3.lock true; then
      echo "[queue] $(date -u +%T) GPU locks held elsewhere; waiting" > $D/$name.log; sleep 60; continue
    fi
    eval /data/muse455/k2/k2safe.sh $args > $D/$name.log 2>&1
    rc=$?
    grep -q "\[k2safe\] refused\|could not take the GPU locks\|\[queue\]" $D/$name.log || break
    [ $rc -eq 95 ] && { echo "$(date -u +%T) $name: SMU refusal, stopping" >> $D/queue.log; exit 95; }
    [ -e /data/muse455/locks/GPU_UNHEALTHY ] && { echo "$(date -u +%T) GPU_UNHEALTHY, stopping" >> $D/queue.log; exit 3; }
    sleep 60
  done
  echo "$(date -u +%T) $name rc=$rc" >> $D/queue.log
  [ $rc -ne 0 ] && { echo "$(date -u +%T) stopping after $name" >> $D/queue.log; exit $rc; }
  touch $D/$name.DONE
done < "$Q"
echo "$(date -u +%T) queue done" >> $D/queue.log
