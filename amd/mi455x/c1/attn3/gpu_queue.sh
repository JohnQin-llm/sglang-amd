#!/bin/bash
# Run queued bench jobs one at a time through k2safe, waiting while the GPUs are refused.
# usage: gpu_queue.sh <jobfile>   (one job per line: <log> <timeout_s> <script> [args...]; '#' comments)
# k2safe exits: 90 unhealthy marker / 91 server running / 92 locks / 93 busy -> wait 120 s and retry;
# 95 (SMU: No response) or 94 (this run left the GPUs unhealthy) -> stop the queue; a failing job stops it too.
JOBS=$1
STATUS=/data/muse455/attn3/logs/queue.status
n=0
while read -r LOG T SCRIPT ARGS; do
  [ -z "$LOG" ] && continue
  case "$LOG" in \#*) continue ;; esac
  n=$((n + 1))
  while true; do
    echo "$(date -u +%T) job $n $LOG: trying" > $STATUS
    /data/muse455/k2/k2safe.sh -t "$T" -w 'python3 /data/muse455/k2/cand/k2v2_poll/s3/mpol_exec.py' \
      /data/muse455/attn3 $SCRIPT $ARGS > "$LOG" 2>&1
    rc=$?
    case $rc in
      90|91|92|93) echo "$(date -u +%T) job $n $LOG: refused rc=$rc, waiting" > $STATUS; sleep 120 ;;
      *) break ;;
    esac
  done
  echo "[gpu_queue] rc=$rc" >> "$LOG"
  echo "$(date -u +%T) job $n $LOG: done rc=$rc" >> $STATUS.hist
  if [ $rc -ne 0 ]; then echo "$(date -u +%T) queue stopped at job $n rc=$rc" > $STATUS; exit $rc; fi
done < "$JOBS"
echo "$(date -u +%T) queue finished ($n jobs)" > $STATUS
