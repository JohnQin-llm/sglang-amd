#!/bin/bash
# Run the draft2 microbenchmarks through k2safe, retrying while an SGLang server holds the GPUs (exit 91/93).
# Stops on any other refusal (90 unhealthy, 95 SMU) and reports it.
# usage: run_all.sh <name> <script> [args...]   (logs to /data/muse455/draft2/logs/<name>.log)
set -u
NAME=$1; shift
LOG=/data/muse455/draft2/logs/$NAME.log
mkdir -p /data/muse455/draft2/logs
for i in $(seq 1 120); do
  /data/muse455/k2/k2safe.sh -t 900 /data/muse455/draft2 "$@" > $LOG 2>&1
  rc=$?
  if [ $rc -eq 91 ] || [ $rc -eq 93 ] || [ $rc -eq 92 ]; then
    echo "$(date -u +%T) refused rc=$rc (busy), retry $i" >> /data/muse455/draft2/logs/$NAME.retries
    sleep 60
    continue
  fi
  echo "$(date -u +%T) rc=$rc" >> /data/muse455/draft2/logs/$NAME.retries
  exit $rc
done
exit 99
