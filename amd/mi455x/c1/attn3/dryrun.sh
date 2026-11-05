#!/bin/bash
# Gate one attn3 variant: separate k2safe -t 60 calls at small contexts and the exact AgentX shape.
# usage: dryrun.sh <variant> [full|swa]; stops at the first failure; any k2safe health refusal / new MES errors
# (exit 90/94/95) or a timeout (124) means: stop GPU work and report.
V=$1; K=${2:-full}
cd /data/muse455/attn3 || exit 1
for ctx in 1024 4096 340000; do
  /data/muse455/k2/k2safe.sh -t 60 -w 'python3 /data/muse455/k2/cand/k2v2_poll/s3/mpol_exec.py' \
    /data/muse455/attn3 dry.py "$V" $ctx $K > logs/dry_${V}_${K}_${ctx}.log 2>&1
  rc=$?
  grep "^\[dry\]\|^\[k2safe\]" logs/dry_${V}_${K}_${ctx}.log
  if [ $rc -ne 0 ]; then echo "[dryrun] $V $K ctx $ctx FAILED rc=$rc"; exit $rc; fi
done
echo "[dryrun] $V $K PASS"
