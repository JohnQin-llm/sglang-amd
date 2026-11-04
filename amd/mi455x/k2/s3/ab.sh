#!/bin/bash
# A/B driver (run inside the container): ab.sh <tl|time|b2b> "ENV=1 ENV2=0" "ENV=0" ...
# tl: probe_b2b --timeline (rank 0 lines); time: run_k2 (numerics + us/layer); b2b: probe_b2b --eager 30.
mode=$1; shift
cd /tmp
for cfg in "$@"; do
  echo "=== $mode [$cfg]"
  case $mode in
    tl) env $cfg K2_DEBUG_TL=1 PYTHONSAFEPATH=1 PYTHONPATH=/data/muse455/k2/ep1 timeout 600 python3 /data/muse455/k2/ep1/s3/probe_b2b.py --layers ${K2_LAYERS:-31} --timeline 2>&1 \
          | grep -v -i warning | grep -E "TL rank 0|gave up|PROBE|Error|error|Trace|layer [0-9]+:" | grep -v "gave up$" ;;
    time) env $cfg PYTHONSAFEPATH=1 PYTHONPATH=/data/muse455/k2/ep1 timeout 600 python3 /data/muse455/k2/ep1/s3/run_k2.py --layers ${K2_LAYERS:-31} --reps 20 --iters 20 2>&1 \
          | grep -v -i warning | grep -E "K2 |pass_target|gave up|Error|error|Trace" ;;
    b2b) env $cfg PYTHONSAFEPATH=1 PYTHONPATH=/data/muse455/k2/ep1 timeout 600 python3 /data/muse455/k2/ep1/s3/probe_b2b.py --layers ${K2_LAYERS:-31} --eager 30 2>&1 \
          | grep -v -i warning | grep -E "gave up|PROBE|Error|error|Trace|layer [0-9]+:" ;;
  esac
done
