#!/bin/bash
# All gfx1250 common-primitive tests on GPU 0 (or $GPU) under its lock; logs -> test_<name>.log.
# Refuses to run while an SGLang server or a campaign bench is up.
set -u
GPU=${GPU:-0}
DIR=/data/muse455/k2/cand/k2v4/port/common
if docker exec muse_mi455x pgrep -f sglang.launch_server >/dev/null; then
  echo "sglang server running: not testing"; exit 2
fi
if tail -3 /data/muse455/campaign/c1/logs/driver.log 2>/dev/null | grep -q 'Harness running bench'; then
  echo "campaign bench running: not testing"; exit 2
fi
flock /data/muse455/locks/gpu$GPU.lock docker exec muse_mi455x bash -c "
  rc=0
  for t in ${*:-lane math block wmma mem}; do
    cd /tmp && PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=$GPU timeout 300 python3 $DIR/test_\$t.py 2>&1 \
      | grep -E '^  (PASS|FAIL)|^==|Error|Traceback' > $DIR/test_\$t.log
    rc=\$((rc | PIPESTATUS[0]))
    tail -1 $DIR/test_\$t.log
  done
  exit \$rc"
