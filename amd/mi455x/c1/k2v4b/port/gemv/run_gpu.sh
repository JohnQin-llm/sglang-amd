#!/bin/bash
# Run a python script of this directory on GPU ${GPU:-0} under its lock, refusing while a server or bench runs.
# usage: [GPU=0] [XENV="K2G_SHUF=1 ..."] [TMO=300] ./run_gpu.sh test_stages.py [args]
set -e
GPU=${GPU:-0}
D=/data/muse455/k2/cand/k2v4/port/gemv
if docker exec muse_mi455x pgrep -f sglang.launch_server >/dev/null; then
  echo "a server is running: not touching the GPUs" >&2; exit 3
fi
if tail -3 /data/muse455/campaign/c1/logs/driver.log 2>/dev/null | grep -q "Harness running bench"; then
  echo "a bench is running: not touching the GPUs" >&2; exit 3
fi
script=$1; shift
exec flock /data/muse455/locks/gpu${GPU}.lock docker exec muse_mi455x bash -c \
  "cd /tmp && ${XENV:-} PYTHONSAFEPATH=1 HIP_VISIBLE_DEVICES=${GPU} timeout ${TMO:-300} python3 $D/$script $*"
