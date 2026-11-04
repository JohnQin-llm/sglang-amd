#!/bin/bash
# Run one K2 GPU job safely: refuse if the GPUs are unhealthy or busy, hold all four GPU locks,
# hard-timeout the job, then check the GPUs drained and the driver logged no MES/queue errors.
# Usage: k2safe.sh [-t seconds] [-e VAR=val ...] [-w 'wrapper cmd'] <tree> <script under tree> [args...]
#   -w runs the job under a wrapper inside the container (e.g. -w 'rocprofv3 --pmc SQ_WAVES -d /tmp/p --'
#   e.g. k2safe.sh /data/muse455/k2/ep1 s3/run_k2.py --layers 31 --reps 20 --iters 20
set -uo pipefail
T=300
ENVS=()
WRAP=
while [ $# -gt 0 ]; do
  case "$1" in
    -t) T=$2; shift 2 ;;
    -e) ENVS+=(-e "$2"); shift 2 ;;
    -w) WRAP=$2; shift 2 ;;
    *) break ;;
  esac
done
TREE=$1; SCRIPT=$2; shift 2
L=/data/muse455/locks
BAD=$L/GPU_UNHEALTHY
C=muse_mi455x

mes_errors() { dmesg | grep -ciE "MES.*(failed|ring buffer is full|unrecoverable)|Failed to (evict|remove|restore) queue"; }
gpu_busy() {  # prints "busy" if any GPU is >2 GB used or >0% GFX
  timeout 20 docker exec $C bash -c 'amd-smi metric -m -u 2>/dev/null' | awk '
    /GFX_ACTIVITY/ {g=$2+0; if (g>0) b=1}
    /USED_VRAM/ {v=$2+0; if (v>2048) b=1}
    END {if (b) print "busy"}'
}

[ -e $BAD ] && { echo "[k2safe] refused: $(cat $BAD)"; exit 90; }
docker exec $C pgrep -f "[s]glang.launch_server" >/dev/null && { echo "[k2safe] refused: an SGLang server is running"; exit 91; }

exec 10>$L/gpu0.lock 11>$L/gpu1.lock 12>$L/gpu2.lock 13>$L/gpu3.lock
for fd in 10 11 12 13; do flock -w 900 $fd || { echo "[k2safe] could not take the GPU locks in 15 min"; exit 92; }; done

[ -n "$(gpu_busy)" ] && { echo "[k2safe] refused: GPUs busy before the run"; exit 93; }
m0=$(mes_errors)
echo "[k2safe] $(date -u +%T) start: $SCRIPT $* (timeout ${T}s)"
timeout -k 20 $T docker exec "${ENVS[@]}" $C bash -c "cd /tmp && PYTHONSAFEPATH=1 PYTHONPATH=$TREE timeout -k 10 $((T - 10)) $WRAP python3 $TREE/$SCRIPT $*"
rc=$?
for i in $(seq 1 12); do [ -z "$(gpu_busy)" ] && break; sleep 5; done
m1=$(mes_errors)
if [ "$m1" -gt "$m0" ] || [ -n "$(gpu_busy)" ]; then
  echo "$(date -u +%FT%TZ) after $SCRIPT $*: rc=$rc, new MES errors $((m1 - m0)), busy='$(gpu_busy)'" > $BAD
  echo "[k2safe] GPUs UNHEALTHY after the run; all further runs refused (see $BAD)"
  exit 94
fi
echo "[k2safe] $(date -u +%T) done rc=$rc, GPUs drained, no new MES errors"
exit $rc
