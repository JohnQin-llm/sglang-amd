#!/bin/bash
# One serving arm: the C1 served env/argv (campaign/c1/session/iter_12/effective_config.json), optionally under
# MPOL_LOCAL (mpol_exec.py), single-stream decode load, KFD quiesce trace. Holds the four GPU locks throughout.
#   serve_ab.sh <arm: base|mpol> [decode seconds]
set -uo pipefail
ARM=$1; SECS=${2:-180}
D=/data/muse455/k2/cand/k2v2_poll
C=muse_mi455x
L=/data/muse455/locks
LOG=$D/logs/server_$ARM.log
PIDF=/tmp/k2poll_server_$ARM.pid
OUT=$D/s3/serve
mkdir -p $OUT
exec 10>$L/gpu0.lock 11>$L/gpu1.lock 12>$L/gpu2.lock 13>$L/gpu3.lock
for fd in 10 11 12 13; do flock -w 900 $fd || { echo "[serve] could not take the GPU locks"; exit 92; }; done
docker exec $C pgrep -f "[s]glang.launch_server" >/dev/null && { echo "[serve] refused: a server is running"; exit 91; }
busy=$(timeout 90 docker exec $C bash -c 'amd-smi metric -m -u 2>/dev/null' | awk '/GFX_ACTIVITY/{if ($2+0>0) b=1} /USED_VRAM/{if ($2+0>2048) b=1} END{if (b) print "busy"}')
[ -n "$busy" ] && { echo "[serve] refused: GPUs busy"; exit 93; }
[ -e $L/GPU_UNHEALTHY ] && { echo "[serve] refused: GPU_UNHEALTHY"; exit 90; }

ENVS=$(python3 -c "
import json, shlex
d = json.load(open('/data/muse455/campaign/c1/session/iter_12/effective_config.json'))
print(' '.join('-e ' + shlex.quote(f'{k}={v}') for k, v in d['env'].items()))")
ARGV=$(python3 -c "
import json, shlex
d = json.load(open('/data/muse455/campaign/c1/session/iter_12/effective_config.json'))
print(' '.join(shlex.quote(a) for a in d['argv']))")
PRE=""
[ "$ARM" = "mpol" ] && PRE="python3 $D/s3/mpol_exec.py"

bpftrace $D/s3/trace/kfd_evict3.bt > $OUT/trace_$ARM.txt 2>&1 10>&- 11>&- 12>&- 13>&- &
echo $! > /tmp/k2poll_bt_$ARM.pid
echo "[serve] $(date -u +%T) start arm=$ARM"
RUN=$D/logs/server_$ARM.run.sh
printf '#!/bin/bash\ncd /tmp\necho $$ > %s\nexec %s %s > %s 2>&1\n' "$PIDF" "$PRE" "$ARGV" "$LOG" > $RUN
eval docker exec -d $ENVS $C bash $RUN
ok=0
for i in $(seq 1 180); do
  sleep 5
  grep -q "fired up" $LOG 2>/dev/null && { ok=1; break; }
  docker exec $C test -e /proc/$(docker exec $C cat $PIDF 2>/dev/null)/status 2>/dev/null || { echo "[serve] server exited early"; break; }
done
echo "[serve] $(date -u +%T) ready=$ok"
if [ $ok = 1 ]; then
  docker exec -e NO_PROXY=127.0.0.1,localhost $C bash -c "unset https_proxy HTTPS_PROXY http_proxy HTTP_PROXY; python3 $D/s3/serve_client.py --out $OUT/client_$ARM.json --seconds $SECS" 2>&1 | grep -v Warning
fi
SP=$(docker exec $C cat $PIDF)
echo "[serve] $(date -u +%T) stopping server pid $SP (container ns)"
docker exec $C kill -INT $SP
for i in $(seq 1 60); do sleep 2; docker exec $C test -e /proc/$SP/status || break; done
docker exec $C test -e /proc/$SP/status && { echo "[serve] still up after SIGINT, SIGTERM"; docker exec $C kill -TERM $SP; sleep 20; }
kill -INT $(cat /tmp/k2poll_bt_$ARM.pid) 2>/dev/null
for i in $(seq 1 24); do
  busy=$(timeout 90 docker exec $C bash -c 'amd-smi metric -m -u 2>/dev/null' | awk '/GFX_ACTIVITY/{if ($2+0>0) b=1} /USED_VRAM/{if ($2+0>2048) b=1} END{if (b) print "busy"}')
  [ -z "$busy" ] && break; sleep 5
done
docker exec $C pgrep -af "[s]glang" | head -3
echo "[serve] $(date -u +%T) done arm=$ARM gpus=${busy:-idle}"
