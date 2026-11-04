#!/bin/bash
# Serve the workspace's current arbor/config.json in muse_mi455x for a microbenchmark inside an agent turn.
#   bash start_probe_server.sh            start, wait for health (about 4 min), print the log path
#   bash start_probe_server.sh stop       stop the server this script started
# It writes /tmp/arbor-server-30000.pid in the container, the file the harness reads to stop its own server, so the
# harness stops it before its own launch. It refuses when port 30000 already answers.
set -euo pipefail
C=muse_mi455x
CFG=/sgl-workspace/sglang/python/sglang/arbor/config.json
PIDF=/tmp/arbor-server-30000.pid
LOG=/tmp/probe_server.log

if [ "${1:-}" = stop ]; then
  docker exec $C bash -c "[ -f $PIDF ] && kill -TERM -- -\$(cat $PIDF) 2>/dev/null; for i in \$(seq 60); do pgrep -f '[s]glang.launch_server' >/dev/null || break; sleep 2; done; rm -f $PIDF; echo stopped"
  exit 0
fi

curl -s -m 3 -o /dev/null http://127.0.0.1:30000/health && { echo "port 30000 answers: a server is already up" >&2; exit 3; }
docker exec $C bash -c "python3 - <<'PY' > /tmp/probe_launch.sh
import json, shlex
c = json.load(open('$CFG'))
lines = ['cd /tmp'] + [f'export {k}={shlex.quote(str(v))}' for k, v in c['server_env'].items()]
lines.append('exec python3 -m sglang.launch_server --model-path /data/Muse-Spark-Open-816B-A42B-Quark-MXFP4 '
             '--served-model-name amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 --host 0.0.0.0 --port 30000 ' + c['server_flags'])
print('\n'.join(lines))
PY
setsid nohup bash /tmp/probe_launch.sh > $LOG 2>&1 < /dev/null & echo \$! > $PIDF"
for i in $(seq 120); do
  curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health 2>/dev/null | grep -q 200 && { echo "probe server up (log $LOG in $C)"; exit 0; }
  docker exec $C bash -c "[ -d /proc/\$(cat $PIDF) ]" || { echo "probe server died; tail:" >&2; docker exec $C tail -20 $LOG >&2; exit 1; }
  sleep 5
done
echo "probe server not healthy after 10 min" >&2; exit 1
