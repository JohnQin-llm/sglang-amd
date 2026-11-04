#!/bin/bash
# Refuse to start a GPU run while a server is up or the campaign bench runs; then take the locks given
# (in order) and run the command.  Usage: gpu_gate.sh "0 1 2 3" <command...>
set -u
gpus=$1; shift
if docker exec muse_mi455x pgrep -f sglang.launch_server >/dev/null; then
    echo "GATE: sglang server up ($(docker exec muse_mi455x pgrep -f sglang.launch_server | tr '\n' ' ')); not running"
    exit 75
fi
if tail -3 /data/muse455/campaign/c1/logs/driver.log | grep -q "Harness running bench\|Harness starting server"; then
    echo "GATE: campaign bench / server start in progress; not running"
    exit 75
fi
cmd=("$@")
for g in $(echo "$gpus" | tr ' ' '\n' | sort -r); do
    cmd=(flock "/data/muse455/locks/gpu$g.lock" "${cmd[@]}")
done
exec "${cmd[@]}"
