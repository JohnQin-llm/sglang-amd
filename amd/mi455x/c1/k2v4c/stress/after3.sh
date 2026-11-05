#!/bin/bash
# start the k2v4c queue once the k2v4b round-3 queue has ended
Q3=/data/muse455/k2/cand/k2v4b/stress/out/queue.log
until tail -1 $Q3 | grep -q "queue done\|stopping"; do sleep 20; done
grep -q "GPU_UNHEALTHY\|SMU" <(tail -1 $Q3) && exit 3
cd /data/muse455/k2/cand/k2v4c/stress && exec ./queue.sh jobs4.txt
