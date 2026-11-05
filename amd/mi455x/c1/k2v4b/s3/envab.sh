#!/bin/bash
# Env-switch A/B on this tree: envab.sh <layers> "<A=1 B=1>" "<A=0>" ...  (each arm one run_k2.py through k2safe)
T=/data/muse455/k2/cand/k2v4
LAYERS=$1; shift
mkdir -p /tmp/k2v2pre
for arm in "$@"; do
  E=(); for kv in $arm; do E+=(-e "$kv"); done
  log=/tmp/k2v2pre/envab_$(date +%H%M%S).log
  /data/muse455/k2/k2safe.sh -t 600 "${E[@]}" $T s3/run_k2.py --layers $LAYERS > $log 2>&1
  echo "[envab] arm '$arm' rc=$?: $(grep -E 'K2 .* us/layer' $log | tr '\n' ' ')"
  grep -E '^layer [0-9]+: \{' $log | sed 's/"vs_k2ref.*//' | cut -c1-260
  python3 - "$T/s3/run_k2_a8.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for k, r in d.items():
    if isinstance(r, dict) and "us_per_layer_median_ranks" in r:
        print(f"  L{k} ranks {r['us_per_layer_median_ranks']} replay_matches_first {r.get('replay_matches_first')}")
EOF
done
