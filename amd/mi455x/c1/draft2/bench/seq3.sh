#!/bin/bash
R=/data/muse455/draft2/bench/run_all.sh
O=/data/muse455/draft2/logs
$R attn_v2check bench/bench_dense.py attn --ctx 131072 348160 --cold --cold-sets 4 --trials 2 --out $O/attn_v2check.json
$R prefetch bench/bench_draft.py prefetch --trials 4 --out $O/prefetch.json
$R ar bench/bench_ar.py $O/ar.json
echo done > $O/seq3.done
