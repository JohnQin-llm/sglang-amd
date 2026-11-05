#!/bin/bash
# First measurement round (each step waits for free GPUs through run_all.sh).
R=/data/muse455/draft2/bench/run_all.sh
O=/data/muse455/draft2/logs
$R dense_warm bench/bench_dense.py all --trials 4 --out $O/dense_warm.json || exit $?
$R dense_cold bench/bench_dense.py all --cold --cold-sets 8 --trials 4 --out $O/dense_cold.json || exit $?
$R norm_cold1024 bench/bench_dense.py norm --cold --cold-sets 1024 --trials 4 --out $O/norm_cold1024.json || exit $?
$R fc_warm bench/bench_draft.py fc --trials 4 --out $O/fc_warm.json || exit $?
$R fc_cold bench/bench_draft.py fc --cold --cold-sets 4 --trials 4 --out $O/fc_cold.json || exit $?
$R attn_340k bench/bench_dense.py attn --ctx 32768 348160 --cold --cold-sets 4 --trials 4 --out $O/attn_ctx.json || exit $?
echo done > $O/seq1.done
