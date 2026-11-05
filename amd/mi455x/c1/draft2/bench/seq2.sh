#!/bin/bash
R=/data/muse455/draft2/bench/run_all.sh
O=/data/muse455/draft2/logs
$R meta bench/bench_draft.py meta --trials 4 --out $O/meta.json
$R gemm bench/bench_draft.py gemm --out $O/gemm.json
echo done > $O/seq2.done
