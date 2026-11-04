#!/usr/bin/env bash
# AgentX at the campaign's concurrency on the InferenceX protocol (commit 73cc7fc): the AgentX 1.0.12
# client replaying semianalysis_cc_traces_weka_062126, scored by InferenceX's own parser.
#
# Arbor injects PORT, MODEL_NAME, CONCURRENCY, BENCH_OUTPUT. From server_env: DURATION (s, default
# 600; the official hour is 3600) and WARMUP_PER_LANE (default 10). The server is already running.
#
# Chips are read from the live server (tp x pp x dp, dp only without DP attention), not assumed, so
# the per-chip numbers stay right when the topology changes. bench_output.json:
#   tput_per_chip        total (input + output) tokens/s per chip: the leaderboard's throughput axis
#   output_tput_per_chip output tokens/s per chip
#   p90_interactivity    1 / P90 full-response inter-token latency, tok/s per user (dashboard rounding)
set -euo pipefail

: "${PORT:?}" "${MODEL_NAME:?}" "${CONCURRENCY:?}" "${BENCH_OUTPUT:?}"
BLIND=${ARBOR_BLIND_ROOT:-/data/muse455/arbor-blind}
DURATION=${DURATION:-600}
WARMUP_PER_LANE=${WARMUP_PER_LANE:-10}
OUT="$(dirname "$BENCH_OUTPUT")/agentx"
URL="http://127.0.0.1:${PORT}"
rm -rf "$OUT" && mkdir -p "$OUT"

curl -sf "$URL/get_server_info" > "$OUT/server_info.json"
read -r TP EP CHIPS < <(python3 - "$OUT/server_info.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
tp, pp, dp = (int(info.get(k) or 1) for k in ("tp_size", "pp_size", "dp_size"))
chips = tp * pp * (1 if info.get("enable_dp_attention") else dp)
print(tp, int(info.get("ep_size") or 1), chips)
PY
)
OFFLOAD=none
grep -q '"enable_hierarchical_cache": true' "$OUT/server_info.json" && OFFLOAD=dram

HF=(HF_HOME="$BLIND/hf-cache" HF_HUB_CACHE="$BLIND/hf-cache/hub" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1)
args=(profile --url "$URL" --endpoint /v1/chat/completions --model "$MODEL_NAME"
      --tokenizer amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 --tokenizer-trust-remote-code
      --tokenizer-revision c7bc74fd64371315a091bc15f9723882eead0dc1
      --concurrency "$CONCURRENCY" --benchmark-duration "$DURATION"
      --warmup-requests-per-lane "$WARMUP_PER_LANE" --warmup-grace-period 1800
      --public-dataset semianalysis_cc_traces_weka_062126 --server-metrics "$URL/metrics"
      --output-artifact-dir "$OUT/aiperf_artifacts" --record-processors 1 --random-seed 42)
[ "$DURATION" -lt 900 ] && args+=(--unsafe-override)   # AgentX refuses runs shorter than 900 s otherwise
echo "[bench] C$CONCURRENCY ${DURATION}s on TP$TP EP$EP, $CHIPS chips, KV offload $OFFLOAD"
env -i PATH=/usr/bin:/bin HOME="$OUT" TZ=UTC "${HF[@]}" AIPERF_HTTP_TCP_USER_TIMEOUT=900000 \
    AIPERF_REQUIRED_SERVER_METRIC_PREFIX=sglang: PYTHONUNBUFFERED=1 \
    "$BLIND/agentx-venv/bin/agentx" "${args[@]}" > "$OUT/client.log" 2>&1 \
    || { tail -20 "$OUT/client.log" >&2; echo "[bench] client failed" >&2; exit 1; }

env -i PATH=/usr/bin:/bin "${HF[@]}" PYTHONPATH="$BLIND/InferenceX/inferencex-e2e" \
    RESULT_DIR="$OUT" RESULT_FILENAME=inferencex-aggregate AGENTIC_OUTPUT_DIR="$OUT" \
    MODEL=amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 MODEL_PREFIX=musespark FRAMEWORK=sglang PRECISION=fp4 \
    CONC="$CONCURRENCY" TP="$TP" EP_SIZE="$EP" PP_SIZE=1 DCP_SIZE=1 PCP_SIZE=1 DP_ATTENTION=false \
    IS_MULTINODE=false DISAGG=false KV_OFFLOADING="$OFFLOAD" SPEC_DECODING=draft_model RUNNER_TYPE=${RUNNER_TYPE:-mi455x} \
    "$BLIND/agentx-venv/bin/python" -m infx.results.agentic.process_agentic_result > "$OUT/aggregate.log" 2>&1 \
    || { tail -20 "$OUT/aggregate.log" >&2; echo "[bench] parser failed" >&2; exit 1; }

python3 - "$OUT/inferencex-aggregate.json" "$BENCH_OUTPUT" "$CHIPS" "$TP" "$EP" <<'PY'
import json, sys
agg_path, out_path, chips, tp, ep = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:])
agg = json.load(open(agg_path))
m = agg["request_metrics"]
total = m["throughput"]
p90_itl = m["latency"]["full_response_itl"]["p90"]
result = {
    "tput_per_chip": total["total"]["tokens_per_second"] / chips,
    "output_tput_per_chip": total["output"]["tokens_per_second"] / chips,
    "p90_interactivity": round(1 / round(p90_itl, 5), 2),
    "chips": chips, "tp": tp, "ep": ep,
    "num_requests_successful": agg.get("num_requests_successful", 0),
    "num_requests_total": agg.get("num_requests_total", 0),
    "agg_json_path": agg_path,
    "status": "ok",
}
json.dump(result, open(out_path, "w"), indent=2)
print(f"[bench] {result['tput_per_chip']:.0f} tok/s/chip, P90 {result['p90_interactivity']} tok/s/user, "
      f"{result['num_requests_successful']}/{result['num_requests_total']} requests")
PY
