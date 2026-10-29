#!/bin/bash
# Muse Spark Open TP4/EP4 server on 4x MI355X from this branch, with the env and server args of the official runs.
#
#   muse-c16/launch_server.sh <c16|c32|c28> [server args...]
#
#   c16  B200 TP4 C16 recipe: SPF, chunked prefill and max prefill 8192, prefill-decode interval 48.
#   c32  C32: max running 64, SPF, chunk 8192, interval 80, real mem fraction 0.87, SWA ratio 0.1, KV savings below.
#   c28  B300 C28 recipe: max running 56, SPF, chunk 16384, interval 80, real mem fraction 0.87, SWA ratio 0.1,
#        KV savings below.
#   KV savings (c32, c28): SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1 (no 0.85 scaling of the explicit fraction),
#   SGLANG_MUSE_ULYSSES_ALIAS_QKV=1, SGLANG_MUSE_DENSE_TILED=0, language_model_only.
#
# The client is InferenceX AgentX 1.0.12 at the recipe's concurrency (16, 32 or 28). Extra server args go last and
# replace a recipe value (argparse keeps the last one), e.g. --port 31201. MUSE_MODEL_PATH and MUSE_DRAFT_MODEL_PATH
# move the checkpoints. GPUs come from the caller (ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES). Inherited SGLANG_*
# variables are dropped, as the official runs' launcher does.
#
# Before starting, the aiter overlay in muse-c16/aiter is applied and the aiter files the stack pins are checked
# (aiter is a separate repo; see muse-c16/MANIFEST.json). JIT extensions build under $SGLANG_CACHE_DIR/muse_jit.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
PYTHON=${PYTHON:-/opt/venv/bin/python}
recipe=${1:?usage: launch_server.sh c16|c32|c28 [server args...]}
shift

# The official runs start the server without the image's SGLANG_* defaults (SGLANG_MOE_PADDING,
# SGLANG_SET_CPU_AFFINITY, ...): drop every inherited SGLANG_* variable and LD_PRELOAD.
for name in $(compgen -e); do
  case $name in SGLANG_*) unset "$name" ;; esac
done
unset LD_PRELOAD
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-lo}

# aiter: apply the overlay (unified_attention.py with num_stages=2 for gfx950 uniform verify at 1..8 requests, else
# the installed file) and check the files the stack pins.
AITER=$("$PYTHON" -c 'import importlib.util; print(importlib.util.find_spec("aiter").submodule_search_locations[0])')
sha() { sha256sum "$1" | cut -d' ' -f1; }
overlay=$AITER/ops/triton/attention/unified_attention.py
case $(sha "$overlay") in
  7e7c7d305ffff48ba7f5ee0fe5d1f541948025e3bce07305341cb2fcb5fe4782) ;;
  6419893c6fced2a1a71ea29cea3773219fc277762d3ac052daf5f72c0273b02f)
    cp -p "$overlay" "$overlay.pre-muse-c16"
    cp "$HERE/aiter/unified_attention.py" "$overlay"
    echo "launch_server.sh: applied muse-c16/aiter/unified_attention.py to $overlay" >&2 ;;
  *) echo "launch_server.sh: $overlay is neither the overlay nor the aiter copy it applies to" >&2; exit 1 ;;
esac
while read -r digest rel; do
  [ "$(sha "$AITER/$rel")" = "$digest" ] || { echo "launch_server.sh: aiter/$rel does not have sha256 $digest" >&2; exit 1; }
done <<'EOF'
7e7c7d305ffff48ba7f5ee0fe5d1f541948025e3bce07305341cb2fcb5fe4782 ops/triton/attention/unified_attention.py
cd2f9cb29358b5582c98d299808a2eecbf643488cfcbc0fd8350bfdc1a494bfa ops/triton/_triton_kernels/attention/unified_attention.py
a48dc8c104562ca3212ec35560d7f087423f28c07ebd40340970a526f2ec038e ops/flydsl/kernels/moe_2stage_a16wmix/__init__.py
299fc24bc8c9b89fe5181d07cdf371d1d5fa51268eae0e1f5f019b24f07db73c configs/model_configs/muse_spark_bf16_tuned_gemm.csv
EOF

configs=bf16_tuned_gemm.csv
for name in dsv3 dsv4 flux2 glm47 glm53 glm5 gptoss kimi kimik2 kimik3 llama405B llama70B minimax_m3_eagle muse_spark \
    q3vl qwen32B qwen3_5_397b qwen3_8b; do
  configs+=:model_configs/${name}_bf16_tuned_gemm.csv
done
export AITER_CONFIG_GEMM_BF16="$(echo "$configs" | sed "s|[^:]*|$AITER/configs/&|g"):$HERE/aiter/muse_c16_tp4_qkv_bf16_tuned_gemm.csv"

export SGLANG_TIMEOUT_KEEP_ALIVE=900 HF_HUB_OFFLINE=1
export SGLANG_USE_AITER=1 SGLANG_USE_AITER_UNIFIED_ATTN=1 MUSE_MOE_BACKEND=flydsl
# Benchmark acceptance of the official runs: the draft's tokens are accepted to an average length of 4.05.
export SGLANG_SIMULATE_ACC_LEN=4.05 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
export SGLANG_MUSE_A4W4_MOE=1 SGLANG_MUSE_MOE_SINGLE_LAYOUT=1 SGLANG_MUSE_MXFP4_BF16_EXPERTS=1
export SGLANG_MUSE_AITER_A16_G1=0 SGLANG_MUSE_AITER_A16_G1G2_SLOT=1 SGLANG_MUSE_EP4_PLACEMENT=fit
export SGLANG_MUSE_TP4_QK_ROPE_CACHE=1 SGLANG_MUSE_TP4_FUSED_QKV_CACHE=1 SGLANG_MUSE_TP4_QKV_TILE_EPILOGUE=1
export SGLANG_MUSE_TP4_PREFILL_QKV_LAYOUT=1 SGLANG_MUSE_TP4_PREFILL_MAXSCALE=1
export SGLANG_MUSE_TP4_VERIFY_FULL_V2=1 SGLANG_MUSE_TP4_VERIFY_SWA_SPLITKV=1 SGLANG_MUSE_TP4_VERIFY_OUTPUT_GATE=1
export SGLANG_MUSE_TP4_VERIFY_ROUTER_POOLER=1 SGLANG_MUSE_TP4_VERIFY_PAD_MASK=1 SGLANG_MUSE_TP4_COMPACT_A4_VERIFY=1
export SGLANG_MUSE_TP4_K2=1 SGLANG_MUSE_TP4_K2_LATENCY_TRIM=1 SGLANG_MUSE_TP4_SPLIT_POOL_UNPOOL=1
export SGLANG_MUSE_TP4_FUSED_FINALIZE_AR=1 SGLANG_MUSE_TP4_FUSED_AR_BOUNDARY=1 SGLANG_MUSE_ULYSSES_LEAN_A2A=1
export SGLANG_MUSE_DFLASH_SPLITKV=1 SGLANG_MUSE_DFLASH_DRAFT_FUSE=1

args=(
  --model-path "${MUSE_MODEL_PATH:-/share_nfs/yanyuqin/huggingface/Muse-Spark-Open-816B-A42B-Quark-MXFP4}"
  --served-model-name amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4
  --tensor-parallel-size 4 --expert-parallel-size 4 --attention-backend aiter --page-size 16 --kv-cache-dtype fp8_e4m3
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "${MUSE_DRAFT_MODEL_PATH:-/share_nfs/yanyuqin/muse-sglang/models/Muse-Spark-Open-DFlash2}"
  --speculative-draft-attention-backend triton --speculative-num-draft-tokens 8 --speculative-draft-window-size 2048
  --reasoning-parser muse --tool-call-parser muse --watchdog-timeout 1800 --enable-metrics
  --scheduler-recv-interval 20 --tokenizer-worker-num 4 --random-seed 42
  --enable-hierarchical-cache --hicache-size 253 --hicache-write-policy write_through --hicache-io-backend kernel
  --hicache-mem-layout page_first
  --schedule-policy shortest-prefill-first
  --host 127.0.0.1 --port 31101
)
case $recipe in
  c16)
    args+=(--max-running-requests 128 --chunked-prefill-size 8192 --max-prefill-tokens 8192 --prefill-decode-interval 48
           --mem-fraction-static 0.85 --swa-full-tokens-ratio 0.2) ;;
  c32|c28)
    export SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1 SGLANG_MUSE_ULYSSES_ALIAS_QKV=1 SGLANG_MUSE_DENSE_TILED=0
    args+=(--prefill-decode-interval 80 --mem-fraction-static 0.87 --swa-full-tokens-ratio 0.1
           --json-model-override-args '{"language_model_only":true}')
    if [ "$recipe" = c32 ]; then
      args+=(--max-running-requests 64 --chunked-prefill-size 8192 --max-prefill-tokens 8192)
    else
      args+=(--max-running-requests 56 --chunked-prefill-size 16384 --max-prefill-tokens 16384)
    fi ;;
  *) echo "launch_server.sh: unknown recipe $recipe (c16, c32 or c28)" >&2; exit 2 ;;
esac

exec "$PYTHON" -m sglang.launch_server "${args[@]}" "$@"
