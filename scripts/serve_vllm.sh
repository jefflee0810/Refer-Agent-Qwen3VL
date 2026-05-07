#!/bin/bash
# Launch vLLM OpenAI-compatible API server for Qwen3-VL-8B-Thinking.
#
# Usage:
#   bash scripts/serve_vllm.sh <gpu_id> [<tp_size>] [<port>]
#
# Examples:
#   bash scripts/serve_vllm.sh 4          # GPU 4, tp=1, port 10000
#   bash scripts/serve_vllm.sh 0,1 2      # GPUs 0+1, tp=2, port 10000
#
# After it boots ("Application startup complete." in logs), Refer-Agent
# scripts can hit it via REFER_VLLM_BASE_URL=http://localhost:<port>/v1.

set -euo pipefail

GPU_ID="${1:?gpu_id required (e.g. 4 or 0,1)}"
TP_SIZE="${2:-1}"
PORT="${3:-10000}"

MODEL_PATH="${REFER_VLLM_MODEL_PATH:-/home/cvlab18/media/data1/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b}"
SERVED_NAME="${REFER_VLLM_SERVED_NAME:-qwen3-vl-8b-thinking}"
MAX_MODEL_LEN="${REFER_VLLM_MAX_MODEL_LEN:-70000}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[serve_vllm] CUDA_VISIBLE_DEVICES=${GPU_ID}  tp=${TP_SIZE}  port=${PORT}"
echo "[serve_vllm] model: ${MODEL_PATH}"
echo "[serve_vllm] served-as: ${SERVED_NAME}"

exec /data/anaconda3/envs/sam3_jaeho/bin/python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}"
