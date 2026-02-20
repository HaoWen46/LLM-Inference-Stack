#!/usr/bin/env bash
# launch_vllm.sh — start the vLLM OpenAI-compatible server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Load env
if [[ -f "$ROOT_DIR/config/.env" ]]; then
    set -a; source "$ROOT_DIR/config/.env"; set +a
else
    echo "[ERROR] config/.env not found. Copy config/.env.example to config/.env and fill it in."
    exit 1
fi

# Activate venv
source "$ROOT_DIR/.venv/bin/activate"

export HF_HOME="${MODEL_CACHE_DIR:-$ROOT_DIR/models}"
export TRANSFORMERS_CACHE="${MODEL_CACHE_DIR:-$ROOT_DIR/models}"

# If CUDA_VISIBLE_DEVICES is empty string (common in some environments), set explicit devices
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" || -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES="0,1"
fi

# NCCL tuning
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Triton kernel cache (persist across runs for faster cold start)
export TRITON_CACHE_DIR="$ROOT_DIR/.triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

TP_SIZE="${TP_SIZE:-1}"
QUANT_ARGS=""
if [[ "${QUANTIZATION:-none}" != "none" ]]; then
    QUANT_ARGS="--quantization ${QUANTIZATION}"
    if [[ "${QUANTIZATION}" == "bitsandbytes" ]]; then
        QUANT_ARGS="$QUANT_ARGS --load-format bitsandbytes"
    fi
fi

PREFIX_CACHE_ARG="--enable-prefix-caching"
CHUNKED_PREFILL_ARG="--enable-chunked-prefill"

echo "[vllm] Starting model: ${MODEL_NAME}"
echo "[vllm] TP size: ${TP_SIZE}"
echo "[vllm] Quantization: ${QUANTIZATION:-none}"
echo "[vllm] Max context: ${MAX_MODEL_LEN:-8192}"
echo "[vllm] GPU memory util: ${GPU_MEMORY_UTILIZATION:-0.90}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --served-model-name "${SERVED_MODEL_NAME:-$(basename $MODEL_NAME)}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype "${DTYPE:-bfloat16}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --max-num-seqs "${MAX_NUM_SEQS:-256}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
    --swap-space "${SWAP_SPACE_GB:-8}" \
    --host "${VLLM_HOST:-0.0.0.0}" \
    --port "${VLLM_PORT:-8000}" \
    --api-key "${VLLM_API_KEY}" \
    --uvicorn-log-level warning \
    --disable-log-requests \
    ${QUANT_ARGS} \
    ${PREFIX_CACHE_ARG} \
    ${CHUNKED_PREFILL_ARG} \
    ${VLLM_EXTRA_ARGS:-}
