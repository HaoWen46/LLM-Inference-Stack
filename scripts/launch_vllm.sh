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

# ── cublas fix ──────────────────────────────────────────────────────────────
# torch 2.10.0 bundles nvidia-cublas-cu12==12.8.4.1 which breaks all BF16
# GEMMs on Ampere (CC 8.6) when the CUDA driver is 12.9+.
# Fix: upgrade to 12.9.1.4.  Checked on every launch; takes <1s if already
# correct, ~5s to install if not (e.g. after a fresh uv sync).
_CUBLAS_NEED="12.9.1.4"
_CUBLAS_HAVE=$(python3 -c \
    "import importlib.metadata; print(importlib.metadata.version('nvidia-cublas-cu12'))" \
    2>/dev/null || echo "unknown")
if [[ "$_CUBLAS_HAVE" != "$_CUBLAS_NEED" ]]; then
    echo "[vllm] cublas fix: nvidia-cublas-cu12 ${_CUBLAS_HAVE} → ${_CUBLAS_NEED}"
    pip install --quiet "nvidia-cublas-cu12==${_CUBLAS_NEED}" --no-deps
else
    echo "[vllm] cublas OK: nvidia-cublas-cu12==${_CUBLAS_HAVE}"
fi
unset _CUBLAS_NEED _CUBLAS_HAVE

# HF cache — HF_HOME is the current standard; TRANSFORMERS_CACHE is deprecated in v5
export HF_HOME="${MODEL_CACHE_DIR:-$ROOT_DIR/models}"

# Redirect vLLM's torch_compile_cache and other caches away from home (limited quota)
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$ROOT_DIR/.cache/vllm}"
mkdir -p "$VLLM_CACHE_ROOT"

# Ensure consistent GPU indexing on mixed-GPU machines (4090 + 3090)
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# If CUDA_VISIBLE_DEVICES is empty, fall back to first two GPUs
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" || -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES="0,1"
fi

# NCCL tuning — IB disabled (no InfiniBand on consumer hardware),
# P2P enabled, blocking waits so NCCL errors surface instead of hanging
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

# Consumer GPUs (RTX 4090 / 3090) don't support NVIDIA custom all-reduce
# (requires NVLink / NVSwitch); disable it explicitly to suppress the warning.
ALLREDUCE_ARG="--disable-custom-all-reduce"

# Use vLLM's own generation defaults instead of the HuggingFace config,
# so sampling params are not silently overridden by model-card suggestions.
GEN_CONFIG_ARG="--generation-config vllm"

PREFIX_CACHE_ARG="--enable-prefix-caching"
CHUNKED_PREFILL_ARG="--enable-chunked-prefill"

# Qwen3.5 is a VL model; we run text-only. Cap vision inputs at 0 to prevent
# the vision encoder from being called with empty inputs during the profile run,
# which triggers a 0-size BF16 GEMM on TP>1 (CUBLAS_STATUS_INVALID_VALUE).
VISION_ARG=""
if [[ "${LIMIT_MM_PER_PROMPT:-}" != "" ]]; then
    VISION_ARG="--limit-mm-per-prompt ${LIMIT_MM_PER_PROMPT}"
fi

# LoRA: enable when LORA_MODULES is set (format: "alias=path [alias=path ...]")
LORA_ARG=""
if [[ "${LORA_MODULES:-}" != "" ]]; then
    LORA_ARG="--enable-lora --lora-modules ${LORA_MODULES} --max-lora-rank ${MAX_LORA_RANK:-32}"
    echo "[vllm] LoRA enabled: ${LORA_MODULES}"
fi

echo "[vllm] Starting model: ${MODEL_NAME}"
echo "[vllm] TP size: ${TP_SIZE}"
echo "[vllm] Quantization: ${QUANTIZATION:-none}"
echo "[vllm] Max context: ${MAX_MODEL_LEN:-8192}"
echo "[vllm] GPU memory util: ${GPU_MEMORY_UTILIZATION:-0.90}"
echo "[vllm] GPUs: ${CUDA_VISIBLE_DEVICES}"

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
    --host "${VLLM_HOST:-127.0.0.1}" \
    --port "${VLLM_PORT:-8000}" \
    --api-key "${VLLM_API_KEY}" \
    --uvicorn-log-level warning \
    --no-enable-log-requests \
    --tokenizer-mode slow \
    --enforce-eager \
    ${ALLREDUCE_ARG} \
    ${GEN_CONFIG_ARG} \
    ${QUANT_ARGS} \
    ${PREFIX_CACHE_ARG} \
    ${CHUNKED_PREFILL_ARG} \
    ${VISION_ARG} \
    ${LORA_ARG} \
    ${VLLM_EXTRA_ARGS:-}
