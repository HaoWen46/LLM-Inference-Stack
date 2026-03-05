#!/usr/bin/env bash
# Launch llama-cpp-python's OpenAI-compatible server.
# Reads config from config/.env; all variables can be overridden from the shell.
#
# Usage:
#   bash scripts/launch_llamacpp.sh
#   LLAMACPP_PORT=8002 LLAMACPP_MODEL=... bash scripts/launch_llamacpp.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Load .env if not already sourced
if [[ -f "$ROOT_DIR/config/.env" ]]; then
  set -a; source "$ROOT_DIR/config/.env"; set +a
fi

LLAMACPP_MODEL="${LLAMACPP_MODEL:-}"
LLAMACPP_PORT="${LLAMACPP_PORT:-8001}"
LLAMACPP_GPU_LAYERS="${LLAMACPP_GPU_LAYERS:--1}"   # -1 = all layers on GPU
LLAMACPP_CTX="${LLAMACPP_CTX:-8192}"
LLAMACPP_HOST="${LLAMACPP_HOST:-0.0.0.0}"

if [[ -z "$LLAMACPP_MODEL" ]]; then
  echo "[llamacpp] ERROR: LLAMACPP_MODEL is not set."
  echo "  Set it in config/.env or pass it as an environment variable."
  echo "  Example: LLAMACPP_MODEL=/tmp/user/models/Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf"
  exit 1
fi

echo "[llamacpp] Model:      $LLAMACPP_MODEL"
echo "[llamacpp] Port:       $LLAMACPP_PORT"
echo "[llamacpp] GPU layers: $LLAMACPP_GPU_LAYERS"
echo "[llamacpp] Context:    $LLAMACPP_CTX"

exec uv run python3 -m llama_cpp.server \
  --model "$LLAMACPP_MODEL" \
  --n_gpu_layers "$LLAMACPP_GPU_LAYERS" \
  --n_ctx "$LLAMACPP_CTX" \
  --host "$LLAMACPP_HOST" \
  --port "$LLAMACPP_PORT"
