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
# Bind to localhost only — external access must go through the Rust gateway on :8080
LLAMACPP_HOST="${LLAMACPP_HOST:-127.0.0.1}"
LLAMACPP_API_KEY="${LLAMACPP_API_KEY:-}"

if [[ -z "$LLAMACPP_MODEL" ]]; then
  echo "[llamacpp] ERROR: LLAMACPP_MODEL is not set."
  echo "  Set it in config/.env or pass it as an environment variable."
  echo "  Example: LLAMACPP_MODEL=/home5/user/models/Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf"
  exit 1
fi

if [[ -z "$LLAMACPP_API_KEY" ]]; then
  echo "[llamacpp] WARNING: LLAMACPP_API_KEY is not set — server will accept unauthenticated requests."
  echo "  Set LLAMACPP_API_KEY in config/.env to require a Bearer token."
fi

echo "[llamacpp] Model:      $LLAMACPP_MODEL"
echo "[llamacpp] Host:       $LLAMACPP_HOST"
echo "[llamacpp] Port:       $LLAMACPP_PORT"
echo "[llamacpp] GPU layers: $LLAMACPP_GPU_LAYERS"
echo "[llamacpp] Context:    $LLAMACPP_CTX"
echo "[llamacpp] Auth:       ${LLAMACPP_API_KEY:+enabled (key set)}${LLAMACPP_API_KEY:-DISABLED — set LLAMACPP_API_KEY}"

API_KEY_ARG=""
if [[ -n "$LLAMACPP_API_KEY" ]]; then
  API_KEY_ARG="--api_key $LLAMACPP_API_KEY"
fi

exec uv run python3 -m llama_cpp.server \
  --model "$LLAMACPP_MODEL" \
  --n_gpu_layers "$LLAMACPP_GPU_LAYERS" \
  --n_ctx "$LLAMACPP_CTX" \
  --host "$LLAMACPP_HOST" \
  --port "$LLAMACPP_PORT" \
  ${API_KEY_ARG}
