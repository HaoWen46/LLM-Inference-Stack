#!/usr/bin/env bash
# download_model.sh — pull model weights from HuggingFace
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$ROOT_DIR/config/.env" ]]; then
    set -a; source "$ROOT_DIR/config/.env"; set +a
fi

MODEL_NAME="${1:-${MODEL_NAME:-meta-llama/Llama-3.2-7B-Instruct}}"
CACHE_DIR="${MODEL_CACHE_DIR:-$ROOT_DIR/models}"
mkdir -p "$CACHE_DIR"

source "$ROOT_DIR/.venv/bin/activate"

echo "[download] Model: $MODEL_NAME"
echo "[download] Destination: $CACHE_DIR"

python3 - <<EOF
import os, sys
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None
model = "$MODEL_NAME"
cache = "$CACHE_DIR"

print(f"Downloading {model} ...")
path = snapshot_download(
    repo_id=model,
    cache_dir=cache,
    token=token,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "*.ot", "rust_model*"],
)
print(f"Saved to: {path}")
EOF
