#!/usr/bin/env bash
# download_model.sh — pull model weights from HuggingFace
#
# Usage:
#   ./scripts/download_model.sh                          # uses MODEL_NAME from .env or default
#   ./scripts/download_model.sh meta-llama/Llama-3.2-7B-Instruct
#   ./scripts/download_model.sh TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF glm-4.7-flash-claude-4.5-opus.q8_0.gguf
#
# When a second argument (filename) is given the script downloads that single
# GGUF file directly via wget, bypassing huggingface_hub / hf_xet entirely.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$ROOT_DIR/config/.env" ]]; then
    set -a; source "$ROOT_DIR/config/.env"; set +a
fi

MODEL_NAME="${1:-${MODEL_NAME:-meta-llama/Llama-3.2-7B-Instruct}}"
GGUF_FILE="${2:-}"
CACHE_DIR="${MODEL_CACHE_DIR:-$ROOT_DIR/models}"
mkdir -p "$CACHE_DIR"

echo "[download] Model : $MODEL_NAME"
echo "[download] Dest  : $CACHE_DIR"

# ── Single-file GGUF download (no huggingface_hub required) ──────────────────
if [[ -n "$GGUF_FILE" ]]; then
    DEST_DIR="$CACHE_DIR/$MODEL_NAME"
    mkdir -p "$DEST_DIR"
    URL="https://huggingface.co/$MODEL_NAME/resolve/main/$GGUF_FILE"
    DEST="$DEST_DIR/$GGUF_FILE"
    echo "[download] File  : $GGUF_FILE"
    echo "[download] URL   : $URL"
    wget -c --progress=dot:giga "$URL" -O "$DEST"
    echo "[download] Saved : $DEST"
    exit 0
fi

# ── Full repo snapshot via huggingface_hub ────────────────────────────────────
cd "$ROOT_DIR"
uv run python3 - <<EOF
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None
model = "$MODEL_NAME"
local_dir = os.path.join("$CACHE_DIR", model)

print(f"Downloading {model} -> {local_dir}")
path = snapshot_download(
    repo_id=model,
    local_dir=local_dir,
    token=token,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "*.ot", "rust_model*"],
)
print(f"Saved to: {path}")
EOF
