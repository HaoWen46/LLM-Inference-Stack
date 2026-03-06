#!/usr/bin/env bash
# download_model.sh — pull model weights from HuggingFace
#
# Downloads to MODEL_STAGE_DIR first (fast local disk), then rsyncs to
# MODEL_CACHE_DIR (final destination, e.g. NFS). If MODEL_STAGE_DIR is
# not set, downloads directly to MODEL_CACHE_DIR.
#
# Usage:
#   ./scripts/download_model.sh                          # uses MODEL_NAME from config/.env (required)
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

MODEL_NAME="${1:-${MODEL_NAME:-}}"
if [[ -z "$MODEL_NAME" ]]; then
    echo "error: MODEL_NAME not set. Pass it as an argument or set MODEL_NAME in config/.env." >&2
    exit 1
fi
GGUF_FILE="${2:-}"
FINAL_DIR="${MODEL_CACHE_DIR:-$ROOT_DIR/models}"
STAGE_DIR="${MODEL_STAGE_DIR:-}"

# If no staging dir configured, download directly to final destination
if [[ -z "$STAGE_DIR" ]]; then
    WORK_DIR="$FINAL_DIR"
else
    WORK_DIR="$STAGE_DIR"
fi

mkdir -p "$WORK_DIR"

echo "[download] Model : $MODEL_NAME"
if [[ -n "$STAGE_DIR" ]]; then
    echo "[download] Stage : $WORK_DIR  (fast scratch)"
    echo "[download] Final : $FINAL_DIR"
else
    echo "[download] Dest  : $WORK_DIR"
fi

# ── Single-file GGUF download (no huggingface_hub required) ──────────────────
if [[ -n "$GGUF_FILE" ]]; then
    STAGE_DEST_DIR="$WORK_DIR/$MODEL_NAME"
    mkdir -p "$STAGE_DEST_DIR"
    URL="https://huggingface.co/$MODEL_NAME/resolve/main/$GGUF_FILE"
    STAGE_DEST="$STAGE_DEST_DIR/$GGUF_FILE"
    echo "[download] File  : $GGUF_FILE"
    echo "[download] URL   : $URL"
    wget -c --progress=dot:giga "$URL" -O "$STAGE_DEST"

    if [[ -n "$STAGE_DIR" ]]; then
        FINAL_DEST_DIR="$FINAL_DIR/$MODEL_NAME"
        mkdir -p "$FINAL_DEST_DIR"
        echo "[download] Moving to final: $FINAL_DEST_DIR/$GGUF_FILE"
        rsync -a --info=progress2 "$STAGE_DEST" "$FINAL_DEST_DIR/$GGUF_FILE"
        rm -f "$STAGE_DEST"
        echo "[download] Done: $FINAL_DEST_DIR/$GGUF_FILE"
    else
        echo "[download] Saved: $STAGE_DEST"
    fi
    exit 0
fi

# ── Full repo snapshot via huggingface_hub ────────────────────────────────────
cd "$ROOT_DIR"

if [[ -z "$STAGE_DIR" ]]; then
    # No staging: download directly to final destination in one shot
    uv run python3 - <<EOF
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None
model = "$MODEL_NAME"
local_dir = os.path.join("$WORK_DIR", model)

print(f"Downloading {model} -> {local_dir}")
path = snapshot_download(
    repo_id=model,
    local_dir=local_dir,
    token=token,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "*.ot", "rust_model*"],
)
print(f"Saved to: {path}")
EOF
else
    # Staging mode: download each file to fast scratch, move to NFS immediately,
    # so the slow NFS write overlaps with the next file downloading.
    STAGE_MODEL_DIR="$WORK_DIR/$MODEL_NAME"
    FINAL_MODEL_DIR="$FINAL_DIR/$MODEL_NAME"
    mkdir -p "$STAGE_MODEL_DIR" "$FINAL_MODEL_DIR"

    uv run python3 - <<EOF
import os, fnmatch, shutil
from huggingface_hub import list_repo_files, hf_hub_download

token      = os.environ.get("HF_TOKEN") or None
model      = "$MODEL_NAME"
stage_dir  = "$STAGE_MODEL_DIR"
final_dir  = "$FINAL_MODEL_DIR"
ignore     = ["*.msgpack", "flax_model*", "tf_model*", "*.ot", "rust_model*"]

print(f"Listing files for {model}...")
all_files = list(list_repo_files(model, token=token))
files = [f for f in all_files if not any(fnmatch.fnmatch(f, pat) for pat in ignore)]
print(f"Found {len(files)} files ({len(all_files) - len(files)} ignored)")

for i, filename in enumerate(files, 1):
    dest = os.path.join(final_dir, filename)
    if os.path.exists(dest):
        print(f"[{i}/{len(files)}] skip (exists): {filename}")
        continue

    print(f"[{i}/{len(files)}] downloading : {filename}")
    stage_path = hf_hub_download(
        repo_id=model,
        filename=filename,
        local_dir=stage_dir,
        token=token,
    )

    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    shutil.move(stage_path, dest)
    print(f"[{i}/{len(files)}] moved       : {filename}")

# Clean up any leftover empty dirs / metadata in staging
shutil.rmtree(stage_dir, ignore_errors=True)
print(f"Done: {final_dir}")
EOF
fi
