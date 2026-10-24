#!/usr/bin/env bash
# Fetch the Qwen3.8-Flash-Next DFlash2 head (Q4_K_M, 362 MB) and print its DFlash metadata.
# Scratch path on purpose: this is a wip/ campaign, nothing here belongs under /llm/models.
set -euo pipefail

DEST="${DEST:-/tmp/df2/head}"
REPO="incoai/Qwen3.8-Flash-Next-DFlash2-GGUF"
FILE="Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf"
LLAMA="${LLAMA_DIR:-/home/stew675/llama.cpp}"

mkdir -p "$DEST"
if [ ! -s "$DEST/$FILE" ]; then
    echo "fetching $REPO/$FILE -> $DEST"
    curl -L --fail -o "$DEST/$FILE" "https://huggingface.co/$REPO/resolve/main/$FILE"
fi
ls -la "$DEST/$FILE"

echo "--- DFlash metadata (llama-gguf) ---"
"$LLAMA/build-rocm/bin/llama-gguf" "$DEST/$FILE" r 2>&1 | grep -E "dflash\." | head -20 || \
    echo "(llama-gguf not built; use gguf-py)"

echo "--- expected: block_size 8 (so n-max <= 7), selector_top_k > 0 (DFlash2),"
echo "    target_layers [6,16,26,36,46], fc.weight [2560, 12800]"
