#!/usr/bin/env bash
# Repro for the qwen4exp DFlash2 blocker (TODO #94): Flash-Next + the incoai DFlash2 head.
#
#   ./repro.sh target    target only (works: 432/432 identity fast path ON)
#   ./repro.sh head      target + head   (aborts at llama-context.cpp:3437 on r120)
#   ./repro.sh both      run both, in that order
#
# Traps this script guards against (both cost real time while the campaign was seeded):
#   * a stray llama process from an interrupted run holds ~33 GB/GPU on this box, and the next run then
#     fails with a 2257 MiB cudaMalloc in 0.3 s -- so check VRAM before starting;
#   * `--fit off` is NOT needed (measured): with clean VRAM the with-head arm loads through the normal
#     fit path and aborts later, in extract_layer_inp (the actual blocker).
set -uo pipefail

LLAMA="${LLAMA_DIR:-/home/stew675/llama.cpp}"
BIN="$LLAMA/build-rocm/bin/llama-server"
MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
HEAD="${HEAD:-/tmp/df2/head/Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf}"
GPUS="${GPUS:-0,1,2}"
PORT="${PORT:-8133}"
OUT="${OUT:-/tmp/df2/fn-repro}"
mkdir -p "$OUT"

vram_guard() {
    used=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -m1 "VRAM Total Used Memory" | grep -oE "[0-9]+$" || echo 0)
    if [ "${used:-0}" -gt 2000000000 ]; then
        echo "WARNING: device 0 already has ${used} B in use.  Stray processes?"
        pgrep -a llama | grep -v slot-proxy || true
        echo "Kill them (pkill -f build-rocm/bin/llama-) before trusting an OOM in the run below."
    fi
}

run_target() {
    echo "== target only"
    vram_guard
    HIP_VISIBLE_DEVICES="$GPUS" timeout 600 "$BIN" \
        -m "$MODEL" -ngl 99 -sm tensor -mg 0 -ncmoe 48 -c 16384 -ub 2048 -b 2048 \
        -fa on -ctk q8_0 -ctv q8_0 -t 8 -np 1 --port "$PORT" --no-webui -lv 3 \
        > "$OUT/target.log" 2>&1 &
    local srv=$!
    for _ in $(seq 1 240); do
        curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null && break
        sleep 1
    done
    echo "   identity tables: $(grep -c 'identity fast path ON' "$OUT/target.log") (expect 432)"
    echo "   health: $(curl -sf -m 3 "http://127.0.0.1:$PORT/health" || echo FAILED)"
    kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null
}

run_head() {
    echo "== target + DFlash2 head (expect the r120 abort at llama-context.cpp:3437)"
    vram_guard
    HIP_VISIBLE_DEVICES="$GPUS" timeout 600 "$BIN" \
        -m "$MODEL" -md "$HEAD" --spec-type draft-dflash --spec-draft-n-max 7 \
        -ngl 99 -sm tensor -mg 0 -ncmoe 48 -c 16384 -ub 2048 -b 2048 \
        -fa on -ctk q8_0 -ctv q8_0 -t 8 -np 1 --port "$PORT" --no-webui -lv 3 \
        > "$OUT/head.log" 2>&1
    echo "   rc=$?"
    grep -nE "GGML_ASSERT|draft acceptance|F1:" "$OUT/head.log" | tail -4
}

case "${1:-both}" in
    target) run_target ;;
    head)   run_head ;;
    both)   run_target; run_head ;;
    *)      echo "usage: $0 [target|head|both]" >&2; exit 2 ;;
esac
