#!/usr/bin/env bash
# One measurement arm for the ngram-mod x host-experts campaign (TODO #59, closed in r123).
#   run-arm.sh <label> <spec-type> [extra llama-cli args...]
# Prints the llama-cli timings + the -v graph/cache counters for the arm.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LLAMA=${LLAMA:-$HOME/llama.cpp}
BIN=${BIN:-$LLAMA/build-rocm/bin/llama-cli}
MODEL=${MODEL:-/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf}
DRAFT=${DRAFT:-/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
N=${N:-128}
PROMPT=${PROMPT:-$REPO/prompts/recall.txt}
OUTDIR=${OUTDIR:-$REPO/archive/work/ngram-mod-host-experts/logs}
mkdir -p "$OUTDIR"

label=$1; shift
spec=$1; shift

out="$OUTDIR/$label.log"
# LV unset/empty = no -lv (text-purity runs want the bare output); LV=3 = the stats lines
LVOPT=""
if [ -n "${LV+x}" ] && [ "${LV}" != "none" ]; then LVOPT="-lv ${LV}"; fi
HIP_VISIBLE_DEVICES=${GPUS:-0,1} timeout 1800 "$BIN" \
  -m "$MODEL" -md "$DRAFT" \
  -ngl 99 -ngld 99 -sm tensor -ncmoe "${NCMOE:-36}" \
  -c "${CTX:-16384}" -b 2048 -ub 512 \
  -t 8 --cpu-range 0-7 --cpu-strict 1 ${LVOPT} \
  -f "$PROMPT" -n "$N" --temp 0 --seed 42 \
  --reasoning off --no-display-prompt --single-turn \
  --spec-type "$spec" \
  "$@" 2>&1 | tee "$out"

echo "=== $label ($spec) ==="
grep -E "graph|reused|realloc|cache_report|moe_cache|prompt eval time|\beval time\b|total time|accept" "$out" | tail -25
