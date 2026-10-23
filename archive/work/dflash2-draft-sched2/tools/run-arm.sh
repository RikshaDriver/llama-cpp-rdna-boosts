#!/bin/bash
# run-arm.sh <label> <lib: fix|base|asis> <n_gen> <gpus> <env-string> [extra llama-cli args...]
# one arm = one llama-cli run; writes $OUT/<label>.{out,err} and prints one RESULT line
set -u

LLAMA=/home/stew675/llama.cpp
BIN=$LLAMA/build-rocm/bin/llama-cli
PROMPT=/home/stew675/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt
OUT=${OUT:-/tmp/df2/out}
mkdir -p "$OUT"

label=$1; lib=$2; ngen=$3; gpus=$4; envs=$5; shift 5
case "$lib" in
    fix)  cp -f /tmp/df2/libllama-fix.so  $LLAMA/build-rocm/bin/libllama.so.0.6.0 ;;
    base) cp -f /tmp/df2/libllama-base.so $LLAMA/build-rocm/bin/libllama.so.0.6.0 ;;
    asis) : ;;
    *) echo "bad lib: $lib" >&2; exit 2 ;;
esac

MODEL=${MODEL:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf}
DRAFT=${DRAFT:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}
SPEC=${SPEC:-draft-dflash}

out=$OUT/$label
HIP_VISIBLE_DEVICES=$gpus timeout 2400 env $envs $BIN \
    -m "$MODEL" -md "$DRAFT" --spec-type "$SPEC" \
    -ngl 99 -sm ${SM:-tensor} -mg ${MG:-0} -c ${CTX:-16384} -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 -t 8 \
    -s 42 -n "$ngen" --temp 0 --no-display-prompt --single-turn \
    -f "$PROMPT" "$@" \
    > "$out.out" 2> "$out.err"
rc=$?
footer=$(grep -o '\[ Prompt:.*\]' "$out.out" | tail -1)
txt=$(/home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py "$out.out" 2>/dev/null || echo "EXTRACT-FAIL")
acc=$(grep -o 'draft acceptance = [0-9.]* ([^)]*)' "$out.err" "$out.out" 2>/dev/null | tail -1)
printf 'RESULT %-26s rc=%d %s | %s\n' "$label" "$rc" "$footer" "$txt"
