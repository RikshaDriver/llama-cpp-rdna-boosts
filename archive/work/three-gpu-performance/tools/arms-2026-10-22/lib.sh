#!/usr/bin/env bash
# Shared runner for the TODO #83 remaining-arms validation (r109 build).
# Canonical 4-axis command (benchmarks/2026-09-13-adaptive-mtp-4-axis.md "Reproducing"):
#   -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt -c 32768 -b 2048 -ub 2048
#   -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4     (default sampling; --reasoning on for R only)
REPO=/home/stew675/llama-cpp-rdna-boosts
BIN=/home/stew675/llama.cpp/build-rocm/bin/llama-cli
L=${ARMS_LOG_DIR:-/tmp/arms-2026-10-22}
D27=/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf

# run <logname> <model> <prompt-file> <reasoning:on|off> <trim> <n> <extra args...>
run() {
    local name=$1 model=$2 prompt=$3 rea=$4 trim=$5 n=$6; shift 6
    echo "=== $name  $(date +%H:%M:%S)  trim=$trim n=$n $*"
    HIP_VISIBLE_DEVICES=${GPUS:-0,1,2} ROCBLAS_USE_HIPBLASLT=0 timeout 3600 \
        "$BIN" -m "$model" -ngl 99 -sm tensor -mg 0 -fa auto -ctk "${KV:-f16}" -ctv "${KV:-f16}" \
        -c "$CTX" -b "${BATCH:-2048}" -ub "${BATCH:-2048}" \
        -n "$n" --seed 42 --temp 0 --single-turn --no-display-prompt \
        --reasoning "$rea" \
        --spec-type "${SPECTYPE:-draft-mtp-adaptive,ngram-mod}" \
        ${NGMATCH+--spec-ngram-mod-n-match "$NGMATCH"} \
        --spec-draft-n-max "${NMAX:-7}" ${NSTART:+--spec-draft-n-start "$NSTART"} \
        --spec-draft-mtp-vocab "$trim" ${EXTRA:-} "$@" -f "$prompt" > "$L/$name.log" 2>&1
    local rc=$?
    printf '    rc=%s  ' "$rc"
    grep -o "draft acceptance = [0-9.]* ([^)]*), mean len = *[0-9.]*" "$L/$name.log" | tail -1
    grep -o "\[ Prompt: [0-9.]* t/s | Generation: [0-9.]* t/s \]" "$L/$name.log" | tail -1
    [ "$rc" -ne 0 ] && echo "    !! FAILED rc=$rc (tail: $(tail -2 "$L/$name.log" | tr '\n' ' '))"
    return 0
}
