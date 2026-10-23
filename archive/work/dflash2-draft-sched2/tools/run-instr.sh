#!/bin/bash
# run-instr.sh <label> <lib> <n_gen> <gpus> <env-string> [extra args...]
# same as run-arm.sh but with -lv 4 --log-file, so the DRAFT_GRAPH trace is captured
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
esac
out=$OUT/$label
HIP_VISIBLE_DEVICES=$gpus timeout 2400 env LLAMA_DRAFT_GRAPH_DEBUG=1 $envs $BIN \
    -m "${MODEL:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf}" \
    -md "${DRAFT:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}" \
    --spec-type "${SPEC:-draft-dflash}" \
    -ngl 99 -sm ${SM:-tensor} -mg ${MG:-0} -c ${CTX:-16384} -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 -t 8 \
    -s 42 -n "$ngen" --temp 0 --no-display-prompt --single-turn -lv 4 --log-file "$out.log" \
    -f "$PROMPT" "$@" > "$out.out" 2> "$out.err"
rc=$?
footer=$(grep -o '\[ Prompt:.*\]' "$out.out" | tail -1)
txt=$(/home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py "$out.out" 2>/dev/null || echo "EXTRACT-FAIL")
python3 - "$out.log" <<'PY'
import re, sys, collections
rows = [l for l in open(sys.argv[1], errors='replace') if 'DRAFT_GRAPH' in l]
# only the decode phase: the last 400 draft ubatches
tail = rows[-400:]
kinds = collections.Counter()
reuse = collections.Counter()
for l in tail:
    m = re.search(r'outputs=(\d+) embd=(\d+) inject=(\d+).*slot_alt=(\d+) reused=(\d+)', l)
    if not m: continue
    out, embd, inj, slot, r = m.groups()
    kind = 'inject' if (embd == '1' or inj == '1') else ('step-out' if out != '0' else 'step-tok')
    kinds[kind] += 1
    reuse[(kind, slot)] += int(r)
tot = sum(kinds.values())
print('  trace: %d ubatches (last 400): %s | reused: %s (%.1f %%)' % (
    tot, dict(kinds), dict(reuse), 100.0*sum(reuse.values())/max(1,tot)))
PY
printf 'RESULT %-22s rc=%d %s | %s\n' "$label" "$rc" "$footer" "$txt"
