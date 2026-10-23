#!/bin/bash
# llama-cli arm runner for the r116 3-GPU campaign (dense 27B + controls).
# stdout = generated text only (hashable), stderr = the log. Prints a CSV row.
#   TAG=d-12-none GPUS=0,1,2 SPEC=none N=400 SM=tensor ./cli116.sh <model> <prompt> [extra args...]
# Env: GPUS SM SPEC NMAX TRIM CTX BATCH KV N REASONING NGMATCH ARPROF EXTRA
set -u
MODEL=${1:?usage: cli116.sh <model> <prompt> [extra]}
PROMPT=${2:?usage: cli116.sh <model> <prompt> [extra]}
shift 2
BIN=${BIN:-/home/stew675/llama.cpp/build-rocm/bin/llama-cli}
TAG=${TAG:-arm}
GPUS=${GPUS:-0,1}
SM=${SM:-tensor}
SPEC=${SPEC:-none}
NMAX=${NMAX:-7}
TRIM=${TRIM:-0}
CTX=${CTX:-32768}
BATCH=${BATCH:-2048}
KV=${KV:-f16}
N=${N:-400}
NSTART=${NSTART:-}
REASONING=${REASONING:-off}
NGMATCH=${NGMATCH:-}
ARPROF=${ARPROF:-}
EXTRA=${EXTRA:-}
LOGD=${LOGD:-/tmp/srr}; mkdir -p "$LOGD"

export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib:${LD_LIBRARY_PATH:-}
[ -n "$ARPROF" ] && export GGML_CUDA_AR_PROFILE=1

HIP_VISIBLE_DEVICES=$GPUS ROCBLAS_USE_HIPBLASLT=0 timeout 3600 \
  "$BIN" -m "$MODEL" -ngl 99 -sm "$SM" -mg 0 -fa on -ctk "$KV" -ctv "$KV" \
  -c "$CTX" -b "$BATCH" -ub "$BATCH" -t 8 -fit off \
  -n "$N" --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning "$REASONING" \
  --spec-type "$SPEC" ${NSTART:+--spec-draft-n-start $NSTART} \
  --spec-draft-mtp-vocab "$TRIM" ${NGMATCH:+--spec-ngram-mod-n-match $NGMATCH} \
  ${EXTRA} "$@" -f "$PROMPT" \
  > "$LOGD/${TAG}.txt" 2> "$LOGD/${TAG}.log"
rc=$?
# stdout carries the banner + spinner control frames + the perf line; hash only the generated text
read H L <<<"$(python3 - "$LOGD/${TAG}.txt" <<'PYX'
import re,sys,hashlib
s=open(sys.argv[1],encoding='utf-8',errors='replace').read()
i=s.rfind('[ Prompt:');  s=s[:i] if i>=0 else s
s=re.sub(r'(?:[-|/\\\\]\x08)+','',s); s=re.sub(r'\x08','',s); s=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',s)
# drop the load banner: the generated text starts after the last banner rule
j=s.rfind('▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀')
if j>=0: s=s[j+40:]
if '▀▀▀▀▀' in s: s=s.split('▀▀▀▀▀')[-1]
s=s.strip('\n')
print(hashlib.sha256(s.encode()).hexdigest()[:12], len(s))
PYX
)"
T=$(grep -o "\[ Prompt: [0-9.]* t/s | Generation: [0-9.]* t/s \]" "$LOGD/${TAG}.log" | tail -1)
PN=$(grep -oE "prompt eval time = *[0-9.]+ ms / *[0-9]+ tokens" "$LOGD/${TAG}.log" | tail -1)
ACC=""
if [ -n "$NGMATCH" ] || [ "$SPEC" != none ]; then
  ACC=$(grep -aoE "draft acceptance = [0-9.]+ \( *[0-9]+ accepted / *[0-9]+ generated\), mean len = *[0-9.]+" "$LOGD/${TAG}.log" | tail -1)
fi
echo "$TAG,gpus=$GPUS,sm=$SM,spec=$SPEC,trim=$TRIM,n=$N,rc=$rc,text_sha=$H,text_len=$L,$T,${ACC:-},${PN:-}"
[ "$rc" -ne 0 ] && echo "   !! rc=$rc  $(tail -2 "$LOGD/${TAG}.log" | tr '\n' ' ')"
exit 0
