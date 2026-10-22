#!/bin/bash
# Reproduction harness for TODO #65: IQ4_NL Flash-Next field config, --host-experts pool vs pinned.
# Usage: run-field.sh <tag> <hostmode> [prompt] [extra args...]
set -u
tag="$1"; mode="$2"; prompt="${3:-/tmp/prompt70k.txt}"; shift 3 2>/dev/null || shift 2
BIN=${BIN:-$HOME/llama.cpp/build-rocm/bin/llama-server}
HOSTFLAG=""
[ "$mode" = pool ] && HOSTFLAG="--host-experts pool"
[ "$mode" = pinned ] && HOSTFLAG="--host-experts pinned"
log="$HOME/llama-cpp-rdna-boosts/wip/iq4nl-field-pool-decode/logs/$tag.log"
port=${PORT:-8035}
env HIP_VISIBLE_DEVICES=${GPUS:-0,1} NCCL_PROXY_CPUSET=8,9,10,11,12,13,14,15 GGML_CUDA_ALLREDUCE=ce ${DIAG_EXTRA_ENV:-} \
  timeout ${TIMEOUT:-3600} "$BIN" \
  --model /llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf \
  --alias m --fit off --top-k 20 --port "$port" --threads 7 --parallel 1 --top-p 0.95 --min-p 0.001 \
  --verbosity ${VERB:-3} --host 127.0.0.1 --cpu-strict 1 --predict 98304 --cpu-range 1-7 --threads-http 4 \
  --load-mode ${DIAG_LOADMODE:-mmap} --cache-ram ${CACHERAM:-16384} --ctx-size ${CTX:-131072} \
  --flash-attn auto --temperature ${TEMP:-1.0} --seed ${SEED:-42} --batch-size 2048 --ubatch-size 2048 --n-gpu-layers all \
  --no-kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --ctx-checkpoints 64 --image-min-tokens 1024 \
  --cache-idle-slots --reasoning-budget 65536 --reasoning-preserve --checkpoint-min-step 4096 \
  --lazy-mode off --n-cpu-moe ${NCMOE:-48} --split-mode tensor --spec-type draft-mtp-adaptive \
  --spec-draft-n-max 7 --spec-draft-p-min 0.001 \
  --spec-draft-model /llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' $HOSTFLAG "$@" > "$log" 2>&1 &
pid=$!
echo "[$tag] pid=$pid loading..."
for i in $(seq 1 ${LOADWAIT:-900}); do grep -q "listening on" "$log" && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
if ! grep -q "listening on" "$log"; then echo "[$tag] FAILED TO START"; tail -30 "$log"; pkill -f "port $port" ; exit 1; fi
echo "[$tag] loaded; sending prompt $(wc -c < "$prompt") B"
iolog="$(dirname "$log")/$tag.io"; : > "$iolog"
( tgt=""
  while kill -0 $pid 2>/dev/null; do
    if [ -z "$tgt" ] || [ ! -r /proc/$tgt/io ]; then tgt=$(pgrep -P $pid | head -1); fi
    if [ -n "$tgt" ] && [ -r /proc/$tgt/io ]; then
      { grep -E "^(read_bytes|write_bytes|rchar)" /proc/$tgt/io | tr '\n' ' '; echo "t=$(date +%s.%N)"; } >> "$iolog"
    fi
    sleep 2
  done ) &
samp=$!
python3 - "$prompt" "${NTOK:-32}" "${TEMP:-1.0}" "${SEED:-42}" > /tmp/iq4nl-req-$tag.json <<'PY'
import json,sys
p=open(sys.argv[1]).read()
print(json.dumps({"model":"m","messages":[{"role":"user","content":p+"\n\nSummarise the above in one sentence."}],"max_tokens":int(sys.argv[2]),"temperature":float(sys.argv[3]),"seed":int(sys.argv[4])}))
PY
curl -s -m ${CURLTO:-1800} http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/iq4nl-req-$tag.json -o /tmp/iq4nl-resp-$tag.json
echo "[$tag] done; kill"
kill $pid 2>/dev/null
wait $pid 2>/dev/null
kill $samp 2>/dev/null
grep -E "prompt eval time|eval time =|host pool|prewarmed|re-ranked|arena .* MiB of|slab|ring|narrow|WORK_ALLOC|shortfall|draft acceptance|n_drafted|n_accept" "$log" | tail -40
