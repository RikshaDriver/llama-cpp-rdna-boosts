#!/bin/bash
# Field/MoE arm on r116, server-driven: one arm per invocation, prints a CSV row.
# Derived from archive/work/moe-verify-fusions/tools/field3.sh (same command line) with the
# knobs this campaign's §1 matrix needs (spec type, split mode, forced arena size).
#
#   GPUS=0,1,2 SM=tensor SPEC=draft-mtp-adaptive NCMOE=48 CACHE=23000 \
#     PROMPT=/tmp/srr/mixed30k.txt TAG=t3-adaptive-full ./field116.sh <build_dir>
#
# Env:
#   GPUS   HIP_VISIBLE_DEVICES (default 0,1)
#   SM     tensor|layer                                   (default tensor)
#   SPEC   none|draft-mtp|draft-mtp-adaptive              (default draft-mtp-adaptive)
#   NMAX   --spec-draft-n-max                             (default 7)
#   NCMOE  -ncmoe                                         (default 48)
#   CACHE  MOE_EXPERT_CACHE_MIB, "" = auto                (default auto)
#   N      n_predict                                      (default 1000)
#   CTX    -c (default 40000; 204800 was the r73 field config and OOMs the 3-GPU full/MTP arm)
#   PROMPT prompt file                                    (default /tmp/srr/mixed30k.txt)
#   TAG    log/tag name                                   (default "arm")
#   EXTRA  extra server args (e.g. "--fit off" is always passed)
#   ARPROF "1" sets GGML_CUDA_AR_PROFILE=1
set -u
BUILD=${1:?usage: field116.sh <build_dir>}
TAG=${TAG:-arm}
M=${M:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
D=${D-/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
GPUS=${GPUS:-0,1}
SM=${SM:-tensor}
SPEC=${SPEC:-draft-mtp-adaptive}
NMAX=${NMAX:-7}
NCMOE=${NCMOE:-48}
CACHE=${CACHE:-}
N=${N:-1000}
CTX=${CTX:-40000}
PROMPT=${PROMPT:-/tmp/srr/mixed30k.txt}
EXTRA=${EXTRA:-}
ARPROF=${ARPROF:-}
PORT=${PORT:-8942}
LOGD=${LOGD:-/tmp/srr}; mkdir -p "$LOGD"

[ -x "$BUILD/bin/llama-server" ] || { echo "no llama-server in $BUILD/bin"; exit 2; }
[ -r "$PROMPT" ] || { echo "no prompt $PROMPT"; exit 2; }

export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=$GPUS
export ROCBLAS_USE_HIPBLASLT=0
[ -n "$CACHE" ] && export MOE_EXPERT_CACHE_MIB=$CACHE
[ -n "$ARPROF" ] && export GGML_CUDA_AR_PROFILE=1

DRAFT_ARGS=()
SPEC_ARGS=(--spec-type "$SPEC")
if [ "$SPEC" != none ]; then
    [ -n "${D:-}" ] && DRAFT_ARGS=(--spec-draft-model "$D")
    SPEC_ARGS+=(--spec-draft-n-max "$NMAX")
fi

"$BUILD/bin/llama-server" -v \
  -m "$M" ${DRAFT_ARGS[@]+"${DRAFT_ARGS[@]}"} \
  -sm "$SM" -ncmoe "$NCMOE" -ub 6144 -b 6144 -c "$CTX" --no-kv-unified \
  -ctk q8_0 -ctv q8_0 -fa on -t 8 --fit off \
  "${SPEC_ARGS[@]}" ${EXTRA} \
  --host 127.0.0.1 --port "$PORT" --no-webui --cache-ram 0 \
  > "$LOGD/${TAG}.out" 2> "$LOGD/${TAG}.err" &
pid=$!

ok=0
for i in $(seq 1 420); do kill -0 $pid 2>/dev/null || break
  if curl -sf --max-time 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then ok=1; break; fi; sleep 1; done
if [ $ok -ne 1 ]; then
  echo "$TAG,FAIL,health-timeout,,,,"; grep -aE 'error|GGML_ASSERT|out of memory' "$LOGD/${TAG}.err" | tail -4
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; exit 1
fi

# prime (warm the lazy paths, size nothing): tiny request first
curl -s --max-time 600 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","n_predict":8,"temperature":0,"cache_prompt":false}' >/dev/null

NS=$(python3 -c "print($N)")
python3 - "$PORT" "$NS" "$PROMPT" > "$LOGD/${TAG}.timing" 2>&1 <<'PY'
import json,sys,urllib.request
port,npre,pf=sys.argv[1],int(sys.argv[2]),sys.argv[3]
prompt=open(pf).read()
req=urllib.request.Request("http://127.0.0.1:%s/completion"%port,
  data=json.dumps({"prompt":prompt,"n_predict":npre,"temperature":0,"cache_prompt":False}).encode(),
  headers={"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=3600))
t=r.get("timings",{})
import hashlib
c=r.get("content","") or ""
print("prompt_n=%s prompt_tps=%.1f predicted_n=%s predicted_tps=%.1f text_sha=%s text_len=%d"%(
  t.get("prompt_n"),t.get("prompt_per_second",0),t.get("predicted_n"),t.get("predicted_per_second",0),
  hashlib.sha256(c.encode()).hexdigest()[:12], len(c)))
open("/tmp/srr/%s.text"%__import__('os').environ.get('TAG','arm'),"w").write(c)
PY
# graceful stop FIRST: the cache summary (h=, arena=, residency) is printed at atexit
kill -TERM $pid 2>/dev/null
for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null
TIM=$(cat "$LOGD/${TAG}.timing")
ACC=$(grep -aoE 'draft acceptance = [0-9.]+ \( *[0-9]+ accepted / *[0-9]+ generated\), mean len = *[0-9.]+' "$LOGD/${TAG}.err" | tail -1)
H=$(grep -aoE 'MoE expert cache: h=[0-9.]+ \([0-9]+/[0-9]+ reaches\), fills=[0-9]+ evictions=[0-9]+.*arena=[0-9.]+ MiB total \(budget [^)]*\)' "$LOGD/${TAG}.err" | tail -1)
RES=$(grep -aoE 'MoE expert cache \(auto\): arena [0-9.]+ MiB of [0-9.]+ MiB host experts \([0-9.]+% residency\)' "$LOGD/${TAG}.err" | tail -1)
EV=$(grep -acE 'moe_cache_evict_slab_range|released the MoE expert cache arena' "$LOGD/${TAG}.err")
echo "$TAG,gpus=$GPUS,sm=$SM,spec=$SPEC,ncmoe=$NCMOE,cache=${CACHE:-auto},$TIM,${ACC:-no-acceptance},${H:-no-h},${RES:-no-auto-res},evict_lines=$EV"
