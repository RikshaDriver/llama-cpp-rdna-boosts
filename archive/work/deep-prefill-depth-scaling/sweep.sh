#!/bin/bash
set -u
MODEL=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
OUT=$HOME/up-ab
run() { # name binary extra_args...
  local name="$1" bin="$2"; shift 2
  echo "== $name ($bin)" > "$OUT/$name.log"
  HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=0 timeout 2400 "$bin" -m "$MODEL" -ngl 99 -fa 1 \
     -ctk q8_0 -ctv q8_0 -t 8 -b 2048 -ub 1024 -p 512 -n 0 \
     -d 0,32768,65536,98304 -r 1 "$@" >> "$OUT/$name.log" 2>&1
  echo "exit=$?" >> "$OUT/$name.log"
}
for build in r125 base head; do
  case $build in
    r125) BIN=$HOME/llama.cpp/build-rocm/bin/llama-bench;;
    base) BIN=$HOME/llama-up-base/build-rocm/bin/llama-bench;;
    head) BIN=$HOME/llama-up-head/build-rocm/bin/llama-bench;;
  esac
  run "${build}-resident" "$BIN"
  run "${build}-stream24" "$BIN" --n-cpu-moe 24
done
