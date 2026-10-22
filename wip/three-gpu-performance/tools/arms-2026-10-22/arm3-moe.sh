#!/usr/bin/env bash
# Arm 3: MoE with host experts (-ncmoe) + expert cache active.
# Qwen3.6-35B-A3B Q8_0 (embedded MTP head), 2 GPU `-sm tensor -ncmoe 40`, q8_0 KV, cache auto.
set -u
source "$(dirname "$0")/lib.sh"
M35=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
CTX=32768 BATCH=2048 NGMATCH=45 KV=q8_0
export CTX BATCH NGMATCH KV
P=$REPO/prompts
for trim in 0 98304; do
  GPUS=0,1 run "moe-P-trim$trim" "$M35" "$P/prose-rdna-boosts.txt" off "$trim" 3000 -ncmoe 40 --log-verbosity 4
done
for trim in 0 98304; do
  GPUS=0,1 run "moe-K-trim$trim" "$M35" "$P/recall.txt" off "$trim" 3000 -ncmoe 40 --log-verbosity 4
done
echo "=== arm3 done $(date)"
