#!/usr/bin/env bash
# Arm 2: depth-16384 decode. Prompt = prose-rdna-boosts.txt x3 (15746 tok, sha256 6fec9806b375...),
# -c 20480 so decode runs from depth ~15746 to ~18700.
set -u
source "$(dirname "$0")/lib.sh"
CTX=20480 BATCH=2048 NGMATCH=45
export CTX BATCH NGMATCH
P="$(dirname "$0")/prose16k.txt"
for g in 0,1 0,1,2; do
  tag=$(echo $g | tr -d ',')
  for trim in 0 98304; do
    GPUS=$g run "d16-${tag}-trim$trim" "$D27" "$P" off "$trim" 3000 --log-verbosity 4
  done
done
# purity at depth (3 GPU, hashable text, no -lv)
GPUS=0,1,2 SPECTYPE=none run "d16-012-none" "$D27" "$P" off 0 3000
for trim in 0 98304; do
  GPUS=0,1,2 SPECTYPE=draft-mtp run "d16-012-mtp-trim$trim" "$D27" "$P" off "$trim" 3000
done
echo "=== arm2 done $(date)"
