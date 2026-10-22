#!/usr/bin/env bash
# Four-axis adaptive gate at 2 and 3 GPU, trim 0 vs 98304 (n-max 7, n-start default), canonical 4-axis config.
set -u
source "$(dirname "$0")/lib.sh"
CTX=32768 BATCH=2048 NGMATCH=45
export CTX BATCH NGMATCH
P=$REPO/prompts
for g in 0,1 0,1,2; do
  tag=$(echo $g | tr -d ',')
  for trim in 0 98304; do
    for ax in R P C K; do
      case $ax in
        R) pf=$P/reasoning.txt;        rea=on;;
        P) pf=$P/prose-rdna-boosts.txt; rea=off;;
        C) pf=$P/code-python.txt;      rea=off;;
        K) pf=$P/recall.txt;           rea=off;;
      esac
      GPUS=$g run "g4-${tag}-$ax-trim$trim" "$D27" "$pf" "$rea" "$trim" 3000 --log-verbosity 4
    done
  done
done
echo "=== gate4 done $(date)"
