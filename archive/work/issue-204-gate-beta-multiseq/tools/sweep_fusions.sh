#!/bin/bash
# Full multi-sequence sweep over the GDN gate/beta fusion family (issue #204 follow-up).
#
# For every switch:
#   * run the lockstep probe with the switch ON (default) and OFF, diff the per-sequence logits
#     (a "DIFFERS" is a multi-sequence shape bug in that arm);
#   * diff the *fusion-skip signature* of the two runs (GGML_CUDA_SKIP_DEBUG=1 build) - a changed
#     signature proves the arm actually fired, so an "identical" logits result is not vacuous.
#
# usage: sweep_fusions.sh <model.gguf> <n_seqs> [layer|tensor]
set -u
MODEL="$1"
NSEQ="${2:-3}"
SPLIT="${3:-layer}"
OUT="$HOME/scratch/fix-204/sweep"
mkdir -p "$OUT"
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib:${LIBDIR:+$LIBDIR:}$HOME/llama.cpp/build-rocm/bin
export ROCBLAS_USE_HIPBLASLT=0
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2}
export SPLIT

SWITCHES=(
  GGML_CUDA_FUSE_GATE_BETA_BF16
  GGML_CUDA_FUSE_GATE_BETA_VERIFY
  GGML_CUDA_FUSE_GDN_GATE
  GGML_CUDA_FUSE_GDN_BETA_SIGMOID
  GGML_CUDA_FUSE_GDN_CONV_NORM
  GGML_CUDA_FUSE_CONT_GATE_Q8_1
  GGML_CUDA_FUSE_GDN_NORM_GATE
)

TAG=$(basename "$MODEL" .gguf)_ns${NSEQ}_${SPLIT}${LIBDIR:+_$(basename "$LIBDIR")}
PROBE="$HOME/scratch/fix-204/multiseq_probe"

run() { # $1 label, $2.. env
    local label="$1"; shift
    env "$@" "$PROBE" "$MODEL" "$NSEQ" 4 > "$OUT/$TAG.$label.txt" 2> "$OUT/$TAG.$label.log" || echo "  !! $label failed"
}
sig() { grep skipdbg "$1" | sed 's/i=[0-9]* //; s/-[0-9]*)/)/' | sort | uniq -c; }

[ -s "$OUT/$TAG.on.txt" ] || run on
echo "=== $TAG"
for sw in "${SWITCHES[@]}"; do
    run "$sw.off" "$sw=0"
    logits=DIFFERS;  cmp -s "$OUT/$TAG.on.txt" "$OUT/$TAG.$sw.off.txt" && logits=identical
    fused=no
    if ! diff -q <(sig "$OUT/$TAG.on.log") <(sig "$OUT/$TAG.$sw.off.log") > /dev/null; then fused=yes; fi
    printf "  %-34s fired=%-3s logits=%s\n" "$sw" "$fused" "$logits"
done
