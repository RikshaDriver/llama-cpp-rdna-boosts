#!/usr/bin/env bash
# Fresh-process logits-class counter -- the determinism instrument for wip/hipblaslt-bypass.
#
# WHY: a greedy-text compare can be identical 10/10 across arms while the prefill logits are
# already in different classes (see ../FINDINGS-2026-10-21-todo-75.md §4).  This counts classes
# at the logits level: one fresh process per sample, hash the `llama-perplexity` logits dump.
#
# The arm is whatever the environment says, so the hipBLASLt control is the same command:
#   ./hbl-classes.sh                                  # default (hipBLASLt on)
#   ROCBLAS_USE_HIPBLASLT=0 ./hbl-classes.sh          # the control: expect ONE class, N/N
#
# Usage:
#   ./hbl-classes.sh [N]           (default N=8 samples)
#
# Environment:
#   M        model gguf   (default: 35B-A3B Q4_K_M)
#   CORPUS   corpus       (default /llm/models/wikitext-2-raw/wiki.test.raw)
#   CTX      default 512      CHUNKS default 1     NGL default 99
#   FA       default on       THREADS default 8
#   PPL      llama-perplexity binary (default ~/llama.cpp/build-rocm/bin/llama-perplexity)
#   OUTDIR   where the per-sample dumps go (default /var/tmp/hbl-classes; ~126 MiB each)
#   KEEP=1   keep every dump (default: keep the first of each class only)
#   ROCM_LIB override LD_LIBRARY_PATH's ROCm root (e.g. /opt/rocm-10.0.0-gfx120X/lib) to run the
#            same binary against another runtime's hipBLASLt -- how the 10.0.0 arm was taken.
#
# Needs: a ROCm GPU, ~1-2 GB of disk per kept dump, one GPU job at a time (never run beside a bench).
set -uo pipefail

N="${1:-8}"
M="${M:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
CORPUS="${CORPUS:-/llm/models/wikitext-2-raw/wiki.test.raw}"
CTX="${CTX:-512}"; CHUNKS="${CHUNKS:-1}"; NGL="${NGL:-99}"; FA="${FA:-on}"; THREADS="${THREADS:-8}"
PPL="${PPL:-$HOME/llama.cpp/build-rocm/bin/llama-perplexity}"
OUTDIR="${OUTDIR:-/var/tmp/hbl-classes}"
KEEP="${KEEP:-0}"
ROCM_LIB="${ROCM_LIB:-}"
DEVICE="${HIP_VISIBLE_DEVICES:-0}"

[ -x "$PPL" ] || { echo "no llama-perplexity at $PPL (set PPL=)" >&2; exit 2; }
mkdir -p "$OUTDIR"

LP="$ROCM_LIB"
if [ -z "$LP" ] && [ -n "${LD_LIBRARY_PATH:-}" ]; then LP="$LD_LIBRARY_PATH"; else LP="${LP:-$HOME/llama.cpp/build-rocm/bin}"; fi

declare -A seen=()
order=()

echo "arm: ROCBLAS_USE_HIPBLASLT=${ROCBLAS_USE_HIPBLASLT:-<default/on>}  libs=${LP}  N=${N}" >&2
for i in $(seq 1 "$N"); do
    f="$OUTDIR/sample$i.kld"
    rm -f "$f"
    HIP_VISIBLE_DEVICES="$DEVICE" LD_LIBRARY_PATH="$LP" timeout 1800 \
        "$PPL" -m "$M" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" -ngl "$NGL" -fa "$FA" -t "$THREADS" \
        --kl-divergence-base "$f" >"$OUTDIR/sample$i.log" 2>&1 \
        || { echo "sample$i: RUN FAILED (see $OUTDIR/sample$i.log)" >&2; continue; }
    [ -f "$f" ] || { echo "sample$i: no logits file written" >&2; continue; }
    h="$(sha256sum "$f" | cut -c1-12)"
    seen["$h"]=$(( ${seen["$h"]:-0} + 1 ))
    [ "$KEEP" = 1 ] || { [ "${seen["$h"]}" -eq 1 ] && order+=("$h") || rm -f "$f"; }
    printf 'sample%-3d %s\n' "$i" "$h"
done

echo
echo "== distinct logits classes: ${#seen[@]} (N=$N) =="
for h in "${!seen[@]}"; do printf '  %s  x%d\n' "$h" "${seen[$h]}"; done | sort -k2
[ "${#seen[@]}" -eq 1 ] && echo "RESULT: DETERMINISTIC across fresh starts" \
                        || echo "RESULT: NONDETERMINISTIC (per-start class lottery)"
