#!/bin/bash
# Cost one row variant of archive/work/fa-rdna3-head512-purity at BOTH measurement points.
# Usage: measure2.sh <tag>            (uses the current build)
# Writes $HOME/m2-<tag>.txt
set -u
tag="$1"
BIN=${BIN:-$HOME/llama-kld/build-kld/bin}
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1100/lib
export ROCBLAS_USE_HIPBLASLT=0
export HIP_VISIBLE_DEVICES=0
M=/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
W=/llm/models/wikitext-2-raw/wiki.test.raw
S=/home/stew675/selfcont.txt
WA=/home/stew675/b-fulltile.kld        # wikitext, fully tile
WB=/home/stew675/b-r102-256only.kld    # wikitext, delivery minus the 512 rows
SA=/home/stew675/s-A-fulltile.kld      # confident (PPL 13.7), fully tile
SB=/home/stew675/s-B-256only.kld       # confident, delivery minus the 512 rows
out=$HOME/m2-$tag.txt
: > "$out"
run() { "$BIN/llama-perplexity" -m "$M" -f "$1" -c 512 --chunks 40 -ngl 99 -fa on -t 8 "${@:2}" 2>/dev/null; }
kj() { grep -E "Mean    KLD|Same top p"; }

{
echo "=== arm $tag   (build $(date -Is))"
echo "-- throughput, q8_0 K/V, pp2048 @ d32768"
"$BIN/llama-bench" -m "$M" -p 2048 -n 0 -d 32768 -r 2 -fa on -ctk q8_0 -ctv q8_0 -t 8 2>/dev/null | grep pp2048
echo "-- wikitext (base PPL 607): KLD vs fully tile"
run "$W" --kl-divergence --kl-divergence-base "$WA" | kj
echo "-- wikitext: KLD vs 256-only (marginal)"
run "$W" --kl-divergence --kl-divergence-base "$WB" | kj
echo "-- confident self-continuation (base PPL 13.7): KLD vs fully tile"
run "$S" --kl-divergence --kl-divergence-base "$SA" | kj
echo "-- confident: KLD vs 256-only (marginal)"
run "$S" --kl-divergence --kl-divergence-base "$SB" | kj
echo "-- op oracle, head-512 prefill shape (kv=32768 nb=512 GQA8), NMSE vs CPU"
for v in 0 1; do
    FA_DBG_SEED=1 FA_DBG_NMSE=1 GGML_CUDA_FA_RDNA3_WMMA_512=$v "$BIN/test-backend-ops" -o FLASH_ATTN_EXT \
        -p 'hsk=512.*kv=32768.*nb=512' 2>/dev/null \
        | grep -E "^\[NMSE\] FLASH_ATTN_EXT" \
        | sed -E 's/^.*type_K=([a-z0-9_]+),type_V=[a-z0-9_]+.*err=([0-9.e+-]+).*/WMMA512='"$v"' type_K=\1 NMSE=\2/' \
        | sed "s/^/   /"
done
} | tee -a "$out"
