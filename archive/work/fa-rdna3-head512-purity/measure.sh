#!/bin/bash
# Measure one arm of archive/work/fa-rdna3-head512-purity.  Usage: measure.sh <tag> [extra env pairs...]
# Records throughput, KLD against BOTH base files, plain PPL and the head-512 op oracle into $HOME/kld-<tag>-*.
set -u
tag="$1"; shift
BIN=${BIN:-$HOME/llama-kld/build-kld/bin}
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1100/lib
export ROCBLAS_USE_HIPBLASLT=0
export HIP_VISIBLE_DEVICES=0
M=/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
C=/llm/models/wikitext-2-raw/wiki.test.raw
BA=/home/stew675/b-fulltile.kld       # fully tile
BB=/home/stew675/b-r102-256only.kld   # delivery minus the head-512 rows
out=$HOME/kld-$tag
ppl() { "$BIN/llama-perplexity" -m "$M" -f "$C" -c 512 --chunks 40 -ngl 99 -fa on -t 8 "$@"; }

echo "== $tag: throughput (q8_0 K/V, pp2048 @ d32768)" | tee "$out.txt"
"$BIN/llama-bench" -m "$M" -p 2048 -n 0 -d 32768 -r 2 -fa on -ctk q8_0 -ctv q8_0 -t 8 2>/dev/null \
    | grep -E "pp2048" | tee -a "$out.txt"

echo "== $tag: KLD vs fully-tile base" | tee -a "$out.txt"
ppl --kl-divergence --kl-divergence-base "$BA" 2>/dev/null \
    | grep -E "Mean    KLD|Same top p|Mean PPL\(Q\)" | tee -a "$out.txt"

echo "== $tag: KLD vs 256-only base (marginal)" | tee -a "$out.txt"
ppl --kl-divergence --kl-divergence-base "$BB" 2>/dev/null \
    | grep -E "Mean    KLD|Same top p|Mean PPL\(Q\)" | tee -a "$out.txt"

echo "== $tag: plain PPL (f16 K/V)" | tee -a "$out.txt"
ppl 2>/dev/null | grep -E "Final estimate" | tee -a "$out.txt"

echo "== $tag: head-512 op oracle (NMSE vs CPU, tile vs this arm)" | tee -a "$out.txt"
for v in 0 1; do
    GGML_CUDA_FA_RDNA3_WMMA_512=$v FA_DBG_NMSE=1 "$BIN/test-backend-ops" -o FLASH_ATTN_EXT \
        -p 'hsk=512.*nb=(24|64)' 2>&1 \
        | grep -E "^\[NMSE\] FLASH_ATTN_EXT" \
        
        | sed "s/^/WMMA512=$v /" | tee -a "$out.txt"
done
echo "== done: $out.txt"
