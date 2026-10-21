#!/bin/bash
# Drive the row sweep of archive/work/fa-rdna3-head512-purity: apply one row variant, rebuild, measure, record, revert.
# Usage: sweep.sh <tag> "<row args>"     e.g.  sweep.sh V1-fa64 "128, 1, 64, 64, 64, 64, 1, false"
set -eu
tag="$1"; row="$2"
WT=/home/stew675/llama-kld
WIP=/home/stew675/llama-cpp-rdna-boosts/archive/work/fa-rdna3-head512-purity
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1100/lib
cd "$WT"
python3 - "$row" <<'EOF'
import sys, re
row = sys.argv[1]
p = 'ggml/src/ggml-cuda/fattn-mma-f16.cuh'
s = open(p).read()
old = "        if (ncols == 32) { return fattn_mma_config(128, 1, 32, 64, 64, 64, 1, false); }"
assert s.count(old) == 1, "row not found"
new = "        if (ncols == 32) { return fattn_mma_config(%s); }" % row
open(p, 'w').write(s.replace(old, new))
print("applied:", new.strip())
EOF
git diff > "$WIP/variants/$tag.patch"
cmake --build build-kld --target llama-bench llama-perplexity -j 16 2>&1 | grep -E "error|Error" && exit 1
"$WIP/measure2.sh" "$tag" > /dev/null
grep -E "^=== |^-- |Mean    KLD|Same top p|pp2048|NMSE=" "$HOME/m2-$tag.txt"
git checkout ggml/src/ggml-cuda/fattn-mma-f16.cuh
python3 - <<'EOF'
# restore the VKQ_C sizing fix (it is part of the experiment tree, not of any variant)
p = 'ggml/src/ggml-cuda/fattn-mma-f16.cuh'
s = open(p).read()
old = "#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)\n    T_C_VKQ VKQ_C[DV % 32 != 0       ? DV/T_C_VKQ::J : DV/(2*T_C_VKQ::J)];"
new = """#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    // An fp32 C tile spans T_C_VKQ::J output columns, an fp16 one spans 2*T_C_VKQ::J (two wmma ops,
    // one per half).  Keying the size on DV % 32 only works for the fp32 rows whose DV is not a
    // multiple of 32 (head 80/112); a fp32 tile at DV % 32 == 0 needs the DV/T_C_VKQ::J size, which is
    // also the bound the fp32 rescale and combine loops use.  Identical to the old expression for
    // every fp16 row, so no shipped kernel changes.  (archive/work/fa-rdna3-head512-purity.)
    T_C_VKQ VKQ_C[std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]> ? DV/T_C_VKQ::J :
                  (DV % 32 != 0 ? DV/T_C_VKQ::J : DV/(2*T_C_VKQ::J))];"""
assert s.count(old) == 1
open(p, 'w').write(s.replace(old, new))
EOF
echo "reverted to the shipped row (sizing fix kept)"
