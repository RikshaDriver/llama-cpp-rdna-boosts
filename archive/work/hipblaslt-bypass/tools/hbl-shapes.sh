#!/usr/bin/env bash
# hipBLASLt shape inventory -- which GEMMs does a run actually send through hipBLASLt?
#
# This is campaign step P0 (archive/work/hipblaslt-bypass/README.md §4): the (shape, calls) table below IS
# the scope of any "bypass hipBLASLt" work, because ggml-cuda only reaches hipBLASLt through
# rocBLAS's non-quantized `mul_mat` fallback (ggml_cuda_mul_mat_cublas) for batches wider than the
# native GEMM cap (mmf.cu: `src1_ncols > 16 -> false`).
#
# Usage:
#   ./hbl-shapes.sh -- <any llama.cpp command>            # e.g.
#   ./hbl-shapes.sh -- ~/llama.cpp/build-rocm/bin/llama-cli -m M -ngl 99 -n 32 -f prompt.txt ...
#   ./hbl-shapes.sh --log FILE                            # just aggregate an existing log
#
# Environment:
#   HBL_LOG  where to put the hipBLASLt log when running (default /var/tmp/hbl-shapes.log)
#
# The command runs with HIPBLASLT_LOG_LEVEL=4 HIPBLASLT_LOG_MASK=1 HIPBLASLT_LOG_FILE=$HBL_LOG,
# so LD_LIBRARY_PATH must already point at the ROCm you mean to inspect.
set -uo pipefail

mode="run"; log="${HBL_LOG:-/var/tmp/hbl-shapes.log}"
case "${1:-}" in
    --log) mode="agg"; log="$2"; shift 2 ;;
    --)    shift ;;
    *)     [ $# -eq 0 ] && { sed -n '2,20p' "$0"; exit 2; } ;;
esac

if [ "$mode" = run ]; then
    [ $# -gt 0 ] || { echo "no command given (use: $0 -- <cmd...>)" >&2; exit 2; }
    rm -f "$log"
    HIPBLASLT_LOG_LEVEL=4 HIPBLASLT_LOG_MASK=1 HIPBLASLT_LOG_FILE="$log" "$@" || true
fi
[ -f "$log" ] || { echo "no log at $log" >&2; exit 2; }

HBL_LOG="$log" python3 - <<'PY'
import collections, os, re
log = os.environ["HBL_LOG"]
c = collections.Counter()
for L in open(log, errors="replace"):
    if "rocblaslt_matmul]" not in L:
        continue
    m = re.search(r"Adesc=\[type=(\S+) rows=(\d+) cols=(\d+)", L)
    b = re.search(r"Bdesc=\[type=(\S+) rows=(\d+) cols=(\d+)", L)
    t = re.search(r"transA=(\S+) transB=(\S+) epilogue=(\S+)", L)
    if m and b and t:
        c[(m.group(1), int(m.group(2)), int(m.group(3)),
           b.group(1), int(b.group(2)), int(b.group(3)),
           t.group(1), t.group(2), t.group(3))] += 1
total = sum(c.values())
print(f"{total} hipBLASLt dispatches, {len(c)} distinct shapes   ({log})")
print(f"{'calls':>7}  A[type rows x cols]        B[type rows x cols]        op           epilogue")
for k, v in c.most_common():
    print(f"{v:7d}  A[{k[0]:6s} {k[1]:5d} x {k[2]:4d}]   B[{k[3]:6s} {k[4]:5d} x {k[5]:4d}]   "
          f"{k[6]:3s},{k[7]:3s}  {k[8]}")
PY
