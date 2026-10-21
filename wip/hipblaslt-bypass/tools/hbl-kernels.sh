#!/usr/bin/env bash
# Which hipBLASLt kernel did THIS start pick? -- kernel-trace companion to hbl-classes.sh.
#
# Runs N fresh `llama-perplexity` starts under `rocprofv3 --kernel-trace`, and prints, per start,
# the logits class (hash of the dump) and the TensileLite kernel(s) hipBLASLt actually dispatched.
# The class is decided by that kernel -- see ../FINDINGS-2026-10-21-todo-75.md §3.
#
# Usage:
#   ./hbl-kernels.sh [N]        (default N=4; each start costs a trace, keep N small)
#
# Environment: as hbl-classes.sh (M, CORPUS, CTX, CHUNKS, NGL, FA, THREADS, PPL, OUTDIR, ROCM_LIB),
# plus:
#   ROCPROF   rocprofv3 binary.  DEFAULT IS THE ROCm 10.1 ONE ON PURPOSE: /usr/bin/rocprofv3 (the
#             7.x system install) runs but writes NO CSV against a 10.x app, silently.
#
# Note: the tracer perturbs timing/heap, so the classes here generally differ from an untraced run.
# That is fine -- what this reports is the class<->kernel MAPPING, not a flip rate.
set -uo pipefail

N="${1:-4}"
M="${M:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
CORPUS="${CORPUS:-/llm/models/wikitext-2-raw/wiki.test.raw}"
CTX="${CTX:-512}"; CHUNKS="${CHUNKS:-1}"; NGL="${NGL:-99}"; FA="${FA:-on}"; THREADS="${THREADS:-8}"
PPL="${PPL:-$HOME/llama.cpp/build-rocm/bin/llama-perplexity}"
OUTDIR="${OUTDIR:-/var/tmp/hbl-kernels}"
ROCM_LIB="${ROCM_LIB:-}"
DEVICE="${HIP_VISIBLE_DEVICES:-0}"
ROCPROF="${ROCPROF:-/opt/rocm-10.1.0-gfx120X/bin/rocprofv3}"
PYTHON="${PYTHON:-python3}"

[ -x "$PPL" ]    || { echo "no llama-perplexity at $PPL" >&2; exit 2; }
[ -x "$ROCPROF" ]|| { echo "no rocprofv3 at $ROCPROF (set ROCPROF=)" >&2; exit 2; }
mkdir -p "$OUTDIR"
LP="$ROCM_LIB"; [ -z "$LP" ] && LP="${LD_LIBRARY_PATH:-$HOME/llama.cpp/build-rocm/bin}"

for i in $(seq 1 "$N"); do
    d="$OUTDIR/run$i"; rm -rf "$d"; mkdir -p "$d"
    HIP_VISIBLE_DEVICES="$DEVICE" LD_LIBRARY_PATH="$LP" timeout 3600 \
        "$ROCPROF" --kernel-trace -f csv -d "$d" -- \
        "$PPL" -m "$M" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" -ngl "$NGL" -fa "$FA" -t "$THREADS" \
        --kl-divergence-base "$d/logits.kld" >"$d/run.out" 2>&1
    [ -f "$d/logits.kld" ] || { echo "run$i: no logits dump (see $d/run.out)" >&2; continue; }
done

OUTDIR="$OUTDIR" "$PYTHON" - <<'PY'
import collections, csv, glob, hashlib, os, sys
outdir = os.environ["OUTDIR"]
for d in sorted(glob.glob(os.path.join(outdir, "run*")), key=lambda p: int(p.rsplit("run", 1)[1])):
    kld = os.path.join(d, "logits.kld")
    if not os.path.exists(kld):
        continue
    h = hashlib.sha256(open(kld, "rb").read()).hexdigest()[:12]
    csvs = glob.glob(os.path.join(d, "**", "*_kernel_trace.csv"), recursive=True)
    if not csvs:
        print(f"{os.path.basename(d)}  logits={h}  NO KERNEL TRACE (wrong rocprofv3?)")
        continue
    rows = list(csv.DictReader(open(csvs[0])))
    c = collections.Counter(r["Kernel_Name"] for r in rows if r["Kernel_Name"].startswith("Cijk"))
    print(f"{os.path.basename(d)}  logits={h}")
    for k, v in c.items():
        spine = "_".join(k.split("_")[:6])            # Cijk_Alik_Bljk_S_B_Bias_HA
        mt = next((t for t in k.split("_") if t.startswith("MT")), "MT?")
        print(f"    {v:5d} x {spine} [{mt}]")
PY
