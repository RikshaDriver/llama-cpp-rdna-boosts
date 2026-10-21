# P3(ii) status — paused mid-diagnosis, 2026-10-21

**Tree state: clean.** `~/llama.cpp` is back to the release tree, both `build-rocm` (ROCm 10.1) and
`build-rocm-714` are rebuilt from it, and the mmf32 experiment is preserved as a patch —
**`tools/experiments/mmf32-splitk-WIP.patch`** (237 lines: `mmf32.cu` + `mmf32.cuh` + a 12-line hook in
`ggml_cuda_mul_mat_cublas_impl`'s F32 branch).  Re-apply with `git apply` in `~/llama.cpp`; nothing else
is needed (the new `.cu` is picked up by the existing `file(GLOB ... ../ggml-cuda/*.cu)`).

## What works (measured, gfx1201, ROCm 10.1.0, 1 GPU unless stated)

| item | result |
|---|---|
| numerical correctness | 16 shapes vs a double reference: **NMSE 1e-13**, including `k=2051` (not a multiple of BK or of the k chunks), odd `m`/`ldc`, `n < m`, `n > m`, `n = 17`; `test-backend-ops -o MUL_MAT` **2444/2444** |
| determinism, 1 GPU | `llama-perplexity -c 512 --chunks 1` logits dump: **1 class in 8/8**; `--chunks 8`: **4/4 byte-identical dumps** |
| determinism, 2 GPU `-sm tensor` | **1 class in 6/6** |
| throughput, 35B-A3B, 3 interleaved visits | **pp512 +4.4 %** (5060 vs 4849), **pp2048 +4.3 %** (4995 vs 4790), tg128 flat |
| vs arm G (the hipBLASLt pin) | pin +5.2 %/+4.8 % — the kernel is **within ~0.8 %** and needs no hipBLASLt, no solution table, no per-runtime validation |
| Flash-Next / dense 27B | zero intercepts (F32-only, and neither has a qualifying shape) |
| kernel time, ssm shape `32×512×2048` | **0.032 ms (2.07 TFLOP/s)** — vs the best hipBLASLt class 0.046 ms; the un-split kernel was 0.27 ms, so the k-split is the whole win |

Layout facts that cost a debug cycle and are worth keeping: ggml `src0` is `ne={k,m}` with `nb0=4` →
`x[gm*k + k]` (**m rows of k**); `src1` is `ne={k,n}` with `nb0=4` → `y[gn*k + k]` (**n rows of k**, i.e.
transposed from the naive `[k][n]` assumption); `dst` is `ne={m,n}` with `nb1=m*4` → `dst[gn*ldc + gm]`
(**n rows of m**, so stores run along `gm`).  A split-K temp buffer is `ks*n*ldc` floats — `ldc*max(m,n)`
is wrong when `n < m` (it silently read the wrong chunk).

Config that ships in the patch: 32 output rows per block, `TMR=4`×`TNC=8` thread sub-tile, `BN=256` for
`m <= 64` (the GDN `ssm_alpha`/`ssm_beta` shapes) and `BN=64` above it (the MoE routers), `ks` chosen to
reach ~256 blocks (capped 16, never a chunk shorter than one `BK=32` block).

## The open question I was paused in the middle of

`ggml-cuda`'s **self-KLD control is not exactly 0 with this kernel**, and it should be:

```
mmf32 vs mmf32 (base recorded with the same binary, -c 512 --chunks 8)
    mean KLD ~0            Same top p: 99.951 ± 0.049 %      RMS Δp 0.001 %   (min Δp -0.004 %)
```

while the *dump files* of four independent runs of that same command are **byte-identical**
(`1e96b6704676` ×4), and the same self-control measured exactly `0.000000 / 100.000 %` for arm G's pin
and for `ROCBLAS_USE_HIPBLASLT=0`.  So either the compare path and the dump path differ in what they
compare, or something in the kernel is reproducible in the dump but not in the compare run — I had just
started to bisect it (per-chunk table, two compare runs diffed, `--record` self-test) when the session
was paused.

**Why it matters:** §5b (decided today) says the pinned/kernel base is re-recorded on the shipped arm and
the pinned-vs-pinned control must read exactly 0.000000 — that is what makes the family-switch number
interpretable.  99.951 % would break that rule for this arm.

**Next diagnostic (cheap, ~4 runs):** rerun the compare twice and diff the full output (the per-chunk
table shows *which* chunk differs); then `--record` with the base path absent twice and compare those two
dumps; if the dumps agree but the compare does not, the residual is in the perplexity compare path, not
in the GEMM.

## Interlock with the parallel `fingon` work

A parallel agent is changing the perplexity scripts.  Everything above that uses
`llama-perplexity --kl-divergence-base` is exactly the surface that change touches, so **do not record
any prefill-logit base for this arm until that lands** — the open anomaly above may be the same thing
they are fixing, and a base recorded either side of it is not comparable.
