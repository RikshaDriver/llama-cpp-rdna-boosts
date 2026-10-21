# TODO #75 — closure evidence: the 35B-A3B per-start nondeterminism is the hipBLASLt solution flip

**Closed 2026-10-21 as external** (issue #67 / [rocm-libraries#12126](https://github.com/ROCm/rocm-libraries/issues/12126),
fixed upstream by [rocm-libraries#11340](https://github.com/ROCm/rocm-libraries/pull/11340), in no release we
run).  No delivery change follows from it.  The follow-on work — removing hipBLASLt from our path instead of
living with the lottery — is the rest of this campaign ([`README.md`](README.md), `TODO.md` #82).

Everything below is gfx1201 (`soar`), the r102 tree (`rdna-boosts` branch tip `7fa283979`, release
`v12-097cf33d9-r102`), 35B-A3B Q4_K_M, `-ngl 99`, 1 GPU, fresh process per sample, one GPU job at a time.
Raw summaries: [`evidence/`](evidence/).

## 1. The claim that was open

`TODO.md` #75 (filed 2026-10-19 at r90): *"35B-A3B greedy text is not run-to-run deterministic with
`GGML_CUDA_ASYNC_INPUT=0`"* — 2-3 different 400-token greedy texts across fresh processes, on 1 GPU and on
2-GPU `-sm tensor`, on the plain default (non-async) path.  The gfx1151 box could not reproduce it on either
side of r97's async fix (`WORKLOG.md` 261020.013), so the open question was what the actual mechanism was.

## 2. Result

**The mechanism is the ROCm hipBLASLt per-process solution selection**, on the F32 GDN
`ssm_alpha`/`ssm_beta` GEMM.  It reproduces on this box **on the release runtime** (ROCm 10.1.0, which
predates the upstream fix), and it disappears bit-exactly with `ROCBLAS_USE_HIPBLASLT=0`.

`llama-perplexity -m 35B-A3B-Q4_K_M -f wiki.test.raw -c 512 --chunks 1 -ngl 99 -fa on -t 8
--kl-divergence-base <file>`, hash of the dump:

| arm | samples | logits classes |
|---|---|---|
| ROCm 10.1.0, default (hipBLASLt on) | 8 | **3** — `3629d90733a1` x5, `a855c2000912` x2, `6d32945460bd` x1 |
| ROCm 10.1.0, `ROCBLAS_USE_HIPBLASLT=0` | 8 | **1** — `2d9ed9ce03cb` x8 |
| ROCm 10.0.0 libs, default | 10 | **2** — `227ce92cb01c` x8, `6d32945460bd` x2 |
| ROCm 10.0.0 libs, `ROCBLAS_USE_HIPBLASLT=0` | 6 | **1** — `81277fdef6573a2e` x6 |

The two `=0` arms agree with each other and with the `llama-cli`/probe arms, i.e. rocBLAS's own path is
deterministic; the whole residual is hipBLASLt.

## 3. The mechanism, named

`rocprofv3 --kernel-trace` on four fresh traced starts (the tracer perturbs the heap so the classes differ
from the untraced runs — that is expected and does not weaken the point, the *mapping* is what matters):

| start | the 61 hipBLASLt kernels that run | logits class |
|---|---|---|
| 1 | `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT128x64x8...` + `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU2_VW4` | `83091b86d325` |
| 2 | `..._MT64x32x32...` + `VW4` | `eafe2e3ca6cc` |
| 3 | `..._MT128x128x8...` (epilogue fused: **no** `VW4` kernel) | `9dc60a34bb72` |
| 4 | `..._MT64x32x32...` + `VW4` | `eafe2e3ca6cc` |

One shape, three different TensileLite spines, three different logits.  With `HIPBLASLT_LOG_LEVEL=4`:

- the only hipBLASLt shape in this run is `A[R_32F 2048x32] B[R_32F 2048xN] OP_T OP_N` (the GDN F32
  `ssm_alpha`/`ssm_beta`), **1080 dispatches per run**;
- `getAllSolutions()` offers the same 18 candidates (`140215`..`140232`) **in a different order in 7 of 8
  fresh starts**.  That order is the heap-address-dependent container the upstream fix (#11340) stably sorts.

## 4. Why the first attempts found nothing (and the methodological lesson)

The `test-logits-width-probe` config was single-valued on 10.1.0 in 10/10 starts (W=1 `a5589707a12dea4f`),
8/8 at 2-GPU `-sm tensor` (`68b41f06a5c1ec35`) and 8/8 at P=1024 (`5dcdfb5b5a4e3167`), while the
perplexity config gives 3 classes in 8 starts.  On ROCm 10.0.0 the *probe* does flip (10 starts: `a558…`
x4, `458b…` x6) — so the instability is per-shape and per-run, and **a config that does not flip is not
evidence about the mechanism**.  Any determinism claim must be made on a config that is *known* to flip on
the build under test, or on the logits dump of a real generation.

**A greedy-text compare is the wrong instrument.**  With the logits demonstrably class-varying, 200-token and
500-token greedy text on the same model/prompt was **identical in 10/10 and 12/12** runs (10.0.0 arm,
`--ignore-eos`), because no near-tie happened to be crossed.  That is the same lesson as issue #151: judge
at the logits level, per position, not on a terminal text hash.

## 5. How big is the difference

`llama-perplexity --kl-divergence` between a class-A base and a class-B run (10.0.0 arm):

| metric | value |
|---|---|
| Mean KLD | **0.003479 ± 0.001195** |
| Same top p | **98.431 ± 0.780 %** |
| RMS Δp | 1.901 ± 0.470 % |
| Max KLD | 0.278 |
| same-class control | 0.000000 / 100.000 % |

The flip therefore sits **just inside** the prefill-logit gate (`mean KLD <= 0.005`, `same-top-p >= 98 %`):
the gate cannot flag it, and a nondeterministic arm can consume the gate's whole tolerance by accident.
That is a calibration fact worth keeping (see `benchmarks/prefill-logit-methodology.md`).

## 6. Why it appears as "greedy text changes"

Only the prefill is affected: the F32 GEMM is called with more than 16 columns (see the dispatch table in
[`README.md`](README.md) §1.1); single-token decode uses `mul_mat_vec_f` and never touches hipBLASLt.  A
start that lands on a different spine therefore enters generation with slightly different KV/state, and a
near-tie later in the generation flips the token — the same "only through the prefill" shape the #67 record
describes, and the reason `GGML_CUDA_ASYNC_INPUT` was never implicated (all arms above run with the ring
irrelevant; the `=0` control is what closes it).

## 7. What this does NOT say

- It does not say the release is defective.  The flip reproduces on stock upstream with none of the patches;
  it is the ROCm runtime.
- It does not close issue #134 (unified-KV period residue), which is a different, in-set mechanism.
- It does not license `ROCBLAS_USE_HIPBLASLT=0` as a shipping default; that is campaign P2/P4 in
  [`README.md`](README.md), and the cost is model-dependent (`GREEDY-PURITY.md` §44 records up to -15 %
  decode on other models, while 35B-A3B on this box measures **+1-3 % prefill / flat decode**).
