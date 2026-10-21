# RESOLUTION — hipBLASLt bypass (closed 2026-10-21, r104)

**Outcome: shipped as block 06's `mmf32`** — our own F32 SIMT prefill GEMM for the shapes that used to reach
hipBLASLt, so the 35B-A3B / gemma4 F32 prefill is deterministic by construction. Rule: `GREEDY-PURITY.md`
§47. Delivery notes: `patches/README.md` (block 06), `ENVIRONMENT.md` §10.1, `MANIFESTS.md` (the block-06
`mmf32` row), gate point: `prompts/kld-corpus-35b-in-domain.txt`. Release record: `WORKLOG.md` 261021.011.

## What was true and what shipped

| question | answer |
|---|---|
| what reaches hipBLASLt | only rocBLAS's non-quantized `mul_mat` fallback: dense 27B **0** dispatches, gemma4 **1 F32 shape** (the router), 35B-A3B **3 F32 shapes** (`ssm_alpha`/`ssm_beta` + router), Flash-Next **27 BF16 shapes / 2790 dispatches** |
| why it mattered | **correctness**, not VRAM: hipBLASLt picks its solution per process start, so the prefill logits change between runs (3 classes in 8 starts; `TODO` #75). Its outside-slab footprint is a constant **~70 MiB**, measured, and was dropped as a motivation |
| the fix that shipped | `mmf32`: a 32-row block tile / 4×8 thread sub-tile SIMT GEMM, k split into contiguous chunks summed in increasing z (a fixed reduction tree) — **12/12 fresh starts one logits class**, 6/6 at 2 GPU, and **+4.2 % pp512 / +4.1 % pp2048**, decode flat |
| cost | none measured; it is also faster than the `ROCBLAS_USE_HIPBLASLT=0` control by 3.6 %, because the heuristic's solution is the *slowest* in the pool for these shapes |

## The alternatives, and why they did not ship

* **Arm A — pin the solution with the library's own switches. REFUTED.** `TENSILE_SOLUTION_INDEX` has no
  effect on the rocBLAS→hipBLASLt path (3 and 2 classes in 6 starts each), and
  `HIPBLASLT_TUNING_OVERRIDE_FILE` is rejected: *"The hipBLASLt git version and the override file git version
  are not the same"* — it is locked to one ROCm build. Evidence: `evidence/p2-arm-a-pinning-attempts.txt`.
* **Arm B — raise `mmf`'s 16-column cap. REFUTED as cheap.** RDNA has no F32 MMA form (`amd_mfma_available`
  is CDNA-only, `amd_wmma_available` is f16/bf16), and 16 is a *kernel* limit (`GGML_ASSERT(ids ||
  ncols_dst <= 16)`), not a predicate. Evidence: `evidence/p2-mmf-cap-attempt.txt`.
* **Arm G — call `hipblasLtMatmul` ourselves with one validated solution. WORKS, did not ship.** Deterministic
  and **+5.2 %/+4.8 %** (≈1 % more than the kernel), but it keeps hipBLASLt, adds a `hipblaslt` link, and owes
  a **per-ROCm-release solution index table** (10.1.0: `m ≤ 64 → 140219`, else `140232`; 7.14.1 advertises only
  four solutions from `140723` and falls back). The kernel buys almost all of the win with none of that
  obligation — maintainer decision, recorded in `WORKLOG.md` 261021.009/.011. Evidence:
  `evidence/p3-arm-g-pin-implemented.txt`, prototype `tools/experiments/hipblaslt-pin-ATTEMPT.patch`.
* **Arm D — requantize the F32 weights (Q8_0). Not pursued:** it changes the model, not the kernel, and needs
  its own accuracy case. **Arm E — refuse hipBLASLt (`ROCBLAS_USE_HIPBLASLT=0`):** kept as the *tester's*
  control; it costs Flash-Next ~+51 % prefill and is model-dependent elsewhere.

## What was, and was not, delivered

Delivered: the F32 skinny shapes (`m ≤ 256`, `17 ≤ n ≤ 65536`, `256 ≤ k ≤ 8192`, `ldc == m`), default-on for
RDNA4, `GGML_CUDA_MMF32=0` to disable.

Not delivered, deliberately: **BF16 prefill** (Flash-Next keeps both hipBLASLt's +51 % and its own per-start
lottery — a column-tiled non-ids MMF path is its own campaign, not a gap in this one); **full-F32 FFN GEMMs**
(`m` ≫ 256, still on rocBLAS; accepted limitation in `TODO.md`); **RDNA3 / RDNA3_5** are opt-in until those
boxes run the determinism + bench pair.

## The instrument lesson (worth carrying to any gate read)

The prefill-logit gate's **self-KLD control is not exact**, and it is the *tool*, not the kernel: the base file
stores **uint16-quantised log-probs** (`scale = (max-min)/65535` over a ≤16-nat window ⇒ step ≈ 2.4e-4 nats),
so an arm compared against its own base reads `mean KLD −0.000000` but `same-top-p 99.951 %` *while both dumps
are byte-identical*; the stock arm carries the same RMS Δp floor.  **Judge determinism on the dump hash, not on
the self-KLD top-p.**  `benchmarks/prefill-logit-methodology.md` §5b records the rule that follows (a base is
only valid for the arm it was recorded on; a family switch re-baselines).

## Files

`README.md` (the campaign record: design space, the full measurement log), `evidence/p0`-`p4`,
`tools/hbl-classes.sh` + `hbl-kernels.sh` + `hbl-shapes.sh` (the determinism/kernel-trace/shape instruments),
`tools/experiments/` (both prototypes, the two refutations, and `mmf32-standalone/` — the kernel's numerical
oracle and tile bench), `FINDINGS-2026-10-21-todo-75.md` (the `TODO` #75 closure).
