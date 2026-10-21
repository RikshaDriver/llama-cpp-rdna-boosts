# wip/hipblaslt-bypass — can the delivery stop calling hipBLASLt entirely?

**Status: OPEN (seeded 2026-10-21, r102).**  Scoping + a measured starting point.  Nothing here is applied
by `scripts/apply-all.sh`, nothing is in `patches/`, and no delivery change was made when this was seeded.

**Tracker:** `TODO.md` #82 (this campaign) — and it is the closure home of `TODO.md` #75, which was closed
on the evidence below (see §1 and [`FINDINGS-2026-10-21-todo-75.md`](FINDINGS-2026-10-21-todo-75.md)).

**Why it exists.**  Two things happened on 2026-10-21:

1. **TODO #75 is the hipBLASLt per-start solution flip**, not a delivery defect.  Reproduced on this box on
   the **release runtime**: 35B-A3B Q4_K_M, `llama-perplexity` logits dump, **3 logit classes in 8 fresh
   starts** with hipBLASLt on and **1 class in 8/8** with `ROCBLAS_USE_HIPBLASLT=0`; the class is decided by
   *which TensileLite kernel spine* hipBLASLt picks for the F32 **`ssm_alpha`/`ssm_beta`** GEMM
   (`MT128x64x8` / `MT64x32x32` / `MT128x128x8` -> three different logit hashes).  Same defect as issue #67 /
   [rocm-libraries#12126](https://github.com/ROCm/rocm-libraries/issues/12126), fixed upstream by
   [rocm-libraries#11340](https://github.com/ROCm/rocm-libraries/pull/11340) (`c50ea60130f`, TheRock
   nightlies from 2026-10-06) which **no ROCm release we run contains** (10.1.0 pins `39d8d60`, branched
   2026-09-09).  Evidence: `FINDINGS-2026-10-21-todo-75.md`.
2. **The flip is not the only cost.**  hipBLASLt is the reason the movable-boundary slab has to leave
   headroom it cannot account for, and it is the single largest source of "the same command gave a different
   answer" reports we handle.  Bypassing it would make the whole delivery start-to-start deterministic by
   construction, remove a binary dependency from the hot path, and (to be **measured**, not assumed) hand
   the slab back whatever VRAM hipBLASLt reserves.

The maintainer's decision (2026-10-21): investigate **removing hipBLASLt from the delivery's path entirely**
rather than only pinning a solution.  This campaign is that investigation; integration is a later session.

---

## 1. What is already measured (r102, gfx1201 `soar`, ROCm 10.1.0 unless stated)

All numbers below are one GPU job at a time, fresh process per sample, 35B-A3B Q4_K_M
(`/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`).  Raw summaries in
[`evidence/`](evidence/).

### 1.1 hipBLASLt is reached only through rocBLAS, and only for a handful of shapes

`ggml-cuda` contains **no** hipBLASLt code (only two comments mention it) and never calls it; `libhipblaslt`
arrives transitively through `librocblas.so.5`, which dispatches to it under `ROCBLAS_USE_HIPBLASLT` (default
on).  Everything hipBLASLt ever does for us therefore enters through `ggml_cuda_mul_mat_cublas()` (ggml-cuda.cu,
the last branch of the `mul_mat` chain).  The dispatch chain that decides it:

| step | predicate | cap that matters |
|---|---|---|
| `mmvf` (decode family) | `ggml_cuda_should_use_mmvf` | `ne11 <= MMVF_MAX_BATCH_SIZE` (**8**, or 32 flattened) |
| `mmf` (native MMA GEMM) | `ggml_cuda_should_use_mmf` | **`src1_ncols > 16 -> false`** (`mmf.cu`) |
| `mmvq` / `mmq` | quantized `src0` only | — |
| `cublas` -> rocBLAS -> **hipBLASLt** | everything else | F32/F16/BF16 weights with **more than 16 columns** |

So the hipBLASLt surface is exactly: **non-quantized weights × a prefill-width batch (> 16 tokens)**.
Observed shapes on 35B-A3B (`llama-cli`, 551-token prompt + 32 decode tokens, 1 GPU):

| calls | A (weight) | B (activations) | op |
|---:|---|---|---|
| 60 | `F32 2048 x 32` | `F32 2048 x 44` | `T`,`N` |
| 60 | `F32 2048 x 32` | `F32 2048 x 512` | `T`,`N` |
| 39 | `F32 2048 x 256` | `F32 2048 x 44` | `T`,`N` |

`2048 = n_embd` and `32 = ssm.time_step_rank` (both from the model's GGUF metadata, which also gives
`expert_count = 256`): the first two shapes are the GDN **`ssm_alpha`/`ssm_beta`** F32 projections, the third
is the F32 **MoE router** (`ffn_gate_inp`, `[n_embd x n_expert]`) — i.e. exactly the §44 call site, on a
second model.  Which ggml tensor each shape belongs to should be confirmed from the graph in P0 rather than
inferred from the numbers.  On a dense 27B everything is quantized, so hipBLASLt is never called there;
`llama-perplexity -c 512 --chunks 1` on 35B-A3B calls hipBLASLt **1080 times per run** for one shape.

### 1.2 The per-start flip, on the release runtime

`llama-perplexity -m <35B-A3B Q4_K_M> -f wiki.test.raw -c 512 --chunks 1 -ngl 99 -fa on -t 8
--kl-divergence-base <file>`, hash of the logits file, fresh process per sample:

| arm | samples | classes |
|---|---|---|
| ROCm 10.1.0, default (hipBLASLt on) | 8 | **3** — `3629d90733a1` x5, `a855c2000912` x2, `6d32945460bd` x1 |
| ROCm 10.1.0, `ROCBLAS_USE_HIPBLASLT=0` | 8 | **1** — `2d9ed9ce03cb` x8 |
| ROCm 10.0.0 libs, default | 10 | **2** — `227ce92cb01c` x8, `6d32945460bd` x2 |
| ROCm 10.0.0 libs, `ROCBLAS_USE_HIPBLASLT=0` | 6 | **1** — `81277fdef6573a2e` x6 |

and the mechanism is visible in the chosen kernel (traced, 4 starts, 61 hipBLASLt dispatches each):

| start | TensileLite kernel for the F32 ssm GEMM | logits class |
|---|---|---|
| 1 | `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT128x64x8` + `VW4` epilogue | `83091b86d325` |
| 2 | `..._MT64x32x32` + `VW4` | `eafe2e3ca6cc` |
| 3 | `..._MT128x128x8` (epilogue fused, no `VW4`) | `9dc60a34bb72` |
| 4 | `..._MT64x32x32` + `VW4` | `eafe2e3ca6cc` |

`HIPBLASLT_LOG_LEVEL=4` shows the mechanism directly: `getAllSolutions()` returns the same 18 candidates
(`140215`..`140232`) **in a different order in 7 of 8 fresh starts** — the heap-address-ordered container
that #11340 fixes.  Note the ordering is *stable* in some configs (the `test-logits-width-probe` config was
identical in 6/6 starts and 10/10 single-valued, which is why the first day of the #75 hunt found nothing):
**a config that does not flip proves nothing about the mechanism**, only about that config.

Magnitude between two classes (`llama-perplexity --kl-divergence`): **mean KLD 0.003479 ± 0.001195,
same-top-p 98.431 ± 0.780 %, RMS Δp 1.901 ± 0.470 %, max KLD 0.278**; same-class control 0.000000 / 100.000 %.
So the flip lands **just inside** the prefill-logit gate (`<= 0.005` / `>= 98 %`) — the gate cannot see it
either, by construction.

### 1.3 What disabling it costs on this box: nothing (and that is the campaign's opening)

`llama-bench -p 512,2048 -n 128 -r 3 -fa 1 -ub 512 -b 2048`, two interleaved visits:

| arm | pp512 | pp2048 | tg128 |
|---|---|---|---|
| hipBLASLt **on** (default) | 4749.72 / 4840.90 | 4804.49 / 4798.11 | 108.23 / 108.02 |
| `ROCBLAS_USE_HIPBLASLT=0` | 4899.77 / 4887.22 | 4841.88 / 4830.28 | 107.92 / 107.97 |

i.e. **+1-3 % prefill and flat decode with it off** on 35B-A3B/1 GPU — the opposite sign to the -15 %
decode cost recorded in `GREEDY-PURITY.md` §44 for other models.  The cost is therefore **model- and
runtime-dependent**, and the campaign must re-measure per model rather than quote either number.

Peak VRAM (`rocm-smi`, `-c 4096 --chunks 2`): **21822 MiB on vs 21752 MiB off, ~70 MiB** — so hipBLASLt's
*own* footprint is not the "~2 GB per card" hypothesis from the 2026-10-21 discussion thread.  If the slab
really loses ~2 GiB to it, the loss is in the **headroom the slab leaves** (an unaccounted allocator that
must not be allowed to fail), not in what hipBLASLt allocates.  **Measuring that on the `-ncmoe`/slab path is
campaign step P1**, and it is the item that decides whether the campaign is worth a delivery change at all.

## 2. Design space (what "bypass" can mean)

Ordered by how much of the delivery they touch, cheapest first.  Nothing here is a decision yet.

**A. Pin the solution instead of removing it (`hipBLASLT_TUNING_OVERRIDE_FILE`).**  hipBLASLt reads a tuning
file that pins a solution index per problem size.  This fixes determinism without touching a kernel, and it
is upstream-supported.  It does **not** remove the dependency, the VRAM headroom question or the binary, so
it is the fallback if A/B/C all lose throughput — cheap enough to quantify first.

**B. Raise the native-GEMM cap (`mmf.cu`: `src1_ncols > 16`).**  `mmf` already handles F32 `src0` on RDNA
(`amd_mfma_available`); only the 16-column cap keeps prefill off it.  Cost of raising it is unknown and this
is why the cap is there upstream, so it needs a real prefill A/B at 512/2048/8192 columns, not a smoke test.
This is the smallest possible patch and it removes hipBLASLt from the *f32 weighted-GeMM* path only.

**C. A dedicated small-M x wide-N F32 kernel (the "GDN alpha/beta GEMM").**  The shapes that hurt are
`M = 32/128/256, K = 2048/2816/5120, N = tokens`.  M is tiny, so this is a skinny GEMM: a block-per-token
kernel with the weight streamed once could beat both hipBLASLt and `mmf` here, and it would be *ours* (so
bit-stable by construction, and the K-split order is ours to pin — the band-uniformity rules in
`GREEDY-PURITY.md` §19/§24/§25 apply).  This is the only arm that can plausibly also be a **win**.

**D. Stop shipping the F32 weights (requantize `ssm_alpha`/`ssm_beta` to Q8_0).**  DanoPTT's suggestion: with
those tensors at Q8_0 the GEMM goes through MMQ and hipBLASLt is never called, at the same speed (their
measurement, Windows).  For the delivery this means either a `llama-quantize` recipe change (a consumer
action, not a patch) or a loader-side shadow copy (extra VRAM, a purity question: it changes the model, not
the kernel).  Needs an accuracy measurement (KLD vs the F32 arm) before it can be recommended at all.

**E. Refuse to use hipBLASLt at all (`setenv("ROCBLAS_USE_HIPBLASLT","0")` in our ROCm init).**  The
blunt instrument.  It works today (§1.3 says it is free on this model) but it also changes behaviour for
every non-quantized GEMM in every model, including the ones where §44 measured a real decode loss.  Only
viable **together with** B/C so that the loss disappears for the shapes we actually own.

**F. Keep hipBLASLt, document the lottery.**  Status quo + `ROCBLAS_USE_HIPBLASLT=0` for reproducible runs.
Free, but leaves #75-class reports arriving forever, and the "which kernel family runs my prefill" question
unanswerable in a benchmark record.

## 3. Gates for any candidate (do not accept a bypass without all of them)

1. **The flip is gone**: 12 fresh `llama-perplexity` logits dumps (commands in §4) -> **one** class, and the
   same for `test-logits-width-probe` on a config *known to flip* (the perplexity config, not the probe's
   default one).  Also run the arms that flip today at 2-3 GPU (`-sm tensor`).
2. **Prefill-logit purity**: `scripts/gate-prefill-logits.sh` against a base recorded on the *current*
   (hipBLASLt) build, mean KLD and same-top-p reported.  A bypass changes the GEMM summation order, so a
   non-zero KLD is expected and must be **recorded against the §113 calibration** (is it a family switch or a
   rounding-level change?), not hidden behind "it is deterministic now".
3. **Throughput**: pp512 / pp2048 / pp2048@d40K and tg128, interleaved visits, both arms, per model
   (35B-A3B, dense 27B `Q4_K_XL`, gemma4 26B-A4B, Flash-Next).  No throughput is claimed from a single visit.
4. **Text/state gates**: same-seed coherence, `none == draft-mtp` on the 4-axis adaptive-MTP gate (long runs,
   `--reasoning` set), `test-backend-ops` on the touched ops, `scripts/validate-set.sh`.
5. **Cross-backend**: the change must be RDNA/ROCm-gated and leave CUDA/MUSA/SYCL/Vulkan/CPU instantiation
   sets consistent (`GREEDY-PURITY.md` §20); non-AMD backends are out of scope for performance.

## 4. Plan

**P0 — inventory (cheap, do first).**  Enumerate every call site that can reach `mul_mat_cublas` for a model
with F32 weights, per model in the fleet, with the shape and the column counts that trigger it.  The tool to
use is the hipBLASLt log, not the source (`HIPBLASLT_LOG_LEVEL=4 HIPBLASLT_LOG_MASK=1
HIPBLASLT_LOG_FILE=<f>`), aggregated exactly as in [`tools/hbl-shapes.sh`](tools/hbl-shapes.sh).  Deliverable:
a table of (model, shape, calls/run, N values) — that table *is* the scope of the whole campaign.

**P1 — is the VRAM/slab claim real?**  On the `-ncmoe` + slab path, measure (a) peak VRAM, (b) the slab's
reported reservation, (c) whether an unaccounted hipBLASLt allocation ever forces the OOM fallback
(`ggml_cuda_device_malloc` arena drop), hipBLASLt on vs off.  If the delta is ~70 MiB as in §1.3, drop that
motivation from the campaign and say so in `TODO.md`; the remaining drivers (determinism, one less binary,
one less lottery) still stand on their own.

**P2 — cheapest viable arm.**  Measure A (tuning override) and B (raise the `mmf` cap) on 35B-A3B +
gemma4 26B-A4B.  A is a config file; B is a one-line predicate change behind a kill switch.  Both are cheap
enough to answer in one session, and either might end the campaign with "no kernel work needed".

**P3 — the real kernel (C) or the requantization (D)**, only if P2 cannot hold throughput.  C is the block
08/10 kind of work (skinny F32 GEMM, band-uniform K-split, verify widths) and inherits every purity rule for
those blocks; D is a model-side change and needs an accuracy story before it is even on the table.

**P4 — default.**  Whichever arm wins becomes **default-on with a kill switch** (`default-on policy` in
`AGENTS.md`), the old path is the kill switch value, and `ENVIRONMENT.md` gains the variable.

## 5. Reproduce the #75 witness (also the P3/flip gate)

```bash
cd ~/llama.cpp
export LD_LIBRARY_PATH=$PWD/build-rocm/bin:/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=0
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

# one fresh process per sample; prints the logits-file hash (= the class)
for i in $(seq 1 8); do
  ./build-rocm/bin/llama-perplexity -m $M -f /llm/models/wikitext-2-raw/wiki.test.raw \
    -c 512 --chunks 1 -ngl 99 -fa on -t 8 --kl-divergence-base /var/tmp/lg$i.kld >/dev/null 2>&1
  sha256sum /var/tmp/lg$i.kld | cut -c1-12
done

# the same with the library path pinned: expect ONE hash, 8/8
for i in $(seq 1 8); do
  ROCBLAS_USE_HIPBLASLT=0 ./build-rocm/bin/llama-perplexity ... --kl-divergence-base /var/tmp/off$i.kld
  sha256sum /var/tmp/off$i.kld | cut -c1-12
done
```

[`tools/hbl-classes.sh`](tools/hbl-classes.sh) does the loop and the class table;
[`tools/hbl-kernels.sh`](tools/hbl-kernels.sh) adds the `rocprofv3 --kernel-trace` pass that names the
TensileLite kernel per class (needs the **10.1** `rocprofv3`; the `/usr/bin` 7.x one silently writes no
CSV).  Both take the arm from the environment, so the `=0` control is the same command.

## 6. Files

| file | what |
|---|---|
| `FINDINGS-2026-10-21-todo-75.md` | the closure record for TODO #75 (what was measured, what it rules out, the exact classes) |
| `NEXT-SESSION-PROMPT.md` | cold-start prompt for the next session (P0/P1/P2) |
| `tools/hbl-classes.sh` | fresh-process logits-class count (the determinism instrument) |
| `tools/hbl-kernels.sh` | adds the kernel-trace pass: which TensileLite kernel per start |
| `tools/hbl-shapes.sh` | aggregate the hipBLASLt shape inventory from a log |
| `evidence/` | the dated measurement summaries quoted in §1 |
