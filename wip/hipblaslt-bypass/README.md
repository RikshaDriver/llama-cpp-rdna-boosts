# wip/hipblaslt-bypass — can the delivery stop calling hipBLASLt entirely?

**Status: OPEN (seeded 2026-10-21, r102).**  Scoping + a measured starting point.  Nothing here is applied
by `scripts/apply-all.sh`, nothing is in `patches/`, and no delivery change was made when this was seeded.

**Why it exists.**  Three things happened on 2026-10-21:

1. **TODO #75 is the hipBLASLt per-start solution flip**, not a delivery defect.  Reproduced on this box on
   the **release runtime**: 35B-A3B Q4_K_M, `llama-perplexity` logits dump, **3 logit classes in 8 fresh
   starts** with hipBLASLt on and **1 class in 8/8** with `ROCBLAS_USE_HIPBLASLT=0`; the class is decided by
   *which TensileLite kernel spine* hipBLASLt picks for the F32 **`ssm_alpha`/`ssm_beta`** GEMM
   (`MT128x64x8` / `MT64x32x32` / `MT128x128x8` -> three different logit hashes).  Same defect as issue #67 /
   [rocm-libraries#12126](https://github.com/ROCm/rocm-libraries/issues/12126), fixed upstream by
   [rocm-libraries#11340](https://github.com/ROCm/rocm-libraries/pull/11340) (`c50ea60130f`, TheRock
   nightlies from 2026-10-06) which **no ROCm release we run contains** (10.1.0 pins `39d8d60`, branched
   2026-09-09).  Evidence: `FINDINGS-2026-10-21-todo-75.md`.
2. **The headroom hypothesis was measured and is small.**  The 2026-10-21 discussion thread attributed part
   of the slab's ~2 GiB headroom floor to hipBLASLt.  Measured on the model that motivated it
   (Qwen3.8-Flash-Next, `-ncmoe 48 -sm tensor`, `GGML_CUDA_SLAB_STATS=1`): hipBLASLt holds a **constant
   ~70 MiB outside the slab** (53-88 MiB), **independent of ubatch (512..8192), prompt length (602..68 262
   tokens), context (16K..131K) and model**.  That is ~3.5 % of the floor, not the driver.  Evidence:
   `evidence/p1-hipblaslt-outside-slab.txt`.
3. **So the campaign's driver is CORRECTNESS** (maintainer decision 2026-10-21): hipBLASLt is the last
   per-start lottery in the set, and a bypass makes every run reproducible by construction instead of by
   environment variable.  VRAM is a side effect to be measured, not the goal.

**Tracker:** `TODO.md` #82 (this campaign) — and it is the closure home of `TODO.md` #75, which was closed
on the evidence below (see §1 and [`FINDINGS-2026-10-21-todo-75.md`](FINDINGS-2026-10-21-todo-75.md)).
Integration is a later session; this campaign records what a bypass must clear.

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
second model.  `llama-perplexity -c 512 --chunks 1` on 35B-A3B calls hipBLASLt **1080 times per run** for
one shape.

**P0 inventory, done 2026-10-21** (`evidence/p0-inventory-all-models.txt`, `p0-shapes-flash-next.txt`).  One
`llama-cli` run per model, `HIPBLASLT_LOG_LEVEL=4`, 5246-token prompt:

| model | dispatches | shapes | dtype | what |
|---|---:|---:|---|---|
| Qwen3.8-27B UD-Q4_K_XL | **0** | — | — | dense + quantized: hipBLASLt never called |
| Gemma4-26B-A4B QAT | 29 | 1 | F32 | `A[2816 x 128]` = MoE router (n_embd 2816, 128 experts) at N=375 |
| Qwen3.6-35B-A3B Q4_K_M | 159 | 3 | F32 | `A[2048 x 32]` x2 = `ssm_alpha`/`ssm_beta`, `A[2048 x 256]` = router |
| Qwen3.8-Flash-Next GSQ | **2790** | **27** | **BF16** | attention / PLE projections, N in {42, 44, 512} |

Three consequences for the campaign:

* A dense quantized model has **no hipBLASLt exposure at all**, so a bypass cannot regress it.
* The **correctness** problem (per-start classes) lives where **F32** weights meet the prefill band: 35B-A3B
  alpha/beta (3 classes / 8 starts) and the gemma4 router (§44: 3 / 18).
* Flash-Next is the big consumer by **count**, and it is **BF16** — the one model where our own MMA path
  (WMMA covers f16/bf16 on RDNA3/4) already exists, but see §2 arm B, which refutes the cheap version.

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

### 1.3 How much VRAM hipBLASLt actually holds OUTSIDE the slab (P1, answered)

This was the opening motivation ("the slab cannot get below ~2 GiB of headroom and hipBLASLt is one
contributor") and it is now sized.  Method: Qwen3.8-Flash-Next GSQ-IQ3_XXS, 2 GPU `-sm tensor -ncmoe 48 -fa on`,
`--load-mode mmap --lazy-mode off` (the maintainer's fast-load/most-consistent setting for this model),
`GGML_CUDA_SLAB_STATS=1`, and compare the slab's own `free_min` (= free VRAM outside the slab) between the
two arms at the same config:

| config | point | hipBLASLt ON | OFF | delta |
|---|---|---:|---:|---:|
| 602 tok, `-c 16384` | wide-arm | 735.0 | 805.0 | **+70** |
| 602 tok, `-c 16384` | exit | 351.5 | 421.5 | **+70** |
| 5298 tok, `-c 32768 -ub 2048` | exit | 175.5 | 245.5 | **+70** |
| 5298 tok, `-c 32768 -ub 8192` | exit | 175.5 | 245.5 | **+70** |
| 68 262 tok, `-c 131072 -ub 8192` | wide-arm | 749.0 | 836.0 | **+87** |
| 68 262 tok, `-c 131072 -ub 8192` | exit | 371.5 | 441.5 | **+70** |
| headroom 512 / 256 (`-c 16384`) | exit | 375 / 275 | 445 / 445 | **+70** |

**A constant ~70 MiB (53-88), independent of ubatch (512..8192), prompt (602..68 262 tokens), context
(16K..131K) and model** (35B-A3B measures the same 70 MiB by peak-VRAM).  It does **not** scale, which rules
out a per-call workspace: the hipBLASLt log shows each call requests 0 / 12 KiB / 24 KiB / 64 KiB.  The
constant is **TensileLite code objects** (one set per distinct solution — Flash-Next loads 27 shapes) plus a
small fixed arena.  So hipBLASLt is ~3.5 % of a 2 GiB headroom floor; the rest is the MTP draft buffer, the
FA-QSA workspace, the pool high-water mark and the model's own PLE/attention allocations.  `ENVIRONMENT.md`
"Why the cards sit near-full" lists hipBLASLt among those consumers; this sizes it, and the campaign should
**not** claim headroom as a reason to bypass it.  (The "~2 GiB is hipBLASLt" attribution was **retracted in
discussion #196** by the maintainer on 2026-10-21; this measurement agrees with the retraction.)

Single-visit throughput of the same runs (indicative, repeats needed): Flash-Next prefill **2405.7 t/s with
hipBLASLt vs 1595.1 without (+51 %)** at 68K tokens/`-ub 8192`, 966.8 vs 823.9 at 5298 tokens/`-ub 8192`, and
630.9 vs 823.1 at 5298 tokens/`-ub 2048` (that last one is unexplained — do not quote it as a shape until it
is repeated).  Decode is flat.  **So on Flash-Next a BF16 bypass has a real bar to clear, while on 35B-A3B
the F32 path is free** (`evidence/bench-hipblaslt-on-vs-off.txt`).

### 1.4 What disabling it costs on this box: nothing for the F32 models (and that is the opening)

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

Each arm carries its 2026-10-21 status.  A and G are the no-kernel arms (G is the recommended one), B/C are
kernel work, D is a model-side change, E is the blunt instrument, F is the status quo.

**A. Pin the solution instead of removing it — REFUTED as a shipping mechanism (2026-10-21).**  Two attempts,
both rejected by the runtime (`evidence/p2-arm-a-pinning-attempts.txt`): `TENSILE_SOLUTION_INDEX=140215/140216`
(the library's own "force a solution" variable) changes **nothing** on the rocBLAS→hipBLASLt path (3 and 2
classes in 6 fresh starts each), and `HIPBLASLT_TUNING_OVERRIDE_FILE`, written in the documented column layout
for the exact shape hipBLASLt logs (`m=32 n=512 k=2048`, all `R_32F`, `T,N`), is rejected with *"The hipBLASLt
git version and the override file git version are not the same"* — the file is locked to one ROCm build.

**G. Make the pick deterministic in OUR code — the recommended practical fix (not yet prototyped).**  Call
`hipblasLtMatmul` directly for the shapes we care about with an algo we choose deterministically (e.g. the
lowest `solution_id` from `hipblasLtMatmulAlgoGetHeuristic`/`getAllAlgos`, or a fixed index validated once).
This keeps hipBLASLt's **+51 % Flash-Next prefill** and removes the lottery at the same time, at the cost of
adding hipBLASLt as a direct dependency (it is already linked transitively through rocBLAS) and of a fixed
algo that must be re-validated per ROCm line.  Cheaper than an F32 GEMM to write, but less of a win: it does
not remove the binary, and an implicit fallback path can still pick a different kernel.  Prototype this
*before* building the F32 kernel if the goal is "reproducible shipped binary" rather than "no hipBLASLt".

**B. Raise the native-GEMM cap (`mmf.cu`: `src1_ncols > 16`) — REFUTED as a cheap arm (2026-10-21).**  Tried
it (`GGML_CUDA_MMF_MAX_COLS`, patch kept as `tools/experiments/mmf-max-cols-ATTEMPT.patch`).  Two blockers:
(a) the F32 branch of `ggml_cuda_should_use_mmf` requires `amd_mfma_available`, which is **CDNA-only**, and
`amd_wmma_available` covers only f16/bf16 — so **RDNA has no native F32 MMA GEMM**, and the 35B-A3B
`ssm_alpha`/`ssm_beta` GEMM cannot move to `mmf` by any predicate change; (b) on Flash-Next (BF16) raising the
cap **aborts** at `mmf.cuh:771  GGML_ASSERT(ids || ncols_dst <= 16)` — 16 is a **kernel** limit
(`cols_per_block` is a template argument) and only the `mul_mat_id` launch tiles columns
(`block_nums_ids.y *= col_tiles`).  The tractable version of B is therefore: **port that column tiling to the
non-ids MMF launch** (bf16/f16 only, the kernel body already carries `col_base` for the ids case) — that is a
real kernel change, not a predicate, and it must beat 2405 t/s prefill on Flash-Next.  Full record:
`evidence/p2-mmf-cap-attempt.txt`.

**C. A dedicated small-M x wide-N F32 kernel (the "GDN alpha/beta GEMM") — the only arm that removes the
F32 lottery in-set.**  The shapes that matter are `M = 32/256, K = 2048, N = tokens` (35B-A3B) and
`M = 128, K = 2816` (gemma4's router).  M is tiny, so this is a skinny GEMM: a block-per-token kernel with the
weight streamed once could beat both rocBLAS and anything we could get from `mmf` here (there is no F32 MMA
form on RDNA at all — §2 arm B).  It would be *ours*, so bit-stable by construction, and the K-split order is
ours to pin (band-uniformity rules in `GREEDY-PURITY.md` §19/§24/§25 apply).  This is campaign step P3(ii)
and the arm that makes every 35B-A3B/gemma4 run reproducible without an environment variable.

**D. Stop shipping the F32 weights (requantize `ssm_alpha`/`ssm_beta` to Q8_0).**  DanoPTT's suggestion: with
those tensors at Q8_0 the GEMM goes through MMQ and hipBLASLt is never called, at the same speed (their
measurement, Windows).  For the delivery this means either a `llama-quantize` recipe change (a consumer
action, not a patch) or a loader-side shadow copy (extra VRAM, a purity question: it changes the model, not
the kernel).  Needs an accuracy measurement (KLD vs the F32 arm) before it can be recommended at all.

**E. Refuse to use hipBLASLt at all (`setenv("ROCBLAS_USE_HIPBLASLT","0")` in our ROCm init).**  The
blunt instrument.  It is free on 35B-A3B (§1.4) and saves ~70 MiB/device, but it costs **-37 % Flash-Next
prefill** (1595 vs 2406 t/s at 68K, §1.3) and it is the arm `ENVIRONMENT.md` records as corrupt in the
wide-prefill config at low headroom (needs re-testing).  Only viable **together with** the BF16 kernel work
(P3(i)) so that the loss disappears for the shapes we actually own.

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

**P0 — inventory (DONE 2026-10-21).**  Four models measured, see §1.1: dense 27B = 0 dispatches, gemma4 =
1 F32 shape (router), 35B-A3B = 3 F32 shapes, Flash-Next = 27 BF16 shapes / 2790 dispatches.  Remaining for a
future pass: Flash-Next IQ4_NL (and the MTP draft's own GEMMs, which need a run with `-md`).

**P1 — the VRAM/slab claim (DONE 2026-10-21, §1.3).**  hipBLASLt holds a constant ~70 MiB outside the slab and
it does not scale with ubatch or prompt length.  Headroom is **not** a reason to bypass it; correctness is.

**P2 — the cheap arms.**  Arm B is refuted as cheap (§2): F32 has no MMA path on RDNA and the 16-column cap is
a kernel limit.  **Arm A is refuted as a shipping mechanism** (§2): `TENSILE_SOLUTION_INDEX` has no effect and
the tuning override file is locked to the exact hipBLASLt git version.  **Arm G (explicit algo in our code) is
the recommended next step** for the correctness goal, because it keeps the Flash-Next +51 % prefill.  **Arm E**
(`ROCBLAS_USE_HIPBLASLT=0` from our ROCm init) is free on 35B-A3B but -37 % Flash-Next prefill, and
`ENVIRONMENT.md` records that `=0` was measured CORRUPT at `GGML_CUDA_SLAB_HEADROOM_MIB=2048` in the wide-prefill
config (`-ub 6144 -c 204800`) — that claim predates this campaign and must be re-tested before any arm relies
on `=0`; at 16384..131072 context and 896..256 headroom it is clean here.

**P3 — the real kernels.**  Two separate jobs: **(i) BF16/f16 wide N** = port the column tiling the
`mul_mat_id` MMF launch already has to the non-ids launch (WMMA exists on RDNA for f16/bf16) for Flash-Next;
**(ii) F32 wide N** = a SIMT tiled GEMM for 35B-A3B alpha/beta + the gemma4 router (no MMA form exists on
RDNA), or arm D (stop shipping those weights as F32).  The F32 job is the one that removes the last per-start
lottery and has a low bar (the F32 path is free on 35B-A3B today); the BF16 job is a performance proposition
first, against a **+51 %** hipBLASLt prefill advantage.

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
| `NEXT-SESSION-PROMPT.md` | cold-start prompt for the next session (P2 arm A / P3) |
| `tools/hbl-classes.sh` | fresh-process logits-class count (the determinism instrument) |
| `tools/hbl-kernels.sh` | adds the kernel-trace pass: which TensileLite kernel per start |
| `tools/hbl-shapes.sh` | aggregate the hipBLASLt shape inventory from a log |
| `tools/experiments/mmf-max-cols-ATTEMPT.patch` | the arm-B attempt (env knob on `mmf.cu`'s 16-column cap); refuted, kept so nobody repeats it |
| `evidence/` | the dated measurement summaries quoted in §1 — `p0-inventory-all-models.txt`, `p0-shapes-flash-next.txt`, `p1-hipblaslt-outside-slab.txt`, `p2-mmf-cap-attempt.txt` are this session's |
