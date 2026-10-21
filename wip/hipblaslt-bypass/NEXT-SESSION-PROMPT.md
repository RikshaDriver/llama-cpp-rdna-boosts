# Next-session prompt — `wip/hipblaslt-bypass`

Paste this in as the first message of a fresh session.

---

We are continuing the **hipBLASLt bypass** campaign in `~/llama-cpp-rdna-boosts/wip/hipblaslt-bypass/`.
Read, in this order, before doing anything:

1. `wip/hipblaslt-bypass/README.md` (the campaign: why, what is measured, the design space, the gates),
2. `wip/hipblaslt-bypass/FINDINGS-2026-10-21-todo-75.md` (the closure evidence that started it),
3. `wip/hipblaslt-bypass/evidence/p0-*.txt`, `p1-*.txt`, `p2-*.txt` (this session's measurements),
4. `AGENTS.md` (policies), `GREEDY-PURITY.md` §44 (hipBLASLt doctrine) and §19/§24/§25 (band uniformity).

**State at handover (2026-10-21, end of session).**

* **Driver = CORRECTNESS** (maintainer decision).  hipBLASLt's per-start solution lottery is the last source of
  run-to-run nondeterminism in the set.  The VRAM/headroom motivation was **measured and dropped**: hipBLASLt
  holds a constant **~70 MiB** outside the slab, independent of ubatch (512..8192) and prompt length
  (602..68 262 tokens) — see `evidence/p1-hipblaslt-outside-slab.txt`.
* **P0 (inventory) is done** for 4 models: dense 27B = **0** hipBLASLt dispatches; gemma4 26B-A4B = 1 F32 shape
  (the router); 35B-A3B = 3 F32 shapes (`ssm_alpha`/`ssm_beta` + router); Flash-Next GSQ = **27 BF16 shapes /
  2790 dispatches**.  Not yet done: Flash-Next IQ4_NL, and any run with an MTP draft (`-md`).
* **P1 (VRAM) is done** — ~70 MiB, constant.
* **P2 arm B is REFUTED as cheap** (`evidence/p2-mmf-cap-attempt.txt`): RDNA has **no F32 MMA GEMM**
  (`amd_mfma_available` is CDNA-only, `amd_wmma_available` is f16/bf16), and the 16-column cap in `mmf` is a
  **kernel** limit (`mmf.cuh:771 GGML_ASSERT(ids || ncols_dst <= 16)`), not a heuristic.
* Nothing is in `patches/`; the experiment patch is `tools/experiments/mmf-max-cols-ATTEMPT.patch` (applied to
  a working tree, measured, reverted).

**This session's goal — in this order:**

1. **P2 arm A (cheap, might end the campaign): `HIPBLASLT_TUNING_OVERRIDE_FILE`.**  Pin a solution index for the
   35B-A3B F32 alpha/beta shape and for the gemma4 router shape, then run 12 fresh `llama-perplexity -c 512`
   starts (`tools/hbl-classes.sh`) and confirm **one class**, plus prefill-logit KLD vs the unpinned default and
   an interleaved `llama-bench` visit.  If a shipped tuning file is enough, that is a delivery-shaped answer
   that needs no kernel at all — report it to the maintainer before writing anything.
2. **P3(ii) the F32 skinny GEMM** (the correctness win in-set): `M = 32` (35B-A3B alpha/beta) and `M = 128`
   (gemma4 router), `K = 2048/2816`, `N = ubatch`.  No MMA form exists, so this is a SIMT tiled kernel
   (block-per-token, weight streamed once, our own K-split order).  The bar is low: the F32 path is **free** on
   35B-A3B (`ROCBLAS_USE_HIPBLASLT=0` measures +1-3 % prefill there).  First milestone is not speed — it is
   *bit-identical across fresh starts* plus a prefill-logit KLD number against the current default.
3. Only if there is time: **P3(i)** the column-tiled non-ids MMF path for BF16 (Flash-Next), where the bar is
   high (**2405 vs 1595 t/s prefill**, `-ub 8192`, 68K prompt).

**Discipline that bit previous sessions.**  A config that does not flip proves nothing — validate every
determinism claim on the `llama-perplexity -c 512 --chunks 1` config (which flips reliably here) with 8-12
fresh starts, never on a greedy text hash.  One GPU job at a time.  Flash-Next runs must use
`--load-mode mmap --lazy-mode off` (fastest load, most consistent: the PLE weights are SSD-resident).  Report
findings to the maintainer before any delivery-affecting change.
