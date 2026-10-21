# Next-session prompt — `wip/hipblaslt-bypass`

Paste this in as the first message of a fresh session.

---

We are continuing the **hipBLASLt bypass** campaign in `~/llama-cpp-rdna-boosts/wip/hipblaslt-bypass/`.
Read, in this order, before doing anything:

1. `wip/hipblaslt-bypass/README.md` (the campaign: why, what is measured, the design space, the gates),
2. `wip/hipblaslt-bypass/FINDINGS-2026-10-21-todo-75.md` (the closure evidence that started it),
3. `AGENTS.md` (policies: default-on, report-findings-first, no delivery patch changes without a go-ahead),
4. `GREEDY-PURITY.md` §44 (hipBLASLt doctrine) and §19/§24/§25 (band uniformity, for any kernel arm).

**State at handover (2026-10-21).** TODO #75 is closed as the hipBLASLt per-start solution flip: 3 logits
classes in 8 fresh `llama-perplexity` starts on ROCm 10.1.0, 1 class in 8/8 with `ROCBLAS_USE_HIPBLASLT=0`,
and the class equals the TensileLite kernel spine hipBLASLt picked.  On this box, for 35B-A3B, disabling
hipBLASLt is **free** (+1-3 % prefill, flat decode) and costs ~70 MiB of VRAM — so the campaign's remaining
motivation is determinism, not throughput.  Nothing is in `patches/`; nothing is to be put there this session
without the maintainer's explicit go-ahead.

**This session's goal: complete P0 and P1, and answer P2 if time allows.**

* **P0 — inventory.**  For each model below, produce the hipBLASLt shape table with
  `wip/hipblaslt-bypass/tools/hbl-shapes.sh -- <llama-cli ...>`: model, shape (A/B), calls per run, and the
  token widths that reach it.  Models: `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/...` (GDN alpha/beta, known),
  `/llm/models/Qwen3.8/27B/Q4_K_XL/...` (dense, expected **none** — confirm), the Gemma4 26B-A4B QAT tree
  (F32 MoE router, known), and a Flash-Next GSQ model.  Deliverable: one table in the campaign README —
  that table is the scope of the whole campaign.
* **P1 — is the VRAM/slab claim real?**  On the `-ncmoe` + slab path (35B-A3B, e.g. `-ncmoe 30 -sm layer`),
  measure peak VRAM and the slab's reported reservation with hipBLASLt on vs off, and check whether an
  unaccounted hipBLASLt allocation ever triggers the `ggml_cuda_device_malloc` arena-drop fallback.  If the
  delta stays ~70 MiB, say so in the README and in `TODO.md` #82 (the `~2 GB` motivation is then dead and
  must not be repeated).
* **P2 — cheapest viable arm.**  (a) Pin the solution with `HIPBLASLT_TUNING_OVERRIDE_FILE` and confirm the
  flip is gone without any code change; (b) raise `mmf.cu`'s `src1_ncols > 16` cap behind a kill switch and
  measure prefill (512/2048/8192 columns) on 35B-A3B + gemma4.  Both arms: 12 fresh logits-class samples
  (`tools/hbl-classes.sh`) + prefill-logit KLD vs the current default + `llama-bench` interleaved visits.

**Discipline that bit previous sessions.**  A config that does not flip proves nothing — validate every
determinism claim on the `llama-perplexity -c 512 --chunks 1` config (which flips reliably here) with at
least 8-12 fresh starts, never on a greedy text hash.  One GPU job at a time, no background benches.  Report
findings to the maintainer before any delivery-affecting change.
