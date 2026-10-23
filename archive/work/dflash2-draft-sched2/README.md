# `dflash2-draft-sched2` — issue #208: a DFlash2 draft block never reached the second scheduler

**Status: RESOLVED (2026-10-23), shipped in `v14-097cf33d9-r120` as an amendment to block 09** (the
second draft scheduler lives there).  Nothing stays in `wip/`: the campaign was closed in the same
session it was opened.

Issue: [stew675/llama-cpp-rdna-boosts#208](https://github.com/stew675/llama-cpp-rdna-boosts/issues/208)
(`@overdoingism`, r112, 1x gfx1201).  `fix-208.patch` here is the delivered source delta
(`src/llama-context.cpp` + `src/llama-context.h`, against the pre-fix block-09 tree);
`tools/` holds the harnesses used for the numbers below.

## What was wrong

The draft context's second scheduler (r65, `LLAMA_DRAFT_SCHED2`) was selected with
`sched_alt_select(n_outputs > 0)`, and the graph-reuse slot with `gf_res_prev[n_outputs > 0]`.  That
split fits MTP: a no-output ingest (KV injection) graph, then a 1-token draft step that has an output.

A DFlash2 draft block has **no outputs either**.  `common_speculative_impl_draft_dflash::draft()` adds
the noise-block tokens with `output = !is_dflash2`, because the selector lattice is read from the nextn
embeddings (`h_nextn`), not from logits.  So for a DFlash2 draft *both* graphs have `n_outputs == 0`:
the KV injection and the draft step shared one scheduler **and** one `gf_res_prev` slot, overwrote each
other's graph result every cycle, and the draft context rebuilt and re-allocated both graphs on every
verify step.  r84's #183 fix (the allocator layout fingerprint) is unrelated and does not help here.

Reproduced on the pre-fix r119 tree with a temporary env-gated trace in `process_ubatch` (0 of 362
draft ubatches reused, see the table):

```
DRAFT_GRAPH n_tokens=6 outputs=0 embd=0 inject=0 slot_alt=0 reused=0 ...   <- draft step
DRAFT_GRAPH n_tokens=6 outputs=0 embd=1 inject=0 slot_alt=0 reused=0 ...   <- KV injection
```

## The fix

`sched_alt_for(const llama_ubatch &)` (new, `src/llama-context.cpp`) owns the decision, and
`get_gf_res_prev(bool alt)` takes it as a parameter so the scheduler and the graph slot cannot disagree:

* `LLM_ARCH_DFLASH` — the **batch kind** decides: the injection is the batch that carries the target
  features (host `embd` rows, or the F1 device injection with `cparams.lf_dev_inject`), so
  `ubatch.embd == nullptr && !cparams.lf_dev_inject` is the draft step.  The partition is total and
  exclusive for a DFlash draft context, and it is behaviour-preserving for a DFlash (non-2) or DSpark
  draft, whose step graph has outputs and already landed on the second scheduler.
* every other arch — `n_outputs > 0`, unchanged.  The batch kind must **not** be used for MTP: its
  ingest is also a token batch, so the two MTP graphs would be merged instead of separated.

No new environment variable: the r65 `LLAMA_DRAFT_SCHED2=0` kill switch covers this path too, and now
reproduces the pre-fix behaviour exactly (measured below).  `LLAMA_DRAFT_SCHED2` on remains the default
per the default-on policy.

## Measurements (gfx1201, 3x R9700, ROCm 10.1.0)

Qwen3.8-27B-UD-Q4_K_XL target + Qwen3.8-27B-DFlash2-Q4_K_M draft, `--spec-type draft-dflash
--spec-draft-n-max 5`, `-c 16384 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 -t 8`, prompt
`prompts/prose-rdna-boosts.txt` (5298 tokens), 512 greedy tokens, seed 42.  Interleaved base/fix
repeats on the same binary pair (only `libllama.so` differs).  `base` = the fix reverted.

Every row is a clean base build against a clean fix build (only `libllama.so` differs, md5
`cf8a96ae…` / `b9de5964…`), arms interleaved; the mean of the repeats is shown.

| arm | base t/s | fix t/s | delta | text |
|---|---:|---:|---:|---|
| 27B, 2 GPU `-sm tensor` (3 repeats) | 94.07 | 95.50 | **+1.5 %** | identical `7cce40e8f7b8` |
| 27B, 3 GPU `-sm tensor` (3 repeats) | 110.20 | 112.13 | **+1.8 %** | identical `5d5af6473255` |
| 27B, 1 GPU `-sm tensor` (2 repeats) | 66.05 | 66.15 | +0.2 % | identical `c05a1461c412` |
| 27B, 2 GPU `-sm layer` (2 repeats) | 60.35 | 60.70 | +0.6 % | identical `c05a1461c412` |
| 27B, 2 GPU, `GGML_LF_DFLASH_DEV=0` (host injection, 2 repeats) | 94.30 | 95.75 | +1.5 % | identical `7cce40e8f7b8` |
| 35B-A3B-UD-Q4_K_M `-ncmoe 40` + DFlash2-Q8_0, 2 GPU (3 repeats) | 203.23 | 210.83 | **+3.7 %** | identical `18b8b5868cbd` |
| 27B, 2 GPU, `-np 2` single request (2 repeats) | 94.20 | 95.60 | +1.5 % | identical `7cce40e8f7b8` |
| 27B, 2 GPU, 2-slot unified server, 2 concurrent requests | 52.50 / 97.21 | 53.05 / 99.05 | +1.0 / +1.9 % | identical per slot, acceptance 0.66894 / 0.73801 |
| 27B, 2 GPU, `LLAMA_DRAFT_SCHED2=0` (2 repeats) | 94.25 | 94.20 | 0 | identical `7cce40e8f7b8` |

Draft graph reuse (trace, last 400 draft ubatches of a 512-token run):

| arm | base | fix |
|---|---|---|
| 27B 2 GPU: injection / draft step | 0 / 0 | 177 / 177 of 362 (**97.8 %**) |
| 35B-A3B `-ncmoe 40`: injection / draft step | 0 / 0 | 164 / 164 of 335 (**97.9 %**) |
| 27B 2-slot server, two concurrent requests | 29.9 % | 86.4 % (injection) / 95.0 % (draft step) |

The base numbers match the reporter's ("draft context reused 0 of 100 graphs", 0.27-0.30 ms per draft
decode): the win is a per-verify-step host cost, so it scales with how fast the target decodes (it is
largest on the MoE host-expert arm and smallest on one GPU).  The reporter's 1-GPU -1.0 % at 48 ms/step
is a slower per-step shape than any arm measured here (245760 context, 2 slots); the mechanism and the
reuse result reproduce exactly.

## Gates

* **Same-seed coherence**: generated text byte-identical base vs fix in *every* arm above; dense
  `Qwen3.5-4B Q8_0` 3-GPU `-sm tensor` `-n 20` gate hash `1c5d32ac537d` in both builds.
* **Adaptive-MTP four-axis gate** (27B UD-Q4_K_XL, embedded MTP head, `-n 3000`, reasoning pinned per
  axis, 1 GPU): identical t/s, acceptance and text on all four axes (R 59.3/59.4, P 75.2/75.3,
  C 81.1/81.1, K 87.5/87.6; acceptance 0.57070 / 0.82903 / 0.90119 / 0.99200 in both arms; text
  `89b8dc5d4c70` / `65b3ea19d90f` / `6536aa147f62` / `a87c4318b649` in both).  MTP still takes the
  `n_outputs` branch, so this is the regression check for the shared code path.
* **Two-slot concurrency** (the #183 shape): identical text and per-slot acceptance with both slots
  active, so the extra scheduling split does not reintroduce the r82/r84 aliasing class.
* **Per-block build**: `gate-per-block-build.sh --changed-from 5efc1e9f1` **3/3**.  The first run of it
  exposed a *pre-existing* gap: block 09 called `heat_rank_vec()`, which block 12 introduces with the
  r114 band-split heat prior, so block 09 had not built on its own since r117 (recent releases used
  `--changed-from` and never selected it).  Block 09 now ranks that layer-drop site by `t.heat` (identical
  in block 09's tree, which has no decode counter) and block 12 owns the switch-over; the merged tree is
  byte-identical.
* `scripts/validate-set.sh` green (fresh tarball, strict 14/14 `git am`, applied tree `0fd9827d…` ==
  `release.json.tree`); `check-gpu-faults.sh` clean on the kernel journal and the campaign logs.
* Not run: `gate-prefill-logits.sh`.  No prefill kernel, no approximate path and no numerics changed
  (host-side scheduler selection only, byte-identical generated text); the same reasoning as r84.

## Files

| path | what |
|---|---|
| `fix-208.patch` | the delivered `src/llama-context.{cpp,h}` delta (block 09) |
| `tools/run-arm.sh` | one decode arm (label, base/fix library, n, GPUs, env, extra args) |
| `tools/run-instr.sh` | the same with the `LLAMA_DRAFT_GRAPH_DEBUG` trace (temporary instrumentation) and the reuse summary |
| `tools/two_slot_check.py` | 2-slot unified server, two concurrent requests, per-slot acceptance/reuse |
