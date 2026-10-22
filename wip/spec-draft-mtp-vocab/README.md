# spec-draft-mtp-vocab — `--spec-draft-mtp-vocab N`: the MTP draft scores a trimmed LM-head row subset

**Status: IMPORTED (2026-10-21) + 3 campaign patches, gated on this box, NOT part of the delivery.**  Nothing
here may be applied to the fork or folded into `patches/` without the maintainer's explicit go-ahead (see the
WIP and promotion rules in `AGENTS.md`).  Origin: discussion **#202** (martino2k6), branch
[`martino2k6/llama-cpp-rdna-boosts:wip/spec-draft-mtp-vocab`](https://github.com/martino2k6/llama-cpp-rdna-boosts/tree/wip/spec-draft-mtp-vocab)
at commit `4534e32` (`007da0f4`): six format-patches cut on `v12-097cf33d9-r101`.  Patches **0007-0009** are
this campaign's own (see §2).  Reporter's box: gfx1151 / Strix Halo, ROCm 7.2.4.

**Where it stands (gfx1201, 2x R9700, ROCm 10.1.0, 2026-10-21).**  Nine patches apply to **r106** with
`git am -3`, no conflict, and build clean.  Summary:

| question | answer |
|---|---|
| 27B Q8_0 `-sm tensor` (the config that must work) | **works** — the abort was a `ggml-backend-meta` limitation on reading a row range out of a vocabulary-split tensor, fixed by **0007** |
| `--spec-type draft-mtp-adaptive --spec-draft-n-max 9`, 27B Q8_0 2 GPU `-sm tensor` | **+3.4…+5.5 %** decode on the delivery's phase-switching prompt (69.7 -> 72.1 t/s); +5.5 % on a short synthetic prompt |
| gemma4 (`gemma4-assistant` sidecar) | **supported** (0008 + 0009) — but **not beneficial**: its useful cutoff is 245760 of 262144 rows, i.e. a no-op; below that acceptance halves and it is a 30 % *loss* |
| greedy purity | identical text when the acceptance trajectory is the same; across trajectories it can differ — and the **untrimmed delivery build already differs between `--spec-draft-n-max 3` and `9`**, so that is a schedule effect, not the trim (F9) |
| trim off | inert: the patched build reproduces the r106 delivery build's text exactly at n-max 1, 9 and adaptive-9 |

Detail: §3 (numbers), §4 (gate plan), §5 (findings).  Scratch tree used for every number below:
`/tmp/lc-vocab` (= the 9-patch tree, `tree 60fc64c6d21cacd58542d3a5fc2c1860092d0442`).

> **Cold start.**  Read `AGENTS.md` (WIP and promotion rules; the default-on policy; the adaptive-MTP gate and
> the prefill-logit gate), then `benchmarks/mtp-adaptive-methodology.md`.  This is a **decode-side, draft-only**
> change: the prefill-logit gate does not apply, the adaptive-MTP gate and the greedy-purity grid do.

## 0. Why it is interesting

The MTP draft head is an **LM-head matmul over the full vocabulary** (151k-262k rows on the current Qwen /
Flash-Next / gemma4 payloads) executed once per draft step — up to `--spec-draft-n-max` times per target
token.  Trimming the scored rows cuts the draft step (reporter: 4.3 ms -> 1.6 ms on gfx1151) and the whole
token by x1.0648 there.  The draft is only a *proposal*: verification is untouched, an id the trim drops
simply loses its draft win.  It is a decode win with no kernel work — a smaller `mul_mat` plus a
`fill`/`set_rows` scatter.

## 1. How it works

**(a) The flag.**  `--spec-draft-mtp-vocab N` (`common/arg.cpp`, `common_params_speculative_draft::mtp_vocab`,
plus the `LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB` env alias from 0005) is copied into
`llama_context_params::mtp_draft_vocab` **only for the MTP draft context** — `common/common.cpp:1283` (the
`-md`/`--model-draft` sidecar path) and `common/speculative.cpp:2884` (the in-tree speculator path).  The
target context never sees it, so the target's own `output` projection is untouched.  `0` = off = today's
code path byte-for-byte.

**(b) The subset.**  When the MTP context is created (`src/llama-context.cpp:802`), it asks the model for an
immutable `llama_mtp_draft_vocab` (`llama_model::mtp_draft_vocab_get`, `src/llama-model.cpp:2895`):

- **Row selection**: token ids `t < N` **plus** every id carrying `LLAMA_TOKEN_ATTR_CONTROL` or
  `LLAMA_TOKEN_ATTR_USER_DEFINED` (so `</s>`, `<|im_end|>`, tool-call ids stay draftable — without them the
  draft could never propose a stop token).
- **Why a plain cutoff is legitimate**: BPE ids are assigned in merge order, so low ids are the short,
  frequent merges.  A guard (`mtp_draft_vocab_ids_follow_merge_order`) walks the merge table and requires a
  BPE vocab, at least `n_vocab/2` merges, and each merge-produced id to be >= the previous one; otherwise the
  flag is **ignored with an `LLAMA_LOG_ERROR`** (`the tokenizer is not BPE` / `token ids do not follow BPE
  merge order`).  **0009** fixes that check: it compared against the *running maximum*, which a single
  early high id poisons (gemma4 maps a few leading-space pieces to ~255998, so it read "236336 of 236339
  out of order" while its ids actually track merge rank with corr **+0.992**).  It now compares against the
  previous merge-produced id and tolerates <= 1 % inverted pairs (gemma4: **30 of 236339, 0.013 %**;
  qwen35: 0).  A sentencepiece/UGGUF vocab is still rejected.
- **The subset tensors**: `head` is `[n_embd, n_sel]`, a **row copy in the LM head's own quant type** (no
  dequant/requant/repack — which is why it cannot change a logit), plus `ids`, an `I64[n_sel]` map back to
  real token ids.  Rows are copied host-side from `model.output` in **contiguous runs** (one copy per run),
  into a buffer of the LM head's own buffer type, tagged `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`.
- **Guards**: the MTP graph must be one of `qwen35`, `qwen35moe`, `qwen4exp` (nextn block or borrowed target
  head) or `gemma4`/`gemma4-assistant`; the block must score with the model LM head; probe contexts
  (`common_fit_params`, unallocated weights) return `nullptr`.  See **0008** for the gemma4 arm.
- **Ownership (0003)**: per context — `llama_context` holds a `shared_ptr<const llama_mtp_draft_vocab>`, the
  model keeps weak references keyed by `(n_keep, buffer_type)` under a mutex, the graph sees it through
  `llm_graph_params::mtp_draft`.  Two contexts with the same N share one copy; `N` then `0`, or `N1` then
  `N2`, do not leak into each other; the copy is freed with the last holder; graph reuse compares the pointer
  and the cached `cparams.mtp_draft_vocab`.

**(c) The graph.**  Only in the MTP graph (`src/models/qwen35.cpp:666`, `qwen35moe.cpp:739`,
`qwen4exp.cpp:1314`, `gemma4-assistant.cpp:196`), when a subset exists **and** the head the graph would score
with is the subset's source tensor **and** `n_outputs == 1`:

```c
sub = build_lora_mm(mtp_draft->head, cur, head_s);              // [n_sel, 1]  -- the smaller matmul
cur = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, F32, 1, n_row), -INFINITY);   // full-vocab row of -inf
cur = ggml_set_rows(ctx0, cur, ggml_reshape_2d(ctx0, sub, 1, n_sel), mtp_draft->ids);  // scatter
cur = ggml_reshape_2d(ctx0, cur, n_row, 1);
```

The speculator still receives a **full-vocab logit row** — same shape, same consumers, same sampler chain —
with `-inf` exactly at the ids the draft may not propose.  So a token that was never drafted can never be
accepted, and a token is accepted only when the *target* agrees; the trim can only lose proposals.

**(d) 0006 — borrowed head.**  Shared-NextN sidecars (`nextn_shared_target_tensors`) carry no
`output.weight`; the draft borrows the target's head, so the donor hook was inert.  0006 falls back to the
target via `cparams.ctx_other`, skips the target's own-MTP-block checks (a shared-NextN target has
`n_layer_nextn == 0`, so 0003's `layers[n_layer()]` access would read past its layer array) and identifies
the tensor by `llama_mtp_draft_vocab::src` instead of `model.output`.

**(e) 0008 — gemma4.**  The gemma4 MTP sidecar (`arch gemma4-assistant`) scores its draft step with its own
**tied token embedding** (`output = token_embd.weight`, `TENSOR_DUPLICATED`), so it takes the same hook.
The arch guard allows `LLM_ARCH_GEMMA4_ASSISTANT` and skips the nextn checks for it (its layer array has
exactly `n_layer()` entries and no nextn sub-struct).  Works; see §3.3 for why it does not pay.

## 2. Patch inventory

Imported (unchanged, authors preserved) + three campaign patches.  All 9 apply to r106 with `git am -3`,
zero conflicts; applied tree `60fc64c6d21cacd58542d3a5fc2c1860092d0442` (verified on a fresh worktree of the
r106 tree).

| patch | author | what |
|---|---|---|
| `0001-spec-MTP-draft-vocabulary-subset-spec-draft-mtp-voca.patch` | Simon Teixidor | the flag, the subset build, the `qwen35`/`qwen35moe` scatter |
| `0002-tests-per-context-MTP-draft-vocabulary-regression-te.patch` | Simon Teixidor | `tests/test-mtp-draft-vocab.cpp` (built, **not** registered with ctest — it needs `-m <qwen35 MTP model>`) |
| `0003-spec-make-the-MTP-draft-vocabulary-subset-per-contex.patch` | Simon Teixidor | per-context ownership, model-side weak-ref cache, `llm_graph_params::mtp_draft` |
| `0004-qwen4exp-MTP-draft-vocab-subset-spec-draft-mtp-vocab.patch` | Laurent Zuijdwijk | the `qwen4exp` (Flash-Next) MTP graph + the arch guard |
| `0005-common-LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB-env-var-for-sp.patch` | Laurent Zuijdwijk | `LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB` env alias |
| `0006-qwen4exp-build-the-MTP-draft-vocab-subset-from-the-b.patch` | Martino B (martino2k6) | shared-NextN sidecar: build the subset from the borrowed target head |
| **`0007-meta-read-a-row-range-out-of-a-row-split-2-D-tensor.patch`** | **campaign** | **F2 fix**: `ggml_backend_meta_get_tensor_impl` can splice a row range of a *row-split* 2-D tensor (a vocabulary-split LM head) device by device |
| **`0008-qwen4exp-MTP-draft-vocab-subset-for-gemma4-assistant.patch`** | **campaign** | gemma4-assistant MTP graph + arch guard |
| **`0009-spec-accept-near-merge-ordered-token-ids-not-strictl.patch`** | **campaign** | merge-order guard: previous-id comparison + 1 % tolerance (gemma4 = 0.013 %) |

Applied-state footprint: 18 files, +~560/-15 lines incl. the 243-line test.  **One ggml file now**
(`ggml-backend-meta.cpp`) — promotion is no longer graph-only, so §4 step 7 is a real gate.

Provenance (reporter's words, kept because it matters for upstreaming): 0001-0003 originated in
Aristo94/GSQHalo.cpp (Simon Teixidor, 2026-09-13/15), were cherry-picked into agentionai/llama.cpp where
0004-0005 were added (Laurent Zuijdwijk, 2026-10-01), 0006 is martino2k6's own.  MIT, authorship preserved.
**Not** the same feature as ggml-org/llama.cpp#30380 (that is a *dynamic* trimmed-vocab draft).

## 3. Measurements (gfx1201, 2x AI PRO R9700, ROCm 10.1.0, 2026-10-21)

`--temp 0 --seed 42`, greedy text hashed with `scripts/extract-generated.py`.  **Shallow depth** (prompts are
296-5246 tokens, context sized 16384) — the delivery's decode work is normally validated at depth 16384, so
treat these as shallow-depth numbers.

### 3.1 27B Q8_0 (`qwen35`, vocab 248320, n_embd 5120) — the config that must work

Prompt `prompts/code-reasoning-mixed.txt` (296 tok), `-n 512`, 2 GPU:

| speculator | split | trim | gen t/s | greedy text |
|---|---|---|---|---|
| `draft-mtp-adaptive`, n-max 9 | tensor | 0 | **69.7** (delivery build: 69.3) | `5612aa128fc3` (= delivery build) |
| | tensor | 64000 | **72.1** (+3.4 %) | `59f85f3d2ef9` (see F9) |
| | tensor | 98304 | 70.8 | `5612aa128fc3` |
| `draft-mtp`, n-max 9 (fixed) | tensor | 0 | 67.0 | `ec1a197011c4` (= delivery build) |
| | tensor | 64000 | **69.9** (+4.3 %) | `ec1a197011c4` — **identical** |
| `draft-mtp`, n-max 1 (fixed) | tensor | 0 | 53.8 | `92010829c9f6` (= delivery build) |
| | tensor | 64000 | 54.4 (+1.1 %) | `92010829c9f6` — **identical** |

Short synthetic prompt (`The capital of France is`, `-n 256`), 2 GPU tensor, adaptive n-max 9:
0 -> **61.8**, 64000 -> **65.2 (+5.5 %)**, 98304 -> 63.0, 131072 -> 60.9; **all four text hashes identical**
(`da2e2d192e21`).

`-sm layer` (2 GPU, `-n 256`): 0 -> 40.6 / 64000 -> 43.5, 42.9 / 98304 -> 42.8, 42.9; text identical
throughout.

Acceptance (adaptive n-max 9, tensor, mixed prompt): trim 0 = **0.4851** (374/771, mean len 3.73), trim
64000 = **0.4757** (372/782, mean len 3.70); per-position 0.825/0.664/0.555/0.409/0.219/0.036/... vs
0.833/0.645/0.529/0.391/0.239/0.029/... — the trim costs ~1 pt of acceptance, deepest positions were already
~0.

4B Q8_0, 1 GPU, `-n 256`: 0 -> 203.9, 64000 -> 216.1 (+6.0 %), 98304 -> 231.9 (+13.7 %), two rounds each,
text identical.  (A small model has a proportionally bigger draft step — the cutoff sweep is per payload.)

### 3.2 `-sm tensor` before 0007 (the F2 abort, for the record)

```
/tmp/lc-vocab/ggml/src/ggml-backend-meta.cpp:2319: GGML_ASSERT(size % chunk_size_full == 0) failed
```

`llama-cli` 27B Q8_0 + `--spec-draft-mtp-vocab 64000 -sm tensor` died at MTP-context creation: the subset
build reads a row range out of the **vocabulary-split** LM head, and the Meta buffer's splice path serves a
2-D axis-1-split tensor only as a whole tensor (its chunk is `nb[axis + 1]`, which for a 2-D tensor *is* the
tensor).  It was a clean host assert before decode (no GPU fault, `check-gpu-faults.sh` clean), and the same
binary was fine at `-sm layer` and at `-sm tensor` with the trim off.  0007 adds the row-granular path; after
it, `-sm tensor` + trim runs at every N in the tables above.

### 3.3 gemma4 26B-A4B QAT + `gemma-4-26B-A4B-it-Q4_0-MTP.gguf` (`gemma4-assistant`)

`gemma4-assistant` **does not implement `-sm tensor`** (load-time rejection), so this is `-sm layer`, 2 GPU,
`-n 256`, mixed prompt, `draft-mtp` n-max 3:

| trim | rows kept | gen t/s | acceptance | mean len |
|---|---|---|---|---|
| 0 (full vocab) | 262144 | **150.3** | 0.669 | 3.00 |
| 64000 | 64006 (35.2 MiB q4_0) | 104.5 | 0.339 | 2.02 |
| 131072 | 131078 (72.0 MiB) | 105.7 | 0.355 | 2.07 |
| 196608 | 196614 (108.0 MiB) | 106.9 | 0.369 | 2.11 |
| **245760** | 245766 (135.0 MiB) | **150.5** | **0.669** | **3.00** |
| 64000 + last 32768 (two bands) | 96768 | 144.6 | 0.610 | 2.82 |

**So gemma4 is supported but not worth trimming on this payload.**  The reason is the tokenizer layout: of
762 sampled draft candidates, 70 % are ids < 64000 and **24 % sit in the 229376..262143 band** (gemma4's
space-prefixed pieces live at the top of the vocabulary), and the *top-1* is split across both — so any cut
below ~245760 halves acceptance, and at 245760 there is nothing left to save and the `fill`/`set_rows`
overhead makes it a wash (150.5 vs 150.3).  A two-band subset (keep `< N` **and** `>= n_vocab - M`, measured
with a temporary env knob, not committed) recovers acceptance to 0.610 at 37 % of the rows but is still
0.4 % *slower* than full vocab: gemma4's draft step is cheap relative to its 26B MoE target, so the row
saving cannot pay for the scatter.  If gemma4 support is wanted for its own sake, the two-band idea is the
route — as a follow-up, with a real CLI knob, not a hidden env.

## 4. Gate plan / what is left

Done here: **1** build + activation line, **2** greedy purity (all same-trajectory comparisons), **3** the
adaptive gate's decode/acceptance axes on one prompt, **4** a first N sweep (27B + 4B + gemma4), **5**
kill-switch identity (untrimmed arms reproduce the delivery build's text at n1/n9/adaptive-9; `0` = off),
**6** 27B `-sm tensor` (2 GPU) — works.

Still open before any ship decision:

1. **Adaptive-MTP gate, properly** — `-n 3000` (rule 0), `--reasoning on` for R and `off` for P/C/K, the
   full R/C/K + phase-switching set, and the *climb/drop trajectory* comparison (`n_accepted` drives the
   controller; the trim lowers acceptance by design).  Our numbers are one prompt at `-n 512`.
2. **Depth** — repeat at depth 16384 (the delivery's decode validation depth).
3. **3 GPU** — `-sm tensor` on 3 devices, and the `-sm layer` arm for comparison.
4. **`GGML_META` regression** for **0007**: `test-backend-ops`, `test-logits-width-probe SPLIT=tensor`, plus a
   direct full-read-vs-row-range equality check on a split weight.  This is a `ggml/` change now.
5. **`--fit` accounting** (F3) and the default-on policy decision (F4) if it is ever promoted.

## 5. Review findings

**F2 — FIXED (0007), verified.**  See §3.2.  `-sm tensor` + trim works at 27B Q8_0 on 2 GPU.  The fix is a
row-granular splice in `ggml_backend_meta_get_tensor_impl` for a 2-D tensor split on axis 1 (per
segment/repeat, so a repeated split works too); everything else keeps its old path.  It is a general
meta-buffer improvement, not a special case for this feature.

**F9 — greedy text is a function of the acceptance trajectory, in the delivery too (NEW, pre-existing).**
With **no campaign code at all**, the r106 delivery build produces *three different* greedy texts on the
mixed prompt depending only on draft depth:

| delivery r106, `-sm tensor`, mixed prompt, 512 tok | greedy text |
|---|---|
| `draft-mtp` n-max 3 | `92010829c9f6` |
| `draft-mtp` n-max 9 | `ec1a197011c4` |
| `draft-mtp-adaptive` n-max 9 | `5612aa128fc3` |

(`-sm layer`: n-max 3 `348b4bb78c8a` vs n-max 9 `921afe655051` — same story.)  Verify-width-dependent
numerics flip a near-tie somewhere in a 500-token generation, so "greedy bytes are identical" is only
meaningful **at a fixed trajectory**.  Consequences:

- The trim's purity is verified the right way here: identical text at **fixed** n-max (1 and 9) and on the
  short prompt under adaptive-9.  The single divergence is the adaptive arm on the 512-token prompt, where the
  trim changes acceptance -> the controller changes depth -> the trajectory changes: the two arms produce
  **identical text for the first 1952 characters (76.5 %)** and then a near-tie flips on a code/reasoning
  phase switch (2560 vs 2492 chars, EOS at a different point).  That is the same effect as the untrimmed
  n3-vs-n9 difference above, and it is a property of the *schedule*, not of the trimmed subset.
- A promotion should therefore **not** gate on "trim == no-trim bytes" across trajectories.  Gate on
  same-trajectory identity plus the structural argument (target graph untouched; `-inf` means a draft cannot
  propose outside its own subset).  This belongs in `GREEDY-PURITY.md` as a §entry either way: it is the first
  measurement in this repo of greedy text depending on `--spec-draft-n-max` alone.

**F1 — arch coverage (partly done).**  `qwen35`, `qwen35moe`, `qwen4exp` (imported) + `gemma4-assistant`
(0008, not beneficial).  Not wired: the DFlash / deepseek4 / glm5-next MTP families — unknown whether worth
it (their draft heads are not necessarily the LM head).

**F3 — the subset copy is unaccounted VRAM.**  Measured: **332.20 MiB** of q8_0 for 64033 of 248320 rows on
the 27B (`n_embd` 5120), 166.10 MiB on the 4B, 35.16 MiB of q4_0 for gemma4.  It is allocated when the MTP
context is created, i.e. **after `--fit`** has committed its estimate (probe contexts are skipped on
purpose).  On tight per-card configs that can tip an allocation into shared memory.  Options: teach the
estimator, account it as draft weight bytes, or document a size.

**F4 — opt-in vs the default-on policy.**  A numeric knob whose sweet spot is payload-specific (27B/4B:
64000-98304 wins; gemma4: only a no-op cutoff keeps acceptance).  "Default 0 + flag" is defensible only if a
recommended per-payload N is published and validated; a default-on trim would also change every published MTP
decode number and require re-recording the adaptive-MTP baseline.  Decision at promotion.

**F5 — purity bookkeeping.**  Needs the `GREEDY-PURITY.md` §entry from F9 (draft-side restriction cannot change
an accepted token; a trim can only drop proposals; greedy text is trajectory-dependent).

**F6 — public struct field mid-`llama_context_params`** (ABI break for prebuilt bindings).  Fine in-tree;
upstream-PR material wants it at the tail.

**F7 — small.**  The flag is silently inert outside MTP spec types (no warning); `LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB`
needs an `ENVIRONMENT.md` row on promotion (kind: opt-in tuning value, `0` = off).

**F8 — confirmed on our tree.**  The activation line is `LLAMA_LOG_INFO` and our `llama-cli` hides INFO at
default verbosity (a 4B A/B printed no loader or core INFO lines at all), so a working trim and a silently
rejected one look identical.  Every A/B here was run with `-v`.  Promote the line to `LLAMA_LOG_WARN`, or add
an `mtp_draft_vocab` gauge next to the draft counters.

## 6. Open questions for the maintainer

1. **Do we want it at all?**  On the delivery's own recommended config (27B Q8_0, 2-3 GPU `-sm tensor`,
   adaptive n-max 9) it is **+3.4 %** on the phase-switching prompt and **+5.5 %** on a short synthetic one,
   for a 332 MiB extra copy and a new `ggml-backend-meta` read path.  gemma4 gains nothing.  The remaining
   gates (§4) are cheap except the meta regression suite and the depth/3-GPU arms.
2. **`-sm layer`-only is no longer a fallback** — 0007 makes `-sm tensor` work, so the question is whether the
   `ggml/` change is acceptable as part of the delivery (it is small and general, but it *is* core ggml).
3. **If it gates green**: new block 12, or a fold into **block 02** (adaptive MTP)?  It is decode-side and
   independent of the controller, so block 02 is the natural home.  F4 decides whether a default-on variant
   is even on the table.
