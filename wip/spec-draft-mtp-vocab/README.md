# spec-draft-mtp-vocab — `--spec-draft-mtp-vocab N`: the MTP draft scores a trimmed LM-head row subset

**Status: IMPORTED (2026-10-21), partial gate done here, NOT part of the delivery.**  Nothing here may be
applied to the fork or folded into `patches/` without the maintainer's explicit go-ahead (see the WIP and
promotion rules in `AGENTS.md`).  Origin: discussion **#202** (martino2k6), branch
[`martino2k6/llama-cpp-rdna-boosts:wip/spec-draft-mtp-vocab`](https://github.com/martino2k6/llama-cpp-rdna-boosts/tree/wip/spec-draft-mtp-vocab)
at commit `4534e32` (`007da0f4`), six format-patches cut on `v12-097cf33d9-r101` (r101 tree `4df89305`,
applied tree `c4391f33`).  Reporter's box: gfx1151 / Strix Halo, ROCm 7.2.4.

**Where it stands on this box (gfx1201, 2x R9700, 2026-10-21):** the series applies to **r106** with
`git am -3` (no conflict) and builds clean; the trim is **active and greedy-identical**, and it reproduces
the reporter's win — **+5.7…+7.1 % decode** on `Qwen3.8-27B-Q8_0` (`-sm layer`, 2 GPU) and **+6…+14 %** on
the 4B, with the greedy text hash equal to the r106 delivery build's own.  **It hard-aborts under
`-sm tensor`** (`ggml-backend-meta.cpp:2319`, F2) because the LM head is vocabulary-split there — that
finding, not the speed, decides this campaign.  Detail: §3.1 (numbers), §5 (findings).

> **Cold start.**  Read `AGENTS.md` (WIP and promotion rules; the default-on policy; the adaptive-MTP gate
> and the prefill-logit gate), then `benchmarks/mtp-adaptive-methodology.md`.  This is a **decode-side,
> draft-only** change: the prefill-logit gate does not apply, the adaptive-MTP gate and the greedy-purity
> grid do.

## 0. Why it is interesting

The MTP draft head is an **LM-head matmul over the full vocabulary** (151k-256k rows on the current Qwen /
Flash-Next payloads) executed **once per draft step** — i.e. up to `--spec-draft-n-max` times per target
token.  On gfx1151 the reporter measures that draft step at **4.3 ms -> 1.6 ms**, and the whole token at
**x1.0648** (46.25 -> 49.24 t/s, Qwen3.8 Flash-Next Q5_K_M, ctx 131k, `--spec-draft-n-max 3`).  The draft
is only a *proposal*: verification is untouched, so greedy output is unchanged **by construction**, and an
id the trim drops simply loses its draft win (acceptance 0.6908 -> 0.6786 in their run).

It is a **decode** win with no purity risk to prefill and no kernel work: a smaller `mul_mat` plus a
`fill`/`set_rows` scatter.  That makes it a cheap candidate for a delivery that already ships
adaptive MTP.

## 1. How it works

Two mechanisms, one knob.

**(a) The flag.**  `--spec-draft-mtp-vocab N` (`common/arg.cpp`, `common_params_speculative_draft::
mtp_vocab`, plus the `LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB` env alias from 0005) is copied into
`llama_context_params::mtp_draft_vocab` **only for the MTP draft context** — `common/common.cpp:1283` (the
`-md`/`--model-draft` sidecar path) and `common/speculative.cpp:2884` (the in-tree speculator path).  The
target context never sees it, so the target's own `output` projection is untouched.  `0` = off = today's
code path byte-for-byte.

**(b) The subset.**  When the MTP context is created (`src/llama-context.cpp:802`), it asks the model for
an immutable `llama_mtp_draft_vocab` (`llama_model::mtp_draft_vocab_get`, `src/llama-model.cpp:2895`):

- **Row selection**: token ids `t < N` **plus** every id carrying `LLAMA_TOKEN_ATTR_CONTROL` or
  `LLAMA_TOKEN_ATTR_USER_DEFINED` (so `</s>`, `<|im_end|>`, tool-call and other special ids stay
  draftable — without them the draft could never propose a stop token at all).
- **Why a plain cutoff is legitimate**: BPE ids are assigned in merge order, so low ids are the short,
  frequent merges.  A guard (`mtp_draft_vocab_ids_follow_merge_order`, line 2857) walks the merge table and
  requires the vocab to be BPE, to have at least `n_vocab/2` merges, and every merge-produced id to be
  >= the previous one; otherwise the flag is **ignored with an `LLAMA_LOG_ERROR`** (`the tokenizer is not
  BPE` / `token ids do not follow BPE merge order`).  A sentencepiece/UGGUF-style vocab is therefore
  rejected rather than silently trimmed to noise.
- **The subset tensors**: `head` is a `[n_embd, n_sel]` copy of the LM-head rows **in the same quant type**,
  `ids` is an `I64[n_sel]` map back to real token ids.  The rows are copied host-side from
  `model.output` with `ggml_backend_tensor_get` in **contiguous runs** (so a run of adjacent ids is one
  copy, not one per row), into a buffer of the LM head's own buffer type, tagged
  `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`.  Nothing is repacked, requantized or dequantized — this is a
  straight copy of a row range, which is why it cannot change a single logit.
- **Guards**: arch must be `qwen35`, `qwen35moe` or `qwen4exp` (the three graphs that were taught the
  scatter), `hparams.n_layer_nextn > 0`, and the MTP block must score with the **model** LM head
  (`layers[n_layer()].nextn.shared_head_head == nullptr`) — a block with its own head would ignore the
  subset, so it logs a warning instead of building an unused copy.  `n_sel >= n_vocab` is a no-op.
  Probe contexts (`common_fit_params`, unallocated weights) return `nullptr` before touching anything.
- **Ownership (0003)**: the subset is **per context**, not model-global state — `llama_context` holds a
  `shared_ptr<const llama_mtp_draft_vocab>`, the model keeps weak references keyed by
  `(n_keep, buffer_type)` under a mutex, and a context's graph sees it through
  `llm_graph_params::mtp_draft` (`src/llama-graph.{h,cpp}`) and `llama_cparams::mtp_draft_vocab` (the N
  actually in use).  Two contexts with the same N share one copy; `N` then `0`, or `N1` then `N2`, on the
  same model do not leak into each other; the copy is freed with the last holder.  Graph reuse compares the
  pointer and the cparams field (`llm_graph_params::equals`), so a captured graph is never replayed against
  the wrong head.

**(c) The graph.**  In the MTP graph only (`src/models/qwen35.cpp:666`, `qwen35moe.cpp:739`,
`qwen4exp.cpp:1314`), when a subset exists **and** the head the graph would score with is the subset's
source tensor **and** `n_outputs == 1`, the single `build_lora_mm(head_w, cur, head_s)` becomes:

```c
sub = build_lora_mm(mtp_draft->head, cur, head_s);              // [n_sel, 1]  -- the smaller matmul
cur = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, F32, 1, n_row), -INFINITY);   // full-vocab row of -inf
cur = ggml_set_rows(ctx0, cur, ggml_reshape_2d(ctx0, sub, 1, n_sel), mtp_draft->ids);  // scatter
cur = ggml_reshape_2d(ctx0, cur, n_row, 1);
```

The tensor handed to the speculator is therefore still a **full-vocab logit row** — same shape, same
consumer, same sampler chain — with `-inf` exactly at the ids the draft may not propose.  This is the whole
correctness argument:

- **Greedy text is preserved by construction** because a speculative token is accepted only when it equals
  what the *target* produces, and the target still scores the full vocabulary.  The trim can only make the
  draft *miss* a proposal (a lost speed opportunity), never change an accepted token.  The reporter's
  gate agrees: greedy bytes identical on every fresh load, logprob fingerprint `kl_mean 0.0`, top-1
  agreement 1.0.
- The `-inf` is what makes it safe for the sampler: a chain that samples the draft logits (top-k/top-p,
  `--spec-draft-p-min`, the adaptive controller) cannot pick a `-inf` row.
- It *is* lossy for **acceptance** whenever the full-vocab argmax id is >= N.  Their measurement is honest
  about it: 0.6908 -> 0.6786, net +6.5 % anyway.  The campaign's second field note is a warning that the
  sweet spot is payload-specific (64000 wins on ROCm, 98304 on their vulkan build — at 98304 the q6_K row
  gather's requant noise flips near-tie draft argmaxes and acceptance drops, still lossless, just slower).

**(d) 0006 — the borrowed-head case.**  Every shared-head MTP sidecar on the hub
(`nextn_shared_target_tensors = True`) carries **no** `output.weight`: the draft graph borrows the
*target's* LM head (`head_w = other.output`).  The donor hook ran on the draft model, whose `output` is
`nullptr`, so it returned silently and the flag was completely inert — the reporter's sixth patch is the
adaptation: fall back to the target model via `cparams.ctx_other`, build the subset from that head, skip
the target's own-MTP-block checks (a shared-NextN target has `n_layer_nextn == 0`, so 0003's
`layers[n_layer()]` access would read past the layer array), and identify the tensor by a new
`llama_mtp_draft_vocab::src` pointer instead of `model.output`, so both shapes match.

## 2. Patch inventory (unchanged, as imported)

Applied on r106 with `git am -3`, **six commits, zero conflicts, zero manual resolution** (verified
2026-10-21; see §3).

| patch | author | sha256 (first 16) | what |
|---|---|---|---|
| `0001-spec-MTP-draft-vocabulary-subset-spec-draft-mtp-voca.patch` | Simon Teixidor | `77a20529aac46268` | the flag, the subset build, the `qwen35`/`qwen35moe` graph scatter |
| `0002-tests-per-context-MTP-draft-vocabulary-regression-te.patch` | Simon Teixidor | `a746a68e0367056a` | `tests/test-mtp-draft-vocab.cpp` (built, **not** registered with ctest — needs `-m <qwen35 MTP model>`) |
| `0003-spec-make-the-MTP-draft-vocabulary-subset-per-contex.patch` | Simon Teixidor | `6f8e5f7b29afd33c` | per-context ownership, model-side weak-ref cache, `llm_graph_params::mtp_draft` |
| `0004-qwen4exp-MTP-draft-vocab-subset-spec-draft-mtp-vocab.patch` | Laurent Zuijdwijk | `20dcc9035d9bc100` | the `qwen4exp` (Flash-Next) MTP graph + the arch guard |
| `0005-common-LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB-env-var-for-sp.patch` | Laurent Zuijdwijk | `c4b02955a90996e9` | `LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB` env alias |
| `0006-qwen4exp-build-the-MTP-draft-vocab-subset-from-the-b.patch` | Martin B (martino2k6) | `be79cfcc7f0dedf9` | shared-NextN sidecar: build the subset from the borrowed target head |

Provenance (reporter's words, kept because it matters for upstreaming): 0001-0003 originated in
Aristo94/GSQHalo.cpp (Simon Teixidor, 2026-09-13/15), were cherry-picked into agentionai/llama.cpp where
0004-0005 were added (Laurent Zuijdwijk, 2026-10-01), and 0006 is martino2k6's own (MIT, authorship
preserved in every commit).  **Not** the same feature as ggml-org/llama.cpp#30380 (that is a *dynamic*
trimmed-vocab draft — different mechanism, different flag).

Applied-state footprint on r106: **17 files, +491/-5 lines** (incl. the 243-line test).  No ggml/kernel
change, no new `.cu`, nothing in the CUDA backend — the whole feature is `src/` graph construction plus one
host-side row copy.

## 3. Import record (2026-10-21, on r106)

- Base mismatch: patches are cut on **r101**, the delivery is at **r106**.  The intervening releases touched
  blocks 06 (r104/r105), 10 (r105) and 11 (r106) — none of them a file this series edits, and
  `git am -3` on a detached worktree of `rdna-boosts` (`tree d3c1e6c5…` = `release.json.tree`) applied all
  six with **no conflict marker, no fuzz**.
- **Build: clean.**  Fresh configure + full build of the patched tree with `~/bin/build-llama-rocm`
  (ROCm 10.1.0, `GPU_TARGETS=gfx1201`) succeeded, `test-mtp-draft-vocab` included, and **no warning in any
  file the series touches**.
- `release.json`/`patches/` are untouched by this import; the delivery tree is unchanged.
- Gate status: **step 1-2 done, step 6 blocked** — see §3.1 and F2.

### 3.1 First local measurements (gfx1201, 2x AI PRO R9700, ROCm 10.1.0, 2026-10-21)

`--temp 0 --seed 42 --spec-type draft-mtp --spec-draft-n-max 3`, greedy text hashed with
`scripts/extract-generated.py`.  These are **smoke arms, not the gate**: short prompts, single config, no
adaptive controller, no deep context.

| model / split | trim N | gen t/s | greedy text | note |
|---|---|---|---|---|
| Qwen3.5-4B Q8_0, 1 GPU, `-n 256` | 0 | 203.9 / 204.3 | identical | baseline, two rounds |
| | 64000 | 216.1 / 215.4 | identical | +6.0 % |
| | 98304 | 231.9 / 231.3 | identical | +13.7 % (4B has a small draft step; see gate step 4) |
| Qwen3.8-27B Q8_0, 2 GPU `-sm layer`, `-n 256` | 0 | 40.6 / 40.6 | `da2e2d192e21` | equals the **r106 delivery build** (40.7, same hash) |
| | 64000 | 43.5 / 42.9 | `da2e2d192e21` | **+5.7…+7.1 %**, reproduces the reporter's x1.0648 |
| | 98304 | 42.8 / 42.9 | `da2e2d192e21` | no better here |
| Qwen3.8-27B Q8_0, 2 GPU `-sm tensor`, `-n 64` | 0 | 50.5 | — | baseline OK |
| | 64000 | — | — | **aborts**: `ggml-backend-meta.cpp:2319 GGML_ASSERT(size % chunk_size_full == 0)` (F2) |

Activation lines observed (`-v`; invisible at default verbosity, see F8):
`mtp_draft_vocab = 64000: MTP draft head uses 64033 of 248320 rows (166.10 MiB of q8_0, ROCm0)` for the 4B
(n_embd 2560) and `(332.20 MiB of q8_0, ROCm1)` for the 27B (n_embd 5120) — i.e. **64000 + 33 control/user
rows**, and the extra weight copy F3 quantifies.  No GPU faults after the abort
(`scripts/check-gpu-faults.sh`, kernel + log: clean); it is a host-side assert, not a device fault.

## 4. Gate plan for this box (feasible, unlike most campaigns)

Unlike the reporter's sidecar setup we do not need a download: **the MTP head is inside the models we
already have**.  `qwen35.nextn_predict_layers` is present in `Qwen3.8-27B-Q8_0.gguf` and
`Qwen3.5-4B-Q8_0.gguf`, and `qwen35moe.nextn_predict_layers` in `Qwen3.6-35B-A3B-Q8_0.gguf`, so
`--spec-type draft-mtp` scores with the **model** LM head — exactly the shape 0001/0003 wire up (0006's
shared-NextN fallback is the one arm we cannot exercise locally; it needs a hub sidecar GGUF).

Order (each step is a gate, not a measurement):

1. **Build + smoke** — done (§3.1): the patched tree builds and `--spec-draft-mtp-vocab 64000` prints the
   activation line `mtp_draft_vocab = 64000: MTP draft head uses N of M rows`.  If it prints
   `... ignored: <why>`, stop: the trim is not active and everything downstream is a false A/B.  **Pass `-v`
   to see it.**
2. **Greedy purity (the criterion)** — same seed, `--temp 0`, `-n 300` minimum, trim arm vs
   `--spec-draft-mtp-vocab 0` on the same binary, **and** vs the r106 release build, per the
   `GREEDY-PURITY.md` §36 grid shape (f16/bf16/q8_0 exact; coarse quants logits-level).  Byte-identical
   output is the promise; a difference is a defect, not a trade.
3. **Adaptive-MTP gate** (`benchmarks/mtp-adaptive-methodology.md`, rule 0: `-n 3000`, `--reasoning on`
   for R, `--reasoning off` for P/C/K, plus the phase-switching `prompts/code-reasoning-mixed.txt`), with
   our **recommended** config (`--spec-type draft-mtp-adaptive` and also `-adaptive,ngram-mod`), not the
   reporter's static `n-max 3`.  The controller keys on `n_accepted`, and the trim *deliberately* lowers
   acceptance for ids >= N, so the climb/drop trajectory — not just the mean — has to be compared.
4. **Sweep N** (e.g. 32768 / 49152 / 65536 / 98304 / 131072 / 0) per model and quant, because the reporter's
   own field note says the sweet spot is payload-specific and a too-large N buys back nothing while
   costing acceptance.
5. **Kill-switch identity** — `--spec-draft-mtp-vocab 0` must reproduce the r106 build bit-for-bit and
   within noise on t/s (the "inert at default" claim).
6. Then the delivery's own gates on the multi-device shapes: `-sm tensor` (2 and 3 GPU) — **blocked today**,
   see F2.  `-sm layer` works (measured above), so a `-sm layer`-only ship is possible if F2 is not fixed.

## 5. Review findings (raised before any delivery-affecting work — maintainer decides)

**F1 (correctness/scope) — the arch list is three entries, but the delivery ships more MTP users.**
Only `qwen35`, `qwen35moe` and `qwen4exp` are wired.  Every other arch with an MTP path falls through to
`"not supported for this model"`.  Two of ours matter: **`gemma4`** (our `-MTP.gguf` sidecars, the
`fix-gemma-4` campaign, block 11) and the **DFlash / deepseek4 / glm5-next** families.  Gemma4's MTP graph
scores with a borrowed head too, so porting it to this hook is the same ~10-line block 0004 adds — but it
is **not** done here, and gemma4 is the MTP model the maintainer actually benches.  Decision needed:
extend the campaign to gemma4 (and which others), or keep it a Qwen/Flash-Next-only probe.

**F2 (CONFIRMED 2026-10-21 — blocks the `-sm tensor` config) — the subset build aborts under a
vocabulary-split LM head.**  `-sm tensor` places the `output` projection **split along the vocab axis**, and
building the subset reads a *row range* out of that split Meta buffer.  `llama-cli` 2x R9700 on
`Qwen3.8-27B-Q8_0` + `--spec-draft-mtp-vocab 64000 -sm tensor` dies at MTP-context creation with

```
/tmp/lc-vocab/ggml/src/ggml-backend-meta.cpp:2319: GGML_ASSERT(size % chunk_size_full == 0) failed
```

which is `ggml_backend_meta_buffer_get_tensor`'s `SPLIT_AXIS_0/1/2` path: it splices a split tensor only in
whole `nb[axis+1]` chunks, so for a 2-D vocab-split tensor *any* partial read is rejected (the `MIRRORED`
branch, immediately below it, does support it).  Same binary, `-sm tensor` with `--spec-draft-mtp-vocab 0`:
50.5 t/s, fine; `-sm layer` + trim: fine (43.5 t/s, §3.1).  So the failure is the feature, not the config,
and it is a **clean host-side abort before decode** — no GPU fault, no silent wrong output.  Users have
`-sm layer` as the escape hatch today; `-sm tensor` is the delivery's recommended multi-GPU mode, so this
must be fixed before any promotion.

Fix routes, in order of preference: (a) build the subset **device by device** through
`ggml_backend_meta_buffer_simple_tensor()` (or by asking the Meta buft for a per-device view) instead of one
range read — small host helper, keeps the maximum win because the subset head keeps the Meta buft and is
then placed vocab-split like the head it was copied from; (b) then verify the two *graph* ops on a split row —
`ggml_fill` of the `-inf` full-vocab row and `ggml_set_rows` with an id list that spans both devices — against
the single-device arm; if the scatter cannot be made correct per-device, gather the draft logits row the way
block 11 does for the DFlash selector (`GGML_ALL_GATHER`, r106); (c) fallback: reject `--spec-draft-mtp-vocab`
with `-sm tensor` loudly (a warning + full-vocab draft) rather than aborting, and ship `-sm layer` only.

**F3 (memory) — the subset copy is unaccounted VRAM, and `--fit` cannot see it.**  Measured: **332.20 MiB**
for 64033 of 248320 q8_0 rows on the 27B (n_embd 5120), **166.10 MiB** on the 4B (n_embd 2560) — a second
copy of `N/n_vocab` of the LM head in its own quant type, allocated when the **MTP context** is created,
i.e. after `--fit` has committed its estimate (probe contexts are deliberately skipped, 0006).  On the tight
24 GiB-per-card configs we gate at (27B Q4 + 128k ctx + expert cache) that is enough to tip an allocation
into shared memory.  Options: teach the estimator, account it as part of the draft's weight bytes, or accept
it and document a size.

**F4 (policy) — opt-in vs the default-on policy.**  `AGENTS.md` says a beneficial, gated feature ships ON
with an env kill-switch.  This is a numeric tuning knob whose sweet spot is payload-specific (their two
field notes), so "default 0 + `--spec-draft-mtp-vocab N`" is defensible **only if** we publish a
recommended N per payload family (and validate it).  Alternatively pick a conservative default (e.g.
`min(n_vocab, 65536)` or the tokenizer's trained prefix) and make `0` the kill-switch.  Note that a
*default-on* trim changes every existing MTP decode number we have published, so the adaptive-MTP baseline
in `benchmarks/` would have to be re-recorded.  Decision needed at promotion, not now.

**F5 (purity bookkeeping) — needs a `GREEDY-PURITY.md` §entry when it is promoted.**  The claim is "the
draft may be arbitrarily lossy; the verified sequence is the target's greedy sequence", i.e. the same
family as the existing §36 statement, but the acceptance loss and the `-inf` scatter deserve an explicit
rule ("a draft-side vocabulary restriction never changes an accepted token; it can only drop proposals").
Also: with `--temp > 0` the sampled output distribution is untouched (acceptance is still evaluated
against the target), but our §36 grid is greedy-only — write the statement down rather than infer it.

**F6 (ABI/API) — a new field mid-struct in the public `llama_context_params`.**  Fine in-tree, but it is an
ABI break for prebuilt bindings, and the field sits between `n_threads_batch` and `ctx_type` (not at the
end).  Upstream-PR material would want it at the tail and documented in `ENVIRONMENT.md` terms.  No action
for a wip import; note it when cutting an `upstream/` copy.

**F7 (small) — the flag is silently inert outside MTP.**  `--spec-draft-mtp-vocab N` with
`--spec-type ngram-mod`/`none`/`eagle3` is accepted and does nothing (no warning).  Also
`LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB` is a `common` env alias only — for promotion it needs an
`ENVIRONMENT.md` row (kind: opt-in tuning value, `0` = off) and the `--spec-draft-mtp-vocab` help text
carries the payload-specific sweet spot caveat.

**F8 (confirmed on our tree, nit) — the activation line is `LLAMA_LOG_INFO`, and our `llama-cli` does not
show it by default either.**  A 4B A/B run without `-v` printed no loader or core INFO lines at all (only
the banner and the timing footer), so a working trim and a silently rejected one look identical — exactly
the trap the reporter fell into twice.  Promoting the line to `LLAMA_LOG_WARN` (or adding an
`mtp_draft_vocab` gauge next to the existing draft counters) makes gate step 1 self-verifying; until then,
every A/B in this campaign is run with `-v`.

## 6. Open questions for the maintainer

1. Is this a campaign we want at all (a decode-only win for a flag, no kernel change), and if yes, is F1's
   gemma4 extension in scope or is Qwen/Flash-Next enough?
2. Who fixes F2, and is `-sm layer`-only acceptable as a first ship?  With F2 fixed the feature is usable in
   our recommended 2-3 GPU `-sm tensor` config; without it, it is a `-sm layer` feature only.  The rest of
   the gate plan (step 3-5) does not depend on the answer and is cheap on this box.
3. If it gates green, is it a new block (12) or a fold into **block 02** (adaptive MTP)?  It is
   draft-decode-side and independent of the controller, so block 02 is the natural home; F4's
   default-on question decides whether it can ship as a block amendment in a patch release.
