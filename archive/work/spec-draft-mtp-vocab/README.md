# spec-draft-mtp-vocab — `--spec-draft-mtp-vocab N`: the MTP draft scores a trimmed LM-head row subset

**Status: PROMOTED -- shipped as block 12 in `v13-097cf33d9-r107` (opt-in, default off).**  This directory is
the campaign's gate record; `RESOLUTION.md` is the promotion summary.  The delivery carries the nine patches as
one block commit (co-authored by all three contributors); `patches/README.md` is the block reference.  Origin: discussion
**#202** and **PR #205** (martino2k6), six format-patches cut on `v12-097cf33d9-r101`; patches **0007-0009** are
this campaign's own.  Reporter's box: gfx1151 / Strix Halo, ROCm 7.2.4; our gates: gfx1201, 2x R9700,
ROCm 10.1.0.

Applied tree of all nine on r106: `91bcefc49ef42220ba97a937821fb5e1d4fda6a8` (verified on a fresh worktree,
`git am -3`, no conflict; it is `release.json.tree` for r107, and `scripts/validate-set.sh` reproduces it from
the shipped 13-block set).

**Result in one line:** with the recommended `--spec-draft-n-max 7` + `--spec-draft-mtp-vocab 98304` on
Qwen3.8-27B Q8_0 / 2 GPU `-sm tensor`, all four adaptive-MTP axes improve (**+0.8…+2.4 %**, +1.6 % aggregate),
the greedy-purity promise of `n_max <= 7` is **verified with the trim on** (three-way text identity against
`--spec-type none`), and the `ggml` change passes the regression pass.  Default is still `0` (off).

## 1. What it does, and how

`--spec-draft-mtp-vocab N` makes the MTP draft head score only the LM-head rows for token ids `< N` (+ control
and user-defined ids), then scatters those logits into a full-vocab row of `-inf`; the speculator still sees a
full-vocabulary logit row and the **target's verification is untouched**, so the draft can only *lose*
proposals, never change an accepted token.

**(a) The flag.**  `--spec-draft-mtp-vocab N` (`common/arg.cpp`, `common_params_speculative_draft::mtp_vocab`,
env alias `LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB`) becomes `llama_context_params::mtp_draft_vocab` **for the MTP draft
context only** — `common/common.cpp:1283` (`-md` sidecar path) and `common/speculative.cpp:2884` (in-tree
speculator).  `0` = off = today's code path byte-for-byte.

**(b) The subset.**  `llama_model::mtp_draft_vocab_get` (`src/llama-model.cpp`) copies the selected rows of
`model.output` into an immutable `llama_mtp_draft_vocab`:

- ids `t < N` plus every `CONTROL`/`USER_DEFINED` token (so stop/tool ids stay draftable);
- a **verbatim row copy in the LM head's own quant type** (no dequant/requant/repack — it cannot change a
  logit), `I64[n_sel]` id map, contiguous runs copied in one call each;
- a merge-order guard (BPE vocab, ≥ `n_vocab/2` merges, ids non-decreasing with merge rank): low ids must be
  the frequent merges, otherwise the flag is ignored with an error.  **0009** fixes that check — it compared
  with the running maximum, which one early high id poisons (gemma4: "236336 of 236339 out of order" while its
  ids track merge rank at corr +0.992); it now compares with the previous merge-produced id and tolerates
  ≤ 1 % inverted pairs (gemma4 30/236339 = 0.013 %, qwen35 0).
- archs: `qwen35`, `qwen35moe`, `qwen4exp` (nextn block or borrowed target head, 0006) and
  `gemma4`/`gemma4-assistant` (0008); probe/`--fit` contexts (unallocated weights) return `nullptr`.
- per-context ownership (0003): `shared_ptr` in the context, model-side weak-ref cache keyed by
  `(n_keep, buffer type)`, graph reuse compares the pointer.

**(c) The graph.**  In the MTP graph only (`qwen35.cpp:666`, `qwen35moe.cpp:739`, `qwen4exp.cpp:1314`,
`gemma4-assistant.cpp:196`), when a subset exists **and** the head the graph scores with is the subset's source
**and** `n_outputs == 1`:

```c
sub = build_lora_mm(mtp_draft->head, cur, head_s);                           // smaller matmul
cur = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, F32, 1, n_row), -INFINITY);   // full-vocab -inf row
cur = ggml_set_rows(ctx0, cur, ggml_reshape_2d(ctx0, sub, 1, n_sel), mtp_draft->ids);
cur = ggml_reshape_2d(ctx0, cur, n_row, 1);
```

**(d) 0007 — the `ggml` fix that makes `-sm tensor` work.**  Under tensor split the LM head is
**vocabulary-split**, and `ggml_backend_meta_get_tensor_impl` could only serve a *whole-tensor* read of a 2-D
axis-1-split tensor (its splice chunk is `nb[axis + 1]`, which for a 2-D tensor *is* the tensor), so the
subset build aborted:

```
ggml-backend-meta.cpp:2319: GGML_ASSERT(size % chunk_size_full == 0) failed
```

The patch splices an arbitrary **row range** block by block (segment → repeat → device, with the per-device
byte offsets advanced exactly as the setter does), so uneven splits, split-straddling ranges and repeated
(segmented) splits all work.  Whole-tensor reads and every other axis keep their old paths.

**(e) gemma4 (0008).**  The `gemma4-assistant` sidecar scores with its own tied token embedding
(`output = token_embd.weight`); the graph hook is the same, the arch guard skips the nextn checks (its layer
array has exactly `n_layer()` entries and no nextn sub-struct — indexing `layers[n_layer()]` there is
out of bounds).

## 2. Patches

| patch | author | what |
|---|---|---|
| `0001` spec: MTP draft vocabulary subset | Simon Teixidor | the flag, the subset build, `qwen35`/`qwen35moe` scatter |
| `0002` tests: per-context regression test | Simon Teixidor | `tests/test-mtp-draft-vocab.cpp` (built, not ctest-registered: needs `-m <qwen35 MTP model>`) |
| `0003` spec: subset per context | Simon Teixidor | per-context ownership, weak-ref cache, `llm_graph_params::mtp_draft` |
| `0004` qwen4exp: subset | Laurent Zuijdwijk | the `qwen4exp` (Flash-Next) graph + arch guard |
| `0005` common: env alias | Laurent Zuijdwijk | `LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB` |
| `0006` qwen4exp: borrowed target head | Martino B | shared-NextN sidecars (`nextn_shared_target_tensors`) |
| **`0007` meta: row-range read of a row-split 2-D tensor** | **campaign** | the `-sm tensor` fix (general, not a special case for this feature) |
| **`0008` qwen4exp: gemma4-assistant subset** | **campaign** | gemma4 arm + arch guard |
| **`0009` spec: accept near-merge-ordered ids** | **campaign** | merge-order guard: previous-id comparison + 1 % tolerance |

18 files, +~620/-15 lines (incl. the 243-line test).  Provenance of 0001-0006 (kept for upstreaming):
Aristo94/GSQHalo.cpp → agentionai/llama.cpp → martino2k6's shared-head adaptation; MIT, authorship preserved in
every commit, and PR #205's originals are archived verbatim in `archive/work/pr205-spec-draft-mtp-vocab/`.
**Not** ggml-org/llama.cpp#30380 (dynamic trimmed-vocab draft, different mechanism).

## 3. Measurements (gfx1201, 2x R9700, ROCm 10.1.0)

All `--temp 0 --seed 42`, `-sm tensor` 2 GPU, `draft-mtp-adaptive,ngram-mod --spec-draft-n-max <N>
--spec-ngram-mod-n-match 45`, `-n 3000` (methodology rule 0), delivery prompt files,
greedy text hashed with `scripts/extract-generated.py`.  Shallow depth (32k context, 5k-token prompts).

> **Correction (2026-10-22).**  This line used to write `--spec-draft-n-start <N>` beside `--spec-draft-n-max <N>`.
> `n-start` is the *initial* adaptive depth (default **3**), not a ceiling — setting it to n-max pins the
> controller at the ceiling and changes the trajectory (2-GPU P 94.3 -> 87.6 t/s, acceptance 0.670 -> 0.627).
> The rows below were measured with `-n-start` at its default.  The remaining arms and the reproduction
> check are in [`VALIDATION-2026-10-22.md`](VALIDATION-2026-10-22.md).

### 3.1 Four-axis adaptive gate — Qwen3.8-27B Q8_0 (`qwen35`, vocab 248320)

`-n 3000` per axis, R = reasoning (reasoning on), P = prose, C = code, K = recall (reasoning off):

| axis | n-max 9, trim 0 | n-max 9, 64000 | n-max 9, 98304 | **n-max 7, trim 0** | **n-max 7, 98304** |
|---|---|---|---|---|---|
| R | 67.8 | 69.7 | 67.9 | 68.1 | **69.0 (+1.3 %)** |
| P | 92.7 / 92.9 | 89.4 / 89.5 | 95.2 / 95.0 | 93.8 / 93.7 | **95.4 / 95.3 (+1.7 %)** |
| C | 115.0 | 113.4 | 118.3 | 120.6 | **123.5 (+2.4 %)** |
| K | 406.7 | 413.7 | 408.2 | 368.2 | **371.2 (+0.8 %)** |
| aggregate (3000 tok/axis) | 109.3 t/s | 109.0 | 110.7 | 110.1 | **111.8 (+1.6 %)** |

`--spec-type none` on the same build: P 34.1, K 34.3 t/s (MTP is 2.7x / 10.8x) — rule 2 satisfied.

- **n-max 9 shows why the ceiling matters**: at 64000 the trim loses acceptance on prose (−3.6 %) while gaining
  on R/K; at 98304 it is +0.4…+2.9 % on every axis.  But 9 is outside the purity band.
- **n-max 7 is the recommendation**: every axis improves, and it is inside `n_max <= 7`.  It also beats n-max 9
  in aggregate, because the acceptance credit above position 7 is ~0 anyway (per-position acceptance at 98304:
  0.906, 0.771, 0.670, 0.584, 0.512, 0.308, 0.090, 0.011, 0.007 — positions 8-10 contribute 0.7 %).
- **The cut is payload-specific**: 64000 is the reporter's gfx1151 value; on our 27B Q8_0 it costs acceptance
  (P 92.7 -> 89.4) while 98304 keeps it (`draft acceptance` 0.61925 -> 0.61941, mean accepted len 4.77 -> 4.86).
  Sweep it, do not copy a number.
- **MoE is neutral**: Qwen3.6-35B-A3B Q8_0, 2 GPU tensor, P 3000: `none` 122.9, `draft-mtp` n3 trim 0 233.5,
  trim 98304 233.4.

### 3.2 Purity — verified *inside* the promised band, with the trim on

`benchmarks/mtp-adaptive-methodology.md` rule 4 / `GREEDY-PURITY.md` §11: purity is promised for `n_max <= 7`
(an 8-token verify batch); above it the FA tile→WMMA switch and the `mmvq`→MMQ switch change the reduction
order, and adaptive MTP's own n-max 12 is deliberately outside.

| run | arms | greedy text |
|---|---|---|
| P, 3000 tok, `draft-mtp` n-max 7 | `none`, trim 0, trim 64000, trim 98304 | **all `fa4867aa074e`** |
| C, 3000 tok, `draft-mtp` n-max 7 | `none`, trim 98304 | **both `f7256a7b1380`** |

So the feature does not weaken the purity promise: with `n_max <= 7` the trim is text-identical to plain
decode.  **Above the band the text can differ** — at adaptive n-max 9 the trim shifts acceptance, the
controller shifts depth, and the verify widths move; the two arms then agreed for the first 1952 characters and
flipped on a near-tie.  That is the documented `n_max > 7` behaviour, not a defect of the trim (see §5).

### 3.3 gemma4 26B-A4B QAT + `gemma-4-26B-A4B-it-Q4_0-MTP.gguf` — supported, not beneficial

`gemma4-assistant` rejects `-sm tensor` at load, so `-sm layer`, `draft-mtp` n-max 3, P 3000:

| trim | rows kept | t/s | acceptance |
|---|---|---|---|
| 0 | 262144 | **150.3** | 0.669 |
| 64000 | 64006 | 104.5 | 0.339 |
| 196608 | 196614 | 106.9 | 0.369 |
| 245760 | 245766 | **150.5** | **0.669** (a no-op) |
| 64000 + last 32768 (two bands, temp knob) | 96768 | 144.6 | 0.610 |

Its draft top-1 ids split across ids < 64000 and the 229376+ band (24 % of sampled candidates), so any cut below
~245760 halves acceptance and the only lossless cut saves nothing.  Wired up and correct; not worth enabling.

### 3.4 Cost

The subset is a second copy of `N/n_vocab` of the LM head, allocated when the **MTP context** is created —
**98337 rows = 510.17 MiB** of q8_0 on the 27B, 332.20 MiB at 64000, 35.16 MiB of q4_0 on gemma4.  `--fit`
does not see it (probe contexts are skipped on purpose), so budget it by hand on tight configs.

## 4. Gate status

| gate | result |
|---|---|
| `git am -3` of all nine on r106, fresh worktree | PASS, no conflict, tree `91bcefc49e…` |
| build (`~/bin/build-llama-rocm`, ROCm 10.1.0, gfx1201) | PASS, no warning in touched files |
| activation line (`-v`) | `mtp_draft_vocab = 98304: MTP draft head uses 98337 of 248320 rows (510.17 MiB of q8_0, Meta())` |
| greedy purity, `n_max <= 7` (P and C, 3000 tok) | **PASS**, see §3.2 |
| four-axis adaptive gate (`-n 3000`, R/C/K/P) | **PASS**, all axes positive at n-max 7 + 98304 (§3.1) |
| MTP vs plain (rule 2) | PASS (P 95.4 vs 34.1; K 371.2 vs 34.3) |
| acceptance (rule 1) | PASS 0.619 / mean len 4.86 on prose |
| MoE 35B-A3B, 2 GPU tensor | neutral (233.4 vs 233.5) |
| `-sm tensor` 2 GPU (the F2 blocker) | **PASS** (0007) |
| 3 GPU `-sm tensor` | **not run** (TODO) |
| `test-backend-ops` (whole suite, gfx1201) | **21179/21179 passed**, 2/2 backends |
| `test-logits-width-probe SPLIT=tensor` | **PASS**, `W=1..8` `maxdiff_vs_W8=0` |
| prefill-logit KLD vs delivery-recorded base | **PASS**, mean KLD 0.000000, same-top-p 100.000 % |
| targeted meta row-range test (standalone) | **18/18** checks: uneven/even 1-repeat, split-straddling range, repeated (nr=2) split + range crossing a repeat, whole tensor, last row |
| plain 27B tensor decode vs delivery build | text identical (`da2e2d192e21`), 29.8 t/s both |

## 5. Interaction points (documented, "known" by construction)

These are the places where `--spec-draft-mtp-vocab` meets another delivery feature.  None is a defect; each is a
property of the *schedule*, and they are the reason the recommended configuration pins a depth.

1. **× the purity band (`n_max <= 7`).**  The trim changes acceptance, acceptance changes the adaptive
   controller's depth, and depth changes the verify batch width.  Inside the band every width takes the same
   kernels, so text is identical (verified, §3.2).  Above it (adaptive n-max 8-15) the widths select
   different kernels and a near-tie can flip — the same thing that happens without the trim between any two
   depths (the r106 build alone gives three different texts for n3 / n9 / adaptive-9).  **Rule: gate purity at
   `n_max <= 7`; above it use acceptance + MTP-vs-plain, exactly as the methodology already says.**
2. **× the adaptive controller.**  The trim lowers acceptance by a small amount (1 pt at 64000 on the mixed
   prompt), and the controller reacts by holding a shallower depth.  At n-max 7 that costs nothing measurable;
   at n-max 9 it is what makes 64000 lose on prose while 98304 wins.  The trim never changes the *target's*
   decision, only how often the draft guesses it.
3. **× the cut size.**  Payload-specific.  64000 (the reporter's value) is a **loss** on our 27B Q8_0; 98304 is
   the sweet spot; 131072 is neutral.  gemma4's useful cut is ≥ 245760 (a no-op).  Never hard-code a value in a
   doc without saying which payload it came from.
4. **× `-sm tensor`.**  The subset head is allocated from the LM head's buffer type, so on a Meta (tensor-split)
   model it is **mirrored** on every device: the draft matmul is duplicated per device (still 2.5x smaller at
   98304) and the `fill`/`set_rows` run locally, which is why no all-gather is needed.  Cost: the copy is
   per-device (2x 510 MiB at 98304 on 2 GPUs).
5. **× `ngram-mod`.**  `ngram-mod` proposals are unaffected (they never touch the MTP head); with the combo the
   trim's relative share shrinks, which is why the gains here (+1.6 %) are smaller than MTP-only (+3.4 % on the
   mixed prompt).
6. **× `--fit`.**  The subset is not in the estimate (allocated after the fit).  510 MiB at 98304 on the 27B.
7. **× the `inert at 0` promise.**  Trim `0` builds no subset and adds no graph node; measured text and t/s
   identical to the delivery build at n1/n9/adaptive-9 and plain decode.

## 6. Open items

1. **3 GPU `-sm tensor`** (0,1,2) — **DONE 2026-10-22, NEGATIVE: −2.2 % aggregate.**  The Meta path is not
   width-neutral after all: the subset head is *mirrored* per device while the untrimmed head is *split*, so
   the trim only pays when `N · n_devices < n_vocab`.  See [`VALIDATION-2026-10-22.md`](VALIDATION-2026-10-22.md)
   §1 and the width-sweep control in
   [`../three-gpu-performance/FINDING-2026-10-22-dense-27b-and-mirrored-draft-subset.md`](../three-gpu-performance/FINDING-2026-10-22-dense-27b-and-mirrored-draft-subset.md).
2. **Deep-context decode** (depth 16384) — **DONE 2026-10-22, flat** (2 GPU +0.1 %, 3 GPU −1.4 %), purity
   identical at depth.
3. **Default-on?**  It stays opt-in (`0`).  A default-on variant would need a per-payload N (98304 dense Qwen,
   nothing for gemma4/35B-A3B) and would change every published MTP number; the maintainer has decided opt-in
   for the block.
4. **MoE + `-ncmoe`/expert cache** — **DONE 2026-10-22, neutral** (P +0.9 %, K −0.6 % at 100 % arena
   residency).
