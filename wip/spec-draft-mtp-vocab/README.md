<!-- Campaign README for wip/spec-draft-mtp-vocab/README.md — gate numbers
     filled from gate-101 RESULTS.json (r101 gated run 2026-10-21,
     verdict PASS with the G0b activation erratum documented there).
     House style modeled on merged wip campaigns (e.g. PR #200 README). -->

# wip/spec-draft-mtp-vocab

Vocab-trimmed MTP draft decode: the MTP draft head scores only token ids below
a cap (plus control tokens) against an LM-head row subset and scatters back a
full-vocab row of `-inf`, so the draft matmul shrinks while verification is
untouched — greedy output is preserved **by construction**. One opt-in flag,
`--spec-draft-mtp-vocab N` (0 = off = today's behavior), plus a
`LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB` env alias and a per-context regression test.

Six format-patches on `v12-097cf33d9-r101` (tree `4df89305`); applied tree
`c4391f33`. Cherry-pick-clean onto r101 with zero manual resolution.

## What changes

**0001 — spec: MTP draft vocabulary subset (`--spec-draft-mtp-vocab N`)**
(`common/arg.cpp`, `common/common.{cpp,h}`, `common/speculative.cpp`,
`include/llama.h`, `src/llama-{context,cparams}.cpp/h`)
Plumbs the flag to `llama_model_mtp_draft_vocab_get()` in the target model.
The gather builds a vocab-subset view of the MTP head's `output` tensor (first
N rows + control ids), and the graph swaps its LM matmul for the subset matmul
+ a scatter node. Off unless the flag is set; all guard rejections are
diagnosed.

**0002 — tests: per-context MTP draft vocabulary regression test**
(`tests/test-mtp-draft-vocab.cpp`)
Checks the subset's row selection (incl. control-token ids), the scatter's
`-inf` guarantee, and greedy equivalence of trimmed vs full draft scoring.

**0003 — spec: make the MTP draft vocabulary subset per context**
The subset is created per `llama_context` (the draft runs on its own context),
fixing lifetime/ownership when target and draft contexts differ.

**0004 — qwen4exp: MTP draft vocab subset**
Wires the qwen4exp (Flash-Next) graph — whose MTP path is built in
`graph_mtp` with hyper-connections — to the subset hook.

**0005 — common: `LLAMA_ARG_SPEC_DRAFT_MTP_VOCAB` env var**
Env alias so serving stacks that set env per recipe (lemonade-style) can
enable the flag without argument surgery.

**0006 — qwen4exp: build the MTP draft vocab subset from the borrowed target head**
The shared-NextN sidecar case: an MTP draft GGUF with
`nextn_shared_target_tensors = True` carries no `output.weight` — the LM head
is borrowed from the target at graph-build time, so the draft model's own
`output` is null and 0001's guard returns silently. This commit falls back to
the `cparams.ctx_other` model with a `shared_target_head` flag, and matches
the borrowed head against the subset's `src` instead of `model.output`.
(This is the shape every shared-head MTP GGUF on the hub uses, e.g.
`unsloth/...:MTP/...shared-Q8_0.gguf`.)

## Off switch

`--spec-draft-mtp-vocab 0` (the default) — no subset is built, no graph node
changes. Measured inertness: full-vocab arms with the flag absent are
byte-identical to the stock payload's captures.

## Gating (gfx1151, Strix Halo, Qwen3.8-Flash-Next Q5_K_M @ ctx 131072)

Live serving shape: ROCm payload, `--fit off --spec-type draft-mtp
--spec-draft-n-max 3` + shared-Q8 MTP head via `--model-draft`, f16 KV, temp 0,
~24k-token greedy anchor ("deep24"), fresh loads ×2 per arm, order-swapped
timed legs, `/metrics` draft counters as validity guard.

On `v12-097cf33d9-r101` + this campaign:

| gate | result |
|---|---|
| fresh-load determinism (each arm, 2 loads) | PASS — byte-identical within arm |
| draft counters active; activation line on trim arm | PASS — Δ126 draft tokens/load, 4/4 loads. The activation line itself is core `LLAMA_LOG_INFO`, invisible in ROCm payload logs (see field note) — trim activation proven by the ±flag decode delta instead |
| greedy bytes trim == full-vocab, every load | **PASS** — all 4 loads md5 `40e95c70` |
| decode med trim / full-vocab | **×1.0648** (46.25 → 49.24 t/s; the identical port on r85 measured ×1.0777) |
| acceptance (draft_n_accepted / draft_n) | 0.6908 full-vocab → 0.6786 trimmed — the trim arm never wins a draft whose full-vocab argmax is id ≥ 64000; lossless, still net +6.5 % |
| logprob fingerprint trim vs full-vocab (same-shape) | **PASS** — kl_mean 0.0, top-1 agreement 1.0, identical greedy |
| vision `image_url` request through MTP + trim | **PASS** — 512×512 image, 322 prompt tokens, correct output |

Two findings worth documenting in the flag's help text:

- **The trim-size sweet spot is payload-specific.** On the toolbox vulkan
  build 98304 (trained-prefix size) beat 64000; on this ROCm build 64000 wins
  (r85: +7.7 % decode, acceptance unchanged; r101 re-validation: +6.5 %,
  acceptance 0.691 → 0.679 as ids ≥ 64000 lose their draft wins — the
  documented CJK caveat; byte-identical output either way) while 98304 loses
  acceptance to row-gather requant noise — the q6_K block re-layout flips
  near-tie draft argmaxes. Still lossless, just slower. 64000 is the safe
  default value.
- **The activation log is invisible at default verbosity on some builds.**
  A successful gather logs one core `LLAMA_LOG_INFO`; on ROCm payload logs at
  default level no core INFO lines appear at all, and 0001's guard paths
  return silently — a working trim is indistinguishable from a rejected one.
  We misread a working build as inert twice before promoting the line to WARN
  to see it fire. Consider WARN + a `mtp_draft_vocab_rows`-style metrics gauge.

## Provenance

Patches 0001–0005 are the upstream-facing vocab series as published in
`agationai/llama.cpp` (authors credited on the commits; cherry-picked
unmodified except context drift). 0006 is ours (shared-head adaptation),
originally written against `v12-097cf33d9-r85`, gated live on this box, and
re-based onto r101 reproducing the win (results tabulated above). This is
**not** the same feature as
ggml-org/llama.cpp#30380 (a *dynamic* trimmed-vocab draft — different
mechanism, different flag).

## Environment

AMD Ryzen AI Max+ 395 (gfx1151, 128 GB LPDDR5X unified), CachyOS (kernel
7.2.10), ROCm 7.2.4 stack, payload base `097cf33d9`, env stance
`GGML_CUDA_MMB=0 GGML_CUDA_SLAB=0 LLAMA_LAZY_BUF_MB=8192`.
