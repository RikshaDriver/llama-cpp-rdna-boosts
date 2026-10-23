# mtp-tail-readback (PR #206)

Folded into block 02 as the r118 amendment (see `RESOLUTION.md`; the record arrived as `wip/mtp-tail-readback/`
and moved here per the promotion rule).  The submitter's text below is unchanged.

A follow-up to `LLAMA_MTP_INGEST_TAIL` (#143, block 02). With the variable set, the draft ingests only the last N
positions of each prompt, but the target still read back the nextn hidden state (the hidden state the MTP draft head takes as input) of every prompt row and the MTP hook
still synchronized the target after every prompt batch. This patch skips both for prompt batches that lie entirely
before the ingested tail. With the variable unset, nothing changes.

One format-patch on `v14-097cf33d9-r109` (tree `8e67f977`); applied tree `085c457c`.

## What changes

`0001-MTP-prompt-tail-no-target-nextn-readback-or-hook-syn.patch` (`src/llama-context.{h,cpp}`, `src/llama-ext.h`,
`common/speculative.cpp`):

- **Before:** every unmasked prompt ubatch copied its nextn rows to the host (about 21 MB per 2048-row ubatch on
  Flash-Next), and `decode()` then waited for that readback. After each prompt batch the MTP hook called
  `llama_get_embeddings_nextn()`, whose `synchronize()` drained both GPUs. The host therefore built the next ubatch
  only after the GPUs had gone idle.
- **New staging call** `llama_set_embeddings_nextn_from(ctx, seq_id, pos)` in `llama-ext.h`: an unmasked ubatch whose
  every row is below its sequence's threshold skips the nextn readback and the readback wait. The default is 0, so
  every row is read and nothing changes for other callers.
- **MTP impl:**
  - `set_prompt_end` sets the threshold to `ingest_from - 1`;
  - `process()` returns before any getter for a prompt batch that lies entirely below it;
  - both "generation reached" resets put the threshold back to 0.
- **Why nothing the draft reads is skipped:** an ingested row `p` reads `h(p - 1)`. That row is either in the same
  batch or is the previous batch's last row (`pending_h`), and in both cases its position is at least
  `ingest_from - 1`.
- **Embedding (vision) rows are always read back.** Under M-RoPE the text after an image resumes at a jumped
  position, so the last image row can be the `h(p - 1)` of an ingested row although its own position is below the
  threshold.
- **Unchanged:** graph and target compute, and the layer-input readback and its sync.

## Gates

Setup:
- R9700 (gfx1201) x 2, native Linux.
- r109 (`s109st`) against r109 + this patch (`s109nf`), same build dir.
- Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, 2 GPU `-sm tensor --n-cpu-moe 10`, shared Q8_0 MTP head (`--spec-type
  draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0`), default env apart from the
  variable under test.
- Arms with `LLAMA_MTP_INGEST_TAIL=4096` and arms without it.
- One server load per arm, order alternated per pair, plus one uncounted cold load first.

**Greedy hashes** (400 tokens, prose / code): `a7a81f01b3bc` / `4bb43e3339c8` in every arm and load, with and
without the variable.

**Draft inputs:** with `LLAMA_MTP_INGEST_TAIL=4096`, greedy 300 tokens after a 31.5k-token and a 9.1k-token prompt
(both past the 4096 tail) gave identical output hashes and identical draft counts on every load of both arms:

| prompt | output sha | drafted / accepted |
|---|---|---|
| 31,539 tokens | `d5f6887ea007` | 291 / 202 |
| 9,139 tokens | `340a81fcefbe` | 282 / 205 |

So the draft sees the same data. Without the variable, the patched build's draft counts fall within stock's own
load-to-load spread on the 31.5k prompt: stock gave 279 / 206, 282 / 205 and 285 / 204 on three loads, and the
patched build gave 285 / 204 twice.

0 faults, 0 aborts.

## Throughput

31.5k-token prefill (`cache_prompt` off, two requests per load), `LLAMA_MTP_INGEST_TAIL=4096`:

| pair | r109 | r109 + patch | change |
|---|---|---|---|
| 1 | 2160.9 t/s | 2199.4 t/s | +1.8 % |
| 2 | 2141.6 t/s | 2174.2 t/s | +1.5 % |
| 3 | 2115.3 t/s | 2168.9 t/s | +2.5 % |
| mean | | | **+1.9 %** (95 % CI +0.6 to +3.2) |

Decode is flat (400-token prose / code within run-to-run noise).

On our own build (r102 + our stack, IQ3_S `--n-cpu-moe 47`, `GGML_CUDA_ALLREDUCE=ce`) a 5-pair server A/B measured
prefill **+2.44 %** (95 % CI +2.09 to +2.79), decode +0.20 % and 4-stream +0.33 % (both not significant). A kernel trace
showed GPU busy during prefill going from 74.8 % to 76.5 %, and the gap at an ordinary ubatch boundary going from
about 23 ms to about 5 ms.
