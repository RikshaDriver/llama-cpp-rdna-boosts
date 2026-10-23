# RESOLUTION — mtp-tail-readback (PR #206)

**Status:** FOLDED into the delivery as the **r118 amendment of block 02** (`common/speculative.cpp`,
`src/llama-context.{h,cpp}`, `src/llama-ext.h`), 2026-10-23.  **Owner:** `@briansp2020` (PR #206).

The change is the target-side half of `LLAMA_MTP_INGEST_TAIL` (#143, `archive/work/mtp-prefill-tail/`).
With the tail set the draft already ingested only the last N prompt positions, but the target still
copied every prompt ubatch's nextn hidden state to the host (~21 MB per 2048-row ubatch on Flash-Next)
and then drained every backend before the host built the next ubatch.

## What was adopted

- `0001-MTP-prompt-tail-no-target-nextn-readback-or-hook-syn.patch` as submitted, plus two changes of
  ours (see below):
  - new staging API `llama_set_embeddings_nextn_from(ctx, seq_id, pos)` (`src/llama-ext.h`), default
    `0` = every row is needed, so no other caller changes;
  - the MTP impl sets it to `ingest_from - 1` in `set_prompt_end()` and back to `0` at both
    "generation reached" resets, and `process()` returns before any getter for a prompt batch that lies
    entirely below it;
  - `llama_context::decode` skips the nextn `get_async` for an unmasked ubatch whose every row is below
    its sequence's threshold; the `embd_batch_idxs` bookkeeping is still appended (it must stay aligned
    with the dense unmasked host rows, or `output_reorder()`'s nextn permutation would place a later
    ubatch's rows wrongly).  Vision rows are always read: under M-RoPE the text after an image resumes at
    a jumped section-0 position.

## Where the gain actually comes from

The drain is the **hook's getter**, not the end-of-ubatch `sync_readback`: the C wrappers
`llama_get_embeddings_nextn()` / `..._ith()` both call `ctx->synchronize()`, and r117's MTP `process()`
called the former on every prompt batch, so the host waited for both backends to go idle before it built
the next ubatch.  The patch's early return removes that whole-path drain; the target-side readback skip
removes the 21 MB/ubatch copy and its ordering.  A 2x2 on one binary (`bin-dbg`, 31,476-token prompt,
tail 4096, steady-state request, 2 loads per arm) splits it:

| arm | prefill t/s | vs pre-r118 behaviour |
|---|---:|---:|
| both halves off | 1500.3 | - |
| readback skip only | 1502.3 | +0.1 % |
| hook skip only | 1509.6 | **+0.6 %** |
| both (shipped) | 1514.5 | **+0.9 %** |

The 2x2 ran on a diagnostic build (extra env-gated switches) whose both-halves-off arm already sat ~0.4 %
above the r117 binary, so only its *split between the two halves* is used; the shipped form was measured
directly against r117, 2 alternating pairs, 12 requests (identical text in every run):

| arm | prefill t/s | vs r117 |
|---|---:|---:|
| r117 | 1488.3 | - |
| r118 (both halves) | **1506.8** | **+1.24 %** |
| r118, `LLAMA_MTP_TAIL_READBACK=0` | 1490.0 | +0.11 % (i.e. the pre-r118 behaviour, the switch works) |

The earlier 3-pair series measured +1.42 % (95 % CI +0.92..+1.93); the submitter's +1.9 % sits inside that
interval.  Report the gain as **about +1.3 %** and the attribution as "mostly the hook's getter, the
readback copy is small on this box".

Side finding, recorded because the block-00 comment reads as if the readback wait always applies: for a
single-ubatch decode (the server's `-b 2048 -ub 2048`) `decode()` does `if (!mctx->next()) break;`
**before** the `sync_readback` block, so that wait only ever runs when another ubatch follows.

## Our two changes to the submitted form

1. **The per-row scan is gated on `cparams.embeddings_nextn`** (`nextn_rows`).  As submitted it ran on
   every decode of every model, over an all-zero array.  Below the noise floor, but it was new work on
   the path everyone takes.
2. **`LLAMA_MTP_TAIL_READBACK=0`** (default on) disables **both halves** (the hook's early return and the
   target's threshold), i.e. it restores the pre-r118 behaviour with the tail still set.  A bisect knob,
   so a field report on the skip does not force giving up the tail's prefill win.

## Why nothing the draft reads is skipped

An ingested row `p` (pos `>= ingest_from`) reads `h(p - 1)`.  That row is either in the same batch or is
the previous batch's last row (`pending_h`), and its position is at least `ingest_from - 1`.  Any ubatch
containing a row at `pos >= ingest_from - 1` is never skipped, so the source row is always fresh.  A
prompt batch is skipped only when *every* row is below `ingest_from - 1`, which is why the threshold is
`ingest_from - 1` and not `ingest_from`.  (This relies on the pre-existing assumption that a sequence's
positions are contiguous across batches; `--cache-reuse` already breaks that assumption for the draft,
independently of this patch.)

## Validation on the delivery tree (gfx1201, 2x R9700)

Full record: `benchmarks/2026-10-23-pr206-mtp-tail-readback-promotion.md`.

| check | result |
|---|---|
| prefill, 31,476-token prompt, `INGEST_TAIL=4096` | 1488.6 -> **1509.8 t/s (+1.4 %, 95 % CI +0.9..+1.9)**; submitter's +1.9 % is inside the interval |
| prefill, no tail | 1424.6 -> 1423.3 (**-0.1 %**, noise: the change cannot trigger with threshold 0) |
| tail effect itself | +4.5 % on r117, **+6.1 %** with the patch |
| greedy text, 300 tokens after a 31.5k prompt, 2 loads per arm (tail set) | `98c4f67d6ed9` in every arm |
| draft counts, same runs | **177 accepted / 364 generated**, acc/pos (0.754, 0.426, 0.270) in every arm |
| decode, same runs | 98.31 (base) vs 98.22 / 98.13 t/s, flat |
| `LLAMA_MTP_TAIL_READBACK=0` arm | reproduces the base prefill rate (see the benchmark record) |
| kill-switch wiring (debug build, per-ubatch log) | skip on: 5 skipped / 10 read ubatches; `=0`: 0 skipped / 15 read |
| prefill-logit gate (27B Q8_0, base recorded with the r117 binary) | mean KLD **0.000000**, same-top-p **100.000 %** |
| dense coherence, 4B Q8_0 200 tokens | `ba301f021e66` before and after |

Kept **opt-in** (it inherits `LLAMA_MTP_INGEST_TAIL`): the tail is still a prefill/decode trade, and
this patch makes the prefill side of that trade better, it does not flip it.  With the tail unset no
row's threshold is ever above `0`, so the skip cannot trigger and every existing path is unchanged.
