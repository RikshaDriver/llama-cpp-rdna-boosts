# archive/work/server-prompt-cache-lazy — keep idle slots resident, reuse prompt-cache buffers, make room before a failed restore

**Status: promoted (r130, block 12).**  PR #210 (`overdoingism`), one commit against `v14-097cf33d9-r120`
(remade on r120; the first version was on r118), touching only `tools/server/server-context.cpp`,
`tools/server/server-task.cpp` and `tools/server/server-task.h`.  It applied to the r129 tree unchanged
(offset only) and is cut against unadulterated `3f03b1772` cleanly, so the same change is also staged as an
`upstream/` candidate.  Kill switches: `LLAMA_SERVER_LAZY_IDLE_SLOTS=0`, `LLAMA_SERVER_CACHE_REUSE=0`.
Delivery doc: `patches/README.md` (block 12), `ENVIRONMENT.md` §15, `MANIFESTS.md` (block 12 row),
`WORKLOG.md` 261024.011.

## What it does

With a unified KV cache (`--kv-unified`, also what `--parallel auto` selects) and `--cache-idle-slots`
(default on), every new task saved each idle slot to the host prompt cache and cleared it, so two
conversations taking turns on two slots copied their whole state out and back on every switch (27B: 330 MiB
of recurrent state + 3.8 GB of KV at 110K, 1.6-2.7 s per switch in the report).

1. `LLAMA_SERVER_LAZY_IDLE_SLOTS` (on unless `0`): an idle slot keeps its context when a new task starts.
   When the KV cache runs out of room, `try_clear_idle_slots` saves the least recently used idle slot to the
   cache and only then clears it (it used to purge it without saving).
2. A restore can fail for want of cells still held by another idle slot ("failed to find N available
   cells"); the restore now moves idle slots to the cache one at a time, with the entry being restored held
   out of that eviction, and retries.  The same hold covers the slot-selection save, which runs before the
   restore and could otherwise evict the entry the task is about to restore.
   `server_prompt_cache::find()` is split out of `load()` for that.
3. `LLAMA_SERVER_CACHE_REUSE` (on unless `0`): the buffers of restored or evicted states are kept for the
   next save (the smallest sufficient spare), counted against `--cache-ram` together with the cached states.

## Accuracy of the claims (review, 2026-10-24)

* **Mechanism and both headline numbers reproduce.**  2 slots, unified KV, 27B UD-Q4_K_XL, no draft:
  ping-pong at 11.5K tokens 0.5 -> 0.2-0.3 s per switch with byte-identical text (the state copy is
  ~1.6 GB, which is what the 0.2 s buys).  The PR's overflow case also reproduces exactly: with the second
  conversation overflowing a 16K unified cache while the first idles, upstream recomputed the first prompt
  in **8.3 s** and the patch restored it in **0.5 s with 4 tokens evaluated**.  DFlash2 and MTP drafts
  behave the same (0.4-0.5 s -> 0.2 s per switch, text identical).
* **The output-caveat is correct and already covered.**  A different cache history places a sequence's KV in
  other cells, so a near-tie can flip; that is the `--kv-unified` non-determinism in `GREEDY-PURITY.md`
  §38, not something this patch introduces.  The patch's teacher-forced KLD sits at the long-context noise
  floor, and every sequential arm measured here was byte-identical.
* **One correctness hole found and fixed in the delivery.**  `load()` consumes `it_best->data.main` *before*
  restoring the draft buffer.  If the draft restore then fails, the entry stays in `states` with an empty
  main buffer; `find()` would match it again, `llama_state_seq_set_data_ext(ctx, nullptr, 0, ...)` returns
  `0 == size` and is read as success, and the server adopts the entry's token list on top of whatever KV the
  failed attempt left.  The entry could already be matched on a later task before this PR (pre-existing),
  but the new retry makes it immediately reachable.  `find()` now skips entries with an empty main buffer.
* **A default-on behaviour change with an upstream test that encoded the old semantics.**
  `tools/server/tests/unit/test_kv_keep_only_active.py` asserted the eager save-and-clear
  (`__TEST_TAG_CACHE_IDLE_SLOT__` on every new task) and failed 1/3 with the patch on the r129 tree.  It is
  updated in the block: the default arm now asserts that an idle slot stays resident and is saved+cleared
  only when the KV cache needs the room, and the old assertions live under `LLAMA_SERVER_LAZY_IDLE_SLOTS=0`.
  3/3 green.  The `--cache-idle-slots` help text in `tools/server/README.md` is upstream's and is left as is
  (the delivery's `ENVIRONMENT.md` §15 is the authoritative description).
* **Not covered by the PR, checked here:** a 4-slot unified concurrent storm (13/13 requests green, no
  asserts; a restore that could not be satisfied because every other slot was *busy* correctly falls back to
  a recompute), `--no-kv-unified` (ping-pong unchanged), `--cache-ram 0`, and a cache too small for the state
  (recompute, no crash).  Peak RSS with reuse on is bounded by `--cache-ram` (6938 vs 6898 MiB on a
  six-save workload with `--cache-ram 4096`).

## Gates (r130, gfx1201, ROCm 10.1.0)

| gate | result |
|---|---|
| `unit/test_kv_keep_only_active.py` (updated) | 3/3 |
| `test_basic`, `test_completion`, `test_slot_save`, `test_ctx_shift`, `test_speculative` | 83 passed, 2 failed -- both **pre-existing** (fail identically on the r129 build: an `openai` client-version test, and `test_slot_restore_checkpoints_draft_kv_type_change[q8_0]`, which is the r125 draft-KV-type follow) |
| 2-slot unified ping-pong, 27B plain / MTP / DFlash2 | identical text across arms and repeats; 0.5/0.4 -> 0.3/0.2 s per switch |
| 2-slot unified overflow (A 13.4K, B 4.7K, `-c 16384`, `--cache-ram 8192`) | upstream 8.3 s recompute -> patch 0.5 s restore, text == original; with `--cache-ram 256` (state does not fit) both recompute, no crash |
| `--no-kv-unified`, 2 slots; `--cache-ram 0` | unchanged behaviour, no errors |
| 4 slots unified, 12 concurrent + 1 replay (DFlash2) | 13/13 requests OK, no assert; some recomputes under genuine KV pressure |
| peak RSS, six 13.4K saves, `--cache-ram 4096` | 6938 MiB with reuse vs 6898 MiB with `LLAMA_SERVER_CACHE_REUSE=0` |
| coherence vs r129 (6 configs) / four-axis MTP / prefill-logit / per-block build / `validate-set.sh` | as recorded in `WORKLOG.md` 261024.011 (the server change is not on those paths; all green) |
