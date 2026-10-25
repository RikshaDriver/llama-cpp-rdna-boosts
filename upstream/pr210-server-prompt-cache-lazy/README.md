# upstream/pr210-server-prompt-cache-lazy — keep idle slots resident, reuse prompt-cache buffers, make room before a failed restore

**Re-base status:** cut against **`3f03b1772`** (the current delivery base) from the contributor's
`v14-097cf33d9-r120` patch; applies cleanly to unadulterated `3f03b1772` (`git apply --check`) because no
upstream commit in the 261-commit re-base window touched `tools/server/`.  In the delivery as **block 12**
(r130); this copy is the same change plus one delivery fix (below) for filing upstream.

## What it is

Server-only (`tools/server/server-context.cpp`, `server-task.cpp`, `server-task.h`), no arithmetic change:

* `LLAMA_SERVER_LAZY_IDLE_SLOTS` (on unless `0`): with a unified KV cache and `--cache-idle-slots`, an idle
  slot keeps its context when a new task starts instead of being saved to the prompt cache and cleared.
  The KV-full path saves the least recently used idle slot before clearing it (it used to purge it).
* A restore that fails for want of KV cells held by another idle slot moves idle slots to the prompt cache
  one at a time, with the entry being restored held out of that eviction, and retries.  The same hold covers
  the slot-selection save.  `server_prompt_cache::find()` is split out of `load()` for that.
* `LLAMA_SERVER_CACHE_REUSE` (on unless `0`): buffers of restored or evicted states are reused for the next
  save (smallest sufficient spare), counted against `--cache-ram` with the cached states.

## The one fix on top of the contributor's patch

`find()` skips an entry whose `data.main` is empty.  `load()` transfers the main buffer before the draft
buffer; if the draft restore then fails, the entry stays in the cache with an empty main buffer, and a
re-match compares against zero bytes -- `llama_state_seq_set_data_ext(ctx, nullptr, 0, ...)` returns
`0 == size` and reads as success -- leaving the slot's token list on top of a failed/cleared KV.  The retry
loop this patch adds makes that entry immediately reachable (it was a rare later-task hazard before).

## Validation

See `../../archive/work/server-prompt-cache-lazy/README.md`: the reporter's two headline scenarios reproduce
(ping-pong 0.5 -> 0.2-0.3 s per switch at 11.5K tokens with identical text; the overflow case 8.3 s
recompute -> 0.5 s restore, 4 tokens evaluated), plus a 4-slot concurrent storm, `--no-kv-unified`,
`--cache-ram 0`, a cache too small for the state, and a peak-RSS bound against `--cache-ram`.  The upstream
test `unit/test_kv_keep_only_active.py` must be updated with this change (it asserts the eager
save-and-clear): the delivery's version covers both arms, 3/3.

**What was NOT validated:** more than four slots, a model swap/sleep cycle with resident idle slots,
multimodal prompts, and the router mode's model eviction.
