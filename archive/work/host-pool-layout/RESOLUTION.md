# RESOLUTION — `host-pool-layout` (TODO #87, r113)

**Closed 2026-10-22 in `v14-097cf33d9-r113`.**  Promoted into **block 12** (the block that owns
`moe-expert-cache.cu` since r111's master drop; blocks 00-11 are byte-identical to r112 and block 13 is
re-cut with a new `From` line and an unchanged body).

## What shipped

1. **Pool content = the arena's complement, ranked by the persisted heat prior**
   (`pool_rank_experts_locked`, `pool_rerank_from_heat_locked`).  Laid out in bulk at sizing (after the
   heat/prefill seeds) and again at the first decode-band pass.  Replaces the flat prompt-tally re-rank
   (`pool_rerank_from_tally_locked`, deleted).
2. **The demand path never evicts and never moves an arena victim into the pool**
   (`pool_peek_locked` + `pool_fill_empty_locked`; `expert_host_src(allow_empty_fill)`); the background
   eviction-prefetch worker and `MOE_HOST_POOL_BGFETCH` are gone.
3. **The whole-table prefill staging reads the pool** and fills empty slots (it used to bypass the pool and
   re-read the whole host expert set from the file per ubatch).
4. **`MOE_HOST_POOL_RAM_PCT` (default 80)** caps the whole pinned pool at that % of `MemAvailable`,
   measured once at the first pool build, trimming per table with a WARN; `0` disables.

## Why (see README §1–§3 for the numbers)

The offline two-tier model on a real trace, with the model's own `.moe-heatmap`, puts a **15×** difference
on the pool's *content* (expert-id order 3571 MiB of L3 reads over 451 decode tokens vs 237 MiB for the
heat-ranked complement) and only 0–246 MiB on the *eviction policy* axis.  On the GPU, the most "correct"
policy (`swap-slot`, the literal "swap positions with the entry being pulled from the pool") measured
**2× slower decode** for no L3 benefit, because an `-sm tensor` axis-0 slice makes every hand-over a fine
strided 2-D copy.  So the campaign ships the layout and leaves the pool alone at run time.

## Measured (gfx1201, 2× R9700, IQ4_NL Flash-Next field config, 70K prompt, 128-token decode)

| arm | prefill t/s | decode t/s | pool h |
|---|---:|---:|---:|
| pre-change, `DIO=1` (default) | 39 (unfinished at 27 min) | — | 0.2609 |
| pre-change, `DIO=0` | 581 | 30.6 | — |
| **post-change, `DIO=1` (default)** | **287** | **23.1** | 0.9044 |
| **post-change, `DIO=0`** | **733** | **53.7** | 0.9044 |
| `pinned` (reference) | 836 | 57.5 | — |

10.5K prompt: 447→579 prefill and 46.6→53.7 decode at `DIO=0`; 36→218 and 6.3→22.7 at `DIO=1`; pool hit
0.7999→0.9183, evictions 4600→0.

## Gates

* `validate-set.sh` — green on a fresh tarball (strict 14/14 `git am`, applied tree `186cf701…`).
* `gate-per-block-build.sh --changed-from c9b12e988` — block 12 and the tip (13) clean.
* `gate-prefill-logits.sh` — **PASS**, mean KLD 0.000000 / same-top-p 100.000 % (the dense 27B arm; the
  pool path's purity is shown by text identity, below).
* **Text identity** — 96-token greedy continuation byte-identical between the pre-change and post-change
  binaries on 35B-A3B Q8_0 `-ncmoe 20 --host-experts pool` (the config exercises prewarm, layout, staging
  and the decode fills), with `MOE_HOST_POOL_VERIFY=1` clean (no `POOLREAD MISMATCH`).
* `MOE_HOST_POOL_RAM_PCT=5` forced: ceiling 8693 MiB, pool trimmed to 8714 MiB, coherent output.

## Retired / changed

* **TODO #65 is retired** (see the `WORKLOG.md` entry): the r67 "`--host-experts pool` decode collapses on
  the IQ4_NL field config" residual no longer reproduces — with buffered fills the same config decodes at
  30.6 t/s at 70K (pinned: 34.9 at the time), and what looked like a pool shortfall is the layout + DIO
  interaction this campaign fixed.
* `MOE_HOST_POOL_BGFETCH` — removed (ignored); `MOE_HOST_POOL_RAM_PCT` — new.
* Not shipped, recorded in README §2: `swap-slot` (implemented, 2× decode), `swap-weakest` (1 copy for
  105 MiB of L3), `pool-retire`.

## Follow-ups filed (not this campaign)

`DIO` auto-selection from `MemAvailable`; a pool-aware prefill staging kernel (one launch instead of one
copy per expert); long-session pool adaptation (needs a topic-shifting trace through the two-tier model).
