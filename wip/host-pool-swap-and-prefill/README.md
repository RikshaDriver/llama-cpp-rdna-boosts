# `host-pool-swap-and-prefill` — the pool as the arena's swap partner, and why prefill must not touch it

**Status: OPEN (seeded 2026-10-22, r111) — re-scoped 2026-10-22 (r113).  The release-relevant halves are
DONE: §2 (the SIGSEGV) is moot (the background worker was removed), §3 (prefill must not evict) is satisfied
by construction (the demand path never evicts at all), and the "one shared LRU" question in §1 was answered
by a measurement rather than a design: the pool's CONTENT dominates the disk traffic (15x) and every
eviction/hand-over policy is within 0-246 MiB of the others, so r113 lays the pool out in bulk from the
persisted heat prior and leaves it alone.  Record: `../../archive/work/host-pool-layout/`.  What is still
OPEN is the remainder in §7: `DIO` auto-selection, a pool-aware prefill staging kernel, and long-session
adaptation.**

Seeded from the gfx1100 (fingon, 7900 XTX 24 GB / **30 GiB RAM** / ROCm 10.1.0) session that validated
`--host-experts pool` on a model **larger than RAM** (Qwen3.6-35B-A3B **Q8_0**, 37.8 GB file, host expert
set 32 640 MiB at `-ncmoe 40`).  That session established the feature (see
`../../archive/work/host-expert-master-drop/`) and produced the questions below.

## 1. The design question: is an arena eviction a swap with the pool?

**Yes, already** — but only one direction is guaranteed.  In `access_locked()` the fill path does:

1. take the arena slot (a free slot, or the LRU victim the admission policy chose);
2. **enqueue the victim into the pool** (`pool_bg_enqueue` → the low-priority worker fills a pool slot with
   it; `MOE_HOST_POOL_BGFETCH=0` does it synchronously) — so the evicted expert lands in L2 instead of being
   lost;
3. **source the incoming expert from the pool** (`expert_host_src` → `pool_slot_locked`, a hit if the pool
   already has it, else a fill from the mapping / the GGUF);
4. retire the consumed pool slot (`expert_src_done` → state 2, reaped after its copy drains), so the pool
   stays the complement of the arena.

So the intended shape is exactly "VRAM evict ⟷ pool admit".  What is **not** coordinated, and is the
campaign's work:

* **Two independent LRUs.** The arena evicts by decaying LFU + `last`; the pool evicts by `last` only, and it
  can drop the very expert the arena just handed it if its slots are all reserved/retiring (the request is
  simply dropped — "a lost prefetch, never a correctness issue").
* **The victim enqueue can be dropped**: `pool_bg_enqueue` drops when the queue is > 8192, and the worker
  drops when every slot is state 2/3/4.
* **The pool fill of the incoming expert is on the critical path** when it misses, i.e. a decode token pays
  a memcpy (page cache) or a disk read.  That is the pool's 25 %-hit-rate penalty measured earlier.
* **Nothing feeds pool residency back into the arena's admission policy** (`count`/`ghost` are arena-local),
  so the two tiers can thrash in opposite directions.

Candidates: a single shared LFU/LRU clock across L1/L2; "admit into the arena only what the pool cannot
hold"; a victim-aware pin (mark the evicted victim's pool slot non-evictable for N tokens); sizing the pool
as the complement of the arena *per layer* rather than a global fraction.

## 2. DEFECT (release-relevant): SIGSEGV in the background pool worker, gfx1100, on a memory-tight box

Reproduced on fingon with the **buffered** pool at a high fraction (not with DIO):

```
# crashes (rc = -11, SIGSEGV), 2/2 runs
MOE_HOST_POOL_FRAC=75  llama-cli -m Qwen3.6-35B-A3B-Q8_0.gguf -ngl 99 -ncmoe 36 \
    --host-experts pool -f <~1400-token prompt> -n 32 --ignore-eos -t 8
```

* Backtrace: `pool_bg_worker()` → `__memmove_avx512_unaligned_erms` (the `memcpy(dst, mapped, hb)` that fills
  a slot from the model mapping), `error 4` (read of an unmapped page) in the kernel log.
* Peak `RssShmem` **8.5 GiB** at the crash — i.e. **not** the 75 % (22 GiB) the config asks for, so the pool
  was still being built when it died; `RssFile` 20.6 GiB, `swap` 0.3 GiB.  The box has 30 GiB of RAM, so
  this is the extreme-pressure end.
* **My first hypothesis is refuted**: I suspected the loader's r49 per-tensor `munmap` reclaim
  (`llama_mmap::unmap_fragment` aligns to pages, so unmapping a *device* tensor can eat a page of an adjacent
  *host expert* tensor, which r111 now *reads*).  `LLAMA_LOAD_RELEASE_MMAP=0` (reclaim off) still crashes
  (2/2), so that is not the cause — although the page-alignment hazard is real and worth fixing anyway
  (cheap guard: page-align the `mmaps_used` overlap test in `load_all_data`).
* The same config with `MOE_HOST_POOL_DIO=1` (and `FRAC` 75 **or** 90) ran clean 2/2 and 1/1, and the
  `-ncmoe 40` / `-ncmoe 32` runs at FRAC 25/50 ran clean — so it needs high pressure or a large pool.
* Next steps: (a) reproduce with `MOE_HOST_POOL_BGFETCH=0` to confirm the worker is required;
  (b) `MOE_HOST_POOL_VERIFY=1` (byte-compares every pool fill against the mapping) to see whether the read
  itself is stale; (c) valgrind/ASAN on the pool worker (it is the one path that reads the file/mapping
  **outside `g_mutex`** — the `mapped`/`dst`/`hb` locals are captured under the lock, so a *lifetime* bug
  (a pool or table freed underneath) or a `cudaHostAlloc` failure path returning a bad pointer are the
  suspects); (d) check whether the pressure-triggered `cudaHostAlloc` failure path can leave
  `p->base`/`p->src_host` inconsistent.
* Until fixed, the workaround on a memory-tight box is `MOE_HOST_POOL_DIO=1` (which also needs no page
  cache) or a lower `MOE_HOST_POOL_FRAC`.

## 3. Prefill should not evict pool entries

Two churn sources remain in the pool during a prefill:

1. **Whole-table staging already bypasses the pool** (r111, `moe_cache_stage_table` → `expert_host_src(...,
   pool = nullptr)`): a whole-table staging reads every expert anyway, and using the pool meant the misses
   evicted the decode band's hot set for no gain (measured: bypassing it recovered 70.0 → 79.4 t/s prefill
   on gfx1201).  **Do not re-enable the pool there.**
2. **The used-expert upload (`moe_cache_upload_pageable`) and the pre-sizing prefill's eager fills still fill
   and evict pool slots.**  The rule the campaign should implement: *a PREFILL-band access may read a pool
   slot but must not evict one* (and should not admit into the pool at all); only the decode/verify band owns
   pool residency.  That way the disk (or page cache) only has to load what is not already resident in
   VRAM/host pool, which is the whole point of the tiering.

Related: `pool_prepopulate_locked` (prewarm, ranked by the arena residents + the prefill tally) and
`pool_rerank_from_tally_locked` (the one-shot re-rank at the first decode pass) are the only *deliberate*
prefill-driven pool fills; keep those, gate everything else on the band.

## 4. gfx1100 numbers from the seeding session (r111, 24 GB VRAM, 30 GiB RAM, `-t 8`)

Qwen3.6-35B-A3B **Q8_0** (37.8 GB; host expert set 32 640 MiB at `-ncmoe 40`), greedy, same seed.  **All
configurations produced byte-identical generated text** (verified across `-ncmoe` 28/32/36/40, pool/pinned,
FRAC 10/25/50/75/90, DIO on/off) — the tiering never changed a value.

Decode, 2000 tokens (`--ignore-eos`; the long run is what lets the VRAM arena converge — 48-token runs
under-report by 1.5-2x):

| config | pool | pinned (RssShmem) | page cache | swap | decode | prefill (~1400-token prompt) |
|---|---|---:|---:|---:|---:|---:|
| `-ncmoe 32`, `pinned` | — | 25.9 GiB | 0.4 | 0 | **83.7 t/s** | **492 t/s** |
| `-ncmoe 36`, pool **FRAC 75 + DIO** | 22.0 GiB | 22.3 | 5.8 | **0** | 74.4 t/s | 49.3 t/s |
| `-ncmoe 36`, pool FRAC 90 + DIO | 26.4 GiB | 27.0 | 2.1 | 0.3 | 71.8 t/s | — |
| `-ncmoe 40`, pool FRAC 25 (buffered) | 8.2 GiB | 8.7 | 19.9 | 0 | 48.6 t/s | — |
| `-ncmoe 40`, pool FRAC 25, 48-token runs | | | | | 8-44 t/s (erratic, cache-state dependent) | 2.6-45 t/s |
| `-ncmoe 40`, **`pinned`** | — | 0.8 (alloc FAILED) | | **15.9** | 2.1-2.9 t/s | 1.5-2.3 t/s |

Readings:

* **When the host remainder fits in RAM, `pinned` wins** (`-ncmoe 32`: 83.7 t/s vs the pool's ~48) — the
  extra host memcpy per staged byte and the pool's hit-rate penalty are pure cost there.  That matches the
  default-on policy: `pool` is for when the set does not fit.
* **DIO + a 75 % pool is the stable disk-tier shape** (74.4 t/s, **zero swap**, page cache untouched):
  FRAC 90 is worse (71.8 t/s *and* 0.3 GiB of swap), FRAC 75 is the knee.  This is the configuration the
  campaign should tune against.
* **The page cache is the real performance lever at low FRAC** — on a box where the model exceeds RAM, a
  *small* pool leaves RAM for the page cache and beats a large one (`-ncmoe 40` FRAC 10: 36-45 t/s vs FRAC
  50: 7.8 t/s).  So `MOE_HOST_POOL_FRAC` has an *inverted* optimum between "the set fits in RAM" and "it does
  not", and 25 % is a compromise, not a constant.
* **Prefill on the disk tier is the weak spot** (1.4-1.8 t/s on a short prompt; 49 t/s on a 1400-token one)
  because the pre-sizing prefill stages whole tables through O_DIRECT.  §3 is the fix to pursue.
* `-ncmoe 28` (host 22.3 GiB) + pool FRAC 25 gave 43.3 t/s decode where the same model at `-ncmoe 40` gave
  48.6 — keeping more layers on the GPU is *not* automatically better on this box: it shrinks the VRAM arena
  (21.1 % vs 38.7 % residency) and hands more to the host tiers.  The knobs interact; that is the campaign's
  search space.

## 4b. r112 shipped the default flip this campaign asked for

`MOE_HOST_POOL_FRAC` **25 → 80** and `MOE_HOST_POOL_DIO` **0 → 1** when `--host-experts pool` is used (the
`pinned` default mode is untouched).  Measured on the same gfx1100 / 30 GiB box with the new defaults and no
env at all, 2000-token runs:

| config | pool | pinned | page cache | swap | decode |
|---|---|---:|---:|---:|---:|
| `-ncmoe 40 --host-experts pool` (new defaults) | 26.0 GiB | 26.1 | 2.7 | 0.3 | **78.3 t/s** |
| `-ncmoe 36 --host-experts pool` (new defaults) | 23.4 GiB | 23.6 | 5.0 | 0.3 | 75.2 t/s |
| `-ncmoe 40 --host-experts pool` (r111 defaults, 25 % buffered) | 8.2 GiB | 8.7 | 19.9 | 0 | 48.6 t/s |
| `-ncmoe 40 --host-experts pinned` | — | 0.8 (alloc failed) | | 15.9 | 2-3 t/s |

So the flagship case (whole 31.9 GiB host expert set, 30 GiB box) is **+61 % decode over the r111 defaults**
and ~30x over the degenerate `pinned` arm.  gfx1201 (184 GiB RAM, `-ncmoe 40`, 2-GPU tensor) is unaffected:
115.8 t/s vs 115.2 for `pinned` (arena fully resident, so there is nothing to fill).

Two things this campaign still owns:

* **The first prefill is a full O_DIRECT read** of the host set (~10 s on that box, so a short prompt shows
  ~0.6 "prompt t/s"; a 1400-token prompt measures 49 t/s).  Item (b) — a prefill-band access must not fill
  or evict pool slots — does not remove that *first* pass (the pool starts empty), but it stops later
  prefills from evicting the decode band and lets the very first pass populate the pool for free.
* **The swap/LRU coordination** (item c) is untouched.

## 5. Repro

```sh
# on fingon (gfx1100, 30 GiB RAM, ROCm 10.1.0)
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1100/lib
M=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf

# the stable disk-tier shape
MOE_HOST_POOL_FRAC=75 MOE_HOST_POOL_DIO=1 llama-cli -m $M -ngl 99 -ncmoe 36 \
    --host-experts pool -p "The capital of France is" -n 2000 --ignore-eos --seed 42 \
    --temp 0 --single-turn --no-display-prompt -t 8

# the crash (§2)
MOE_HOST_POOL_FRAC=75 llama-cli -m $M -ngl 99 -ncmoe 36 --host-experts pool \
    -f <a ~1.4k-token file> -n 32 --ignore-eos -t 8
```

`/tmp/memmon.py` in the session is a 30-line peak-sampler (`VmRSS`/`RssShmem`/`RssFile`/`RssAnon`/`VmSwap`);
`scripts/check-gpu-faults.sh --since "1 hour ago" --log run.log` was clean on every non-crashing run.

## 6. Pointers

* The feature and its purity gates: `../../archive/work/host-expert-master-drop/` (`README.md`,
  `RESOLUTION.md`); `WORKLOG.md` 261022.006 (r111).
* Pool knobs: `../../ENVIRONMENT.md` §1.2.1.  The fill path: `moe_cache_fill_slot_locked` /
  `expert_host_src` / `expert_src_done` / `access_locked` in `ggml/src/ggml-cuda/moe-expert-cache.cu`.
* Admission quality (the sibling question, different axis): `../moe-admission-quality/` (TODO #86).
* Issue #139 (why a smaller pinned footprint matters at all): `../../archive/work/issue-139-ar-spin-timeout/`.

## 7. What is still open after r113 (2026-10-22)

r113 (`archive/work/host-pool-layout/`) closed the disk-I/O half of this campaign: the pool is laid out as
the arena's complement ranked by the persisted heat prior, the demand path never evicts, the prefill staging
reads the pool, `MOE_HOST_POOL_RAM_PCT` bounds the pinned total, and the background worker (and therefore
the §2 SIGSEGV) is gone.  Remaining, in priority order:

1. **`MOE_HOST_POOL_DIO` is still a hand-set default** (the "inverted optimum" this campaign named).  On the
   field config (`-ncmoe 48`, 70K prompt, 128-token decode, 2x R9700) `DIO=0` measures 733 t/s prefill /
   53.7 t/s decode against `pinned` 836 / 57.5, while `DIO=1` measures 287 / 23.1 -- on a box whose RAM
   holds the set.  On the 30 GiB field box the opposite holds.  The obvious fix is to choose at the first
   pool build from `MemAvailable` vs the host expert set (the same measurement `MOE_HOST_POOL_RAM_PCT`
   already takes), with a kill switch.
2. **The pool-backed prefill issues one `cudaMemcpyAsync` per expert.**  The pinned-master path stages a
   whole table with ONE `moe_cache_stage_kernel` launch; the pool path loops the host, which is the rest of
   the gap to `pinned` (733 vs 836 t/s at `DIO=0`).  A pool-aware staging kernel needs a per-expert slot map
   upload (the `prs` pattern already exists) and a "no slot" sentinel that falls back to the bounce.
3. **Long-session adaptation.**  Between the two bulk layouts the pool's content is static, so a single
   session that changes topic keeps its pre-prompt layout.  Do not add an eviction policy for this before
   re-running the two-tier model (`scripts/moe-trace-replay.py --pool-frac`) on a *long, topic-shifting*
   trace: on the short trace every eviction policy was within noise, and the one that looked most "correct"
   (`pool-swap-slot`) cost 2x decode for 0 MiB of L3.
4. **§4's gfx1100 numbers predate the layout fix** -- re-measure the 30 GiB box with r113 before quoting
   them anywhere; the pinned-footprint and swap conclusions still hold (they are about `FRAC` and `DIO`),
   but the decode figures should move.
