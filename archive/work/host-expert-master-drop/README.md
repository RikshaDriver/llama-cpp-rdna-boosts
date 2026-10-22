# `host-expert-master-drop` — remove the full pinned `ROCm_Host` master when `--host-experts pool` is on

**Status: IMPLEMENTED + VALIDATED (2026-10-22), not yet folded into `patches/`.**  The change is
`master-drop.patch` in this directory (cut against the r110 tree, `git diff` of the working tree in
`~/llama.cpp` on `rdna-boosts`; it is **not committed there**, so the canonical block chain stays clean).
Nothing is in the delivery set yet.  Promotion (cut it into the owning blocks, regenerate `patches/` +
`release.json`, `validate-set.sh`, refresh the fork branch) needs the maintainer's go-ahead — see §8.

Campaign seeded 2026-10-11 from the r49 `host-memory-reclaim` finding (now
`archive/work/host-memory-reclaim/`) and from issue #139 (@briansp2020).  Phase 3 of the r35
`host-expert-dio-cache` campaign, which the maintainer had withdrawn on 2026-10-08
(`archive/work/host-expert-dio-cache/PHASE2B-HANDOVER.md` §6) and TODO #55 re-opened.  The goal was
restated for this session: **unpin the host expert master so the weights stream from disk (via the model
mapping) into the bounded pinned pool, i.e. the disk becomes a third cache level and a model whose host
expert set exceeds RAM can still run.**

## 1. The design that works

`--host-experts pool` keeps its name and its pinned budget, but it no longer keeps the master:

| tier | before | after (`--host-experts pool`) |
|---|---|---|
| L1 | VRAM arena (movable-boundary slab) | unchanged |
| L2 | pinned master (**whole** host expert set) **+** a bounded pinned pool | **the bounded pinned pool only** (`MOE_HOST_POOL_MIB`/`FRAC`) |
| L3 | — (the master had to fit in RAM) | the read-only **model mapping** → page cache → the GGUF on disk |

* The host expert tensors are placed **in the model's read-only mapping** (a new per-device host buffer
  type `ROCm_HostMap`), not in a `cudaMallocHost` block.  The mapping is the source of truth, it costs no
  pinned memory, and its pages are reclaimable — that is the whole trick.
* The tensor still carries its device, so the scheduler keeps placing the `MUL_MAT_ID` on the owning GPU
  and nothing about the schedule changes.
* **No device copy and no kernel ever reads the mapping.**  Every host read goes through the pinned pool
  (`pool_slot_locked`: a memcpy out of the page cache, or a `pread` with `MOE_HOST_POOL_DIO=1`), and when
  the pool declines (a slot held by the background worker, the geometry does not fit a slot, the budget is
  smaller than one expert) through a **per-table pinned bounce ring** that grows on demand and is never
  reused until its event has drained.  A pageable `cudaMemcpyAsync` is *not* a usable fallback: ROCm
  stages a pageable upload through a shared internal bounce, and back-to-back pageable copies tear (that
  was the whole nondeterminism hunt, §6).

Reader inventory (all made pool-aware, `ggml/src/ggml-cuda/moe-expert-cache.cu`): the eager arena fill
(`access_locked`), the fused/sliced/1-D fill (`fill_slot_from_host`), the identity fill, the prefill and
heat seeds, the whole-table staging (`moe_cache_stage_table`), and the scheduler's used-expert copies
(`moe_cache_upload_pageable`, called from `ggml_backend_cuda_set_tensor_async` / `stage_upload` /
`stage_gather`).  `moe_cache_gather_host` (a device kernel reading the master in place) and the cold-UVA
serve stand down for a pageable master (`host_dev == nullptr`), which is exactly the issue-#116 rail the
r35 campaign built.

`MOE_HOST_POOL_FRAC` stays **25 %**; `MOE_HOST_POOL_DIO=1` is the disk-only tier (nothing is cached).  The
mode requires a mapped load (the default): with `--load-mode none/dio` there is no mapping to place the
tensors in, so the pinned master is kept and a warning says so.

## 2. Headline result — a host expert set larger than RAM now runs

35B-A3B UD-Q4_K_M, `-ncmoe 40` (host expert set **18 662 MiB**), 1x R9700, `-16k` cgroup
(`systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0`):

| limit | `--host-experts pinned` | `--host-experts pool` |
|---|---|---|
| 12 GiB | **OOM-killed (rc 137)** | runs, 93.3 t/s |
| 8 GiB | **OOM-killed (rc 137)** | runs, 93.4 t/s |
| 6 GiB | **OOM-killed (rc 137)** | OOM-killed at FRAC 25 → **runs at `MOE_HOST_POOL_FRAC=10`** (93.6 t/s) |
| 4 GiB | OOM-killed | OOM-killed (non-pool overhead > 4 GiB) |

`RssShmem` (non-swappable) on the same model, no cgroup, 128 decode tokens:

| config | `RssShmem` | `RssFile` (reclaimable page cache) | decode | prefill |
|---|---:|---:|---:|---:|
| `pinned` (master) | **18.6 GiB** | 0.4 GiB | 97.9 t/s | 138.7 t/s |
| `pool`, FRAC 25 | **6.8 GiB** | 18.5 GiB | 97.6 t/s | 134.3 t/s |
| `pool`, FRAC 50 | **10.2 GiB** | 18.5 GiB | (parity) | (parity) |

100 GB Flash-Next GSQ-IQ3_XXS, 2 GPU `-sm tensor -ncmoe 48`, `-n 24`: `RssShmem` **40.8 → 11.2 GiB**,
`RssFile` 0.7 → 26.0 GiB, decode 44.8 → 42.8 t/s, prefill at parity once the page cache is warm.

**The pinning is now a knob, not a property**: FRAC 25 → 6.8 GiB, FRAC 50 → 10.2 GiB, FRAC 100 → the old
18.6 GiB.  That is the whole point of the exercise.

## 3. Gates (all GREEN, 2026-10-22, ROCm 10.1.0, gfx1201, `ROCBLAS_USE_HIPBLASLT=0`)

Text identity `pinned` vs `pool` (same seed, greedy, generated text compared after stripping logs):

| config | text | prefill | decode |
|---|---|---|---|
| 1 GPU, 35B-A3B `-ncmoe 40` | **identical** | 138.7 → 134.3 | 97.9 → 97.6 |
| 2 GPU tensor, same | **identical** | 107.8 → 109.2 | 111.0 → 111.2 |
| 2 GPU tensor + `--spec-type draft-mtp` | **identical** | 80.6 → 79.4 | 216.2 → 216.4 |
| 3 GPU tensor, same | **identical** | 74.3 → 70.1 | 18.5 → 18.7 |
| gemma4 26B-A4B Q8_0, 2 GPU tensor, `-ncmoe 30` (fused tables) | **identical** | 72.6 → 74.3 | 73.0 → 73.2 |
| Flash-Next GSQ-IQ3_XXS, 2 GPU tensor, `-ncmoe 48` | **identical** | 43.6 → 42.1 | 56.5 → 55.7 |

* **Prefill-logit KLD** (the release gate for an approximate prefill path): 2 GPU tensor, 35B-A3B
  `-ncmoe 40`, wikitext `-c 512 --chunks 8`, base recorded with `pinned`, candidate `pool`:
  **Mean KLD -0.000000, Mean Δp 0.000 %, RMS Δp 0.001 %, Same top p 100.000 %**.  (Recorded PPL
  6.2080 — `pinned` and `pool` produce the same number; `moe_cache_report` shows the pool at 4 665.5 MiB.)
* **MTP purity on both arms**: `--spec-type none` and `--spec-type draft-mtp` produce the same text in
  the `pool` arm (and in `pinned`), 2 GPU tensor, `-n 192`.
* `test-backend-ops -o MUL_MAT_ID`: **947/947**, 3/3 backends.
* `llama-batched-bench -npl 1,4,8` (2 GPU tensor, pp512/tg64): decode parity at B=1/4/8
  (91.3/295.7/487.9 etc.), prefill parity at B=4/8.  **B=1 prefill is -34 %** (165.7 → 109.0 t/s): a
  single 512-token ubatch stages the whole expert set once, and dropping the master adds one host memcpy
  of it (see §5).

## 4. The disk tier, measured (`MOE_HOST_POOL_DIO=1`, nothing in the page cache)

1 GPU, 35B-A3B, `-ncmoe 40`, arena forced to 4096 MiB so the pool actually misses, `-n 128`:

| config | prefill | decode | `RssShmem` | `RssFile` |
|---|---:|---:|---:|---:|
| `pinned` | 104.3 | 43.3 | 18.6 GiB | 0.4 GiB |
| `pool` 25 %, buffered (page cache) | 102.5 | 41.8 | 4.9 GiB | 18.0 GiB |
| `pool` 25 %, **DIO** (all disk) | 33.5 | **7.5** | 5.1 GiB | 12.9 GiB |
| `pool` 100 %, **DIO** | 91.5 | 10.2 | 18.8 GiB | 12.9 GiB |

So: with the page cache holding the expert set the mode is at parity; with the disk as the only tier it is
~6x slower on decode — which is the expected shape (and the reason a *bigger pool* is the knob: §2).

## 5. Accepted trade-offs / limitations

* **One extra host memcpy per staged byte on the host-expert prefill path.**  The pinned master fed
  `cudaMemcpyAsync` directly; the mapping must be staged through pinned memory first.  Cost measured:
  B=1 pp512 on the 2-GPU tensor config -34 %, normal (`llama-cli`) prefill -1 to -6 %, decode at parity
  (the decode band is served by the arena/pool, not by staging).  Not recoverable without pinning the
  data again; a fully resident *arena* (the auto default when VRAM allows) already removes it for the
  staged path via the arena→ring D2D copy.
* **Requires `use_mmap`.**  `--load-mode none/dio` keeps the pinned master (warned).  The mapped load is
  the default, and `MOE_HOST_POOL_DIO=1` reproduces a "direct I/O" run *with* the drop.
* **The cold-UVA fast serve and the device-side gather are unavailable** for a pooled master (they read
  the master in place).  Misses are pool fills instead — the same bytes, one extra pinned landing.
* **Integrated (APU) devices keep the pinned master**: they share host memory with the GPU, so there is no
  duplicated pinned copy to drop and their host buffers are directly readable.
* `--host-experts pinned` (the default) is **byte-for-byte unchanged behaviour**: the pinned master, the
  cold path, the gather, the staging kernel and every `MOE_HOST_POOL_*` default are untouched.

## 6. Three PRE-EXISTING bugs found and fixed en route (all in code this campaign runs hard)

1. **`pool_dio_read` used a shared `static` bounce buffer** while the background prefetch worker reads the
   GGUF *without* `g_mutex` (by design).  The race corrupted the tail of an expert (~520 KiB into a
   576 KiB expert, variably) — found with `MOE_HOST_POOL_VERIFY=1`, which byte-compares every pool slot
   against the mapping.  Fixed: `thread_local`.
2. **`g_host_pools` was a static `std::map`** touched by the detached worker.  At process exit the map was
   destroyed under a worker still inside a (slow, DIO) read → `double free or corruption` in
   `pool_reap_locked` (reproducible with `MOE_HOST_POOL_DIO=1` at any pool size; dormant before because
   the master fallback kept the worker idle).  Fixed: leaked heap object, like the worker's mutex/cv/queue.
3. **The AUTO pool budget was both 3x too large and registration-order dependent.**  `g_preflight_host_bytes`
   was a running `+=` over every preflight call (`--fit` probe + real load + a draft model = 3x), and
   `table_pool_locked` divided the budget by the number of host tensors *registered so far* (pools are
   created lazily while only a few tables exist), so EARLY pools got the whole table.  Measured: FRAC 25
   pinned 75 % of the host set.  Fixed: the budget is derived from the per-device MAX (order-independent)
   and each pool gets `frac %` of ITS OWN experts (`MOE_HOST_POOL_MIB` scaled by the tensor's share).

The first two only matter once the pool is the *only* host source; they are independent of this campaign
and should be kept even if the master-drop is not promoted.  (They are, however, exactly the kind of thing
the campaign was for — the mode makes the pool do the work it was always supposed to do.)

## 7. Environment variables touched

| variable | change |
|---|---|
| `--host-experts pool` | semantics changed: **drops the pinned master** (was: "pinned master + pool") |
| `MOE_HOST_POOL_MIB` | unchanged meaning (process-wide total); now enforced **accurately** (the 3x bug), 0 = disable the pool |
| `MOE_HOST_POOL_FRAC` | unchanged default 25; now the *per-tensor* share of the host set |
| `MOE_HOST_POOL_DIO` | unchanged default 0; `1` = never touch the page cache (the disk tier; also disables the mapping memcpy on the staging bounce path) |
| `MOE_HOST_POOL_PREWARM` / `_BGFETCH` / `_PREFETCH` | unchanged |
| `MOE_HOST_POOL_VERIFY` | **new, diagnostic**: byte-compare every pool fill against the model mapping (catches an O_DIRECT offset/alignment bug at the source) |
| `LLAMA_HOST_MASTER_PINNED` | **new, A/B only**: force the pinned master with `--host-experts pool` (the control arm for a measured comparison) |
| `MOE_EXPERT_CACHE_COLD_UVA` | unchanged (a pooled master has no device mapping, so the cold path stands down on its own) |

`ENVIRONMENT.md` §1.2.1 must be rewritten at promotion time: the "It does not reduce host RSS yet"
paragraph becomes the L2/L3 description above, and the two new variables are appended.

## 8. Promotion checklist (not started — needs the maintainer's go-ahead)

Files changed (uncommitted in `~/llama.cpp`): `ggml/include/ggml-backend.h`,
`ggml/src/ggml-backend-impl.h`, `ggml/src/ggml-backend.cpp` (device hook + `ggml_backend_host_buffer_from_ptr`),
`ggml/src/ggml-cuda/ggml-cuda.cu` (`ROCm_HostMap` buft + device hook + `set_tensor_async` routing),
`ggml/src/ggml-cuda/ggml-cuda-moe-cache-hooks.cu` (staging/gather), `ggml/src/ggml-cuda/moe-expert-cache.{cu,h}`,
`src/llama-model-loader.{cpp,h}`, `src/llama-model.cpp` (mapping-backed buffer for the pooled ctx),
`include/llama.h` + `common/arg.cpp` (enum/help text).

Owning blocks: **09** (host-resident-MoE iface/scheduler/pool: the pool, the fills, the staging, the two
worker bugs, the budget), **10** (the `ROCm_HostMap` buffer type + the device-iface hook live with the
scheduler/backend plumbing there, or stay in 09 if the fold prefers it), **12** (loader/model placement:
`host_expert_buft`, the mapping-backed ctx buffer, the help/enum text).  `patches/README.md` needs a block
note and `ENVIRONMENT.md` §1.2.1 the rewrite; then `make-patches.sh`, `validate-set.sh`,
`gate-per-block-build.sh`, `gate-prefill-logits.sh` on the new tree, the fork `rdna-boosts` refresh, and
the container worker.

## 9. Pointers

* r35 promotion record (the pool, Phase 1/2): `archive/work/host-expert-dio-cache/`.
* r49 reclaim (the mmap half of the `-ncmoe` peak): `archive/work/host-memory-reclaim/`, `WORKLOG.md`
  261011.017.
* issue #139 (the `evict_process_queues` queue stops), `ENVIRONMENT.md` §1.2.1,
  `patches/README.md` block 09, `GREEDY-PURITY.md` §44 (why a pageable upload must never be trusted).
* Measurements and gates: `WORKLOG.md` 261022.006.
