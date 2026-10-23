# RESOLUTION — `host-pool-swap-and-prefill` (r114, 2026-10-22)

**Outcome: closed.  `TODO #87`'s three remaining follow-ups are all resolved — two shipped in r114 (block
12), one measured and declined.**  Nothing in `README.md` §1-§6 is open; it is kept as the record.

## §7(1) `MOE_HOST_POOL_DIO` is now `auto` — SHIPPED (block 12)

The r113 default was a hand-set `DIO=1`, and the optimum is *inverted* between "the host expert set fits in
RAM" and "it does not": on a RAM-rich box O_DIRECT throws away a page cache that could hold the whole set,
on a memory-tight box buffering makes the page cache compete with the pinned pool for the same bytes.

`pool_decide_dio_locked()` decides once, at the **first pool build** (the point where
`MOE_HOST_POOL_RAM_PCT` already reads `MemAvailable`), from three numbers it logs:

```
MemAvailable  (one /proc/meminfo read)
planned pool  = min(MOE_HOST_POOL_MIB, FRAC % of the host expert bytes, RAM_PCT % of MemAvailable)
host expert set (Σ host_bytes * n_experts)
buffered  iff  planned pool + host expert set <= MemAvailable      (the page cache can hold the set)
O_DIRECT  otherwise                                               (it cannot; do not thrash)
```

`MOE_HOST_POOL_DIO=0|1` still forces either arm (an A/B must not depend on the machine), and the variable
now accepts any negative value / unset as `auto`.  No `MemAvailable` (non-Linux, unreadable) keeps r112's
O_DIRECT, which is the safe end.

Measured, 2x R9700, 35B-A3B Q4_K_M `-ncmoe 40 --host-experts pool`, 70K-token prompt, `-ub 2048`:

| arm | prefill t/s | decode t/s |
|---|---:|---:|
| `auto` → buffered (MemAvailable 173666 MiB, planned pool 14930 MiB, set 18662 MiB) | **2103** | **46.6** |
| `MOE_HOST_POOL_DIO=1` forced | 2088 | 22.7 |
| `pinned` (reference) | 2099 | 49.3 |

The 30 GiB gfx1100 box is the other end of the same rule (`planned pool + 32.6 GiB set > ~25 GiB
MemAvailable` → O_DIRECT, the arm that measured clean there in r111/r112).

## §7(2) the pool-backed prefill stages in ONE launch — SHIPPED (block 12)

`moe_cache_stage_table`'s pool branch used to copy the tier-resident experts with one `cudaMemcpyAsync`
each (~200 per table per ubatch), where the pinned-master path has always staged a whole table with a single
`moe_cache_stage_kernel`.  `moe_cache_stage_kernel` now takes a pool source, and the per-expert snapshot it
reads encodes both tiers:

```
prs[e] >= 0    arena slot
prs[e] <= -2   pool slot (-v - 2), read from pool->base_dev (the pinned pool's UVA alias)
prs[e] == -1   in neither tier: the host already copied it (each of those is a separate mapping/disk read)
```

The snapshot is rebuilt on every pool-backed staging call — it also depends on the pool's slot map, which
the demand path and the bulk layout change without bumping `g_write_seq` — for ~1 KiB against a whole-table
copy.  `MOE_HOST_POOL_STAGE_BATCH=0` restores the per-copy arm.

| arm (same config as above) | prefill t/s | decode t/s |
|---|---:|---:|
| one launch (default) | **2103 / 2099** | 46.6 / 46.4 |
| `MOE_HOST_POOL_STAGE_BATCH=0` | 1564 | 46.2 |

+34 % prefill, and the pool now stages within 0.3 % of `pinned` (2103 vs 2099).  This is the item's
predicted "rest of the gap to `pinned`", and it is larger than r113's estimate (733 vs 836 t/s) because the
estimate was taken on a config where the staging was not crossed.

**Purity.** Greedy text byte-identical across `pinned` / pool / `DIO` 0 and 1 / staging on and off (146 chars
at `-n 32` and 735 chars at `-n 200`, one hash each), `MOE_HOST_POOL_VERIFY=1` clean (0 `POOLREAD MISMATCH`),
and `scripts/gate-prefill-logits.sh` green (mean KLD 0.000000, same-top-p 100.000 %).  Pool fills serve
byte-identical copies, so placement cannot move a value.

## §7(3) long-session adaptation — MEASURED AND DECLINED

The rule was "do not add an eviction policy before re-running the two-tier model on a long,
topic-shifting trace".  That run is now done: a real 1024-token decode trace (`MOE_EXPERT_TRACE`, 35B-A3B
Q4_K_M, 2 GPU `-ncmoe 40`, `MOE_EXPERT_CACHE_MIB=4096`) with a different-topic 128-token decode concatenated
after it, replayed through `scripts/moe-trace-replay.py --pool-frac 80 --arena-mib 4096 --devices 2`:

| pool layout (the session moved topic part-way through) | L3 reads |
|---|---:|
| oracle (whole-trace prior) | 7.45 MiB/token (~4.1 ms/token of disk) |
| built from the **new** topic | 7.55 MiB/token (~4.1 ms/token) |
| built from the **previous** topic (the delivered case) | 8.05 MiB/token (~4.4 ms/token) |

So the entire prize for adapting the layout mid-session is ~6 % of the L3 traffic -- **0.5 MiB/token, i.e.
~0.3 ms/token at this box's ~1.8 GB/s SSD peak**, in a band where a decode token already costs ~22 ms (46 t/s on the field config).
Every mechanism that adapts by churning costs several times that on the same trace:

* `pool-retire` (promote-on-demand, refill empty slots) — **24.6 MiB/token** (3.3x the static layout);
* re-admitting the arena victim into the pool (`pool-shipped`) — trades **32.8 GiB of copies** for 8.7 GiB
  of L3 reads, i.e. the 2x-decode result r113 already measured and dropped.

The only candidate left is a **periodic bulk re-rank** of the existing layout (the bulk pass
`pool_rerank_from_heat_locked` already does once at the first decode-band pass), whose ceiling is that ~6 %.
It is not implemented: on the field configs the prefill dominates wall clock, and 6 % of the pool's L3
traffic does not pay for a periodic stall.  If it is ever revisited, `scripts/moe-trace-replay.py` is the
instrument, and the numbers above are the bar it has to beat.

## Pointers

* Code: `pool_decide_dio_locked`, `table_pool_locked`, `moe_cache_stage_kernel`,
  `moe_cache_stage_table` — `ggml/src/ggml-cuda/moe-expert-cache.cu` (block 12).
* Knobs: `ENVIRONMENT.md` §1.2.1 (`MOE_HOST_POOL_DIO`, `MOE_HOST_POOL_RAM_PCT` **85** since r114,
  `MOE_HOST_POOL_STAGE_BATCH`).
* Sibling question on the arena side: [`../moe-admission-quality/`](../moe-admission-quality/) (`TODO #86`,
  closed by the same release; the lever it exposed — the band-split heat prior — shipped in the same release
  too.
* Prior records: [`../host-pool-layout/`](../host-pool-layout/) (r113) and
  [`../host-expert-master-drop/`](../host-expert-master-drop/) (r111).
