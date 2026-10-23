# `host-pool-layout` — the host pool is a LAYOUT, not a churning cache (r113, TODO #87)

**Status: CLOSED (r113, 2026-10-22).**  Promoted into **block 12**.  `RESOLUTION.md` records the decision,
the measured-and-dropped candidates and the gates.  Everything here is reference; nothing in this directory
is part of the delivery.

> **CORRECTION (r114, 2026-10-22) — the `sidecar` rows in §1 below are the ORACLE, not the sidecar.**
> `scripts/moe-trace-replay.py`'s `load_heat_sidecar` did not normalise the role name (a `.moe-heatmap` stores
> the GGUF tensor name `blk.0.ffn_gate_exps.weight`, a trace stores `ffn_gate_exps.weight` + the layer), so
> every lookup missed and `--pool-init sidecar-complement` silently fell back to the whole-trace `heat`
> prior.  The "persisted heat prior" row (237 MiB) and the "oracle prior" row (237 MiB) are therefore the
> same measurement twice, and **this record's "the achievable prior matches the oracle" conclusion is not
> established**.  The r113 *code* is unaffected (the C++ reads the real heat vector); only the offline
> number was wrong.  Fixed in r114 (`archive/work/moe-admission-quality/RESOLUTION.md` §5) — re-run the model
> with the corrected tool before quoting a pool-prior number.  The rest of the record (the pool's content
> beats every eviction/hand-over policy by 0-246 MiB) rests on the `pool-*` policy rows, which never used the
> sidecar, and stands.

## 0. TL;DR

`--host-experts pool` was making the disk do work it did not need to.  Two independent causes, one fix each:

| # | cause | fix |
|---|---|---|
| 1 | the pool's **content** was expert-id order (a prewarm with no tally, then a re-rank by the *prefill* tally, which is nearly uniform because a prefill routes every expert) | lay the pool out as the **arena's complement, ranked by the persisted heat prior**, in bulk (at sizing and at the first decode-band pass) |
| 2 | the **whole-table prefill staging bypassed the pool** and re-read the entire host expert set from the file on every ubatch; the demand path then evicted a pool resident on every arena fill and re-read the evicted arena expert back into the pool | staging reads the pool (and fills EMPTY slots); the demand path **never evicts, never copies the victim into the pool** |

Measured on the gfx1201 field config (Qwen3.8-Flash-Next **IQ4_NL**, 2× R9700 `-sm tensor`, `-ncmoe 48`
= 64 800 MiB of host experts, MTP `--spec-draft-n-max 7`, `--cache-ram 16384`, 70K-token prompt,
128-token decode, one build):

| arm | prefill t/s | decode t/s |
|---|---:|---:|
| pre-change pool, `DIO=1` (the shipped default) | 39 (still prefill after 27 min) | — |
| pre-change pool, `DIO=0` | 581 | 30.6 |
| **r113 pool, `DIO=1` (default)** | **287** | **23.1** |
| **r113 pool, `DIO=0`** | **733** | **53.7** |
| `pinned` (reference, unaffected) | 836 | 57.5 |

Pool hit rate **0.7999 → 0.9183** (10.5K prompt; 0.9044 at 70K), pool evictions **4600 → 0**.

## 1. Why "content" and not "policy": the offline model

`scripts/moe-trace-replay.py` grew a **two-tier** model for this campaign (`--pool-frac`, policies
`pool-none`, `pool-retire`, `pool-shipped`, `pool-swap-slot`, `pool-swap-weakest`, and
`--pool-init sidecar:<file>` which reads a REAL `MHT1` `.moe-heatmap` prior — the `#86` step-0 instrument,
shared with `../moe-admission-quality/` (closed in r114)).  Model: arena (L1) + pinned host pool (L2) + mapping/disk (L3),
one slot map per table, counting L3 reads and every host↔device copy a policy pays for.

Trace: `MOE_EXPERT_TRACE` on the field config, 10.5K-token prompt, 512-token decode (11 523 D records,
864 P records, 144 tables).  Geometry from the same run: arena 369 slots/table (23 414 MiB per device),
pool 409/table (FRAC 80), expert_bytes 460 800 per device.

| pool content | pool hit | L3 (disk) reads over 451 decode tokens |
|---|---:|---:|
| expert id (the pre-r113 behaviour) | 0.7995 | 3571 MiB |
| **persisted heat prior, as the arena's complement** | 0.3923 | **237 MiB** |
| oracle prior (this run's own decode routing) | 0.3923 | 237 MiB |
| expert id, arena heat-seeded (i.e. the complement fixed but unranked) | 0.6813 | 5346 MiB |

And, with the heat-ranked complement fixed, the *policy* axis:

| policy (what happens to the arena victim) | L3 | hand-over copies |
|---|---:|---:|
| `pool-none` (never touch the pool) | 237 MiB | 0 |
| `pool-retire` (free the promoted expert's slot) | 246 MiB | 0 |
| `pool-shipped` (evict pool LRU, admit incoming, re-read victim into the pool) | 0 | 171 MiB |
| `pool-swap-weakest` (victim overwrites the pool's weakest resident) | 132 MiB | 171 MiB |
| `pool-swap-slot` (victim takes the incoming's own slot) | 0 | 342 MiB |

i.e. everything on the policy axis is within 0–246 MiB, against a **15× difference** for the content axis.
Model calibration: `h` 0.9369 vs the run's logged 0.9044–0.9183 (the model has no arena shortfall, as
documented in the tool's caveats); `pool_h` 0.7995 vs the logged 0.7999.

## 2. Measured on the GPU (not modelled)

* **`pool-swap-slot` was implemented and dropped.**  It cost **2× decode** (25.9 t/s vs 52.8 t/s with the
  hand-over disabled, identical text and acceptance, 6259 swaps × 2 extra copies) and bought 0 MiB of L3 —
  the model predicts both.  The copies are 2-D `cudaMemcpy2DAsync` (an `-sm tensor` axis-0 slice is a fine
  strided scatter), which is why they are so expensive per event.
* **Write-over-the-weakest-candidate** (the maintainer's proposal) is the cheapest of the moving policies
  (1 copy, 105 MiB of L3 saved in the model) but still costs a copy per arena eviction for a gain the
  layout makes irrelevant — recorded here, not shipped.
* **The layout alone** reproduces the model: pool hit 0.80 → 0.92, evictions → 0, prefill 581 → 733 t/s
  (`DIO=0`) and 39 → 287 t/s (`DIO=1`), decode 30.6 → 53.7 and 6.3 → 23.1 t/s.

## 3. `MOE_HOST_POOL_RAM_PCT` (default 80)

`MOE_HOST_POOL_FRAC` is a percentage of the **model's host expert set**, not of RAM, and it is capped
there (`slots = min(n_experts * frac/100, n_experts)` — FRAC 100 = the whole set pinned once, never more).
Nothing tied it to what the box can actually pin, and an oversized pool fails *silently*: each table's
`cudaHostAlloc` fails, that pool is marked failed and its reads take the bounce path (the disk).  r113 adds
a ceiling on the WHOLE pinned pool of `MOE_HOST_POOL_RAM_PCT` % of `MemAvailable` (measured once, at the
first pool build), trims per table with one WARN naming both knobs, keeps the one-slot floor, and `0`
disables it.  Verified by forcing `MOE_HOST_POOL_RAM_PCT=5` on 35B-A3B `-ncmoe 20`: ceiling 8693 MiB,
pool trimmed to 8714 MiB (60 tables), coherent output.

## 4. Files

| file | what |
|---|---|
| `run-field.sh` | the reproduction harness (`run-field.sh <tag> pool\|pinned [prompt]`, env: `DIAG_EXTRA_ENV`, `NTOK`, `TEMP`, `SEED`, `NCMOE`, `CTX`, `CACHERAM`, `GPUS`, `BIN`) |
| `logs/r112-*` | the pre-change arms (pinned 70K, pool `DIO=0` at 10.5K and 70K, `DIO=1` 10.5K) |
| `logs/r113-*` | the post-change arms (the same matrix + `pinned` at 70K/128 tok) |
| `logs/trace-field.log` | the traced run (the trace file itself is 60 MB, not committed) |

Repro of the model (needs the trace and a sidecar):

```bash
MOE_EXPERT_TRACE=/tmp/field.trace ./run-field.sh trace pool /tmp/prompt10k.txt   # -lv 4 for the pool lines
./scripts/moe-trace-replay.py /tmp/field.trace --devices 2 --arena-mib 23414 --pool-frac 80 \
    --pool-init sidecar-complement:/llm/models/.../<model>.moe-heatmap --admit always \
    --policies pool-none,pool-retire,pool-shipped,pool-swap-weakest,pool-swap-slot
```

## 5. What is still open (belongs to #87, not to this campaign)

* **`DIO=1` vs `DIO=0` is still a hand-set default** (the "inverted optimum"): on a box where the host set
  fits in RAM the page cache is the right L3 and `DIO=0` is worth +90 % decode on the field config; on a
  memory-tight box `DIO=1` is.  An auto choice from `MemAvailable` at pool build is the obvious next step.
* **The pool-backed prefill still issues one `cudaMemcpyAsync` per expert** (the pinned-master path uses a
  single `moe_cache_stage_kernel` launch).  That per-copy issue cost is the rest of the prefill gap to
  `pinned` (733 vs 836 t/s at `DIO=0`), and a pool-aware staging kernel (per-expert slot map, like `prs`)
  would close it.
* **#87(a)** (SIGSEGV in the background worker at high FRAC, `DIO=0`) is moot — that worker is gone — but the
  `DIO=0` arm on a memory-tight box still deserves a soak before it is recommended again.
* Long-session adaptation: the pool is static between the two bulk layouts, so a single session that
  changes topic keeps its pre-prompt layout.  The model says the alternatives cost more than they save; a
  future campaign should re-run the two-tier model on a long, topic-shifting trace before adding one.
