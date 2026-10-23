# RESOLUTION — `moe-admission-quality` (r114, 2026-10-22)

**Outcome: closed.  The ~11-point gap is FOREKNOWLEDGE, not admission and not eviction churn.**  The
delivered policy (heat map on, its default) is already within ~0.5 pt of an oracle static placement at the
same slot count; the remaining headroom is the *quality of the persisted prior*, and the largest defect in
that prior is that it mixes the prefill and decode bands in one counter.  That finding became `TODO.md` #88
and was **implemented in the same release** (r114, block 12): the arena seed now ranks by a decode-only
counter, and on the GPU with the same map in both arms `h` goes 0.8747 → **0.8862** (284-token prompt) and
0.8698 → **0.8948** (5256-token prompt).  Offline, in the units that actually bound decode on this box, the
long-prompt arm drops from **48.1 → 17.6 MiB of L3 per token**, i.e. from ~26 ms to ~9.5 ms of disk read
per token at the SSD's ~1.8 GB/s.

The campaign's own decision rule (`README.md` §0) was: "if a heat-seeded arena lands near `static`, close the
campaign with that measurement; if it stays near `shipped`, score the candidate admission/prefetch policies".
The heat-seeded arena lands *on* `static`; the unseeded one lands near `shipped`; and the reason the two
differ is the seed's accuracy, which is measurable and is not an admission-policy question.  So: close, and
re-seed the prior question.

## 1. Step 0 — the replayer now models the persisted-prior seed

`scripts/moe-trace-replay.py` grew (r114):

* **`--seed none|heat|probe:N|sidecar:<path>`** — the load-time arena seed, mirroring
  `apply_heat_seed_locked` exactly (rank by the prior, count desc / expert id asc, fill slots
  `0..slots-1` with `count=0, ghost=0, last=0`; a seeded expert must be the first thing eviction gives up).
  `none` = today's unseeded model, `heat` = the trace's own routing (an ORACLE prior — and, as an identity
  check, `--seed heat --churn off` reproduces the `static` policy bit for bit), `probe:N` = the trace's
  first N passes (a same-session prior, the most a run can know before generating), `sidecar:<path>` = a
  REAL `MHT1` `.moe-heatmap` from a previous run — the achievable prior and the one the delivery uses.
* **`--churn on|off`** — `off` lets a miss take an EMPTY slot but never evict a resident, so the arena becomes
  a pure function of its seed and the effect of eviction is separable from the effect of foreknowledge.
* **`--score-band all|decode`** — `all` is what the runtime's logged `h=` counts (keep it for calibration);
  `decode` drops the prefill band, whose whole-table staging is not what an arena policy is about.  The
  two-tier pool model already worked this way.

Two tool defects were found and fixed while doing this, both of which had silently produced wrong numbers
before (see §5).

## 2. The 2x2 decomposition

Qwen3.6-35B-A3B UD-Q4_K_M, 2x gfx1201, `-sm tensor -ncmoe 40`, `MOE_EXPERT_CACHE_MIB=4096`
(**112 of 256 slots/table = 43.8 % residency**), `MOE_EXPERT_HEATMAP=0` for the trace, 128 decode tokens,
`-n 128 --temp 0 --seed 42`.  Two workloads: **A** = `prompts/code-reasoning-mixed.txt` (284-token prompt,
hit `h=0.8248` logged) and **B** = `prompts/prose-rdna-boosts.txt` (5256-token prompt, `h=0.8287` logged).
Decode band, `--arena-mib 4096 --devices 2`:

| seed \ churn | on (the delivery) | off (never evict) |
|---|---|---|
| **none** (unseeded: `MOE_EXPERT_HEATMAP=0`) | A `h=0.8139` / 54.2 MiB/tok &nbsp; B `0.8189` / 52.8 | A `0.7881` / 61.7 &nbsp; B `0.7822` / 63.5 |
| **sidecar** (the real persisted `.moe-heatmap`) | A `0.8324` / 48.8 &nbsp; B `0.8468` / 44.6 | A `0.7180` / 82.1 &nbsp; B `0.8013` / 57.9 |
| **heat** (oracle prior) | A `0.9461` / 15.7 &nbsp; B `0.9464` / 15.6 | A `0.9538` / 13.5 &nbsp; B `0.9535` / 13.6 |
| **static** (oracle, never evicted) | — | A `0.9538` &nbsp; B `0.9535` (0 cold, 0 traffic) |

Readings, in the order they matter:

1. **Eviction is not the problem — it is worth +2.6 pts** at `seed=none` (0.8139 vs 0.7881) and **+11.9 pts**
   with a real prior (0.8324 vs 0.7180).  A frozen arena, seeded or not, is strictly worse than one that
   re-ranks.  So the "churn" half of the campaign's hypothesis is refuted: the delivered policy must keep
   evicting.
2. **Foreknowledge is the whole gap.**  The same policy, same slot count, same trace — only the seed's
   quality differs: 0.8139 (none) → 0.8324 (real prior) → **0.9461** (oracle).  The oracle-seeded arm sits
   within 1 pt of `static` *while still evicting*, which is what "the oracle number is mostly hindsight"
   looks like when the hindsight is given to the policy too.
3. **The delivery is already on the `static` line.**  Whole-trace `static` = 0.8765 on A; the delivered
   configuration (heat map **on**, i.e. seeded from the persisted prior) measures **`h=0.8718` on the GPU**
   and the model predicts **0.8719** for that arm (all-band scoring, the calibration basis).  The
   campaign's 11-pt gap was measured with `MOE_EXPERT_HEATMAP=0`, which is not the shipped default.
4. A per-layer slot quota is worth +0.0003 on the decode band (0.8142 vs 0.8139 on A, 0.8191 vs 0.8189 on
   B, −0.1 % traffic) — confirming r110's `+0.0025` finding from the other side.
5. `window` (pass-start pre-admission) on the **decode** band reads 0.9178 / 23.9 MiB per token, but a decode
   pass *is* one token, so it pre-admits exactly the set it is about to serve: that is the hindsight bound,
   not a policy.  (On the all-band scoring it was the campaign's "worse than shipped, 25 % less traffic"
   result, because there it pre-admits a whole prefill pass.)  The honest one-pass-lookback variant was not
   scored: step 1 of the plan is gated on churn dominating, and it does not.

## 3. Where the remaining 11 points live: the prior's coverage

The decisive measurement is not `h` but what fraction of the *decode* mass a prior's top-112 per table
actually covers (the oracle covers 95.4 % on both traces; the oracle-seeded arm's `covered` = 95.03 % /
95.09 % is the same thing after eviction):

| prior | A (code) | B (prose) |
|---|---:|---:|
| the persisted `.moe-heatmap` (Oct 16, mixed workloads) | **71.80 %** | **80.13 %** |
| a *fresh* map learned from the **same** workload, as stored (prefill + decode) | 92.21 % | 80.28 % |
| the same map restricted to its **decode** component | **95.28 %** | **95.27 %** |
| oracle (the trace's own decode routing) | 95.38 % | 95.35 % |

Two separate defects, both in the prior and neither in the policy:

* **Staleness / workload mix** — the persisted map is a lifetime accumulator from earlier sessions on other
  prompts (its ranking covers 71.8 % of trace A's decode mass against a fresh same-workload map's 92.2 %).
* **Band mixing (the larger one)** — `t.heat` is one counter fed by both bands, `+1` per (token, expert) in
  each.  A prefill ubatch routes ~`n_tok` experts, so on the 5256-token prompt the prefill contributes
  **41x** the decode mass (42016 vs 1080 touches per table on trace B), and the ranking degenerates toward
  "nearly every expert is hot" — which is exactly what a decode-band arena must not be seeded with.  Forcing
  the same map to its decode component takes trace B's coverage from **80.3 % → 95.3 %**, i.e. to the
  oracle's.  The estimate is exact: the trace and the learner saw identical routing (byte-identical text),
  so `sidecar/2 - prefill = decode` per expert.

`h` follows the coverage monotonically (0.8139 → 0.8324 → 0.9461 as coverage goes 71.8 → 71.8+eviction →
95.0), so **the shippable lever is a band-separated heat prior, worth up to ~10 pts of decode hit rate on a
long-prompt workload** — implemented in the same release (r114, block 12; see §7).  It is not an admission or
prefetch policy, which is why
this campaign closes with no C++ change.

## 4. What is explicitly NOT worth doing

* **A prefetch / pre-admission policy** (`window` and its lookback variants) — the decode-band bound is
  foreknowledge the policy cannot have; the honest one-token lookback was not scored because churn does not
  dominate, and r110 already measured pass-start pre-admission as a 6.6-pt `h` *loss* on the all-band view.
* **An eviction-order or admission-rule change** (`admit=value`, heat-weighted victims, a decay sweep) — the
  delivered F-LRU + `admit=touch` beats *no* eviction by 2.6-11.9 pts on these traces, and the 2x2 says the
  remaining variance is seed quality.
* **Slot quota / per-layer sizing** — +0.0003 `h` (r110: +0.0025 all-band).  Closed unshipped, again.

## 5. Tool defects found (both had produced published numbers)

1. **The `.moe-heatmap` sidecar never matched a trace.**  The sidecar stores the GGUF tensor name
   (`blk.0.ffn_gate_exps.weight`); a trace stores the cache's role (`ffn_gate_exps.weight`) plus the layer
   number.  `load_heat_sidecar` did not normalise, so every lookup missed — and `--pool-init
   sidecar-complement` **silently fell back to the whole-trace ORACLE** (`heat`) for every table.  That is
   the arm r113's record quotes as "a real `.moe-heatmap` prior = 237 MiB of L3 reads, the same as the
   oracle"; it was the oracle twice, so **r113's "the achievable prior matches the oracle" claim is not
   established** and its number needs re-measuring before it is quoted again.  Fixed: roles are normalised,
   a sidecar that matches **none** of the trace's tables is a hard error, the matched count is printed, and
   a missing table now degrades to expert-id order (what `pool_rank_experts_locked` would do) instead of to
   the oracle.
2. **`--mass-band` was not applied to the pass loop**, so the arena policies were scored on *all* passes
   while their `covered%` came from the decode band only.  On a long prompt this made `static` look worse
   than `shipped` (0.7553 vs 0.8498 on trace B) because the prefill band dominated the reaches.  Fixed with
   `--score-band`.

A third observation, harmless but worth knowing: under `-sm tensor` every device owns a table for the same
`(layer, role)` key, so the heat map's counters are **N-device multiples** of the true touch counts (the
`86160 = 2 x 43096` in trace B's map).  Ranking is unaffected (both copies scale together), but no absolute
threshold should be read off a map written by a multi-device run.

## 6. Gates

No delivery change, so no gate was owed.  The instrument's own calibration is the check that matters: the
unseeded model reproduces the campaign's published numbers exactly (`shipped` 0.8505, `static` 0.8765 — the
all-band basis), and the *seeded* model predicts a GPU run it did not see: model `sidecar/on` **0.8719** vs
the real `MOE_EXPERT_HEATMAP=load` run's logged **0.8718**.

## 7. Pointers

* Instrument: `scripts/moe-trace-replay.py` (`--seed`, `--churn`, `--score-band`; `scripts/README.md`).
* Code it models: `apply_heat_seed_locked`, `decay_locked`, `access_locked`, `moe_cache_heat_observe`
  (the band-mixing source), `heat_map_save_locked`/`load_heat_map` (the sidecar format) —
  `ggml/src/ggml-cuda/moe-expert-cache.cu`.
* Follow-up (shipped in r114): the band-separated heat prior — `apply_heat_seed_locked` / `heat_rank_vec` /
  `heat_map_save_locked`'s MHT2 format; `ENVIRONMENT.md` (the heat-map note) and `patches/README.md` (block 12).
* Prior record: [`../moe-cache-coverage/RESOLUTION.md`](../moe-cache-coverage/RESOLUTION.md) (the P1
  instrument and the P2 quota decision), [`../moe-heatmap-persistence/`](../moe-heatmap-persistence/) (the
  heat accumulator and its persistence contract).
