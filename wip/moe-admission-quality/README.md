# `moe-admission-quality` — why is the shipped expert-cache policy ~11 pts of hit rate below a static oracle, and how much of that is recoverable?

**Status: OPEN, seeded 2026-10-22 (r110) — offline/measurement first, no delivery change yet.**
Not part of the delivery; nothing here may be applied to the fork without the maintainer's go-ahead
(see `AGENTS.md`).  This is `TODO.md` #86 (a TODO handle — distinct from GitHub issue #86).

Seeded by the P1 instrument that closed `archive/work/moe-cache-coverage/` (the offline routing-trace
replayer, `scripts/moe-trace-replay.py`, `MOE_EXPERT_TRACE`).  That campaign answered its own question
(*slot allocation*: a per-layer quota is worth ~0.25 pt, closed unshipped) and in doing so exposed a
much larger number it was never scoped to explain.

## 0. TL;DR

On a real trace the delivered policy (`h` **0.8505** modelled, 0.8248 logged) sits only a few points
above the naive baseline, and **~11 points of hit rate / ~5 points of covered mass below an oracle static
placement** at the *same* slot count (`static` 0.8765, global-top-f bound 96.14 % vs 91.42 % covered).
The gap is produced by two things that the oracle has and the shipped policy does not:

1. **foreknowledge** — the oracle ranks experts by the *whole* trace's routing; the delivered policy
   learns online and only carries a *persisted* heat prior across runs, and
2. **no churn** — the oracle never evicts; the delivered policy evicts and re-fills (2112 evictions and
   ~37 k cold reads over 130 decode passes on the measured run).

**The replayer does not model the persisted-prior seed**, so it *cannot currently tell those two apart*:
the real run it was calibrated against had `MOE_EXPERT_HEATMAP=0`.  Step 0 is therefore to add the seed
model, because the answer decides whether this campaign has any work in it at all:

* if a heat-seeded arena lands near `static`, the shipped product is already close to the achievable
  bound (the oracle number is mostly hindsight) → **close the campaign with that measurement**;
* if it stays near `shipped`, **eviction churn / admission** is the lever, and the candidate policies are
  measurable offline before a line of C++ changes.

## 1. The measurement that opened this

Qwen3.6-35B-A3B UD-Q4_K_M, 2x gfx1201, `-sm tensor -ncmoe 40`, `MOE_EXPERT_CACHE_MIB=4096`
(112 of 256 slots/table = 43.8 % residency), `prompts/code-reasoning-mixed.txt` (296 tok),
`-n 128 --temp 0`, **`MOE_EXPERT_HEATMAP=0`** (so the replayer's unseeded model is comparable).
130 passes / 15 600 decode records / 120 tables.  Trace + JSON: reproduce with

```bash
MOE_EXPERT_CACHE_MIB=4096 MOE_EXPERT_HEATMAP=0 \
MOE_EXPERT_TRACE=/tmp/trace.txt ./build/bin/llama-cli -m <35B-A3B Q4_K_M> \
  -ngl 99 -sm tensor -ncmoe 40 -c 8192 -f prompts/code-reasoning-mixed.txt \
  -n 128 --seed 42 --temp 0 --no-display-prompt --single-turn
./scripts/moe-trace-replay.py /tmp/trace.txt --arena-mib 4096 --devices 2 --measured-h <logged h>
```

```
policy          h    misses    cold   MiB/tok resident MiB  covered%
shipped    0.8505     58554   42927     42.78        4082.3    91.42
static     0.8765     48351       0      0.00        4082.3    95.38
quota      0.8529     57588   41928     42.09        4095.4    91.43
window     0.7846     84348   81108     61.66        4082.3    90.48
static bound (global top-f, same slot count): 96.14 % of the routed mass
```

Policy definitions (all identical in `scripts/moe-trace-replay.py`, so a new candidate is a small edit
there and zero C++):

| policy | what it models | shippable? |
|---|---|---|
| `shipped` | per-table uniform slots, F-LRU eviction with the 32-token decay, `admit=touch` doorkeeper, misses served cold, **no seed** | yes (this is the delivery) |
| `static` | each table pre-filled with its **whole-trace** hottest experts, never evicted | no — hindsight |
| `quota` | `shipped` with per-layer slot counts from routed mass | yes — measured, **closed unshipped** (`+0.0025 h`, flat mass) |
| `window` | `shipped` + pass-start pre-admission of the pass's whole distinct set | no — needs the routing before the op computes it |

Two side results from the same run, both worth remembering:

* **`window` is *worse* on `h` (0.7846) than `shipped`, while cutting traffic 25 %.**  Pre-admission
  displaces residents it cannot outrank later, so a "prefetch" is not automatically a win — measure it,
  never assume it.
* `--mass-band all` (prefill + decode) **misranks layers**: the prefill pass routes ~`n_tok` experts in
  one pass and one table per pass can be lost to the staged path with a null-buffer `src0`
  (`OFFLINE-audit` note in the resolution).  The tool defaults to the decode band.

## 2. What to measure, in order (offline first)

### Step 0 — make the replayer model the persisted-prior seed (P0, the whole campaign hinges on it)

Add `--seed none|heat|trace`:

* `none` — today's default (matches `MOE_EXPERT_HEATMAP=0`).
* `heat` — pre-fill each table with its top-`slots` by a **prior** distribution.  Two sources:
  (a) a `<model>.moe-heatmap` sidecar (`archive/work/moe-cache-coverage/tools/heatmap-audit.py` already
  parses `MHT1`), so the number is the *real* prior; (b) the trace's own first N passes, as a proxy.
  The prior must be loaded **without** the trace's own counts (otherwise it is the oracle in disguise).
* `trace` — the oracle (`static`); keep it as the upper bound.

Deliverable: a 2x2 `{seed: none, heat} x {churn: on, off}` table that decomposes the 11 points into
*foreknowledge* and *churn*.  That single table decides the campaign.

### Step 1 — only if churn dominates: candidate admission/prefetch policies

Each is a few lines in the replayer, scored on the same trace *and* on a second, different workload
(reasoning vs code) before any C++ is written:

1. **Lookback prefetch** — on a pass, pre-admit the *previous* pass's routed set for that table (the only
   thing a real implementation can know).  This is the honest version of `window`; measure whether
   `window`'s 25 % traffic cut survives a one-token lookback.
2. **Eviction order variants** — heat-weighted victim choice (the persisted prior as a tiebreaker), and
   `admit=value` at the verify widths (the doorkeeper is tuned for n_max 7; a wider verify batch is a
   different admission problem).
3. **Decay period vs verify width** — 32 tokens is a decode-token scale; with adaptive MTP the *pass*
   count is not the token count.  Sweep `--period` offline first.

### Step 2 — the smallest C++ that can win

Only the winning candidate, behind an env kill-switch, with the same-eviction-order purity argument the
existing policy has (`GREEDY-PURITY.md` §19/§24/§25 and the mmvq band rules): **placement/admission must
not change the numerics** (a cold read and an arena fill serve the same bytes), so the gate is
bit-identical same-seed text plus the prefill-logit gate for the moved arena.

## 3. Hard invariants any candidate must respect (copy them into the design review)

* **Layer-uniform slots** — `gate/up/down` of one layer share a slot count, and under `-sm tensor` the
  count is the *minimum over the devices owning the layer* (`alloc_all_locked`).  A per-table count
  breaks the shared remap.
* **`slots >= n_used * n_tok`** for the decode/verify band, else `moe_cache_update_host` declines the
  table every time (`g_decline_all`), the scheduler copies the whole table, and the per-token
  `all`/decline decision stops being constant — which is what a captured graph requires.
* **A residency decision that flips per token cannot be baked into one captured graph** — the hook's
  success must be constant per table.  Any prefetch must be *ordered* on the compute stream, not
  decided on the host mid-capture.
* **No arena address in a launch parameter may move while a kernel is in flight**
  (`moe_cache_sync_devices_locked`), and a captured graph is invalidated by a work-region shrink
  (block 10's `ggml_cuda_graph_mem_freed` bump).
* **A partially-failed cache falls back wholesale** (the fusion guard and `moe_cache_take_over` agree).
* The trace itself has to stay honest: an identity table's decode band is not traced (it cannot miss),
  so a trace from a fully-resident run is empty on decode — trace a partially-resident config.

## 4. Gates (unchanged from the coverage campaign)

* Same-seed coherence (byte-identical text) and `scripts/gate-prefill-logits.sh` — both mandatory for a
  placement change (a moved arena changes which path serves a prefill-band op).
* Adaptive-MTP field gate (`benchmarks/mtp-adaptive-methodology.md`), `-n 3000`, `--reasoning` per axis,
  plus the phase-switching prompt, for anything that touches decode residency.
* `test-backend-ops -o MUL_MAT_ID`, the greedy-purity 8-type x 5-length grid, and
  `scripts/validate-set.sh` if promoted.
* Every policy claim comes with the replayer JSON (`--json`) and the trace's sha256, so it is re-derivable
  without a GPU.

## 5. Pointers

* `scripts/moe-trace-replay.py` — the instrument; `policy=` in `score()` is where a candidate goes.
* `ggml/src/ggml-cuda/moe-expert-cache.cu` — `access_locked` (F-LRU + the admission gate),
  `decay_locked`, `apply_heat_seed_locked` (the persisted-prior seed), `apply_prefill_seed_locked`,
  `pool_rank_from_prefill_locked` (the pool's warm order), `alloc_all_locked` (sizing), `moe_cache_rearm`.
* `archive/work/moe-cache-coverage/RESOLUTION.md` — the measurement that opened this (and the P2
  no-ship decision).
* `archive/work/moe-heatmap-persistence/` — the heat accumulator and its persistence contract.
* `archive/work/slab-transient-absorption/` — the arena-size lever (worth 3-6 pts of coverage); **not**
  this campaign.
