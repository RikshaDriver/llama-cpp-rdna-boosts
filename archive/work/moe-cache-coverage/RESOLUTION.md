# `moe-cache-coverage` — RESOLVED 2026-10-22 (r110)

**Closed.**  P1 (the offline routing-trace replayer) and P3 (metric parity) shipped in
`v14-097cf33d9-r110`; P2 (the per-layer capacity quota) was **measured on the P1 replayer and is not
worth shipping** — which is exactly what P1 existed to decide.  P4 (arena size) was handed to
`archive/work/slab-transient-absorption/` in the 2026-10-17 retarget and stays there.

## What shipped (block 09, r110)

| item | what |
|---|---|
| **P1** | `MOE_EXPERT_TRACE=<path>` — the cache dumps the raw routing of every `mul_mat_id` (DECODE from the scheduler hook where the ids are already on the host; PREFILL from the device entry, one readback per op, safe because prefill is never captured; an identity table's decode band is not traced because it cannot miss).  `MOE_EXPERT_TRACE_MIB` caps the file.  The replayer is `scripts/moe-trace-replay.py` (delivery tooling, not a patch). |
| **P3** | `moe_cache_report` now prints the **covered mass** of the resident set against *this session's* routing (`heat_delta`, not the persisted prior), the **global-top-f static bound** at the same slot count, and the resident/expert slot counts, next to the existing `h` and expert MiB/token. |
| **P2** | *Not shipped — measured, no value.*  See below. |
| — | `TODO #64`'s three residuals closed in the same amendment (host-pool warm order, sampled identity decode tally, dead `moe_cache_observe` removed). |

## The P1 result: per-layer capacity quota is a wash

Trace: Qwen3.6-35B-A3B UD-Q4_K_M, 2x gfx1201, `-sm tensor -ncmoe 40`, `MOE_EXPERT_CACHE_MIB=4096`
(112 of 256 slots/table = 43.8 % residency), prompt `prompts/code-reasoning-mixed.txt` (296 tok),
`-n 128 --temp 0`, `MOE_EXPERT_HEATMAP=0` (the persisted prior is not in a trace, so the replayer
models the unseeded cache).  130 passes, 15 600 decode records, 120 tables.

```
policy          h    misses    cold   MiB/tok resident MiB  covered%
shipped    0.8505     58554   42927     42.78        4082.3    91.42
static     0.8765     48351       0      0.00        4082.3    95.38
quota      0.8529     57588   41928     42.09        4095.4    91.43
window     0.7846     84348   81108     61.66        4082.3    90.48
static bound (global top-f, same slot count): 96.14 % of the routed mass
```

* **Calibration:** the run's logged `h` was **0.8248** against the model's 0.8505 (+0.026).  The model
  has no persisted heat prior, no seed and no arena shortfall, so it is systematically a little
  optimistic; **judge policies by their delta, not by the absolute `h`** (the tool prints this).
* **Quota (P2): +0.0025 h, −1.6 % host traffic.**  Per-layer routed mass is essentially flat in the
  decode band, so the water-filled slot counts come out uniform (113/table) except where the UD quant
  promotes a layer's `down_exps` to Q6_K (+19 % expert bytes ⇒ 106 slots).  This reproduces the
  campaign's own prediction (0.3-0.5 pts on a uniformly tallied map) and is **inside the noise of a
  ~0.8 pt trace-to-trace spread**, for a sizing change that would have to preserve the per-layer
  uniform re-size, the re-arm slot table and the `slots >= n_used*n_tok` decline bound.  Not worth it:
  closed.
* **What *is* large:** the gap between the shipped policy (0.8505) and the oracle static bound (0.965
  hit rate / 96.1 % covered mass) is ~11 pts — an admission/placement-quality gap, not a
  slot-allocation one.  `window` (pass-start pre-admission of the whole routed set) does not capture it
  because a real prefetcher does not know the pass's routing in advance; the honest next question is a
  *history*-driven prefetcher, which is a new campaign, not a P2 variant -- filed as
  **TODO #86**, [`archive/work/moe-admission-quality/`](../../moe-admission-quality/README.md) (closed in r114).
* **Caveat on `--mass-band all`:** the prefill band routes ~`n_tok` experts per layer in a single pass,
  so it dominates any mass ranking; it also loses one table per pass to the staged path (offline
  `src0->buffer == nullptr`), which distorts a per-layer ranking.  The tool defaults to the decode band
  and documents it.

## Files

* `README.md`, `FINDINGS-2026-10-17-coverage-vs-defrag.md` — the campaign's history (unchanged; they
  describe the state before this resolution).
* `tools/` — `heatmap-audit.py` (the P0 health/coverage instrument), plus the pre-P1 scripts.
* The tool that closed the campaign lives in the delivery: `scripts/moe-trace-replay.py`
  (`scripts/README.md`).  Record: `WORKLOG.md` 261022.005.
