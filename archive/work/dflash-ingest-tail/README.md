# archive/work/dflash-ingest-tail — a windowed DFlash draft ingests only the prompt tail

**Status: promoted (r130, block 12).**  PR #212 (`overdoingism`), one commit against `v14-097cf33d9-r120`,
touching `common/speculative.cpp`, `src/llama-ext.h` and `src/llama-model.cpp`.  Re-cut onto the r129 tree
for the r130 set (the r120 version no longer applied: the DFlash impl had gained the DFlash2 selector, the
LiLiCorr head and the F1 device-injection members in between).  Env kill-switch: `LLAMA_DFLASH_INGEST_TAIL=0`.
Delivery doc: `patches/README.md` (block 12), `ENVIRONMENT.md` §3, `GREEDY-PURITY.md` §53,
`MANIFESTS.md` (block 12 row), `WORKLOG.md` 261024.011.

## What it does

Every layer of the Qwen3.8-27B DFlash2 draft attends over a 2048-position sliding window and its KV cache is
sized for the window, but `process()` fed it every prompt row: a 110K prompt encoded 110K rows, all but the
last ~2048 dropped again without ever being read.  Each prompt batch also waited on the target before the
draft could start, which broke the queue of the target's prompt ubatches.

When **every** draft layer has a window, the DFlash impl takes the prompt end from `set_prompt_end()` (the
block-02 hook) and skips the rows before `prompt_end - n_swa`; a batch left with no rows returns before the
target wait, and the first generation row resets the threshold as the MTP impl does.  The window test needs
the per-layer pattern, which `llama_model_meta_val_str()` cannot return (it skips array values), so the
patch adds `llama_model_n_layer_swa()` to the staging header `src/llama-ext.h`.

## Accuracy of the claims (review, 2026-10-24)

* **Mechanism: correct.**  DFlash sets `LLAMA_SWA_TYPE_STANDARD` unconditionally
  (`src/models/dflash.cpp`), so the mask is `p1 - p0 >= n_swa` and a prompt's last `n_swa` positions are
  exactly what a first generation row can attend to.  Its KV cache is the iswa cache with
  `size_swa = min(size_base, n_swa*(unified ? n_seq_max : 1) + n_ubatch)`, and its cell mapping is
  position-modular, so writing only the tail leaves the same cells behind.  The skip is exact, and the
  target verifies every drafted token anyway.
* **The gain is understated for this delivery.**  The PR measured -2.1 / -1.3 / -0.9 % prefill on r120.  On
  r129 (whose prefill is much faster, and where the F1 device path makes the per-batch
  `llama_synchronize(ctx_tgt)` a real pipeline stall) the same switch is **+8-12 %**:
  6.1K 1556 -> 1683 t/s, 26.5K 1591 -> 1787, 88.6K 1271 -> 1395, with no-draft at 1427/…/1415 t/s, i.e. the
  draft's share of the prefill falls from 10.2 % to 1.4 %.  35B-A3B Q4_K_M + DFlash2: 3611 -> 4139 t/s at
  6.1K (+14.6 %).
* **One claim does not hold for our model files.**  The PR says the Qwen3.6-35B-A3B DFlash draft has "5 of 6
  layers windowed", so the tail stays off there, and its cross-test table lists that arm as unchanged.
  `Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf` reports `dflash.attention.sliding_window_pattern = [true x 6]`, so the
  tail **does** engage (the startup log prints the window line).  That arm was therefore validated
  separately here, not taken from the PR: text and draft counts identical, +14.6 % prefill.
* **Not covered by the PR, covered here:** multi-GPU (3 GPU `-sm tensor` throughout), the F1 device path
  (the skip runs before `llama_synchronize()`, which is where most of the r129 gain comes from),
  `-ub 512` (a ubatch smaller than the window) and a 2-slot unified run.
* **API question answered:** keep `llama_model_n_layer_swa()`.  It is one read-only query in the delivery's
  staging header (`src/llama-ext.h`), it mirrors upstream's `llama_model_n_swa()`, and the alternatives are
  worse (a DFlash-specific query duplicates it; reading the metadata array needs a new array-capable
  metadata accessor).  Note this makes the patch delivery-only: it depends on the block-02
  `common_speculative_set_prompt_end()` hook, which upstream does not have, so there is no `upstream/` copy.

## Gates (r130, gfx1201, ROCm 10.1.0, `~/bin/build-llama-rocm`)

| gate | result |
|---|---|
| 3x R9700 `-sm tensor -mg 0`, 27B UD-Q4_K_XL + DFlash2-Q4_K_M, `-fa on -ctk/ctv q8_0`, `-ub 2048`, tail on vs `LLAMA_DFLASH_INGEST_TAIL=0` | 6.1K 1683 vs 1556 t/s, 26.5K 1787 vs 1591, 88.6K 1395 vs 1271; greedy text sha identical in every pair; draft acceptance identical |
| `-ub 512`, 26.5K prompt | 17.4 s vs 19.7 s wall; text and acceptance identical |
| 2 slots, unified KV, A/B ping-pong | text and acceptance identical; 7.9 s vs 8.7 s on the first prefill |
| 35B-A3B Q4_K_M + DFlash2-Q8_0, 6.1K | 4139 vs 3611 t/s (+14.6 %), text and acceptance identical |
| 3 GPU `-sm layer`, 6.1K | 1251 vs 1245 t/s (neutral), text and acceptance identical -- the gain is the draft's own encode plus the per-batch target wait, and neither is larger under a layer split |
| coherence vs the r129 build (dense 4B, dense 27B, 27B MTP, 27B DFlash2, 35B-A3B DFlash2, 35B-A3B `-ncmoe 40`) | text identical in **6/6** configs (only the timing/log lines differ); DFlash2 prefill 1287.7 -> 1398.5 t/s (27B) and 2751.6 -> 2994.2 (A3B) |
| adaptive-MTP four axes (`-n 3000`, r129 vs r130) | text identical on R/P/C/K; 79.3/98.6/106.1/112.9 vs 78.8/98.9/105.6/112.8 t/s (noise) |
| `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (q8_0 KV, dense 27B) | B=1 42.76 vs 42.92, B=4 138.69 vs 139.26, B=8 240.41 vs 239.94 t/s (no worse) |
| prefill-logit KLD | 0.000000 / same-top-p 99.980 % |
| per-block build (`--tip 531d70b23 --changed-from r129-tip`) | 2/2 (block 12, tip 13) |
| `validate-set.sh` | green (checksums, strict 14/14 on a fresh tarball, tree == `release.json.tree`) |
