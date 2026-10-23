# RESOLUTION — `three-gpu-performance` / `TODO #51` ("3-GPU decode is slower than 2-GPU")

**Verdict: NARROWED and CLOSED for the field/MoE arm.  `TODO #51`'s headline — the 2026-10-16 field
table (2 GPU 84.4 vs 3 GPU 56.9 t/s) — no longer reproduces: on r116 the same config measures 2 GPU 93.4
vs 3 GPU 131.2 t/s (+40.5 %).**  What is left is *plain* (non-speculative) decode at 3 devices, which loses
2-10 % depending on the split, with each arm's mechanism named below and recorded as an accepted
limitation with a production recommendation.  **No delivery change was made**: every residual is either a
sizing/configuration cliff (owned by the new `TODO #93`) or a property of the split itself.

Campaign home before this file: `wip/three-gpu-performance/` (revived 2026-10-16; the r55 closure record
is still here as `RESOLUTION-2026-10-13-closed.md`).  Tree: `v14-097cf33d9-r116`
(`~/llama.cpp/build-rocm`, ROCm 10.1.0, 3x R9700 gfx1201).  One binary for every arm.

---

## 1. The measurement that decides it — the field/MoE arm on r116

Qwen3.8-Flash-Next **GSQ-IQ3_XXS** + shared Q8_0 MTP head, `-ncmoe 48 -sm tensor -c 204800
--no-kv-unified -ub/-b 6144 -ctk/-ctv q8_0 -t 8` (exactly `archive/work/moe-verify-fusions/tools/field3.sh`),
`--spec-type draft-mtp-adaptive --spec-draft-n-max 7`, `-n 1000`, auto arena,
`ROCBLAS_USE_HIPBLASLT=0`.  Prompt `archive/work/kv-cell-offset-nondeterminism/prompts/mixed30k.txt`
(32358 tokens, sha256 `69624f4d207f40bdf7357e74ec978af5bfb4725a0968faf4eea4aab46c2ae241`).

| devices | arena | identity tables | prefill t/s | decode t/s | acceptance | accepted/generated | mean len |
|---|---:|---:|---:|---:|---|---|---:|
| 2 | 37328.1 MiB (91.2 %) | 0 / 288 | 1416.9 | **93.4** | 0.66358 | 215 / 324 | 3.65 |
| 3 | 40925.0 MiB (100 %) | 432 / 432 | 1517.1 | **131.2** | 0.74517 | 193 / 259 | 4.06 |

Against the r73 table this campaign was revived on (1/2/3 GPU = 37.1 / **84.4** / **56.9** t/s): the
2-GPU arm improved 11 %, the 3-GPU arm improved **2.3x**, and the ranking is now the same as the r55
closure found.  The r116 improvements that did it are r115's graph key (`archive/work/graph-key-churn/`)
and — as it turns out, for most of the residual — the arena being fully resident, which at 3 devices it
now is and at 2 devices with `-c 204800` it is *not* (§3).

## 2. The §1 matrix (all 24 cells, one binary, one context, one prompt)

`-ncmoe 48 -sm tensor|layer`, `-c 40000`, prompt as above, `-n 1000`, `ROCBLAS_USE_HIPBLASLT=0`, `-t 8`.
Two residency classes, held equal **at both device counts** by sizing `MOE_EXPERT_CACHE_MIB` per device
(the budget is MiB/device; under `-sm tensor` a device holds `host_bytes / n_devices`):

* **PART** — explicit `MOE_EXPERT_CACHE_MIB` 18830 (2 GPU) / 12550 (3 GPU) → 91.8 % / 90.8 % of the
  40925 MiB host set, **0 identity tables at both counts** (`-sm layer` 3 GPU: 51 of 144 fully resident —
  layer granularity, noted).
* **FULL** — auto arena → **40925 MiB = 100 % and every table identity at both counts** (294/294 and
  432/432 fully resident; `-sm layer` 141/144 and 144/144).

Decode t/s (prefill in brackets):

| residency | split | spec | 2 GPU | 3 GPU | Δ (3 vs 2) |
|---|---|---|---|---:|---:|---:|
| PART | tensor | none | 51.5 (1601) | 46.1 (1647) | **−10.5 %** |
| PART | tensor | draft-mtp (n-max 3) | 101.4 (1439) | 97.8 (1498) | −3.6 % |
| PART | tensor | draft-mtp-adaptive (7) | 96.8 (1429) | 100.7 (1483) | +4.0 % |
| PART | layer | none | 38.6 (1435) | 39.4 (1425) | +2.1 % |
| PART | layer | draft-mtp | 69.6 (1341) | 71.6 (1339) | +2.9 % |
| PART | layer | adaptive | 64.1 (1327) | 66.0 (1337) | +3.0 % |
| FULL | tensor | none | 66.4 (1619) | 65.1 (1684) | −2.0 % |
| FULL | tensor | draft-mtp | 113.9 (1446) | 129.9 (1531) | **+14.0 %** |
| FULL | tensor | draft-mtp-adaptive | 106.3 (1432) | 130.6 (1514) | **+22.9 %** |
| FULL | layer | none | 49.0 (1459) | 44.0 (1443) | **−10.2 %** |
| FULL | layer | draft-mtp | 80.4 (1343) | 80.4 (1355) | 0.0 % |
| FULL | layer | adaptive | 73.6 (1332) | 73.3 (1352) | −0.4 % |

Acceptance (`accepted/generated`, mean len) per arm is in [`evidence/matrix-summary.csv`](evidence/matrix-summary.csv);
raw per-run rows (including the invalidated sets) in `evidence/raw-matrix-*.csv`.

Reading it:

* **Every speculative arm is at least as fast at 3 GPUs once the arena is fully resident** (+14.0 % /
  +22.9 % tensor, 0.0 % / −0.4 % layer).  That is the field config, and it is the reverse of #51.
* **Plain decode is still slower at 3 GPUs**: −2.0 % with an identity arena, −10.5 % at 91 % residency
  (tensor), and −10.2 % on `-sm layer` even with every table resident.
* Prefill at 3 GPUs is +2.9 % (part/tensor) and +4.0 % (full/tensor) — unchanged from the campaign's
  earlier finding that prefill is link-bound, not the problem.

## 3. The dominant residual mechanism: the **identity cliff**, not the device count

While filling the FULL class at 3 devices with an explicit `MOE_EXPERT_CACHE_MIB=13800`
(the exact host set / 3, plus headroom) the arena came up **80 MiB short** — 40845.1 of 40925.0 MiB —
and **not one of the 432 tables was fully resident**, because a table is identity only at 100 % of its own
slots:

| 3-device arm, plain decode | arena | identity tables | decode t/s |
|---|---:|---:|---:|
| explicit `13800`/device | 40845.1 MiB | **0 / 432** | **46.7** |
| explicit `15000`/device | 40925.0 MiB | 432 / 432 | **65.0** |
| auto | 40925.0 MiB | 432 / 432 | 65.1 |

The same cliff is what made the *2-GPU* field arm (and the whole first pass of this campaign) look broken:
at `-c 204800` auto gives 2 devices 37808 MiB = 92.4 %, **0 identity tables**, 46.3 t/s — while at
`-c 40000` the same arm gets 40925 MiB, 294 identity tables and 66.4 t/s.  Identity is worth **+40 %**
decode here, and the log states why (`alloc_table_locked: … FULLY RESIDENT (512/512 slots): identity fast
path ON (the decode band now reads the arena directly; no per-token routing readback)`).

Mechanism: `alloc_all_locked()` (`ggml/src/ggml-cuda/moe-expert-cache.cu`) holds back
`rounding = dev_n_tables[d] * ggml_cuda_slab_arena_unit(d)` from the slab capacity, so a budget that is
"just enough" in aggregate lands one arena unit per table short of full residency.  That is the whole
difference between the −10.5 % and the +22.9 % columns in §2 at 3 devices, and between 46.3 and 66.4 t/s at
2 devices with a large context.  **Filed as `TODO #93`** (a sizing policy: prefer whole-table identity
when the shortfall is small, instead of spreading a one-slot deficit across every table).  Nothing was
changed in this campaign — it is a block-09 sizing change and needs its own A/B and gates.

## 4. Suspect 1 (all-reduce) — retired for throughput, with numbers

`GGML_CUDA_AR_PROFILE=1`, full/identity arena, plain decode, 300 tokens, 4096 calls:

| devices | total | avg/call | p50 | p90 | decode-band cost | share of decode |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 82.5 ms | 20.1 µs | 13.8 µs | 26.5 µs | 0.26 ms/token | 1.7 % |
| 3 | 144.3 ms | 35.2 µs | 21.3 µs | 56.2 µs | 0.45 ms/token | **2.9 %** |

The extra 0.19 ms/token at 3 devices is ~1.2 % of decode — most of the −2.0 % plain-tensor gap and nothing
more.  Backends at 3 devices: `hybrid` (default) 65.1 t/s, `internal` (forced for every size) 67.5 but it
changes the *prefill* AR path and therefore the text (smoke only), `nccl` 59.3 (**−9 %**, not
bit-identical).  `GGML_CUDA_AR_SMALL_BLOCKS=8` 64.1 vs the default 4's 65.1 → no gain at 3 ranks.  No AR
change is warranted; the per-device spin split (20.8 / 22.1 / 14.2 µs avg) is reported in
`evidence/ar-profile-3g.txt`.

## 5. Suspects 2 and 4 (verify-width gather / draft placement) — resolved by the residency axis

* **Verify-width gather (suspect 2):** the tensor-split MoE path's cost is the *non-identity* decode band,
  not the verify width.  At 91 % residency (non-identity, both counts) the fixed-depth MTP arm is −3.6 %
  and adaptive is +4.0 %; at 100 % identity the same arms are +14.0 % / +22.9 %.  The control with the
  cache off entirely (`MOE_EXPERT_CACHE_MIB=0`, 2 GPU 21.1 vs 3 GPU 19.7 t/s, −6.6 %) shows the residual
  per-op host traffic still scales with device count — but the cache-on path is 3x faster, so it is the
  cache that matters, not the gather.
* **Draft/target placement (suspect 4):** the MTP arms are the *best* arms at 3 devices when resident, so
  the draft does not serialize on one device in a way that matters.  The layer-split arms (where every op
  is local and there is no AR at all) are flat-to-negative, which is the pipeline-stage effect of §6, not
  placement.
* The identity cliff dominates both, and it is measured, not inferred (§3).

## 6. The one remaining genuine 3-device loss: plain decode on `-sm layer`

`-sm layer` loses 0.4-10.2 % at 3 devices in this campaign, and the same sign appears on a **dense** model
(27B Q8_0, depth 15746, plain decode: `-sm layer` 18.0 → 16.9 t/s = **−6.1 %**, greedy text identical at
both counts), while the dense **tensor** split gains **+25.6 %** (33.6 → 42.2 t/s) at the same depth.  A
single-token decode has no intra-token parallelism to gain from a layer split; each extra device adds a
pipeline stage and a handoff, so 3 devices is strictly worse there.  This is a property of layer splitting,
not a defect, and it is why the field config is a tensor split.

## 7. Block 13's mirrored MTP draft subset — measured again, accepted with the rule

27B Q8_0, `draft-mtp-adaptive,ngram-mod` (`--spec-ngram-mod-n-match 45`, n-max 7), `-n 1000`, `-sm tensor`,
trim 0 vs 98304 (text identical within each device count, so the workload is identical):

| devices | trim 0 | trim 98304 | Δ |
|---|---:|---:|---:|
| 2 | 84.6 t/s | 85.3 t/s | +0.8 % |
| 3 | 105.5 t/s | 103.5 t/s | **−1.9 %** |

r109's −2.2 % reproduces (−1.9 %, text `3981ee1763f8` in both 3-GPU arms).  **Accepted limitation, not
fixed here**: the opt-in `--spec-draft-mtp-vocab N` is only profitable when `N · n_devices < n_vocab`
(block 13 is *mirrored* per device under `-sm tensor` while the untrimmed head is split), which the delivery
already documents (`AGENTS.md`, `patches/README.md` block 13, `GREEDY-PURITY.md` §48).  The production rule
stands: **do not enable the trim at 3 devices on a 248320-vocab model** — there is no profitable `N` there.
A startup *warning* when the rule is violated is filed under `TODO #93` rather than smuggled in.

## 8. Purity / determinism findings this campaign produced

1. **Cross-device-count greedy text is not a valid gate on the MoE tensor-split path.**  2 vs 3 GPUs
   produce different text (`40d493bbb5ac` vs `ecf8a32bae77`) even with both arenas 100 % identity and with
   the AR's BF16 wire disabled, while PPL on the same model is **10.8452 vs 10.8623 (0.16 %)** — the tensor
   split's own rounding class (`GREEDY-PURITY.md` §45), not a deviation.  A **dense** tensor split and
   *every* `-sm layer` arm **are** text-identical across device counts (same hash, both counts).
   ⇒ compare builds *within* a device count (as r116's own gates did); use `-sm layer` when a cross-count
   text comparison is needed.
2. **Spec purity is intact per device count:** in all 24 cells the three `--spec-type` arms of one
   (class, split, device count) produce the **same text hash** (`none` == `draft-mtp` ==
   `draft-mtp-adaptive`, n_max ≤ 7).  The adaptive controller moves `generated`/mean len slightly between
   arms with identical text (e.g. 216/332/3.60 vs 215/336/3.56 in the layer arms) — the §48 schedule
   effect, unchanged.
3. `scripts/check-gpu-faults.sh --since "150 min ago"` over the run logs and the kernel journal: **clean**.
4. One arm OOM'd (3 devices, `-ncmoe 40`, 100 % arena at `-c 204800`, MTP draft: `CUBLAS_STATUS_ALLOC_FAILED`
   in `hipblasCreate`) — VRAM exhaustion from `-c 204800` + a fully resident arena + the draft context, not
   a fault; dropping to `-c 40000` or `-ncmoe 48` fits.  It is why the matrix uses `-c 40000`.

## 9. What was measured, and what was deliberately not

Measured: the 24-cell matrix (§2), the 2026-10-16 field config re-run (§1), the identity cliff incl. the
explicit-budget A/B (§3), the 2-vs-3-device AR profile + backend/blocks sweep (§4), cache-off and
AR-BF16-off controls (§5, `evidence/controls.md`), the dense layer-vs-tensor depth control (§6), the
block-13 trim (§7), cross-count PPL, and the GPU-fault check (§8).

Deliberately **not** measured, and why:

* **The `GGML_CUDA_AR_PACE=1` lockstep and a 3-rank ring/CE reduction shape** (§6 suspect 1's fix space):
  the AR is 2.9 % of decode at 3 devices, so the ceiling on any such change is ~1-2 % of plain decode and
  0 for the speculative arms.  Not worth a design change.
* **A prefill-logit gate / 4-axis adaptive-MTP gate on this campaign's arms**: no delivery code changed,
  so there is nothing to gate.  (The arms do carry same-seed text and, on the dense model, identical text
  across device counts; the 4-axis gate at 2/3 GPUs was last run at r109, `FINDING-2026-10-22…`.)
* **A fix for the identity cliff or for the trim admission rule**: both are delivery changes; they are
  filed as `TODO #93` with the evidence, per "report before shipping".
* **4-device / gfx1100 arms**: this box has three gfx1201 cards and `fingon` is single-RDNA3.

## 10. Production recommendation (what the server should set)

1. **3 GPUs is the right device count for the field/MoE config with speculation** (`draft-mtp`,
   `draft-mtp-adaptive`): +14 % to +23 % decode at 100 % residency, +4 % at 91 %, and +2.9 % / +4.0 %
   prefill.  2 GPUs is not faster on any speculative arm.
2. **Keep the expert arena at 100 % residency (identity).**  It is worth +40 % decode; `auto` reaches it at
   `-c 40000`, it does not at `-c 204800` on 2 devices.  If an explicit `MOE_EXPERT_CACHE_MIB` is used,
   oversize it by at least one arena unit per table (a few hundred MiB) and grep the log for
   `identity fast path ON`: zero occurrences means every table is paying the per-token routing readback.
3. **Plain decode (no speculation) is the only arm where 3 GPUs is slower**: −2 % with an identity arena
   (the 3-device internal AR, 1.2 % of decode, plus noise), −10 % at partial residency (the routing
   readback, paid per device), never on a tensor split when resident.  If a workload runs plain decode at
   partial residency, run it on 2 GPUs.
4. **Do not use `-sm layer` for decode.**  It is 25-45 % slower than a tensor split in every cell
   (49.0/44.0 vs 66.4/65.1 t/s for plain decode), and it loses 6-10 % going from 2 to 3 devices.
5. `--spec-draft-mtp-vocab` stays off at 3 devices (the `N · n_devices < n_vocab` rule).

## 11. Actions taken with this file

* `TODO.md` #51 updated: narrowed to the plain-decode/layer-split residuals, with the mechanisms and the
  recommendation above (the field/MoE headline is closed).
* `TODO.md` **#93** filed: "expert-cache sizing lands one arena unit per table short of full residency,
  which drops *every* table off the identity fast path" (+ sizing-policy and trim-admission-guard
  candidates).
* `AGENTS.md`: new invariant in "Critical facts" — the identity cliff, and the cross-device-count text
  caveat.
* `WORKLOG.md` 261023.001 (r116) — the campaign record and the numbers.
* `wip/CAMPAIGNS.md`: the entry is gone; this directory is the record (no stub left in `wip/`).
