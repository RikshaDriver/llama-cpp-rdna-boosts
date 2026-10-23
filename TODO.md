# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.

**Forward-looking only.**  This file lists open work, items waiting on someone else, accepted limitations
and parked ideas.  **A closed or resolved item is not kept here** — it is one dated entry in
[`WORKLOG.md`](WORKLOG.md) plus the campaign record that entry points to (`archive/work/<campaign>/`).
Details never live here: they live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`,
`GREEDY-PURITY.md`, `ENVIRONMENT.md`, `wip/*` and `benchmarks/`.

**Numbers are never reused.**  `#NN` is a permanent handle: an old `WORKLOG.md` entry that cites `TODO #74`
still means the same subject after that item is closed and removed from this file.  New items append.

**Current state:** the **14-block V2.1 set** (block 00 + blocks 01-13) at fork point **`097cf33d9`**,
release **`v14-097cf33d9-r116`**.  `release.json` is the source of truth for the tip/tree and hashes; the
top of `WORKLOG.md` is the newest record.

| # | subject | kind | status |
|---|---|---|---|
| **80** | Meta split-state cache: 30+ lookups per node per re-allocation | performance | parked (see the section at the end; the lever is the lookup COUNT, not the comparison) |
| **79** | RDNA3_0 verify-band fusions: no multi-device measurement, and gfx1101/gfx1102 untested | platform validation | open (shipped default-on in r95; needs a 2+ card gfx1100 box).  `ssh fingon` is single-RDNA3 (`ROCm0`; its second rocm-smi entry is an iGPU and is not a HIP device), so it does **not** cover the multi-device arm — it validates single-GPU gfx1100 only.  gfx1101/gfx1102 still have no hardware |
| **70** | harden the expert-cache re-arm / wide-aux growth against a starved arena | robustness | open |
| **59** | `ngram-mod` + `draft-mtp-adaptive` halves decode with host-resident experts | performance | open (`wip/ngram-mod-host-experts/`) |
| **51** | 3-GPU decode is slower than 2-GPU | performance | **narrowed to PLAIN decode (2026-10-23, r116)**.  The field/MoE headline (2 GPU 84.4 vs 3 GPU 56.9 t/s, 2026-10-16) is **closed**: on r116 the same config measures **2 GPU 93.4 vs 3 GPU 131.2 t/s (+40.5 %)**, and in the 24-cell matrix every *speculative* arm is at least as fast at 3 devices with the arena fully resident (+14.0 %/+22.9 % `-sm tensor`, 0.0 %/−0.4 % `-sm layer`, `archive/work/three-gpu-performance/RESOLUTION.md`).  What is left is plain decode: **−2.0 %** resident (the 3-device internal all-reduce, 0.45 vs 0.26 ms/token = 1.2 % of decode), **−10.5 %** at 91 % residency (the non-identity per-op routing readback, paid per device) and **−10.2 %** with `-sm layer` (no intra-token parallelism; reproduced on dense 27B at depth, −6.1 %).  Accepted limitation + production rule: 3 GPUs for speculative decode, 2 GPUs for plain decode at partial residency, never `-sm layer` for decode.  The dominant residual *mechanism* was the identity cliff, **fixed in r117** for a marginal arena (`MOE_EXPERT_CACHE_IDENTITY_PREF`) |
| **36** | genuine CPU/GPU overlap for the MoE misses | performance | open, scoping (`wip/moe-cpu-overlap/`) |

Accepted limitations that are deliberately **not** fixed are listed at the end — do not re-report them.

---

## 80. Meta split-state cache: 30+ lookups per node per re-allocation — PERFORMANCE (parked)


r100 removed the walk's heap churn (block 09, issue #184: `wstack`/`wseen` on a reused per-thread
`monotonic_buffer_resource`).  What is left per graph re-allocation, on the reporter's 4-stream workload, is
~165k cache hits -- **more than 30 per node** -- each a hash find on the tensor pointer plus a 336-byte
`memcmp` of the cached `ggml_tensor` copy.

Parked, not attempted: hashing the tensor bytes to skip the `memcmp` would cost about what the `memcmp`
costs (336 B is ~11 vector words), so the lever is the lookup **count** (the recursion/pruning structure of
`calculate_split_state` and the pre-warm walk), not the comparison.  Any change there needs the same proof
the r100 change used: identical states under `GGML_META_SS_VERIFY=1`, identical greedy text and prefill
KLD, and a same-binary A/B (`GGML_META_SS_SCRATCH=0` is the adjacent switch).  Measurements and the
reporter's numbers: `archive/work/issue-184-meta-r80r82/RESOLUTION.md`.

## 79. RDNA3_0 verify-band fusions: the multi-device arm and gfx1101/gfx1102 are unmeasured — PLATFORM VALIDATION


**Status: open, accepted limitation.  Shipped default-on in r95 on the maintainer's decision**
(`WORKLOG.md` 261020.006, record `archive/work/rdna3-verify-fusions/RESOLUTION.md`).  Not a suspected
defect: the change is arch-identical code behind a host predicate and every single-GPU gate is green
(gfx1100 `W=1..8` maxdiff 0 on six model/prefix cells, `none == draft-mtp`, same-binary kill-switch
identity, prefill-logit KLD 0.000000).

Two gaps, both needing hardware the fleet does not have:

1. **RDNA3_0 multi-device.**  The batched f32 `GGML_OP_CPY` fusion is **not** single-device restricted and
   it is an actively used multi-device path: on 3x gfx1201 with the GDN model + MTP,
   `GGML_CUDA_FUSE_CPY_BATCH_DEBUG=1` shows **450 runs under `-sm layer` and 1350 under `-sm tensor`**.
   The RDNA4 multi-device path is therefore heavily exercised, but the same code on a 2+ card gfx1100
   box was never run.  What to run: a GDN model with MTP (`draft-mtp`, n-max 3) under both `-sm tensor`
   and `-sm layer`, plus `test-logits-width-probe` with `SPLIT=tensor`, comparing default against
   `GGML_CUDA_FUSE_CPY_BATCH=0`.  `fingon` is one gfx1100 plus a gfx1036 iGPU and `halo` is gfx1151, so
   neither is a valid host.  Community members running multiple gfx1100 cards are expected to report.
   If a multi-device failure does appear, `GGML_CUDA_FUSE_CPY_BATCH=0` is the immediate workaround and
   restricting that fusion to `get_device_count() <= 1` on RDNA3_0 is the local fix.
2. **gfx1101 / gfx1102.**  `GGML_CUDA_CC_IS_RDNA3_0` covers `0x1100 <= cc < 0x1150`, so gfx1101 (RX
   7800 XT / 7700) and gfx1102 (RX 7600) take the same fusions, but neither the PR author nor the r95
   release ran them.  There is no such card in the fleet.  The same `W=1..8` probe and kill-switch A/B
   is the check if one becomes available.

**Do not re-report as a bug without one of those runs.**

---

## 70. Harden the expert-cache re-arm / wide-aux growth against a starved arena — ROBUSTNESS


Opened 2026-10-19 while fixing r87.  r87 forces the post-prefill drop, so the r86 trigger (the wide layout
was never released and the arena decayed) is gone.  But the failure shape was: a re-arm that had to shrink
one layer to a partial slot count (`layer=47 ... 344 requested, only 202 fit`), a starved wide-aux region
(thousands of `SLAB_WIDE_AUX` attempts), then a GPU page fault in `k_get_rows_float` on the next graph.
The re-arm is supposed to be layer-uniform and safe; make a partial/refused re-arm leave the cache
**wholesale stood down** (the fuse-guard invariant) instead of running a mixed-slot layer, and make the
wide-aux growth refuse once per pass instead of once per allocation.  Repro: the 10-prompt series against
r86 in `WORKLOG.md` 261019.002; validation arm `MOE_EXPERT_CACHE_VALIDATE=1`.

**r94 note:** the issue #187 fix adds the missing *graph* half of this stale-pointer family — a captured
graph is now invalidated when the slab work boundary shrinks (block 10, `WORKLOG.md` 261020.004).  That
covers the observed #187 fault; it does **not** discharge this item, because a re-arm can still change a
table's arena placement under a graph that is not being dropped.  Keep #70 open.

## 59. `ngram-mod` + `draft-mtp-adaptive` halves decode with host-resident MoE experts — PERFORMANCE


Opened 2026-10-14; scoping.  With partially or fully host-resident MoE experts (`-ncmoe`/`-cmoe`, the
expert cache or op-offload path active), the README's recommended
`--spec-type draft-mtp-adaptive,ngram-mod` combo roughly **halves decode throughput**; dropping
`ngram-mod` and using plain `draft-mtp-adaptive` restores the expected rate.  The combo is the measured
best on a fully device-resident model, so the interaction is with the host-expert serve path.  Documented
in the README "Recommended configuration" section.  Working home
[`wip/ngram-mod-host-experts/`](wip/ngram-mod-host-experts/README.md) (hypotheses: verify width against the
per-op host-expert upload, the MoE cache band and verify graph recapture, the staging-ring width; bisect
plan in that README).

## 51. 3-GPU decode is slower than 2-GPU — NARROWED to PLAIN decode (closed for the field/MoE arm)

**Measured to a verdict 2026-10-23 on r116** (`archive/work/three-gpu-performance/RESOLUTION.md`, 24-cell
matrix + controls, one binary, `ROCBLAS_USE_HIPBLASLT=0`).  Campaign history: revived 2026-10-16 from the
r55 `mtp-3gpu-decode` closure; r115 (`archive/work/graph-key-churn/`) fixed the graph-key churn; #89 closed
(one-time HIP code-object load, 0 % of decode).

**Closed half — the field/MoE arm.**  The 2026-10-16 r73 table (2 GPU 84.4 vs 3 GPU 56.9 t/s,
GSQ-IQ3_XXS, `-ncmoe 48 -sm tensor`, adaptive MTP) does **not** reproduce: r116 measures **2 GPU 93.4 vs
3 GPU 131.2 t/s (+40.5 %)** on the same config, and with a 100 %-resident arena every speculative cell of
the matrix is at least as fast at 3 devices (`draft-mtp` +14.0 %, adaptive +22.9 % tensor; 0.0 % / −0.4 %
`-sm layer`; prefill +2.9…+4.0 %).  Three mechanisms did it: r115's graph key, the arena reaching every
table's identity fast path, and (for the 2-GPU arm) the context no longer starving it of arena.

**Open half — plain (non-speculative) decode.**  3 GPUs loses, for three separate reasons:
1. `-sm tensor`, arena fully resident: **−2.0 %** (66.4 → 65.1 t/s).  The 3-device internal all-reduce is
   0.45 ms/token vs 0.26 (2.9 % vs 1.7 % of decode) — i.e. 1.2 % of the gap; the rest is launch/schedule
   noise.  `nccl` is 9 % *worse* at 3 ranks and not bit-identical; forcing `internal` for every size gains
   3.7 % but changes the prefill AR path (smoke only).  **No AR change is warranted**; suspect 1 is retired.
2. `-sm tensor`, 91 % residency: **−10.5 %** (51.5 → 46.1 t/s).  A non-identity table pays a per-token
   routing readback, and at 3 devices it is paid per device.  (A *marginal* arena is fixed in r117 — the cache
  now prefers whole-table identity — but an arena that is 8 % short cannot reach it without dropping layers.)
3. `-sm layer`, any residency: **−0.4…−10.2 %**.  A single-token decode has no intra-token parallelism to
   gain, so the 3rd device only adds a pipeline stage — reproduced on dense 27B Q8_0 at depth 15746
   (−6.1 %, text identical at both counts).  Property of the split, not a defect; `-sm layer` is not the
   decode recommendation (it is also 25-45 % slower than tensor split overall).

**Purity note for anyone re-measuring this:** cross-device-count greedy text is **not** a valid gate on
the MoE tensor-split path (2 GPU `40d493bbb5ac` vs 3 GPU `ecf8a32bae77`, PPL 10.8452 vs 10.8623 = 0.16 %,
the §45 split rounding); dense tensor split and every `-sm layer` arm are text-identical across counts.
Compare builds *within* a device count.

## 36. Genuine CPU/GPU overlap for the MoE misses (Strata's pipeline shape) — PERFORMANCE (scoping)


Opened 2026-10-05.  The delivered cache-on path computes every expert on the GPU (resident from VRAM,
misses via UVA over PCIe); the cache-off path computes them on the CPU but **serialised** with the GPU
splits.  Our own CPU-computes-misses arm was correct and negative above a ~1.2 GiB arena (the loss is
~600 scheduler dispatches + ~3 cross-backend copies per layer/token, not CPU compute; see
`archive/work/moe-expert-cache/WORKLOG.md`).  Strata's custom engine runs the CPU pool *because* it is not
under llama.cpp's serial scheduler.  **Next:** O0 — measure how much of a 52 t/s token is the UVA cold
fraction at 43 % residency; if it is <20 %, park it.  Design space and gates:
[`wip/moe-cpu-overlap/`](wip/moe-cpu-overlap/README.md).

## Documented, deliberately NOT fixed (accepted limitations — do not re-report)


- **`mmf32` covers the skinny F32 shapes only (r104, `TODO` #82 closed) — accepted.**  A full-F32 model's FFN
  GEMMs (`m` ≫ 256: dense F32/F16-weights-heavy prefill) still fall through to rocBLAS/hipBLASLt, so they keep
  that path's per-start solution lottery; the kernel's tile (32 output rows) is the wrong shape to be
  competitive there and no such model is validated.  Also untouched: BF16/f16 prefill (Flash-Next keeps both
  hipBLASLt's +51 % and its own lottery — a BF16 counterpart is its own job) and **RDNA3** (gfx1100), where the
  kernel stays opt-in (`GGML_CUDA_MMF32=1`) because it measured −1.3…−2.2 % at `pp2048` there; the lever would be
  a per-arch tile, not the gate.  RDNA3_5 was admitted and is default-on since r105 (`halo`: 12/12
  deterministic, +1.0 %/+1.6 % prefill, family-switch PASS).  Scope: `patches/README.md` (block 06), rule:
  `GREEDY-PURITY.md` §47.
- **The `launch-ledger` remainder (item 5(c)) — measured, not pursued (2026-09-12 (8)).**  The small-pp
  remainder (+38 `scale_f32`/eval, an `rms_norm<256,true>` count diff) is sub-0.2 %, root-cause-only.
- **The mmq mma `sum[]` accumulator-overflow latent defect (item 5(d)) — accepted, no upstream report
  (2026-09-12 (8)).**  `process_tile` sizes the per-thread accumulator as `J*I/(nwarps*32)` while the
  AMD-WMMA vec_dot indexes up to `J/2-1`, so any config with `I < nwarps*16` silently corrupts
  (deterministic for J=128, racy for J=48/24).  Root cause + full evidence matrix:
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-mmq-j128-latent-defect.md`.
  Upstream ships **no** violating config (every `mmq-config-*.cuh` row keeps `I >= nwarps*16`) and the
  block-13 rows never violate it either, so there is no upstream reproducer to file.
- **V3 prefill cost is arch-dependent (item 5(g)) — accepted.**  gfx1151 measured −3.2 % at pp20480 (4B,
  q8_0) vs the RDNA4 reference −1.3 %, decode flat; still a large net win (−799 MiB compute + −799 MiB
  host) and on by default.
- **Upstream monitor: ROCm unaligned-width split-load (item 13) — standing, no action (2026-09-12 (8)).**
  Fixed locally in block 13; no PR planned (upstream is busy with its own qwen4exp work).  It resolves
  naturally as a re-base conflict if upstream fixes it; nothing to track.
- **Mixed K/V cache types are HARD-REJECTED.**  Every mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`) measured
  1.7-3.6× slower and never smaller; `params.type_k != params.type_v` fails `llama_init_from_model` with
  a message naming both types (block-14 amendment; upstream's MLA/DeepSeek4-only condition is dropped).
  Both types default to f16, so only an explicit `-ctk`/`-ctv` can trigger it.
- **gemma-4-E4B-it + 3-GPU `-sm tensor`** aborts in the meta splitter because its 2 KV heads are fewer
  than the 3 devices (one device gets a zero-extent share: `[1, 1, 0]` over 2 heads).  The abort is the
  split-state **ratio-consistency** assert in `ggml_backend_meta_get_split_state`
  (`split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis] == sum * tensor->ne[split_state.axis]`
  — `ggml-backend-meta.cpp`; 1582 at r101, 1487 at r99: the line number moves, the assert does not, and
  older records that cite `:1177` are one more tree back).  Works on 1/2 GPUs and on 3 GPUs with
  `-sm layer`; maintainer's call: no fix.  r101's split-state work deliberately does not change this
  (`GREEDY-PURITY.md` §45 accepts a slice that is *empty or the whole* extent, and one head of two is
  neither); re-verified on r99 and r101, identical abort.  Every other model is unaffected (a future
  block or upstream report could make the splitter tolerate a zero-extent share).
- **`src/llama-kv-cache.h:274` `-Wunused-private-field` for `v_enabled`** on a full build (the field *is*
  used, in `llama-kv-cache.cpp:232`; clang's per-TU analysis fires).  A `[[maybe_unused]]` one-liner
  silences it; left alone to keep the V5 amendment scoped to the FA kernels.
- **Gemma4's full-attention (head_dim 512) layers stay outside the rope + FWHT + KV-write fusion**
  (documented 2026-10-21, deliberate).  The fused kernel instantiates `fwht_n` in `{64, 128, 256}`, and
  Gemma4 is the only arch with a distinct SWA/full head dim (`key_length_swa` 256 vs `key_length` 512),
  so its 1/6 full-attention layers run `rms_norm_mul_rope_f32<256,true,float,0,false>` + `fwht_cuda<512>`
  for Q and K plus two `set_rows` — 6 launches instead of 2.  Extending it needs 4 new kernel
  instantiations **and** a fix to the fused kernel's V-row write (its `ncols <= block_size` precondition
  would quantize half a 512-wide V row), which edits the shipping 256 fold, for ~0.1 % on gfx1201.
  Accepted as a gap; the analysis (what to change, why the arithmetic is bit-identical by construction)
  is in [`archive/work/fix-gemma-4/ISWA-WRITE-FUSION.md`](archive/work/fix-gemma-4/ISWA-WRITE-FUSION.md) §5.6.
  (The iSWA builder order that made the 256-wide half of this fusion live is r101, `WORKLOG.md` 261021.003;
  `GREEDY-PURITY.md` §45 records the split-state rule that came with it.)
- **3-GPU plain (non-speculative) decode is a few percent slower than 2-GPU — accepted** (2026-10-23,
  r116; `TODO #51`, `archive/work/three-gpu-performance/RESOLUTION.md`).  −2.0 % with a fully-resident
  arena (the 3-device internal all-reduce is 1.2 % of decode and no backend/size knob beats it), −10.5 % at
  91 % residency (the non-identity per-token routing readback, paid per device; the *marginal*-arena case is
  fixed in r117, an 8 %-short one is not), −10.2 % with
  `-sm layer` (no intra-token parallelism; a property of layer splitting, reproduced on dense 27B).
  Speculative decode does **not** lose: +14…+23 % at 3 devices when resident.  Production rule: 3 GPUs for
  speculative/MTP workloads, 2 GPUs for plain decode at partial residency, never `-sm layer` for decode.
- **Not worth pursuing** (measured, no win): the decode fq-inline-quantize port (wash-to-negative — the
  tree already launches fewer kernels/step and sits at wall parity) and the GDN +72-launch 2-kernel split
  (cosmetic).
- **`--spec-draft-mtp-vocab` two-band subset (closed 2026-10-22, measured, not shipped).**  A second band
  (keep `< N` **and** `>= n_vocab - M`) was meant to catch gemma4's space-piece band (229376..255999): it
  measured 0.610 acceptance at 37 % of the rows and was still 0.4 % *slower* on that payload.  With the
  width rule found in the 3-GPU arm (`N · n_devices < n_vocab`) a second band only makes the **mirrored**
  draft matmul wider, so it is a non-goal, not a parked idea.  Record:
  `archive/work/spec-draft-mtp-vocab/VALIDATION-2026-10-22.md` §5.

## Where the current lists live


- **Open campaigns and scoping notes:** [`wip/CAMPAIGNS.md`](wip/CAMPAIGNS.md) (the index for `wip/`).
- **Closed campaigns, with their evidence:** `archive/work/<campaign>/` (`RESOLUTION.md` where the campaign
  closed with a finding).
- **Remaining gfx1151 work + the 2026-09-12 audit:** `archive/work/strix-halo/HANDOVER-2026-09-12-remaining-gfx1151.md`.
- **QSA sparse-regime width purity (items 4/7 disposition):** `archive/work/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`.
- **Environment, instruments, reference hashes and the landing procedure for KV/FA work:**
  `archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-remaining-work.md` (its §0 status and its
  items 1/5 and F3 are **superseded** — see `WORKLOG.md`).
- **Delivery verification contract + dated records:** `MANIFESTS.md`, `patches/README.md`, `WORKLOG.md`,
  `AGENTS.md` headers.
- **Purity/invariant analysis and the instrument rules:** `GREEDY-PURITY.md`.
- **Benchmarks + gates:** `benchmarks/` (the adaptive-MTP baseline gate: `mtp-adaptive-methodology.md`).
- **Upstream PR candidates:** `upstream/README.md`.
