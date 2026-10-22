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
release **`v14-097cf33d9-r113`**.  `release.json` is the source of truth for the tip/tree and hashes; the
top of `WORKLOG.md` is the newest record.

| # | subject | kind | status |
|---|---|---|---|
| **87** | `--host-experts pool` follow-ups: what is left after r113 (DIO auto-selection, a pool-aware prefill staging kernel, long-session adaptation) | performance + tuning | open (`wip/host-pool-swap-and-prefill/` §7).  The disk-I/O half shipped in **r113** (record `archive/work/host-pool-layout/`): the pool is laid out as the arena's complement ranked by the persisted heat prior, the demand path never evicts, the prefill staging reads it, `MOE_HOST_POOL_RAM_PCT=80` bounds the pinned total, and the background worker whose SIGSEGV was (a) is gone.  On the field config that took the pool hit rate 0.80 -> 0.92 and 70K-prompt prefill from 39 -> 287 t/s (`DIO=1`) / 581 -> 733 t/s (`DIO=0`) |
| **86** | MoE expert-cache *admission quality*: the shipped policy is ~11 pts of hit rate below a static oracle — decompose foreknowledge vs churn, then prefetch/admission candidates offline | performance | open, scoping (`wip/moe-admission-quality/`) |
| **80** | Meta split-state cache: 30+ lookups per node per re-allocation | performance | parked (see the section at the end; the lever is the lookup COUNT, not the comparison) |
| **79** | RDNA3_0 verify-band fusions: no multi-device measurement, and gfx1101/gfx1102 untested | platform validation | open (shipped default-on in r95; needs a 2+ card gfx1100 box).  `ssh fingon` is single-RDNA3 (`ROCm0`; its second rocm-smi entry is an iGPU and is not a HIP device), so it does **not** cover the multi-device arm — it validates single-GPU gfx1100 only.  gfx1101/gfx1102 still have no hardware |
| **70** | harden the expert-cache re-arm / wide-aux growth against a starved arena | robustness | open |
| **59** | `ngram-mod` + `draft-mtp-adaptive` halves decode with host-resident experts | performance | open (`wip/ngram-mod-host-experts/`) |
| **51** | 3-GPU decode is slower than 2-GPU | performance | open (`wip/three-gpu-performance/`).  2026-10-22 datapoint (dense 27B Q8_0, `wip/three-gpu-performance/FINDING-2026-10-22-dense-27b-and-mirrored-draft-subset.md`): it is **width**-dependent, not direction-dependent — 3 GPU wins +9.2 % at mean verify len 2.9 and loses **−14.4 %** at mean len 29.2 with *bit-identical* acceptance, and at depth 16384 it loses 8.5 %.  The MoE/field-config arm is still the campaign's target |
| **36** | genuine CPU/GPU overlap for the MoE misses | performance | open, scoping (`wip/moe-cpu-overlap/`) |

Accepted limitations that are deliberately **not** fixed are listed at the end — do not re-report them.

---

## 87. `--host-experts pool` follow-ups — what is left after r113 — PERFORMANCE + TUNING


Opened 2026-10-22 (r111) from the fingon (gfx1100, 24 GB VRAM, **30 GiB RAM**) session that validated the
master drop on a model **larger than RAM**.  All configurations produced byte-identical text, so this was
never a purity item.  **r113 closed its disk-I/O half** (record `archive/work/host-pool-layout/`, `WORKLOG.md`
261022.008); the three items below are what remains, in priority order, with the full detail in
[`wip/host-pool-swap-and-prefill/`](wip/host-pool-swap-and-prefill/README.md) §7.

**Closed by r113.**  (a) the `pool_bg_worker` SIGSEGV is moot — the worker was removed (the demand path no
longer moves an arena victim into the pool).  (b) "prefill must not evict pool entries" now holds by
construction: nothing on the demand path evicts, the prefill staging reads the pool, and a miss takes only an
EMPTY slot.  The measured answer to (c) replaced the design question: the pool's **content** dominates the
disk traffic (15x between expert-id order and the heat-ranked arena complement, on a real trace through the
new two-tier model in `scripts/moe-trace-replay.py`), while every eviction/hand-over policy measured within
0-246 MiB of the others — so the pool is laid out in bulk and left alone, and the literal "swap with the
entry being pulled from the pool" measured **2x slower decode** for no benefit.

**(1) `MOE_HOST_POOL_DIO` is still a hand-set default — the "inverted optimum" is unfixed.**  On the field
config (`-ncmoe 48`, 70K prompt, 128-token decode, 2x R9700) `DIO=0` measures 733 t/s prefill / 53.7 t/s
decode against `pinned` 836 / 57.5, while the shipped `DIO=1` measures 287 / 23.1; on the 30 GiB box the
opposite holds (page cache thrashes).  Fix: choose at the first pool build from `MemAvailable` vs the host
expert set — the same measurement `MOE_HOST_POOL_RAM_PCT` already takes — with a kill switch.

**(2) The pool-backed prefill issues one `cudaMemcpyAsync` per expert.**  The pinned-master path stages a
whole table with ONE `moe_cache_stage_kernel` launch.  This is the rest of the gap to `pinned` (733 vs 836
t/s at `DIO=0`) and needs a per-expert slot map upload (the `prs` pattern) plus a "no slot" sentinel falling
back to the bounce.

**(3) Long-session adaptation.**  Between the two bulk layouts the pool is static, so a single session that
changes topic keeps its pre-prompt layout.  Do not add an eviction policy for this before re-running the
two-tier model on a *long, topic-shifting* trace: on the short trace every policy was within noise.

Related: #86 (admission quality) is the same tiering question on the *arena policy* axis.

## 86. MoE expert-cache admission quality — the ~11 pt gap to a static oracle — PERFORMANCE (scoping)


Opened 2026-10-22 (r110), from the P1 instrument that closed `#61`.  On a real 35B-A3B trace (2 GPU
`-sm tensor -ncmoe 40`, `MOE_EXPERT_CACHE_MIB=4096` = 43.8 % residency, 296-token prompt, 128 decode
tokens, `MOE_EXPERT_HEATMAP=0`) the delivered policy models `h` **0.8505** (0.8248 logged) while an oracle
*static* placement of the **same slot count** reaches **0.8765**, and the global-top-f bound is 96.14 % of
the routed mass against the shipped set's 91.42 %.  A per-layer slot **quota** does not explain it (+0.0025,
closed unshipped — see `archive/work/moe-cache-coverage/RESOLUTION.md`), so the gap is either *foreknowledge*
(the oracle ranks by the whole trace; the delivery carries only a persisted prior) or *eviction churn*
(2112 evictions / ~37 k cold reads over 130 passes).

The replayer does **not** model the persisted-prior seed, so it cannot currently tell those apart.  Step 0
is to add it (`--seed heat|trace|none`, reading a real `.moe-heatmap` sidecar or the trace's first N
passes) and publish the `{seed} x {churn}` decomposition.  If a heat-seeded arena lands near `static`, close
the campaign with that measurement; if it stays near `shipped`, score the candidate admission/prefetch
policies **offline** (lookback prefetch of the previous pass's routed set, heat-weighted victim choice,
`admit=value` at the verify widths, decay-period sweep) before writing any C++.  Note the one result we
already have: pass-start pre-admission (`window`) *lowers* `h` to 0.7846 while cutting traffic 25 %, so a
prefetch is not automatically a win.  Design, invariants and gates:
[`wip/moe-admission-quality/`](wip/moe-admission-quality/README.md).

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

## 51. 3-GPU decode is slower than 2-GPU (MTP and plain, partial and fully resident) — PERFORMANCE


Reopened 2026-10-16; measured.  Formerly closed at r55 (a fresh A/B then found 3-GPU MTP 74.3 vs 2-GPU
65.0 t/s).  A 2026-10-16 measurement on r73 found the opposite: 2-GPU **84.4** vs 3-GPU **56.9** t/s
(13.8K prompt, GSQ-IQ3_XXS, `-ncmoe 48 -sm tensor`, adaptive MTP), and it is present at **100 % arena
residency** (so not an arena-sizing artifact).  3-GPU prefill is the fastest (1258 t/s).  Broader than MTP
(the fresh arm is MTP; the matrix must cover plain and fully-resident too).

Working home [`wip/three-gpu-performance/`](wip/three-gpu-performance/README.md) (revived 2026-10-16 from the
r55 `mtp-3gpu-decode` campaign; that campaign's r55 closure record moved with it and is kept beside the
README as `RESOLUTION-2026-10-13-closed.md`).  Step 0: re-run the r55 harness on the current tree, then the full
`{2,3}` x `{plain,MTP}` x `{partial,fully-resident}` x `{tensor,layer}` grid, then the all-reduce /
identity-path / tensor-split-gather profile.

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
