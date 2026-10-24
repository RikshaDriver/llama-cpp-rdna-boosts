# GOTCHAS — what a fresh session must know before re-cutting onto `3f03b1772`

Companion to [`README.md`](README.md).  Dated survey: [`SURVEY-2026-10-24.md`](SURVEY-2026-10-24.md).

## 1. The replay is semantic, not mechanical

`git am -3` of the r126 set at `3f03b1772` stops in **block 00** (`ggml/src/ggml-cuda/ggml-cuda.cu`,
`src/llama-graph.cpp`, `src/llama-model.cpp`, `src/llama-moe-cache.cpp`).  That is the easy part.  The hard
part is that upstream implemented several of the same things we ship, sometimes differently:

| upstream change (window) | blocks it hits | decide |
|---|---|---|
| `62df1815b` **cuda, hip: GDN chunked kernel (#29353)** | **03** (our chunked GDN prefill), also 10 (`GGML_CUDA_GDN_TAIL`) | **Adopt upstream's kernel and reduce block 03 to the delivery-specific remainder** (the `n_rs_batch` rollback bound, the per-sequence tracker, the `GGML_CUDA_GDN_CHUNKED` kill switch if it still applies), or keep ours only if it measures better.  Do not ship two chunked kernels.  Reconcile `cparams.n_rs_batch` (AGENTS: the rollback bound is enforced by it) and re-run the recurrent-state depth test on a GDN model + the prefill-logit gate (a GDN kernel family change is a §46 family switch). |
| `33d6135cb` **moe-cache : replace MoE LRU cache with a 50/50 SLRU cache (#30637)** | **09** (the scheduler `callback_copy` rewiring, TODO #56), 12 | Re-check the slot-map/callback integration against the new `llama_moe_cache`; the delivery's own `MOE_EXPERT_CACHE_*` engine is independent and should stay.  `--moe-cache-mib` (upstream engine) must be re-validated (it was broken from block 09 to r98). |
| `b6334e37e` **Limit max CUDA pinned host buffer size to 4 GB (#30449)** | **00/09/10** | **Potential hard break.**  The `-ncmoe` host-expert master is a pinned `ROCm_Host` buffer measured at 18.6 GiB (35B-A3B `-ncmoe 40`).  Find where the cap is applied and either exempt the host-expert buft / raise it for it, or re-validate the pinned path end to end.  Do not "fix" it by switching to the pool mode. |
| `6afa9a313` **llama : share the k-pool input views between layers (#30613)** | **04** (BF16 KV), **05** (native KV FA), **09/10** (state save/restore, slab), 11 | KV cache layout change.  Re-apply the BF16/native-KV rules on the new pool; re-run `FLASH_ATTN_EXT`, coherence and prefill-logit.  Mixed K/V is still hard-rejected. |
| `280141e6d` + `9ff0d7e90` **CUDA chunked graphs (#30804)** + `13786fbb3` graph clear after pool flush | **10** (graph key/mem-gen, slab capture rules), 09 | Our graph cache is keyed on `{first node data ptr, n_tokens}` (r115) and `graph_mem_freed()` must be bumped on every slab shrink.  Upstream now chunks graph launches: re-check `ggml_cuda_graph_get_key`, the capture-deferred rules and the slab-shrink bump, and re-measure the 3-GPU graph-churn case (r115: +41 %). |
| `d454a204e` **fused QSA mask op (#30640)**, `5c3114268` **QSA selection without kq_mask (#30965)**, `ff5a23376` **qwen4exp SCALE→SIGMOID→SCALE→hc_post fusion (#30602)** | **11** (qwen4exp/QSA/HC), 12 | Decide adopt-vs-keep per piece.  Keep the QSA CPU oracle (`test-backend-ops -o FLASH_ATTN_QSA`, 18/18 minimum) and the `hc_post` split-mode rule (r97: the fused op only under a Meta device; the chain is not correct across a tensor split). |
| `9a16f1dbe` MTP for Mellum2.1, `d9ee007a0` qwen4exp layer inputs for DFlash, `1ab09d632` DFlash2 probabilistic sampling, `4912ac24f` DFlash FC input scale | **02**, **09/12** (DFlash2 selector/sched), **11** (layer-input tap) | Upstream moved the DFlash/QSA layer-input surface.  Keep our `build_hc_mean()` tap rule (TODO #94) and the `sched_alt_for` DFlash rule (issue #208); re-run the DFlash gates. |
| `f2f667b02` **grouped experts top-k fusion (#29181)** | **12** (`topk-moe.cu`) | Likely a duplicate.  Compare with our `topk-moe.cu` work and keep one; if ours is dropped, re-check the `topk_moe` fusion purity/address-independence rule. |
| `8baabd53b` **MoE down-projection per-expert-token scaling (#30861)** | **08/12** (fused MoE, shared-expert band) | New graph node in the MoE path; check the shared-expert fusion band and the `1 <= nt <= 8` byte-identical rule. |
| `2744e48d1` faster MMVQ low-bit, `1a24df163` DP4A MMQ tile height, `4c1004c49` MMQ Stream-K cap, `b68449857` mmf tile widths, `9d1878a43`/`1ddffe9f7` HIP perm, `9145ddb95` bf16 WMMA operand type, `623c16393` rms_norm float4 | **06/07/08** | Re-apply the delivery's band rules on top: mmvq `nwarps`/VDR uniform across `W=1..8`, tuned at the verify widths; the pinned fusion ops (GDN/SSM, shared-expert, gate) keep plain `calc_nwarps`; RDNA4 dense `nwarps=1`, VECdot VDR=2 (dense) / 4 (MoE).  Re-run the stock-relative `llama-batched-bench -npl 1,4,8` band gate. |
| `5aeac37a3` FA oob-check condition, `99db719bd` FA vec WAR hazard, `9d1e03616` vec dispatch | **04/05** | Keep the RDNA2 checked-tail (`ncols2 > 1` last tile) and the r126 rule: no MMA band row at `ncols == 64` (`GREEDY-PURITY.md` §51).  Re-run the FA oracle and the prefill-logit gate. |
| `39da130fc` ssm_conv kernel size 7 | **03/11** | Check the conv path still matches the GDN/SSM kernels after the adopt decision. |
| `303220975` deny `-sm tensor` for arches with MoE amax rescaling, `73adf89e3` minimax, `412a363c6` qwen3 bidirectional, new model arches | **00** (meta/split dispatch), 11 | Mostly orthogonal, but the meta/offload-op predicates must stay consistent; check the new arch denies do not disable a delivery path. |
| `83c2d012a` save the dynamic ngram cache file, `13b804892` ngram cache by reference | **09** (the ngram band clamp), 02/12 | Re-check the clamp still fires for every enabled ngram type (`common_speculative_clamp_ngram_n_max`) after the speculator types/enums changed. |

## 2. Docs/metadata that must move with the base

`grep -rl 097cf33d9` is the mechanical check.  Current-state files (must change): `release.json`,
`AGENTS.md` (the "Current state" line and any fact the re-base makes stale — block 03's GDN bullet is the
likely one), `BASELINE.md`, `MANIFESTS.md` (identity), `README.md` (fork point), `patches/README.md`,
`TODO.md` (current state), `CONTAINERS.md`, `wip/CAMPAIGNS.md`, `scripts/make-patches.sh` (fallback
defaults only, optional), `upstream/*/README.md` (their base), `beta/README.md` + `beta/*/*.patch` (re-cut).
Dated records (`WORKLOG.md` history, `archive/docs/*`, `archive/work/*`, `benchmarks/*`) keep the hash they
were written against — do **not** rewrite them; add the re-base record in `archive/docs/baseline-history.md`
and a new `WORKLOG.md` entry.

## 3. Release-mechanics gotchas (learned on r126)

* The delivery repo's `main` is the delivery; `~/llama.cpp`'s current branch is **`r125`** (the r126 chain),
  and its `rdna-boosts` branch is stale (r124, `355417a72`).  **Do not rebase the wrong branch** (this bit the
  r126 session once).  Verify with `git worktree list` / `git log -1`.
* After amending a block commit, replay the later blocks with
  `git rebase --onto <new-block> <old-block> <branch>`; the resulting tip SHAs change, so every patch's
  `From <sha>` line changes when you regenerate (bodies must not).
* `make-patches.sh` then `make-release.sh --tip/--tree/--release`; `validate-set.sh` must be green (strict
  apply on a fresh tarball == `release.json.tree`).  Never hand-edit hashes.
* Our r126 process note: `gate-prefill-logits.sh` on the standard point passed at KLD 0.000000 because the
  gate model is head-128 and the changed row is head-256 only.  When a change is shape-specific, run the
  gate with `--model` on a model that actually hits the shape (the r126 head-256 family-switch run is the
  example), or the gate is blind to it.
* `test-backend-ops` scans **all** devices: set `HIP_VISIBLE_DEVICES=0,1,2` or the iGPU gfx1036 aborts with
  `invalid kernel file` and looks like a failure.
* Tag → CI (`validate.yml`, `release.yml`) + the `radiance` container watcher; never build the images on
  this box.  Push the tag only after `release.json` and the patches are in the same commit.
* Fork branch: clean clone of the personal fork at the new base → `scripts/apply-all.sh .` → verify
  `HEAD^{tree} == release.json.tree` → build + gates there → `git push --force-with-lease origin
  rdna-boosts`.  The permitted target is `git@github.com:stew675/llama.cpp.git` only.
* ccache: `ccache --max-size=30G` before `gate-per-block-build.sh`, or block 05 thrashes (45 min vs ~7).

## 4. Session-2 lessons (rungs 7-9, and the ship)

**Per-block build gate — run it, it earns its keep.** With a 30 GiB ccache a block costs ~1.5 min
(early blocks) to ~7 min (FA-heavy blocks 05-12) on this box; the whole 14-block ladder is ~50 min:

```sh
cd ~/llama-cpp-rdna-boosts
./scripts/gate-per-block-build.sh --repo ~/llama-rebase --base 3f03b1772 \
    --tip "$(git -C ~/llama-rebase rev-parse rebase-work)" --keep-going
```

It found three real defects that a tip-only build cannot see: block 08 used
`ggml_cuda_topk_moe_fusion_disabled()` that only block 10 defined, block 09 carried a stray `}`, and every
block carried `ml.` where the delivery renamed the loader parameter to `ml_l`.  **Scanning the tip tree is
not enough** — a block's own tree is what `make-patches.sh` ships.

**`ml` vs `ml_l` (the rename churn).** Upstream commits that land *inside* the re-base window (and so
arrive through the base, not through a block) write `ml.` in functions the delivery renamed to `ml_l`
(`llama_model_base::load_hparams`, `load_tensors`).  Symptom: `member reference type
'llama_model_loader *' is a pointer` at *some* block, not at the tip.  `grep -n '\bml\.' src/llama-model.cpp`
and check each hit is a function that really takes a parameter called `ml`.

**Chunked CUDA graphs (rung 8, #30804).** `ggml_cuda_graph` has no `graph`/`instance` members any more:
`graph->instances` is a vector of per-chunk `cudaGraphExec_t`, and the capture is ended *inside*
`ggml_cuda_graph_launch_chunk()`.  Anything that used to run "right before `cudaStreamEndCapture`" (our
WIP AR-stage hook) now goes **before the last `ggml_cuda_graph_launch_chunk()` call**.  Our HIP
`hipGraphExecUpdate` leak workaround lives inside upstream's new
`ggml_cuda_graph_update_executable(cudaGraphExec_t &, cudaGraph_t)` signature, i.e. it now destroys and
re-instantiates one *chunk* per recapture.

**qwen4exp + MTP needs `-md`.** A Flash-Next GGUF has no `nextn` tensors, so `--spec-type draft-mtp`
without `-md` fails with `failed to create MTP context` (expected, not a regression).  The standalone
`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (halo: in the model dir; this box:
`/llm/models/Qwen3.8/Flash-Next/IQ4_NL/`) works: acceptance 0.53, +48 % decode, text byte-identical to
plain.  Use `--load-mode none --lazy-mode off` for Flash-Next timing runs (PLE preloaded into RAM; the
mapped arm is text-identical, so the load mode is numerically transparent).

**`llama-batched-bench` first-run pp anomaly.** The first run after a model load can report B=1 pp ~20 %
low (1283 vs 1608 t/s here, and a 1151 t/s outlier in session 1).  Always run the band gate twice and quote
the second run.

**halo = gfx1151 (RDNA3.5).** `ssh halo`, `~/llama.cpp` (branch `rdna-boosts`, r126 build in
`build-rocm`), ROCm 10.1.0 at `/opt/rocm-10.1.0-gfx1151`, `~/bin/build-llama-rocm` is the gfx1151 script
(**use `BUILD_DIR=build-r9`-style overrides — `build-rocm` is the live server build**).  Transfer a chain
with `git bundle create /tmp/b 3f03b1772..rebase-work` + `git fetch /tmp/b …` (the base commit is already
in halo's repo).  Recorded gfx1151 gates: GDN ours-vs-MMA KLD 0.000108 / 99.676 %
(`prefill-logit-gfx1151-Qwen3.5-9B-Q8_0.kld`), QSA sparse-vs-dense PPL 2.2320/2.2366, Flash-Next
plain==draft-mtp sha 201502207fcb, `FLASH_ATTN_QSA` 26/26, `GATED_DELTA_NET` 48/48, `FLASH_ATTN_EXT`
6391/6391.  GDN A/B: `GGML_CUDA_GDN_CHUNKED=0` is the upstream-MMA arm on gfx1151 (ours preempts it
otherwise) — interleave the arms, the APU's clocks drift within a run block.

**Cross-device-count greedy text on a DENSE tensor split is NOT identical** (contrary to the AGENTS
`§45` summary): Qwen3.8-27B UD-Q4_K_XL, prose prompt, seed 42 → 3 GPU 850 chars / 2 GPU 876 chars, and
r126 shows the same (897 vs 855).  Within one device count `plain == draft-mtp` holds.  Gate a build
change within a device count.
