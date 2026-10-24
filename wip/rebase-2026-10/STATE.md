Session: 2 (continued)    Date: 2026-10-24
Delivery: old base 097cf33d9, chain r126 (tip 760ac6aaf, tree 6abcc1a92)
Checkpoint: base **3f03b1772** (the FINAL base — rungs 7, 8 and 9 are all replayed)
  chain tip `cd2c8f488`   tree `f46264b1fe565a016940e6fc7858cf50e3b11a9d`
  **14 blocks, 00..13, no fixup commits, no renumbering, no block merged**
Blocks applied: 00..13 (all 14)

Build: ok — `~/llama-rebase/build-rocm` (ROCm 10.1.0, `~/bin/build-llama-rocm`), 0 errors
Per-block build gate: **14/14 built, 0 failed** (`scripts/gate-per-block-build.sh --repo ~/llama-rebase
  --base 3f03b1772 --tip cd2c8f488`, fresh build dir per commit, ccache 30G).  This is the first time in
  the campaign that the whole ladder compiles block by block; it found and fixed two latent breaks
  (block 08 used a helper that only block 10 defined; block 09 carried a stray `}`) — see DECISIONS rung 9.

Coherence (gfx1201, this box):
  * dense 27B (Qwen3.8-27B-UD-Q4_K_XL, `-sm tensor` 3 GPU, prompts/prose-rdna-boosts.txt, -n 200,
    seed 42, temp 0): **plain == draft-mtp**, 850 chars sha=0cd2d424a564 (decode 48.7 / 84.9 t/s)
    — the same text as rungs 3..8, so nothing in rungs 7-9 moved the dense decode numerics.
  * MoE 35B-A3B (UD-Q4_K_M, `-sm tensor -ncmoe 40`, 4 kB prompt, -n 64): **plain == draft-mtp**
    (identical text), decode 19.7 / 33.0 t/s
  * qwen4exp Flash-Next GSQ-IQ3_XXS, 3 GPU tensor `-ncmoe 48`, `--load-mode none --lazy-mode off`,
    -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf: **plain == draft-mtp**, 243 chars sha=e792bee1d6f0
    (identical to rung 7, i.e. upstream's rung-8/9 QSA work did not disturb our derived QSA path);
    -n 200: **plain 58.0 t/s vs draft-mtp 106.7 / 105.5 t/s** (two runs)
  * DFlash2 (35B-A3B Q4_K_M + Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf, `--spec-type draft-dflash`,
    same prompt): text identical to plain, 35.2 t/s (block 11/12 DFlash path intact)
Prefill-logit: PASS — mean KLD 0.000657, same-top-p 98.755 % (unchanged vs rungs 3-8; base still the
  r126-recorded `prefill-logit-gfx1201-Qwen3.8-27B-Q8_0.kld`.  The final session should re-record it
  once with a note that the re-base moved it, as the campaign README requires.)
Op tests (gfx1201, HIP_VISIBLE_DEVICES=0,1,2): FLASH_ATTN_EXT **6402/6402**, FLASH_ATTN_QSA **26/26**
  (floor 18), GATED_DELTA_NET **55/55**, HC_MIX **30/30**, HC_COMBINE 0/0 (no test cases exist in this
  tree — pre-existing), TOP_K_MASK **28/28** (upstream's new op, the union we resolved), MUL_MAT
  **2445/2445**, MUL_MAT_ID **931/931**
test-llama-archs (gfx1201): qwen4exp 5/5 (GPU NMSE 4.63e-14, Meta 2.94e-12), deepseek4 5/5
  (6.79e-08 / 6.16e-06), glm5-next 5/5 (7.17e-09) — the rung-9 QSA/glm5/deepseek changes are green
Band gate (llama-batched-bench -npl 1,4,8, 27B UD-Q4_K_XL 3 GPU tensor, 512/128, 2nd run):
  pp 1608.6/1800.8/1825.4, tg 48.0/156.3/242.7  (rung 3 record: pp 1589/1781/1823, tg 47.7/144.1/230.7)
  -> same band; the tg B=4/B=8 numbers read slightly higher but that is not claimed (different session,
       no interleaved control)

RDNA3.5 (ssh halo, gfx1151, single iGPU) — the campaign's first non-RDNA4 validation:
  * GDN A/B (qwen35 4B Q8_0, pp2048, **interleaved** 4x): ours (bf16 chunked gfx11) **3232 t/s** vs
    upstream MMA (`GGML_CUDA_GDN_CHUNKED=0`) **3123 t/s** -> **+3.5 %**, every pair favours ours.
    pp4096 (arm-blocked, order-biased): 3153 vs 2963 (+6.4 %) — quote the +3.5 % interleaved figure.
  * GDN prefill-logit A/B (qwen35 9B Q8_0, ctx512 x 40 chunks): ours vs the MMA/sequential arm
    **mean KLD 0.000108 / same-top-p 99.676 %** -> PASS (limits 0.005 / 98 %).  Base recorded on halo at
    `~/.cache/rdna-boosts/gates/prefill-logit-gfx1151-Qwen3.5-9B-Q8_0.kld` (base arm CHUNKED=0).
    => DECISION: keep exactly ONE chunked kernel per device class = ours (block 03).  Upstream's MMA
    stays only where our chunked path stands down (kda, S_v outside {16,32,64,128}, or the A/B switch).
  * QSA sparse-vs-dense oracle (Flash-Next GSQ, ctx4096 x 4, `LLAMA_QSA_SPARSE_FA=0` as the oracle,
    plus `LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_DENSE_DECODE_UNTIL=0`): PPL **2.2320 vs 2.2366**
    (+0.21 %) — the shipped sparse QSA path agrees with the dense masked path on RDNA3.5.
  * Flash-Next coherence on gfx1151: **plain == draft-mtp**, 312 chars sha=201502207fcb; 34.5 -> 48.7 t/s.
  * Op tests (gfx1151): FLASH_ATTN_QSA 26/26, GATED_DELTA_NET 48/48, FLASH_ATTN_EXT 6391/6391.
  * NOTE: those halo numbers are from halo's **r126** build (`~/llama.cpp`, build-rocm).  The rung-9
    chain was transferred (`git bundle 3f03b1772..cd2c8f488`, branch `rebase-r9`) and its gfx1151 build
    is `BUILD_DIR=build-r9` (see the session log); re-run the same four checks there before shipping.

Where the work lives (NOT in the delivery repo — see the campaign rules):
  * worktree/branch: `~/llama-rebase`, branch `rebase-work`, 14 block commits
  * durable checkpoints: `rebase-ckpt-1` @ 632a8a568 (base 62df1815b), `rebase-ckpt-2` @ 38eb0ec47
    (base 6afa9a313), `rebase-ckpt-3` @ f11c97861 (base 9d1878a43), `rebase-ckpt-4` @ e1bd6da22
    (base 5aeac37a3), `rebase-ckpt-5` @ e4a863a61 (base 33d6135cb), `rebase-ckpt-6` @ bb4825231
    (base 13786fbb3), `rebase-ckpt-7` @ 1425a07fc (base d454a204e), `rebase-ckpt-8` @ d8cf28e2c
    (base 280141e6d), **`rebase-ckpt-9` @ cd2c8f488 (base 3f03b1772 — the folded, per-block-green chain)**
  * `rebase-ckpt-7/8` are the pre-fold chains: 18 commits including the four fixup commits.  Their tip
    TREE equals ckpt-9's (`f46264b1`) — use ckpt-9 for anything artifact-related.
  * `fold-backup` @ 7e54e6eb4 = the pre-fold rung-9 tip (same tree, kept only as the fold's reference)
  * build dir: `~/llama-rebase/build-rocm`

Per-block tree attribution: **DONE** (session 2).  The four fixup commits were folded into the blocks
that own them and dropped, so the chain is exactly 14 commits:
  * block 00: `ml_l.get_arr(LLM_KV_DECISION_LABELS)` and `!ml_l.no_alloc` (upstream's new lines landed
    with `ml.` while the delivery had renamed the parameter); the DECIMAL_LABELS read was a build error.
  * block 07: the `mmq-load-tiles.cuh` conflict markers (they were committed INSIDE block 07's tree in
    the rung-3 resolution) and the `vec_dot_q6_K_q8_1_impl_mmvq_vdr2` re-add.
  * block 08: the `norm.cu` `rms_norm_f32` dst16 brace and the `ggml_cuda_op_rms_norm_q8_1` fallback
    arg bug (`src0_d, mul_d` -> `src0, weight`) — the latter was a real correctness bug in the
    replay, not just a brace; ALSO the `ggml_cuda_topk_moe_fusion_disabled()` definition in
    `ggml-cuda.cu` (block 08 used it, only block 10 defined it -> block 08 did not build).
  * block 09: a stray `}` in `llama_model_get_tok_embd_rows` (left by the rung-2 concatenation).
  * block 10: the moved-dispatcher `ggml_cuda_topk_moe_fusion_disabled()` (now in
    `ggml-cuda-try-fuse.cu`) plus the rung-5 dedup of that TU.
  * block 11: `GGML_OP_COUNT == 108`, the `ggml_top_k_mask()` function close, the rung-7 rpc assert.
  * block 12: `GGML_OP_COUNT == 109` (its `ALL_GATHER` op) and the rpc assert.

Open questions / findings for the maintainer:
  1. **The AGENTS §45 "a dense tensor split is text-identical across device counts" claim does not hold
     on this model.**  Qwen3.8-27B UD-Q4_K_XL, `-sm tensor`, prose prompt, -n 200, seed 42:
     3 GPU -> 850 chars sha 0cd2d424a564, 2 GPU -> 876 chars sha 8c8fd395402f.  Measured on **r126**
     (the old base) as well: 3 GPU 897 chars sha 28aba455d3c6 vs 2 GPU 855 chars sha 70418dbd473d.
     So it is pre-existing and NOT a re-base defect, but the doc sentence needs correcting (or the
     measurement was made on a different quant/model).  Within each device count `plain == draft-mtp`
     holds on both 2 and 3 GPUs (rung 8 and rung 7 both checked).
  2. **GDN on RDNA3.5 — CLOSED** (see the halo A/B above): keep ours, +3.5 % interleaved and KLD
     0.000108 vs the MMA arm.  The AGENTS block-03 bullet should record the gfx1151 numbers and that
     upstream's MMA is unreachable behind our chunked path except via its kill switch.
  3. **HC_COMBINE has no test cases** in `tests/test-backend-ops.cpp` (0/0 "passed"), so that op has no
     oracle.  Pre-existing (r126 is the same); an oracle would be a nice TODO item.
  4. **qwen4exp + `--spec-type draft-mtp` needs `-md`**: a Flash-Next GGUF carries no `nextn` tensors, so
     the in-model MTP fails with `failed to create MTP context` (expected, not a regression).  The
     standalone `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` works (acceptance 0.53, +48 % decode).
  5. Flash-Next decode speeds are stable with `--load-mode none --lazy-mode off` (the PLE preload); the
     mapped-load arm produces byte-identical text, so the load mode is numerically transparent.
  6. `invalidate_cuda_graphs()` now counts `graph->instances.size()` (upstream's chunked-graph shape);
     the HIP `GGML_HIP_GRAPH_FORCE_UPDATE` workaround now applies per chunk instance (rung 8).

Next session = the SHIP (separate, maintainer-approved):
  * docs current-state headers: `release.json` (via `scripts/make-release.sh`), `BASELINE.md`,
    `MANIFESTS.md`, `README.md`, `patches/README.md`, `AGENTS.md` (fork point `3f03b1772`, the §45
    correction, the block-03 GDN fact), `TODO.md`, `CONTAINERS.md`, `upstream/*/README.md`,
    `beta/*` re-cut, `wip/CAMPAIGNS.md`, plus a new `WORKLOG.md` entry (newest first) and a
    `archive/docs/baseline-history.md` record.
  * re-record the prefill-logit base once (gfx1201) with the "the base moved" note.
  * `scripts/make-patches.sh` from a canonical chain built with `apply-all.sh` at 3f03b1772, then
    `make-release.sh --base 3f03b1772 --tip cd2c8f488 --tree f46264b1... --release v14-3f03b1772-r<N>`,
    then `scripts/validate-set.sh` (fresh tarball, strict 14/14, tree equality).
  * tag -> CI (`validate.yml`/`release.yml`) -> `radiance` container watcher; refresh
    `git@github.com:stew675/llama.cpp.git` `rdna-boosts` only (never upstream).
  * optional, before the tag: re-run the four RDNA3.5 checks against the rung-9 gfx1151 build.

Resume / verify command:
  cd ~/llama-rebase && git checkout rebase-work && git log --oneline -1   # expect cd2c8f488
  git rev-parse HEAD^{tree}                                             # expect f46264b1...
