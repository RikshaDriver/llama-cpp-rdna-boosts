Session: 1 (continued)    Date: 2026-10-24
Delivery: old base 097cf33d9, chain r126 (tip 760ac6aaf, tree 6abcc1a92)
Checkpoint: base 5aeac37a3    chain tip e1bd6da22    tree 78f5c1cc33822994e9d0efa21f0332cd166db85c
Blocks applied: 00..13 (all 14; no renumbering, no block merged)
Build: ok — `~/llama-rebase/build-rocm` (ROCm 10.1.0, `~/bin/build-llama-rocm`), 0 errors
Coherence:
  * dense 27B (Qwen3.8-27B-UD-Q4_K_XL, `-sm tensor` 3 GPU, prose prompt, -n 200):
    plain == draft-mtp, 850 chars sha=0cd2d424a564 (decode 47.0 / 84.0 t/s)
    NOTE: the text differs from rung 2's 897 chars sha=28aba455d3c6.  Cause is upstream's
    float4 rms_norm (#29720, rung 3), which changes the rms reduction order: the prefill-logit
    gate measures mean KLD 0.000657 / same-top-p 98.755 % against the r126 base (PASS, limits
    0.005 / 98 %), i.e. a rounding-level move, and greedy diverges at a near-tie later in the
    stream.  plain == mtp holds on both sides of the change.
  * rung 4 (base 5aeac37a3) re-checked: plain == draft-mtp, 850 chars sha=0cd2d424a564 (same as
    rung 3); prefill-logit identical (0.000657 / 98.755 %); FLASH_ATTN_EXT 6402/6402;
    GATED_DELTA_NET 55/55
  * GDN/MoE 35B-A3B (Q4_K_M, `-sm tensor -ncmoe 40`, 1k-token prompt, -n 64): rung 2 was
    plain==mtp sha=de5cbbeaf065; re-check at rung 3 not re-run (same kernels as rung 2 for the
    decode band; the Q2_K/Q6_K vec_dot changes are bit-identical by construction)
Prefill-logit: PASS — mean KLD 0.000657, same-top-p 98.755 % (r126 base, NOT re-recorded; the
  base should be re-recorded once, in the final session, with the note that the re-base moved it)
Op tests: GATED_DELTA_NET 55/55; FLASH_ATTN_EXT run at this rung (log: /tmp/fa-r3.log)
Band gate (llama-batched-bench -npl 1,4,8, 27B UD-Q4_K_XL 3 GPU tensor, 512/128):
  rung 3 pp 1589/1781/1823, tg 47.7/144.1/230.7  vs  r126 pp 1609/1791/1814, tg 47.7/143.0/230.4
  -> equal within noise (one first-run pp anomaly of 1151 t/s did not reproduce)
Next rung: 33d6135cb (MoE cache SLRU #30637, FA WAR #30573, Stream-K cap #30553)
Next action: `cd ~/llama-rebase && git rebase --onto 33d6135cb 5aeac37a3 rebase-work`

Where the work lives (NOT in the delivery repo — see the campaign rules):
  * worktree/branch: `~/llama-rebase`, branch `rebase-work`, 14 block commits
  * durable checkpoints: `rebase-ckpt-1` @ 632a8a568 (base 62df1815b),
    `rebase-ckpt-2` @ 38eb0ec47 (base 6afa9a313), `rebase-ckpt-3` @ f11c97861 (base 9d1878a43),
    `rebase-ckpt-4` @ e1bd6da22 (base 5aeac37a3)
  * build dir: `~/llama-rebase/build-rocm`

Per-block tree attribution is INCOMPLETE (do it before the per-block build gate):
  the rung-2 and rung-3 resolutions were applied to the working tree at the tip, so blocks
  07/08/10 carry earlier trees.  Concretely, these fixes are only in the tip tree so far:
  * block 07: `VDR_Q6_K_Q8_1_MMVQ_MOE`, the `vec_dot_q6_K_q8_1_impl_mmvq_vdr2` re-add,
    `mmq-load-tiles.cuh`'s closing `#endif`
  * block 08: `calc_rows_per_block_override`/`calc_rows_per_block_weight` taking the new
    `ggml_type` parameter (mmvq-rdna-params.cuh + mmvq.cu call sites), the norm.cu float4 +
    dst16/store_f32 merge, the topk-moe.cu config union
  * block 10: `ggml_cuda_topk_moe_fusion_disabled()` in ggml-cuda-try-fuse.cu
  Use the `/tmp/rewrite_fixes.py` pattern (tree rewrite via `git commit-tree`) to fold each
  fix into the block that owns it before running `scripts/gate-per-block-build.sh`.

Open questions / findings for the maintainer:
  1. **4 GiB pinned-host cap (rung 2, upstream b6334e37e) — INTEGRATED and verified.**  The cap now
     applies to our per-device host buffer type too, so a `-ncmoe 40` host-expert master is
     allocated as 5 pinned parts (4032+4032+4000+4066+2532 MiB on 35B-A3B Q4_K_M) in one
     multi_buffer instead of one 18.6 GiB `hipHostMalloc`.  Load, decode speed and greedy text are
     unchanged (text sha de5cbbeaf065 == rung 1).  Probe evidence: `HBA_PROBE` on
     `ggml_backend_cuda_host_buffer_type_alloc_buffer`.
  2. **Upstream's float4 rms_norm changes greedy text at a near-tie** (see Coherence above).  The
     prefill-logit gate still passes, so this is inside tolerance; it is an upstream INTEGRATE and
     not a delivery regression, but it means the r126 greedy hashes are no longer the reference.
  3. **k-pool input sharing (rung 2, 6afa9a313): the qwen4exp half does not apply.**  Our QSA
     design (block 11) removed the `new_pool_*` kpool inputs entirely (0 occurrences), so upstream's
     optimization of them has no target; the glm5-next half is adopted (auto-merged).
  4. Block 08's `weight_rule`/`rows_per_block`/`small_m` layer (RDNA4/RDNA3_0 dense mmvq) was kept
     and threaded through upstream's new `calc_rows_per_block(ggml_type, ...)`; upstream's
     `mmvq_two_rows_ncols1` low-bit two-row rule only applies to the GENERIC table id, i.e. it is
     inert on RDNA.  Band gate above shows no decode drift.
  5. GDN on RDNA3.5 (rung 1 open question) is still open — needs the halo A/B.
