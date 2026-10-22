# RESOLUTION — `host-expert-master-drop` (TODO #55): shipped in `v14-097cf33d9-r111`

**Promoted 2026-10-22.**  `--host-experts pool` no longer keeps the full pinned `ROCm_Host` master: the
host expert tensors live in the read-only model mapping and every GPU read of them is staged through the
bounded pinned pool.  Folded into **block 12** (blocks 00-11 are byte-identical to r110; block 13 is re-cut
with a new `From` line and an unchanged body).  Release `v14-097cf33d9-r111`, canonical tip
`0349d6b4af5ff2bbcf88d1c4631599a625df0d53`, tree `e51b1023a42b1183c3fca8b93be677ffa4cb194b`.

* Change: `master-drop.patch` (this directory) — the same diff that shipped, also present in
  `patches/0012-…-campaign-remainder-…-patch` (see `patches/README.md` block 12).
* Full design, reader inventory, measurements, gates, trade-offs: `README.md` (this directory).
* Dated record + release note: `WORKLOG.md` 261022.006.
* Why block 12 and not 09: the fold needs block 11's fused / `-sm tensor` slice geometry
  (`table_t::n_slices`); a block-09 fold would have needed a second, pre-fused pool path that block 11
  then had to re-amend.  The repo's documented exception for additive last-block changes covers it
  (`patches/README.md` → "A block owns its code").

**What shipped, in one line each**

1. `ROCm_HostMap`, a per-device host buffer type that carries its device (so the `MUL_MAT_ID` stays on the
   owning GPU) but is backed by the model mapping; `llama_model_loader::host_expert_buft()` selects it in
   POOL mode, and `llama_model::load_tensors` wraps the mapping instead of allocating a pinned block.
2. Every host read made pool-aware: `expert_host_src` / `expert_src_done` (pinned master, pool slot, or a
   grown-on-demand per-table pinned bounce ring), `fill_slot_from_host`, the identity fill, the
   prefill/heat seeds, `moe_cache_stage_table` (which now also serves the pre-sizing prefill) and
   `moe_cache_upload_pageable` (the scheduler's used-expert copies).
3. No pageable H2D for a pooled master anywhere (`set_tensor_async`, `stage_upload`, `stage_gather`), with
   `moe_cache_source_pageable` as the guard and a one-time warning as the tripwire.
4. Three pre-existing pool bugs: the `pool_dio_read` shared static bounce raced by the background worker
   (`thread_local`), `g_host_pools` destroyed under the detached worker at exit (leaked, like the worker's
   mutex/cv/queue), and an AUTO budget that was a 3x running sum plus registration-order dependent.
5. Docs: `ENVIRONMENT.md` §1.2.1 rewritten (the "does not reduce host RSS yet" paragraph was wrong; new
   `MOE_HOST_POOL_VERIFY` and `LLAMA_HOST_MASTER_PINNED`), `AGENTS.md` new critical fact + the pinning fact
   qualified, `patches/README.md` block 12.

**Validation at promotion time:** `validate-set.sh` green (fresh tarball, strict `git am`, tree
`e51b1023…`); `gate-per-block-build.sh --changed-from 8bb85cc82` → blocks 12 and 13 built clean, 0 failed;
all coherence / prefill-logit KLD / MTP / `MUL_MAT_ID` gates as recorded in `README.md` §3.

**Follow-ups (not blockers).**  `MOE_HOST_POOL_FRAC` (25 %) and the prefill staging cost are the two tuning
axes the maintainer parked for a later session; `wip/CAMPAIGNS.md` had no other open item derived from this
campaign.
