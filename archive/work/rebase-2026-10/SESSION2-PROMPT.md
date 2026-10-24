# Continuation prompt for session 2 of the re-base (paste into a fresh session)

You are continuing a **multi-session re-base** of the 14-block delivery set onto the maintainer's fork
`master` (current upstream, base `3f03b1772`).  Session 1 got the chain to **rung 6** with everything
built and gated; rung 7's rebase was started, surveyed and **aborted** (nothing lost).  Your job is to
continue the ladder from rung 7.

## Read first, in this order

1. `AGENTS.md` — every rule applies (report findings before shipping; push only to the personal fork;
   default-on policy; WIP/promotion rules; critical facts).
2. `archive/work/rebase-2026-10/STATE.md` — the current checkpoint, hashes, what was decided, what is pending.
3. `archive/work/rebase-2026-10/RUNG7-SURVEY.md` — the live conflict map for the step you are about to do, plus
   the resolution plan.
4. `archive/work/rebase-2026-10/DECISIONS.md` — the append-only conflict table for sessions so far (use the same
   format; append, never edit old sections).
5. `archive/work/rebase-2026-10/GOTCHAS.md` and `SURVEY-2026-10-24.md` — the collision inventory (what each
   remaining upstream commit does to which block).

## Hard rules (unchanged)

* **Do not ship, tag, push or touch `release.json`, `patches/` or the delivery branch.**  The only repo
  files you may change are `archive/work/rebase-2026-10/*`.  The final re-cut is a separate, maintainer-approved
  session.
* **Never push to upstream.**  The only permitted push target is `git@github.com:stew675/llama.cpp.git`
  and only at release time.
* Do not change the 14-block numbering or merge blocks.  Amend a block, or propose a new one.
* Report findings and open questions **before** any delivery-affecting commit; surface them in the
  session output and wait for a decision.
* Performance claims need measurements on this box.  One bench at a time; `llama-cli` always with
  `--single-turn --no-display-prompt`; wrap blocking commands in `timeout`.
* Op tests: `HIP_VISIBLE_DEVICES=0,1,2` (the gfx1036 iGPU aborts with `invalid kernel file`).
* `ccache --max-size=30G` (already set); cross-build text comparisons with `ROCBLAS_USE_HIPBLASLT=0`.

## Where the work is

```
worktree/branch : ~/llama-rebase, branch rebase-work   (base 13786fbb3, chain tip bb4825231)
checkpoints     : rebase-ckpt-1 .. rebase-ckpt-6 (never work on a bare detached HEAD)
build           : cd ~/llama-rebase && ~/bin/build-llama-rocm        (build-rocm, ROCm 10.1.0)
delivery repo   : ~/llama-cpp-rdna-boosts (docs only this campaign)
```

## Resume command

```sh
cd ~/llama-rebase
git checkout rebase-work && git log --oneline -3      # expect bb4825231 at the tip
git rebase --onto d454a204e 13786fbb3 rebase-work
```

## Session goal

1. **Rung 7** (`d454a204e`: fused QSA mask op #30640, qwen4exp layer inputs #30927, MoE down-projection
   #30861).  Follow `RUNG7-SURVEY.md`: the conflicts are mostly mechanical op-table unions
   (`GGML_OP_TOP_K_MASK` vs the delivery's `GGML_OP_FLASH_ATTN_QSA`/`HC_MIX`/...), the op-count
   `static_assert` arithmetic (recompute, and fix the pre-existing 107-vs-108 message typo), and one
   semantic hunk in `src/models/qwen4exp.cpp` (keep the delivery's derived-visibility QSA design;
   upstream's fused mask op simply stays unused there).  Then build and run the rung gates:
   coherence (plain == draft-mtp), `scripts/gate-prefill-logits.sh`, `test-backend-ops -o
   FLASH_ATTN_EXT`, `-o FLASH_ATTN_QSA` (18/18 minimum), `-o GATED_DELTA_NET`, `-o HC_MIX` if present.
2. If time remains, continue the ladder: **rung 8** `280141e6d` (chunked CUDA graphs #30804, bf16 WMMA
   #30885, DP4A MMQ #30837, QSA no-mask #30965 — re-check the graph key/`graph_mem_freed` rules) and
   **rung 9** `3f03b1772` (the remainder; then the full validation and ship, which is the *final*
   session and needs the maintainer's go-ahead).

## Per-session deliverables (mandatory)

1. Build + gates at the new checkpoint, listed with numbers.
2. `git branch rebase-ckpt-<n>` at the tip (durable), and its tip/tree in `STATE.md`.
3. `STATE.md` rewritten for the new checkpoint (current base, tip/tree, gates, next rung, exact resume
   command, open questions).
4. `DECISIONS.md` — append the session's conflict table (block, upstream commit, class
   SUBSUMES/CORDON/INTEGRATE, evidence, follow-up).
5. Commit the `wip/` changes in the delivery repo.  Never leave `rebase-work` mid-conflict without a
   `STATE.md` note; `git rebase --abort` back to the last checkpoint is the safe fallback.

## Two known outstanding items (do not silently drop)

* **Per-block tree attribution** (STATE.md → "per-block tree attribution"): two fixup commits
  (`28019efb2` rung-3 fixes, `bb4825231` rung-5 fix) carry hunks that belong in blocks 07/08/10.
  Fold them into the owning blocks with the tree-rewrite pass (`git commit-tree`), then drop the fixup
  commits — do this before `scripts/gate-per-block-build.sh` and before `make-patches.sh`.
* **GDN on RDNA3.5 is undecided**: upstream's `gated-delta-net-mma.cu` fires only on gfx1151 at
  >= 2048 tokens; run the halo A/B (`ssh halo`, `GGML_CUDA_GDN_CHUNKED=0` isolates upstream's MMA) and
  keep exactly one chunked kernel per device class before rung 9.  `ssh fingon` is the RDNA3 box.

## Useful mechanics learned in session 1

* During a rung rebase, `<<<<<<< HEAD` is upstream **plus every already-rebased block** — read it that way.
* When a block *moves* code (block 10 moved `ggml_cuda_try_fuse` into `ggml-cuda-try-fuse.cu`), a normal
  conflict resolution cannot work: re-merge the moved region with `git merge-file` using
  `base = the old chain's block-09 file`, `ours = <the previous rung's checkpoint>:<the new file>`,
  `theirs = HEAD's region`.  Using the block commit's own copy as `ours` re-introduces old-base content
  (that is what broke rung 5 once).
* `/tmp/resolve_conflicts.py <file> (head|theirs|both) [hunk...]` resolves markers by side;
  `/tmp/rewrite_fixes.py` shows the tree-rewrite pattern for per-block attribution.  Re-create them if
  `/tmp` was cleared.
* The new upstream CLI swallows `GGML_LOG_INFO` from the backend: "no log line" is not evidence that a
  kernel path did not fire.  Probe with an env-gated `fprintf(stderr, ...)` + rebuild, or measure the
  effect.
