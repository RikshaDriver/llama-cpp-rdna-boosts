# rebase-2026-10 — re-cut the 14-block set onto `3f03b1772`

**Status: CLOSED — shipped as `v14-3f03b1772-r127` (2026-10-24).**  This directory is the campaign's
record (moved out of `wip/` on release, per the WIP rule).  The chain is complete at base `3f03b1772`: 14 blocks, tip `cd2c8f488`,
tree `f46264b1...`, per-block build gate 14/14, all rung gates green (see `STATE.md`).  Nothing has been
tagged, pushed or written to `release.json` / `patches/` by the campaign itself; the ship was a separate,
maintainer-approved session that reused exactly this chain (see `WORKLOG.md` 261024.008).

This is the handover a fresh session should start from.  **Paste
[`KICKOFF-PROMPT.md`](KICKOFF-PROMPT.md) into the new session**; it is the self-contained task brief
(decision framework, the checkpoint ladder, the per-session deliverables).  Then read `AGENTS.md` (policies:
report findings before shipping, push only to the personal fork, WIP/promotion rules), this file,
[`GOTCHAS.md`](GOTCHAS.md) (the collision inventory) and [`SURVEY-2026-10-24.md`](SURVEY-2026-10-24.md)
(what a trial apply did).

## The move

| | old | new |
|---|---|---|
| base | `097cf33d9` (2026-10-12, b11585) | **`3f03b1772`** (2026-10-24) |
| where | upstream `ggml-org/llama.cpp` | the maintainer updated `github.com:stew675/llama.cpp` **`master`** to current upstream; `origin/master` = `3f03b1772` |
| gap | — | **261 commits**, 868 files, +96k/-48k |
| delivery | `release.json` = r126, 14 blocks, tree `6abcc1a92` | to be re-cut: `v14-3f03b1772-rNNN` |
| old base still available | `~/llama.cpp` has `097cf33d9` and the fork still has the r126 `rdna-boosts` (`890cdea8c`, tree `6abcc1a92`) | — |

The old base is 12 days old.  Upstream in that window gained several things that **duplicate or override**
delivery blocks (a chunked GDN prefill kernel, QSA/hc_post fusions, a MoE cache rewritten to SLRU, chunked
CUDA graphs, a 4 GiB pinned-host cap, new MMQ/MMVQ tuning).  Treat this as a **semantic** re-base, not a
mechanical `git am`: several blocks need an adopt-upstream-vs-keep-ours decision before they can be replayed.

## Procedure (what the fresh session should do)

1. **Scratch checkout at the new base.**  Do not reuse `~/llama.cpp`'s r125/r126 branch:
   ```sh
   git -C ~/llama.cpp fetch origin master           # 3f03b1772 (personal fork master)
   git worktree add --detach /home/stew675/llama-rebase 3f03b1772
   cd /home/stew675/llama-rebase && git switch -c rb-work
   ```
2. **Replay the blocks as commits, one at a time**, keeping each block's subject:
   `git am -3 ~/llama-cpp-rdna-boosts/patches/000N-*.patch` (the r126 set is the starting point).  It
   **stops in block 00** (4 files) — that is expected; resolve, `git am --continue`, and repeat per block.
   Where upstream superseded a block (GDN chunked, QSA mask, top-k MoE), decide first (see `GOTCHAS.md`),
   then either replay the block adapted or reduce it to the delivery-specific remainder.  **Never change the
   14-block numbering without the maintainer's decision** (amend the block, or add a new one).
3. **Validate the chain, not just the tip:** `scripts/gate-per-block-build.sh` (clean `all` build per block;
   needs `ccache --max-size=30G`), then the usual gates on the tip (below).
4. **Re-cut the artifacts:** `scripts/make-patches.sh ~/llama.cpp 3f03b1772 <newtip>`, then
   `scripts/make-release.sh --base 3f03b1772 --base-tree "$(git -C ~/llama.cpp rev-parse 3f03b1772^{tree})"
   --tip <newtip> --tree "$(git -C ~/llama.cpp rev-parse <newtip>^{tree})" --release v14-3f03b1772-r<N>`,
   then `scripts/validate-set.sh` (fresh tarball, strict 14/14, tree equality).
5. **Docs in the same commit** (the current-state headers, not the dated records):
   `WORKLOG.md` (new entry, newest first), `BASELINE.md` (+ a re-base record in
   `archive/docs/baseline-history.md`), `MANIFESTS.md` identity section, `patches/README.md` (fork-point /
   block notes), `AGENTS.md` (the "Current state" pointer and any critical fact the re-base changes — e.g.
   block 03's GDN bullet if upstream's kernel supersedes it), `TODO.md` current state, `CONTAINERS.md`,
   `README.md` fork-point phrasing, `upstream/*/README.md` (their base is the old one), `beta/*` (re-cut
   against the new tree), `wip/CAMPAIGNS.md`.
6. **Ship with the full policy:** tag `v14-3f03b1772-r<N>`; push the delivery repo (`main` + tag); refresh
   the fork branch from a clean clone at the new base (`scripts/apply-all.sh`, strict, tree ==
   `release.json.tree`, build + gates there) and `git push --force-with-lease origin rdna-boosts` to
   **`git@github.com:stew675/llama.cpp.git` only**; the `radiance` watcher builds/pushes the images after
   the tag.  Never push to `ggml-org/llama.cpp`.

## Gates to run on the re-based tip

The r126 release set is the baseline (`WORKLOG.md` 261024.007): `FLASH_ATTN_EXT` oracle, coherence
(`plain == draft-mtp` same-seed text), `scripts/gate-prefill-logits.sh`, MTP Protocol A.  Add for a re-base:

* **`scripts/gate-prefill-logits.sh --record` is a decision, not a formality.**  The recorded bases in
  `~/.cache/rdna-boosts/gates/` were recorded from r126 kernels.  A re-base changes prefill kernels
  (upstream's edits to FA/MMQ/GDN are part of the base), so the old base will fail for reasons that are
  *upstream*, not a delivery regression.  Re-record only from a build already validated by coherence + MTP +
  the op tests, and say in the record that the base moved.  The head-256 family-switch files created for
  r126 (`*-familyswitch.kld`, `*-control-ub256.kld`) are r126-row-specific — re-record or drop them.
* `test-backend-ops -o FLASH_ATTN_EXT` (and `-o FLASH_ATTN_QSA`) with **`HIP_VISIBLE_DEVICES=0,1,2`**: the
  iGPU (ROCm3, gfx1036) has no kernels in this build and aborts with `invalid kernel file` — that is not a
  failure of the set.  gfx1201 r126 baseline: 6391/6391.
* Protocol A and the band/verify-width arm for anything touching FA/MoE (`TODO #99`'s caveat).
* The `n_rs_batch` / recurrent-state bound on GDN models (`tests/test-recurrent-state-depth`; baselines are
  per box and model — compare the per-arm FAIL/PASS set, not the count).

## Do not

* Do not `git apply` the concatenated series (drops hunks); `git am` (strict, then `-3`) is the path.
* Do not regenerate patches from `~/llama.cpp`'s working branch without checking its tip/tree — it holds the
  r125/r126 chain at the OLD base and can export upstream commits as block patches (see
  `scripts/make-patches.sh`'s fork-state warning).
* Do not run parallel benches; do not run the container build on this box (off-host watcher).
* Do not treat the reporter's RDNA2 issue (#214) as fixed by the re-base: the r126 row fix is RDNA4-only,
  and `TODO #97`'s RDNA2 half still wants their depth sweep.
