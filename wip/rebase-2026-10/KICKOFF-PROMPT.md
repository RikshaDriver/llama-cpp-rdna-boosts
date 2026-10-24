# Kickoff prompt for the re-base (paste this into a fresh session)

You are working in `~/llama-cpp-rdna-boosts` (the delivery repo) and `~/llama.cpp` (the fork checkout).
Your task is the **re-base of the 14-block delivery set** from base `097cf33d9` onto the maintainer's
fork `master`, which is now current upstream (`3f03b1772`, 2026-10-24; 261 commits, 868 files ahead).
This is a multi-session job. Do **not** try to finish it in one session: this file defines the loop, and
each session ends at a validated checkpoint.

The repo at ~/llama.cpp/ is based from https://github.com/stew675/llama.cpp/ which has just been re-synced
against upstream.  We generally build with ~/bin/build-llama-rocm

We have various models for testing in sub-directories under /llm/models/

## Read first, in this order

1. `AGENTS.md` (report findings before shipping; push policy; default-on policy; WIP/promotion rules;
   critical facts).  Every rule there applies.
2. `wip/rebase-2026-10/README.md`, `GOTCHAS.md`, `SURVEY-2026-10-24.md` (the handover, the collision
   inventory and the trial-apply survey).
3. `wip/rebase-2026-10/STATE.md` and `DECISIONS.md` if they exist (the previous session's checkpoint and
   decisions).  **Create them if this is session 1** (templates below).
4. `BASELINE.md` (drift policy) and `archive/work/r50-rebase/README.md` +
   `archive/work/patch-block-reorg/REBASE-RESILIENCE.md` (the previous re-base and the cordon pass, done
   2026-10-12/13; use their decision vocabulary and their "keep ours cordoned / adopt / union" patterns).

## Hard rules for this task

* **Do not ship, tag, push or touch `release.json`, `patches/` or the delivery branch** during the re-base.
  The only repo files you may change are `wip/rebase-2026-10/*` and (at the final session, with the
  maintainer's go-ahead) the current-state docs, artifacts and release metadata.
* **Never push to upstream** (`ggml-org/llama.cpp`).  The only permitted push target is the personal fork
  (`git@github.com:stew675/llama.cpp.git`), and only at release time.
* **Do not change the 14-block numbering or merge blocks** without the maintainer's explicit decision.
  Amend a block, or propose a new one.
* **Report findings and open questions before any delivery-affecting commit** and wait for a decision.
  Surface them in the session output, not buried in a commit message.
* **Performance claims need measurements on this box.**  "Upstream subsumes ours" requires a measurement
  showing upstream is at least as fast (or a correctness/robustness reason accepted by the maintainer).
  "Ours is better" requires the same in reverse.  Never adopt a change that is slower because it is newer.
* One bench at a time; `llama-cli` always with `--single-turn`; `ccache --max-size=30G`; op tests with
  `HIP_VISIBLE_DEVICES=0,1,2` (the gfx1036 iGPU has no kernels and aborts); cross-build text comparisons
  with `ROCBLAS_USE_HIPBLASLT=0`.

## The decision framework (apply to every conflict, write it down)

For each colliding upstream change, classify it as exactly one of:

1. **SUBSUMES** - upstream now implements what our block does at least as well.  Adopt upstream, retire or
   trim our hunk, and record the measurement that proves it is not slower.  Do not keep two
   implementations of the same thing.
2. **CORDON** - our code is still the active/better path and upstream's version cannot be shown faster.
   Port our change on top of upstream's new shape and keep it isolated: behind its existing env
   kill-switch (default-on per the default-on policy), behind a clear predicate, or as a comment naming
   the upstream commit it diverges from.  Do not let both paths run at once; make the loser dormant or
   absent.  Add a TODO with the exact re-evaluation trigger.
3. **INTEGRATE** - upstream's change is orthogonal and improves us (a faster kernel, a better graph, a
   robustness fix).  Fold it into the owning block, default-on with a kill-switch for A/B only, and gate
   it.

Record per conflict: block, upstream commit(s), files, class (SUBSUMES / CORDON / INTEGRATE), the
evidence (bench command and numbers, or the correctness reason), what was left behind, and the follow-up.
Use `DECISIONS.md` (append-only; one section per session).  If a decision needs the maintainer, stop and
ask rather than guessing.

## Session loop (the fast-forward pattern)

Maintain one work branch in `~/llama.cpp`, `rebase-work`, whose base is the current **checkpoint**
(an upstream commit).  Each session advances the checkpoint to the next rung of the ladder, resolving
whatever conflicts that jump produces:

```sh
cd ~/llama.cpp
git fetch origin master        # 3f03b1772 or newer; if it moved past 3f03b1772, ask first

# session 1 only: build the chain once at the OLD base
git worktree add ~/llama-rebase 097cf33d9
cd ~/llama-rebase && git switch -c rebase-work
git am -3 ~/llama-cpp-rdna-boosts/patches/[0-9][0-9][0-9][0-9]-*.patch
~/bin/build-llama-rocm         # must build cleanly at the old base before you move it

# every session: advance the base to the next ladder rung
cd ~/llama-rebase
git rebase --onto <next-checkpoint> <current-checkpoint> rebase-work
# resolve conflicts with the framework above, then:
~/bin/build-llama-rocm         # full build per checkpoint (ccache warm)
```

The **ladder** (chronological, chosen from `SURVEY-2026-10-24.md` §"Selected upstream commits"; stop at a
rung when it produces the session's first major conflict, resolve it, validate, checkpoint, stop):

| # | checkpoint (upstream) | carries | expected collisions |
|---|---|---|---|
| 1 | `62df1815b` | GDN chunked kernel (#29353) | block 03 (decide adopt-vs-keep; do not ship two chunked kernels), recurrent-state bound |
| 2 | `6afa9a313` | pinned 4 GiB cap (#30449), rms_norm float4 (#29720), k-pool sharing (#30613) | 00/08/09/10/11; the pinned cap can break the 18.6 GiB host-expert master |
| 3 | `9d1878a43` | top-k MoE (#29181), faster MMVQ (#30517), q1_0 perm (#29927) | 07/08/12; re-apply the band-uniformity rules |
| 4 | `5aeac37a3` | FA oob (#30572), hc_post fusion (#30602) | 04/05/11; keep the RDNA2 checked tail and §51 |
| 5 | `33d6135cb` | MoE cache SLRU (#30637), FA WAR (#30573), Stream-K cap (#30553) | 09/12, 05, 06; re-validate the callback/slot map (TODO #56 history) |
| 6 | `13786fbb3` | ssm_conv 7 (#30873), mmf tile widths (#30836), graph flush (#26771) | 03/10/11, 06 |
| 7 | `d454a204e` | QSA mask op (#30640), qwen4exp layer inputs (#30927), MoE down-proj (#30861) | 08/11/12; keep the QSA oracle and the hc_post split rule |
| 8 | `280141e6d` | chunked CUDA graphs (#30804), bf16 WMMA (#30885), DP4A MMQ (#30837), QSA no-mask (#30965) | 10/04/05/06/07/11; re-check the graph key/mem-gen rules |
| 9 | `3f03b1772` | everything else to current fork `master` | the remainder; then the final validation and ship |

If a rung turns out to be clean, keep going within the session; stop at the first **major** conflict, or
when the session's time/context budget says so.  Never leave `rebase-work` mid-conflict without writing
`STATE.md` (a `git rebase --abort` to the last checkpoint is the safe fallback).

## End of every session (mandatory, even if incomplete)

1. **Validate the checkpoint**: full build (`~/bin/build-llama-rocm`), same-seed greedy coherence
   (`plain == draft-mtp`), the `FLASH_ATTN_EXT` oracle, and `scripts/gate-prefill-logits.sh` (standard
   point).  Re-record the gate base only if the checkpoint's kernels legitimately moved, and say so.
2. **Checkpoint the work**: create a local branch `rebase-ckpt-<n>` at `rebase-work` in `~/llama.cpp`
   (durable, never a bare detached HEAD), and record its tip/tree in `STATE.md`.
3. **Write `STATE.md`**: current base checkpoint, chain tip/tree, blocks applied, decisions taken, open
   questions/blockers, the next rung, and exactly how to resume (`cd`, branch, command).
4. **Append to `DECISIONS.md`**: the conflict table for the session (block, upstream commit, class,
   evidence, what was left behind).
5. **Stop and report**: what was decided, what needs the maintainer, what the next session should do.
   Do not start a release.

## Final session (only with the maintainer's go-ahead)

When the chain is at `3f03b1772` and the whole set is green:

1. Re-cut the delivery: `scripts/make-patches.sh ~/llama.cpp 3f03b1772 <newtip>`, then `make-release.sh
   --base 3f03b1772 --base-tree ... --tip <newtip> --tree ... --release v14-3f03b1772-r<N>`, then
   `scripts/validate-set.sh` (fresh tarball, strict 14/14, tree equality).
2. Update the current-state docs (`WORKLOG.md`, `BASELINE.md` + `archive/docs/baseline-history.md`,
   `MANIFESTS.md`, `patches/README.md`, `AGENTS.md` current state, `TODO.md`, `CONTAINERS.md`,
   `README.md`, `upstream/`, `beta/`, `wip/CAMPAIGNS.md`) and retire this campaign to `archive/work/`.
3. Run the full release gate set (per-block build gate, coherence matrix, prefill-logit, MTP Protocol A,
   op tests), then the release mechanics in `AGENTS.md` (tag, CI, fork branch refresh, containers off-host).

## STATE.md template

```
Session: <n>    Date: <YYYY-MM-DD>
Delivery: old base 097cf33d9, chain r126 (tip 760ac6aaf, tree 6abcc1a92)
Checkpoint: base <upstream commit>    chain tip <sha>    tree <sha>
Blocks applied: 00..13 (list any amended/trimmed)
Build: <ok/fail, build dir>    Coherence: <hashes>    Prefill-logit: <KLD/top-p, base recorded?>
Next rung: <upstream commit>    Next action: <command / decision needed>
Open questions: <...>
Decisions this session: see DECISIONS.md <section>
```

## DECISIONS.md template

```
## Session <n> (<YYYY-MM-DD>) - checkpoint <commit>

| block | upstream | class | decision | evidence | follow-up |
|---|---|---|---|---|---|
| 03 | 62df1815b GDN chunked | SUBSUMES/CORDON/INTEGRATE | ... | <bench numbers / reason> | <TODO> |
```
