# Cold-start prompt — `wip/async-input-per-origin` (issue #201) + the RDNA3_5 `mmf32` default, one release (r105)

You are starting a fresh session in `~/llama-cpp-rdna-boosts/`.  Read, in this order, before doing anything:

1. `AGENTS.md` (policies — especially **Pushing policy**, **Report findings before shipping**, **default-on
   policy**, **WIP and promotion rules**),
2. `wip/async-input-per-origin/README.md` (this campaign: the finding, the fix, the gates, the release shape),
3. `GREEDY-PURITY.md` §41 and §44 (the two rules the async-input ring and the hipBLASLt lottery already have),
4. `benchmarks/prefill-logit-methodology.md` §5b + "Family switches" (how a prefill base is judged),
5. `archive/work/async-input-graph-ordering/` (the r90 resolution — this ring is NOT the r90 defect),
6. `archive/work/hipblaslt-bypass/evidence/p5-cross-arch-validation.txt` (the RDNA3_5 numbers you will quote),
7. `patches/README.md` block 06 and block 10, `WORKLOG.md` 261021.002/.003 and 261021.011/.012.

## What is already established (do not re-derive)

* **The defect is real and still in r104.**  `ggml_cuda_async_input_origin_locked()` keys the per-origin map on
  `cudaStreamPerThread`, which is the sentinel `((hipStream_t)2)` in every thread
  (`hip_runtime_api.h:1160`), so one origin per device, so r101's per-origin ordering does not exist.
  Confirmed by a code read against the r104 tree (line anchors in the campaign README §1).  Two live
  consequences: a foreign-thread drain (§1.1) and the wrap path's empty "every other origin" loop (§1.2); plus
  the `o.q.empty() → continue` gap that skips `has_flush` (§1, "secondary gap").
* **`mmf32` is validated on gfx1151** (`halo`): 2444/2444 `test-backend-ops`, NMSE 6.11e-14 vs rocBLAS
  6.83e-13, 12/12 one logits class, +1.0 % pp512 / +1.6 % pp2048, family-switch PASS (0.013016 / 98.673 % vs a
  same-kernel control of 0.019125 / 97.975 %).  The maintainer has decided to make it **default-on for RDNA3_5**,
  bundled into this release.  RDNA3 (gfx1100, `fingon`) stays opt-in (−1.3…−2.2 % at pp2048).

## The work, in order

1. **Fix** (§2 of the campaign README): key on `std::this_thread::get_id()` (keep issuing on
   `cudaStreamPerThread`), correct the misleading comment block, consume `flush_ev` even when the queue is
   empty, add `GGML_CUDA_ASYNC_INPUT_DEBUG` (origin count + thread id, first N prints).  Amend **block 10**.
2. **Prove the fix**: build the two-thread reproducer under `wip/async-input-per-origin/repro/` and show
   **2 origins** on one device where r104 shows 1.  Keep its output as evidence.
3. **The RDNA3_5 default**: the one-line predicate in `ggml/src/ggml-cuda/mmf32.cu` (`IS_RDNA4 ||
   IS_RDNA3_5`), amend **block 06**, and update the comment there + the docs listed in §6 of the campaign
   README (`ENVIRONMENT.md` §10.1, `patches/README.md` block 06, `GREEDY-PURITY.md` §47, `TODO.md`'s accepted
   limitation).
4. **Gates** — all of §3 of the campaign README, one GPU job at a time, never parallel benches.  In particular:
   * re-derive the **r101 gemma4 0 → 8/10 credit** (`10 × -n 300`, `3 × -n 3000`, `ROCBLAS_USE_HIPBLASLT` both
     arms, classes not hashes).  If it does not reproduce, do **not** edit the r101 entry — add a correcting
     `WORKLOG` entry and point `archive/work/fix-gemma-4/RESOLUTION.md` at it;
   * `GGML_CUDA_ASYNC_INPUT=0` A/B (same-seed text + prefill logits identical);
   * perf: no regression vs the ring-off arm on 35B-A3B (3 interleaved visits) and a Flash-Next `-ncmoe` run;
   * re-record the gfx1151 prefill base **on the new default** and log the family-switch magnitude (§5b).
5. **Release r105** with both amendments: `make-patches.sh` → `make-release.sh` → `validate-set.sh` →
   `gate-per-block-build.sh --changed-from <r104 tip f38bfa9dd>` → fork `rdna-boosts` refresh
   (`--force-with-lease`, **personal fork only**) → tag → push.  The container worker on `radiance` builds the
   images from the tag; do not build them on this box.
6. **Close out**: `WORKLOG.md` entry (r105, with the gates and the record correction if warranted),
   `TODO.md` (#84 closes; the #201 issue should be closed on GitHub with a pointer to the entry),
   `wip/CAMPAIGNS.md` (row removed, a closure note added), and this campaign directory moved to
   `archive/work/async-input-per-origin/` with a `RESOLUTION.md` — **no stub left behind**.

## Boxes

| box | what | notes |
|---|---|---|
| `soar` (this host) | gfx1201, ROCm 10.1.0 | `~/bin/build-llama-rocm`; the canonical gates run here |
| `ssh halo` | gfx1151, ROCm 10.1.0 | the RDNA3_5 default is claimed here — re-run the mmf32 gates after the flip |
| `ssh fingon` | **single** gfx1100 (the other `rocm-smi` entry is a gfx1036 iGPU) | the async-input fix should be re-checked here; RDNA3 stays mmf32 opt-in |

## Discipline that has bitten before

* `llama-cli` **always** `--single-turn` (+ `--no-display-prompt` when scripted); wrap blocking commands in
  `timeout`.
* One GPU job at a time; a fresh-process comparison needs `ROCBLAS_USE_HIPBLASLT=0` on the *comparison* run
  (§44), and **never** judge determinism on a greedy text hash — use the logits-dump hash (a dump is
  byte-identical or it is not; the gate's self-KLD top-p has a uint16 floor, methodology §5b).
* `grep`-ing a log for `page fault`/`slab` is not a GPU-fault check — use `scripts/check-gpu-faults.sh`.
* Report every finding to the maintainer **before** any delivery-affecting commit and before the tag.
* A new `.cu` is picked up by `file(GLOB)` only at **configure** time — re-run the build script, not just
  `cmake --build`, or you get `undefined symbol` at load.
