# MANIFESTS — apply order and verification contract

**What this file answers:** in what order the blocks apply, what "verified" means for each one, and what to
do when an apply fails.  The current release identity (fork point, tip/tree, block count, per-artifact
sha256) is [`release.json`](release.json); the dated record of every release and re-validation is
[`WORKLOG.md`](WORKLOG.md).  Dated validation records from the pre-V2.0 era (the 16-block set, its
per-block table and its release-by-release narrative) are preserved in
[`archive/docs/validation-history.md`](archive/docs/validation-history.md) and do **not** describe the
current delivery.

## Current delivery

13 blocks (`block 00` + `blocks 01-12`) against `release.json.base`, canonical chain `release.json.tip` /
`.tree`.  `scripts/validate-set.sh` is green (checksums, then a strict 13/13 apply on a fresh tarball,
tree equality).  Each block is **one** `git am` commit and the series is **linear**: apply in order, never
reorder, never insert.

## Apply order

| # | patch | subject | notes |
|---|---|---|---|
| 00 | `0000-…-block-00-foundation-…` | foundation: band uniformity, Meta wrapper, allocator | assumed by every later block |
| 01 | `0001-…-block-01-hybrid-HIP-all-reduce-…` | hybrid HIP all-reduce | RDNA4 gate, RCCL elsewhere |
| 02 | `0002-…-block-02-adaptive-MTP-draft-depth` | adaptive MTP draft depth | `--spec-type draft-mtp-adaptive` |
| 03 | `0003-…-block-03-chunked-gated-delta-net-prefill` | chunked GDN prefill | two arch-segregated kernels + rollback tracker |
| 04 | `0004-…-block-04-BF16-KV-cache-…` | BF16 KV + native-BF16 FA | |
| 05 | `0005-…-block-05-flash-attention-band-…` | FA band / native quantized K-V / derived kq mask | |
| 06 | `0006-…-block-06-MMB-quantized-prefill-GEMM` | MMB quantized prefill GEMM + `mmf32` (the F32 skinny prefill GEMM, r104; default-on for **RDNA4 and RDNA3_5** since r105, opt-in on RDNA3) | |
| 07 | `0007-…-block-07-k-quant-boosts-…` | k-quant boosts (`mmvq` VDR, Q6_K MMQ, Q1_0, grid pad) | **the decode-numerics block** — see `GREEDY-PURITY.md` §2/§9 |
| 08 | `0008-…-block-08-fused-kernels-…` | fused kernels | fusions + fused MoE gate+up+GLU + per-shape `mmvq` `nwarps` |
| 09 | `0009-…-block-09-host-resident-MoE-…` | host-resident MoE | slab, expert cache, prefill cache staging |
| 10 | `0010-…-block-10-qwen4exp-…` | qwen4exp | model, hyper-connections, QSA/indexer, async graph-input ring |
| 11 | `0011-…-block-11-campaign-remainder-…` | campaign remainder | CPU wins, backend iface fills, DFlash, FA build grouping, mmap reclaim |
| 12 | `0012-…-block-12-MTP-draft-vocabulary-subset-…` | MTP draft vocabulary subset | **opt-in** (`--spec-draft-mtp-vocab N`), default off; draft-only |

`rdna-boosts-all.patch` (repo root) is the entire net as ONE patch.  It is a convenience for a one-off
build at the fork point; it is **not** the reviewable increment and not a substitute for the per-block
flow.

## Apply (fresh checkout at the fork point)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout "$(jq -r .base /path/to/rdna-boosts/release.json)"
bash /path/to/rdna-boosts/scripts/apply-all.sh .     # strict 12/12 git am, tree checked vs release.json
```

`scripts/apply-all.sh` is the only supported flow.  **Do not `git apply` the concatenated series** — it
silently drops hunks (measured: 30 files / 2483 lines instead of 35 / 6094).  The script:

1. refuses a dirty tree or an existing `rdna-boosts` branch;
2. applies the whole series with strict `git am` (one commit per block, original subjects);
3. on failure, aborts and retries the series **once** with `git am -3` (3-way against the recorded blob
   ids), warning that the merged tree may differ from the canonical one;
4. on the strict path, asserts the applied tree equals `release.json.tree` — a stale fork point or a
   partially regenerated set fails here rather than at the build;
5. `--include-beta` then applies `beta/*/*.patch` on top of the *verified* tree (one commit each).  The
   checkout is then **not** the release tree, which the script reports.

## Release gates (all mandatory, recorded in the release's `WORKLOG.md` entry)

`scripts/validate-set.sh` covers patch integrity only (checksums + a fresh-tarball strict apply) and runs
in CI.  The model-level gates are manual:

1. **Coherence** — same-seed `llama-cli` output identical to a known-good build (dense and, for a change
   that touches the MoE path, a host-expert config).  `GGML_CUDA_ALLREDUCE=nccl` is not a bit-identical
   reference under `-sm tensor`; treat it as a smoke comparison.
2. **MTP** — the adaptive-MTP four-axis gate, `-n 3000` (2000 floor), `--reasoning on` for R and `off` for
   P/C/K, plus the stock-relative `llama-batched-bench -npl 1,4,8` arm.  Protocol:
   [`benchmarks/mtp-adaptive-methodology.md`](benchmarks/mtp-adaptive-methodology.md).
3. **Prefill logits** — `scripts/gate-prefill-logits.sh`: mean KLD <= 0.005 and same-top-p >= 98 % against
   a recorded known-good base.  **Required for any change to a prefill kernel** (GDN/SSM chunked, FA,
   MMQ/MMVF/MMB, MoE prefill) or to a default-on approximate path.  Protocol:
   [`benchmarks/prefill-logit-methodology.md`](benchmarks/prefill-logit-methodology.md).
   A **kernel-family switch** (a new tile/WMMA head cap, e.g. block 05's RDNA3_0 head-512 rows) uses the
   script's `--family-switch` mode, which judges the same absolute limits **at a point inside the gate's
   calibrated domain** and always prints a same-kernel reordering control measured on the spot:
   `--record --ab "<feature-off env>" --family-switch --model <head-256/512 model> --corpus
   prompts/fa-family-switch-gemma4-26B-A4B.txt --ctx 2048 --chunks 10`, then the same command without
   `--record` (`--candidate-env` selects the arm, so a single switch can be judged against a fully-tile
   base).  The absolute limits stay the criterion; `--allow-control-pass` is an explicit opt-in for a
   switch that provably cannot meet them, so an insensitive point cannot pass a bad change.  The mode refuses a point whose base PPL is above `--family-max-ppl` (default 10): at a
   point the model is not confident about, *every* reordering measures far outside the limits (a benign
   `-ub 512` vs `-ub 256` same-kernel change measures 0.105 / 85.0 %, so such a point cannot decide the
   question — `GREEDY-PURITY.md` §46, `archive/work/fa-rdna3-head512-purity/RESOLUTION.md`).
   The `FA_DBG_NMSE` op-level accuracy reference (`test-backend-ops` error magnitude vs the CPU backend at
   the model's own prefill shape) is the second, deterministic leg of a family-switch judgement.
4. **Per-block build** — `scripts/gate-per-block-build.sh` on a clean `all` build per commit (bisection and
   review walk the blocks individually, so every block must build on its own).  Use `--changed-from` to
   rebuild only the blocks a release changed.
5. **GPU faults** — `scripts/check-gpu-faults.sh` on the run's journal + log (never grep a log by hand).

Run the GPU gates **alone**: two concurrent GPU jobs change the arena/band layout and invent failures.

## Verification per block

The per-block command is a *targeted* check for a block's own surface; the release gates above still run
on the tip.

| block | verify | expected |
|---|---|---|
| 00 | `test-backend-ops -o FLASH_ATTN_EXT`; `test-logits-width-probe` W=1..8 | pass; one hash across the band |
| 01 | same-seed coherence on 2 and 3 GPUs (`-sm tensor`), MTP `none == draft-mtp` | identical |
| 02 | `tests/test-speculative-adaptive`, `tests/test-arg-parser`; `llama-server --spec-type draft-mtp-adaptive` smoke | pass |
| 03 | `test-backend-ops -o GATED_DELTA_NET`; `test-recurrent-state-depth`, `test-recurrent-state-rollback` | per-arm FAIL/PASS set unchanged from the recorded baseline (the counts differ per box — never gate on zero) |
| 04 | `test-backend-ops` bf16-KV FA cases + a bf16-cache model run | pass; coherence identical to f16 |
| 05 | `test-backend-ops -o FLASH_ATTN_EXT` (full matrix); width probe; prefill-logit gate | 6389/6389 on the r102 tree (gfx1100); probe PASS; KLD <= 0.005 for within-family changes, and the RDNA3_0 head-512 row set passes the absolute limit in `--family-switch` mode (0.001778 / 98.967 % at a base-PPL-2.65 point) — `GREEDY-PURITY.md` §46 |
| 06 | `test-backend-ops -o MUL_MAT` (MMQ cases) + prefill A/B | pass; prefill not slower at depth |
| 06 | **(r104) `mmf32`**: `test-backend-ops -o MUL_MAT` (F32 cases; with `LLAMA_TEST_OP_NMSE=1 LLAMA_TEST_SEED=1` the intercepted cases must be no worse than the rocBLAS path's NMSE) + fresh-start determinism (`llama-perplexity -c 512 --chunks 1`, 12 starts, 1 class; plus 2 GPU `-sm tensor`) + `llama-bench` A/B against `GGML_CUDA_MMF32=0` | 2444/2444 and 11x closer to the CPU reference than rocBLAS (6.1e-14 vs 6.9e-13, threshold 5e-4); 12/12 one class, 6/6 at 2 GPU; `+4.2 % pp512 / +4.1 % pp2048`, decode flat — `GREEDY-PURITY.md` §47.  **(r105) the same three legs on each arch the default is claimed for**; on gfx1151 (RDNA3_5, now default-on) 2444/2444, 12/12 one class, `+1.0 % pp512 / +1.6 % pp2048`, and the `--family-switch` judgement at the shipped confident point (0.013016 / 98.673 % vs a 0.019125 / 97.975 % same-kernel control) |
| 07 | `test-backend-ops -o MUL_MAT`/`MUL_MAT_ID` (K-quant cases); `llama-perplexity` before/after | pass; **PPL must be unchanged** (it is a numerics change, not a quality change) |
| 08 | `test-backend-ops -o MUL_MAT_ID`; MoE decode + `llama-batched-bench -npl 1,4,8`; width probe on a MoE model | pass; one hash across the band; verify widths not slower than stock-relative |
| 09 | host-expert coherence (`-ncmoe`/`-cmoe`), `test-llama-archs`, `MOE_EXPERT_CACHE_VALIDATE=1`; for the batched f32 CPY: `test-logits-width-probe` default vs `GGML_CUDA_FUSE_CPY_BATCH=0` | pass; no `////`, no fault; cache-on == cache-off text; probe maxdiff 0 and identical hashes |
| 10 | `test-backend-ops -o FLASH_ATTN_QSA` (18/18 minimum); `test-llama-archs -a qwen4exp`; sparse-vs-dense perplexity ratio (`LLAMA_QSA_SPARSE_FA=0`); `scripts/gate-qwen4exp-quant-coherence.sh`; for the verify-band fusions: `test-logits-width-probe` default vs `GGML_CUDA_FUSE_Q8_1_VERIFY=0` + `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0`; for the async-input ring (r105): `GGML_CUDA_ASYNC_INPUT=0` A/B (identical text + prefill logits) and the two-thread origin count (`GGML_CUDA_ASYNC_INPUT_DEBUG=1`, `archive/work/async-input-per-origin/repro/`) | pass; ratio close to 1; probe maxdiff 0 and identical hashes (`GREEDY-PURITY.md` §11); ring: 2 origins on one device from two threads, `=0` unchanged, no perf regression vs the ring-off arm |
| 10 | **(r108) gate/beta fusion, multi-sequence**: `build/bin/test-fusion-gate-beta-multiseq` (built by default; `ctest -R test-fusion-gate-beta-multiseq`) - it compares the fused launch against the same chain with the fusion made ineligible for `(n_seq_tokens, n_seqs)` = (1,1)..(2,4) and proves the fusion fired; plus a real multi-sequence decode A/B (`GGML_CUDA_FUSE_GATE_BETA_BF16=0`, e.g. two concurrent `llama-server` slots or a lockstep probe) on a BF16-`ssm_alpha` model | PASS (11/11 fused, 0 failed, exit 0); the fusion-off arm bit-identical to the fusion-on arm at every sequence; single-stream text and prefill logits unchanged (`GREEDY-PURITY.md` §49) |
| 10 | **(r103)** bf16 K/V prefill runs and is bit-identical to the unfused chain: `llama-bench -ctk bf16 -ctv bf16` (must not abort) + a prefill-logit compare against a base recorded with `GGML_CUDA_DISABLE_ROPE_SET_ROWS=1`; `test-backend-ops -o ROPE -o SET_ROWS` | 5136.70 t/s pp512; KLD −0.000000 / 100.000 %; ROPE+SET_ROWS green |
| 11 | dense coherence + the block's own arm (DFlash/CPU/mmap): `-ncmoe` peak-RSS check with and without the reclaim | identical text; RSS as recorded in `patches/README.md` |
| 11 | **(r106) DFlash2 + `-sm tensor`**: a 2- and 3-GPU `--spec-type draft-dflash` run on a selector drafter (coherence, deterministic text, draft acceptance equal to the same config under `-sm layer`), `--parallel 2`, and `-sm tensor` + draft + `-ncmoe` (the slab narrow-2 arm) | all coherent with no override env; acceptance identical to `-sm layer`; `test-backend-ops` 3/3 and `gate-prefill-logits.sh` at KLD <= 0.005 (r106: 2 GPU 84.5 vs 63.4 t/s, 3 GPU 61.1, KLD 0.000000 / 100.000 %) |
| 12 | **(r107) MTP draft vocabulary subset**: `-sm tensor` 2-GPU run with `--spec-draft-mtp-vocab N` on (the model prints `mtp_draft_vocab = N: MTP draft head uses R of V rows` under `-v`); greedy text at `--spec-draft-n-max 7` against `--spec-type none` (prose **and** code, `-n 3000`); the four-axis adaptive gate (R/C/K/P, `-n 3000`) with the flag off vs on; `test-backend-ops`; `test-logits-width-probe SPLIT=tensor`; `scripts/gate-prefill-logits.sh`; a row-range read check for the `ggml-backend-meta` half | text identical inside the purity band; all four axes not slower (r107: R +1.3 %, P +1.7 %, C +2.4 %, K +0.8 %); `test-backend-ops` 21179/21179; probe W=1..8 maxdiff 0; KLD 0.000000 / 100.000 %; `-sm tensor` works without the trim aborting — `GREEDY-PURITY.md` §48 |

## Failure handling (agent instruction)

Apply in the order above, to a branch from `release.json.base`, with `scripts/apply-all.sh`.  If a block
fails to apply:

1. Let the script's `git am -3` retry run; if the merged tree differs from `release.json.tree`, **stop** —
   the checkout was not at the recorded base, or the set is stale.
2. For a true conflict, resolve the hunks **in the block that owns the code** (never in the last block),
   then regenerate the set (`scripts/make-patches.sh`) rather than leaving a hand-edited patch.
3. Do not skip a block.  The series is linear and each block assumes its predecessors; there is no
   supported "apply a subset" flow (including for the k-quant block — see `GREEDY-PURITY.md` §8).
4. Record what happened in `WORKLOG.md`, with the base/tip/tree you ended up with.

## Known notes (working)

- **`test-backend-ops.cpp` is shared by several blocks** (03/04/05/07/08/10).  Their hunks sit in different
  case regions, so a re-base is local to the failing file — but grep every patch that touches the file
  before concluding a case is missing.
- **Block 08 is one diff on purpose.**  Its matchers and the `mmvq` / `mmq` / `norm` / `quantize` kernels
  they dispatch to co-developed over one kernel region; do not try to split it.  (The `ggml_cuda_try_fuse`
  translation unit itself is block 10's.)
- **Blocks own their code.**  A fix goes into the block whose surface it changes, even if that means
  amending an early block; appending it to the last block is what made the per-block build gate necessary.
- **The set must apply whitespace-clean.**  `apply-all.sh` prints no git whitespace warnings; a warning is
  a regeneration bug, not a cosmetic issue.
- **Block-number collisions in old records:** documents written before 2026-10-13 use the
  pre-V2.0 16-block numbering ("block 12" = hybrid all-reduce there, but "block 10" = the k-quant
  umbrella).  `WORKLOG.md`, `archive/` and the dated notes in `archive/docs/patch-notes-history.md` are
  historical and keep their numbering; `patches/README.md` and this file use the V2.0 numbers, and the
  block subjects are the table above.

## History

Dated validation records, the retired block-structure churn and the pre-V2.0 per-block verification table:
[`archive/docs/validation-history.md`](archive/docs/validation-history.md).  Baseline/re-base records:
[`archive/docs/baseline-history.md`](archive/docs/baseline-history.md).
