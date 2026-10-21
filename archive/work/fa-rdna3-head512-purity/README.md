# archive/work/fa-rdna3-head512-purity — why does the gfx1100 head-512 WMMA FA row set move the prefill logits?

**Status: CLOSED 2026-10-21 — resolved as "the point was the problem"; see [`RESOLUTION.md`](RESOLUTION.md).**
The rows pass the absolute limits at a calibrated point (`0.001778 / 98.967 %`).  Kept as the historical brief.
This campaign exists to either (a) find a row set that meets the prefill-logit gate's absolute limits
(mean KLD `<= 0.005`, same-top-p `>= 98 %`), or (b) show the divergence is inherent to a tile→WMMA family
switch and replace the criterion with a real accuracy reference.  Candidate (b) is a legitimate result and
is what the guidance in `GREEDY-PURITY.md` §46 currently assumes.

**NOT delivery work.**  The promoted code is in `patches/0005-*.patch`; nothing here is applied by
`scripts/apply-all.sh`.  Run this on **`fingon` (gfx1100, RX 7900 XTX)**; the RDNA3_0 rows do not exist on
gfx1201/gfx1151.

**Tracker:** `TODO.md` #81.  **Rulebook:** `GREEDY-PURITY.md` §46.  **Upstream PR:** #199 (@briansp2020).

---

## 1. What is already measured

All of it on gfx1100 (`fingon`), r101 vs r101 + PR #199, `ROCBLAS_USE_HIPBLASLT=0`, one GPU job at a time.
Model: `/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (30 layers, 5 global layers
at head 512 / GQA 8, 25 SWA layers at head 256).

| arm | number |
|---|---|
| pp2048 @ d32768, `q8_0` K/V | 1908.3 / 1909.2 -> **2549.1 / 2554.6** t/s (+34 %) |
| pp2048 @ d65536, `q8_0` K/V | 1209.3 / 1207.6 -> **1756.2 / 1757.5** t/s (+45 %) |
| tg128 @ d32768 | 127.74 / 127.74 -> 127.80 / 127.73 (unchanged) |
| `GGML_CUDA_FA_RDNA3_WMMA_512=0` | bit-identical to r101 (`cf175b78ecd1`, 200-token greedy) |
| `test-backend-ops -o FLASH_ATTN_EXT` | 6391/6391, 0 failed (incl. the head-512 shapes) |
| gfx1201 / gfx1151 | byte-identical to r101 (gemma4 26B-A4B both) |

Prefill-logit, gate protocol (`llama-perplexity -c 512 --chunks 40 -ngl 99 -fa on -t 8`, wikitext-2,
f16 K/V), one base file recorded with `GGML_CUDA_FA_WMMA_256=0` (fully tile):

| change | mean KLD | same top p |
|---|---|---|
| **head-512 WMMA vs tile** | **0.0293** | **92.59 %** |
| head-256 WMMA vs fully tile (shipped default-on since r5) | 0.0441 | 90.63 % |
| benign ubatch change, same kernel (tile `-ub 512` vs `-ub 256`) | ~0.12 / chunk | 84.5 % |
| issue #113 BF16 chunked GDN (gfx1100) | 0.62 | — |

PPL, no base file needed: f16 K/V 604.06 (tile) vs 607.20 (WMMA); `q8_0` K/V 617.85 vs 620.59.  Note the
**point is out of the gate's calibrated domain** (PPL 604 vs 5.93 for the pinned 27B Q8_0), which is why the
absolute limit is not reachable by any family switch here.  Reproduce the numbers first before changing
anything.

## 2. How to reproduce (exact commands, on fingon)

```bash
# tree: a worktree at release.json.base with the 12 r102 blocks applied, plus your experiment.
# ~/llama-fa512 may already exist from the PR #199 verification: remove it first (git worktree remove).
cd ~/llama.cpp && git worktree add ~/llama-fa512 097cf33d9 --detach
cd ~/llama-fa512 && git checkout -b fa512
for p in ~/llama-cpp-rdna-boosts/patches/00*.patch; do git am "$p"; done   # 12 commits, strict
git rev-parse HEAD^{tree}   # must equal the r102 tree in release.json (42679a04ff81...)
BUILD_DIR=build-fa512 ~/bin/build-llama-rocm                              # ROCm 10.1.0-gfx1100

export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1100/lib ROCBLAS_USE_HIPBLASLT=0 HIP_VISIBLE_DEVICES=0

# prefill A/B, same binary (the rows are always on; the env only switches the chooser)
M=/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
./build-fa512/bin/llama-bench -m $M -p 2048 -n 0 -d 32768 -r 2 -fa on -ctk q8_0 -ctv q8_0 -t 8

# prefill logits: one base (fully tile), then compare each candidate arm against it
PPL=./build-fa512/bin/llama-perplexity; C=/llm/models/wikitext-2-raw/wiki.test.raw
GGML_CUDA_FA_WMMA_256=0 $PPL -m $M -f $C -c 512 --chunks 40 -ngl 99 -fa on -t 8 \
    --kl-divergence-base /tmp/base-fulltile.kld

$PPL -m $M -f $C -c 512 --chunks 40 -ngl 99 -fa on -t 8 \
    --kl-divergence --kl-divergence-base /tmp/base-fulltile.kld
```

`scripts/apply-all.sh` is the sanctioned apply path, but it wants to create the branch `rdna-boosts`,
which `~/llama.cpp` already owns on fingon: either rename that branch there first or use the explicit
`git am` loop above (identical in effect, and the tree check is the contract).

Two traps: `--kl-divergence-base FILE` alone **writes** the file (add `--kl-divergence` to compare), and
fingon's `/tmp` is a 16 GiB tmpfs (one base file is ~3 GiB; put extra files in `$HOME`).

## 3. Where the divergence can come from (the decomposition to run)

The rows live in `ggml_cuda_fattn_mma_get_config_rdna` (`ggml/src/ggml-cuda/fattn-mma-f16.cuh`), the
enabling predicate in `ggml_cuda_get_best_fattn_kernel` (`fattn.cu`).  The tile→WMMA difference has four
independent dimensions; only the third changes *values* rather than addition order, so start there:

1. **Threads per block / occupancy** — 128 threads, occupancy 1 (tile: 128-256 threads, strip-mined).
2. **VKQ combine width and K/V batch width** — `nbatch_combine` 64, `nbatch_K2`/`nbatch_V2` 64 at 16
   columns and 32 at 32/64 columns.
3. **`nbatch_fa`** — the online-softmax rescaling interval: 64 at 16 columns, 32 at 32/64 columns.  This is
   the `m`/`l`/`acc` rescale cadence, i.e. the one parameter that changes the *arithmetic*, not just the
   tree order.  The tile kernel's strip length is the natural reference point: try a row at
   `nbatch_fa` = the tile's value and nothing else different.
4. **`Q_in_reg`** — false (shared memory) here; the gfx1201 rows use true.  This is what fixes the VGPR
   spills, so it is probably not negotiable, but measure the cost of putting it back (it was 994-1362
   spilled VGPRs per lane, so the measurement is also the "is the win really the spills" check).

Method: one variant at a time, KLD against the same fully-tile base, plus `llama-bench` for the throughput
so each variant is costed.  If a variant lands under 0.005 while keeping most of the +34 %, that is the
deliverable.  Variants can be forced with a temporary env var in
`ggml_cuda_fattn_mma_get_config_rdna` (do not commit it) or by editing the row directly and rebuilding
(`cmake --build build-fa512 --target llama-cli llama-bench llama-perplexity -j 16` is ~2 min with ccache).

## 4. If it is inherent: replace the criterion (option b)

The gate's instrument is a KLD against a sibling kernel.  For a family switch the honest instrument is
accuracy against an independent reference.  Options, cheapest first:

* `test-backend-ops -o FLASH_ATTN_EXT` currently reports pass/fail only.  Add an error magnitude to its
  console/csv output (the `nmse()` helper is already there) and compare the head-512 arms; the arm closer to
  the CPU reference is the more accurate one, which settles the direction of the change.
* Run the same model/text at `f16` K/V on both arms and compare each against a `-ngl 0` CPU run (slow but
  exact for a 40-chunk job).
* Then either write the calibration into `scripts/gate-prefill-logits.sh` as a `--family-switch` mode
  (record one fallback base, measure the candidate and the shipped-equivalent switch, require the candidate
  to be no worse) or record the deviation for good.

Whichever way it goes, update `GREEDY-PURITY.md` §46 and close `TODO.md` #81 with the numbers.

## 5. Reporting

Keep the measurement record here (`MEASUREMENTS-YYYY-MM-DD.md`), one file per session, with the arm
definitions, the raw per-round numbers and the base-file sha256.  On resolution: move this directory to
`archive/work/fa-rdna3-head512-purity/`, add a `RESOLUTION.md`, delete the `wip/` entry, and add a
`WORKLOG.md` entry (newest first).  If a better row set is found it is a block 05 amendment, which means
the full promotion path (`make-patches.sh` -> `make-release.sh` -> `validate-set.sh` ->
`gate-per-block-build.sh` -> the fork refresh).
