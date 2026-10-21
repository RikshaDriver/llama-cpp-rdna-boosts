# Next-session prompt (for the fingon agent)

Paste the block below into a fresh session on `fingon`.  It is self-contained: the campaign record it
points at is `wip/fa-rdna3-head512-purity/README.md`, the tracker is `TODO.md` #81, the rulebook section is
`GREEDY-PURITY.md` §46.

---

You are picking up a live campaign in `/home/stew675/llama-cpp-rdna-boosts` on this box (`fingon`, gfx1100).
Read `AGENTS.md` first, then `wip/fa-rdna3-head512-purity/README.md`, `TODO.md` #81 and `GREEDY-PURITY.md`
§46. Then drive the campaign to resolution.

**Why this exists.**  PR #199 gave RDNA3_0 its own head-512 WMMA FA rows and was promoted default-on in
r102: it is worth +34 % deep prefill at d32768 and +45 % at d65536 on this box with decode unchanged, by
fixing ~1.0-1.4k VGPR spills per lane on gfx1100.  But it is a tile→WMMA *family* switch, so it moves the
prefill logits against the tile kernel: **mean KLD 0.0293, same-top-p 92.59 %** on gemma-4-26B-A4B QAT
(gate limits 0.005 / 98 %).  We shipped it because we calibrated that point: the head-256 WMMA rows already
shipped default-on for gfx1100 measure **0.0441 / 90.63 %** against fully tile, a benign ubatch change on
the same kernel measures ~0.12 per chunk, and issue #113's BF16 GDN measured 0.62.  So it is more faithful
than a switch we already run, while the absolute limit is unreachable by *any* family switch there (base
PPL at that point is 604, against 5.93 for the model the gate was pinned on).

**Mission.**  Resolve it.  Two outcomes are acceptable:

* **A. A row set that passes the absolute gate** (mean KLD <= 0.005, same-top-p >= 98 %) while keeping a
  meaningful share of the win, i.e. pp2048 @ d32768 still at least ~+20 % over tile.
* **B. A defensible demonstration that the divergence is inherent to the family switch**, plus a
  replacement criterion that is actually decidable: an accuracy reference (op-level error magnitude for
  `FLASH_ATTN_EXT` at head 512 against the CPU reference, or a CPU-backend perplexity reference) that says
  which kernel is closer to the truth, wired into `scripts/gate-prefill-logits.sh` as a documented
  family-switch mode.  B is a real result, not a consolation prize; it also re-justifies the r5/r99 caps.

**Work in this order.**

1. **Reproduce before changing anything.**  Rebuild the r102 reference build and reproduce the three
   numbers: pp2048 @ d32768 (tile vs default), the KLD/top-p against the re-usable base file below, and PPL
   for both arms.  Report them to the maintainer, then start.
2. **Decompose the divergence one dimension at a time**, cheapest-to-reason-about first.  The row set
   differs from the tile kernel in four independent ways:
   * **`nbatch_fa`** (the online-softmax rescale interval: 64 at 16 columns, 32 at 32/64 columns).  This is
     the only dimension that changes *arithmetic* rather than addition order, so start here.  Try a row at
     the tile kernel's strip length and nothing else changed.
   * `Q_in_reg` (false = shared memory here; true on the gfx1201 rows).  This is what fixes the spills, so
     measure the cost of putting it back.  That measurement doubles as the "is the win really the spills"
     check.
   * `nbatch_K2` / `nbatch_V2` (64 at 16 columns, 32 at 32/64 columns).
   * `nthreads` (128) / `occupancy` (1).
3. For **every** variant record: the throughput, the KLD/top-p against the same base file, and the PPL, so
   each variant is costed rather than just measured.  Rebuild is
   `cmake --build build-kld --target llama-cli llama-bench llama-perplexity -j 16` (2-4 min warm).

**Environment (exact; these are the traps we already hit).**

* fingon: gfx1100 RX 7900 XTX, 24 GiB, ROCm 10.1.0 at `/opt/rocm-10.1.0-gfx1100`, build script
  `~/bin/build-llama-rocm` (takes `BUILD_DIR=<relative dir>`).
* `~/llama.cpp` is already at the **r102** tree (`42679a04ff81…`, 12 blocks), but `build-rocm` is an r101
  build: rebuild it (`cd ~/llama.cpp && BUILD_DIR=build-rocm ~/bin/build-llama-rocm`).  Expect a full
  `ggml-hip` rebuild (~15 min warm) because r102 changed `common.cuh`, which every TU includes.
* `~/llama-fa512` is the old PR #199-only worktree and is superseded by r102: remove it
  (`git -C ~/llama.cpp worktree remove --force ~/llama-fa512`).
* For experiments use a throwaway worktree and its own build dir:
  `git -C ~/llama.cpp worktree add ~/llama-kld <r102 tip> --detach`, then
  `cd ~/llama-kld && BUILD_DIR=build-kld ~/bin/build-llama-rocm`.
* **ccache**: keep it at 30 GB (`ccache -s`).  If it ever reports 5 GiB, run `ccache --max-size=30G`; at the
  tool default this work is ~6x slower.
* Model: `/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (5 global layers at head
  512, 25 SWA layers at head 256).
* **Reusable KLD base** (fully-tile, recorded with the gate protocol): `/home/stew675/b-fulltile.kld`,
  sha256 `d871235cd52f7b1c90cfe171e6e30280aaef2722fc496a19a6f1b7d9406d7439`, 5347901140 bytes.  Verify the
  sha256 before trusting it, and delete the truncated `/tmp/b-tile-ub256.kld` (2.9 GB).
* Always: `ROCBLAS_USE_HIPBLASLT=0` (issue #67: per-start hipBLASLt solution picks otherwise confound every
  A/B), `HIP_VISIBLE_DEVICES=0`, **one GPU job at a time**, and `llama-cli` always with `--single-turn
  --no-display-prompt`.  Never run benches in parallel.  `/tmp` is a 16 GiB tmpfs (a KLD file is ~3 GiB);
  put extra files in `$HOME`.

**Commands.**

```bash
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1100/lib ROCBLAS_USE_HIPBLASLT=0 HIP_VISIBLE_DEVICES=0
M=/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
B=/home/stew675/b-fulltile.kld

# throughput, same binary (the rows are always on; the env only switches the chooser)
./build-kld/bin/llama-bench -m $M -p 2048 -n 0 -d 32768 -r 2 -fa on -ctk q8_0 -ctv q8_0 -t 8
GGML_CUDA_FA_RDNA3_WMMA_512=0 ./build-kld/bin/llama-bench -m $M -p 2048 -n 0 -d 32768 -r 2 -fa on -ctk q8_0 -ctv q8_0 -t 8

# prefill logits against the shared base (note: --kl-divergence-base ALONE writes the file)
PPL=./build-kld/bin/llama-perplexity; C=/llm/models/wikitext-2-raw/wiki.test.raw
$PPL -m $M -f $C -c 512 --chunks 40 -ngl 99 -fa on -t 8 --kl-divergence --kl-divergence-base $B

# PPL, no base file needed
$PPL -m $M -f $C -c 512 --chunks 40 -ngl 99 -fa on -t 8
```

**Code locations.**

* `ggml/src/ggml-cuda/fattn-mma-f16.cuh` — `ggml_cuda_fattn_mma_get_config_rdna`, the RDNA3_0 head-512
  block (search for the `is_rdna3_5 && !is_rdna4 && DKQ == 512` guard).
* `ggml/src/ggml-cuda/fattn.cu` — `ggml_cuda_get_best_fattn_kernel` (the `rdna3_512` predicate) and
  `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1` (caps `ncols1` at 32 for heads above 256).
* `ggml/src/ggml-cuda/fattn-tile.cuh` — the kernel you are comparing against: its strip length and rescale
  cadence are the reference point for `nbatch_fa`.

**Acceptance.**

* Outcome A: KLD <= 0.005 and top-p >= 98 % against the base; pp2048 @ d32768 still >= ~+20 % over tile;
  tg128 @ d32768 unchanged; `test-backend-ops -o FLASH_ATTN_EXT` 6391/6391; with
  `GGML_CUDA_FA_RDNA3_WMMA_512=0` the text still bit-identical to r102; gfx1201 and gfx1151 still unchanged.
* Outcome B: the decomposition table (every variant's KLD, top-p, t/s, PPL), the argument that no reachable
  variant passes, and the criterion landed in `scripts/gate-prefill-logits.sh` + `MANIFESTS.md` +
  `GREEDY-PURITY.md` §46 with a recorded accuracy comparison.

**Rules of engagement.**

* Do **not** edit `patches/` or `release.json`.  `wip/` is not the delivery: nothing is promoted without the
  maintainer's explicit go-ahead.
* Every claim must be reproducible from a recorded command in a fresh process.
* Record each session in `wip/fa-rdna3-head512-purity/MEASUREMENTS-<date>.md`: arm definitions, the base
  file sha256, raw per-round numbers.
* On resolution follow `AGENTS.md`: move the campaign to `archive/work/fa-rdna3-head512-purity/`, write
  `RESOLUTION.md`, add a `WORKLOG.md` entry (newest first, header `YYMMDD.NNN`), close `TODO.md` #81 (a
  closed item is a WORKLOG entry, not a TODO row), delete the `wip/` entry and update `wip/CAMPAIGNS.md`.
  If a better row set exists, the promotion path is: amend block 05 → `make-patches.sh` →
  `make-release.sh` → `validate-set.sh` → `gate-per-block-build.sh` → fork refresh.  Report findings to the
  maintainer before any delivery-affecting commit.
* Never push anywhere except the personal fork (`git@github.com:stew675/llama.cpp.git`), and only with the
  maintainer's go-ahead.  Never push to upstream `ggml-org/llama.cpp`.

Start by reporting the reproduction (throughput, KLD vs the base, PPL) before you change any code.
