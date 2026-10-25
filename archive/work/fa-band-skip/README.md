# fa-band-skip -- the decode/verify band skips fully masked KV groups

Status: **PROMOTED (r133, 2026-10-25) into block 05.**  Contributor PR
[#223](https://github.com/stew675/llama-cpp-rdna-boosts/pull/223) by
[@overdoingism](https://github.com/overdoingism), filed as one `git am` file against
`v14-3f03b1772-r131` plus a WIP note.  The delivery takes the change with three additions
(decode-width op-test coverage, the launcher evaluating the band predicate once, and the stale
"prefill only" comment rewritten); the submitted patch is preserved verbatim as
[`fa-band-skip-as-submitted.patch`](fa-band-skip-as-submitted.patch).

Kill switches: `GGML_CUDA_FA_BAND_SKIP=0` (this arm) and `GGML_CUDA_FA_MASK_SKIP=0` (the whole
mechanism).  Rule: `GREEDY-PURITY.md` §37.  Release record: `WORKLOG.md` 261025.003.

## What it does

The fully-masked KV-group skip (issue #48, upstream #28495) classified 256-cell KV groups and let the
kernels skip the ones that are `-INF` for every query row of a tile, but only on the prefill path
(`Q->ne[1] > 8` for the packed mask; the derived `cell_pos` form is already prefill-only).  The
decode/verify band -- `n_q <= 8`, the RDNA4 WMMA band with its round-robin KV split -- read every cell
of the shared cache, including the other slots' masked cells.

The change is two lines of gate plus one argument:

* `launch_fattn` also builds the packed per-(stream, query tile, group) bitmap
  (`flash_attn_mask_to_KV_blocks`) for the band, under the band's own predicate
  (`ggml_cuda_fattn_band_wmma_applies`: RDNA4, head 256, GQA 5..12, `n_q <= 8`, quantized-native or f16
  K/V) and `GGML_CUDA_FA_BAND_SKIP`.
* the band entry of `flash_attn_ext_f16` passes that tile's row to `flash_attn_ext_f16_process_tile`
  instead of `nullptr`.

The band's skip loops already existed in block 05 and had never been reached (the launcher never built a
bitmap for the band).  Block i still owns iterations `i, i+P, i+2P, ...` of its tile and only omits the
fully masked ones, so its accumulation order over the remaining iterations is unchanged; a block that
owns no unmasked iteration writes the same neutral partial (`max -FLT_MAX/2`, rowsum 0, VKQ 0) that the
`flash_attn_stream_k_fixup_uniform` combine already treats as an exact no-op.  The result is
bit-identical, not merely equivalent.

## Why now: the r130 interaction

`LLAMA_SERVER_LAZY_IDLE_SLOTS` (r130, PR #210) keeps an idle slot's context resident in a unified KV
cache.  A conversation whose cells land after a long resident one then pays for reading those masked
cells on **every** decode step, and the band is where that happens.  The reporter measured it and asked
whether to send the fix; this is that fix.

**This is why the 2026-09-26 verdict in `archive/work/unified-cache-decode/` ("implemented and
break-even, close it") does not apply here.**  That campaign implemented the same option (b) and measured
it on a 2-slot `llama-batched-bench` cell whose foreign-cell fraction was small, so the per-layer prepass
(~0.6-1 %) cancelled the saving (~1 %).  At the r130 shape the foreign fraction is ~92 % of the range and
the saving is an order of magnitude larger.

## Evidence (gfx1201, ROCm 10.1.0, 27B UD-Q4_K_XL, `q8_0` K/V, 2 slots, `--kv-unified`, greedy, seed 42)

| arm | band skip off | band skip on | result |
|---|---|---|---|
| resident ~105K conversation, active 9.8K one, 384 tokens, plain decode | 50.84 ms/step (19.67 t/s) | **41.69 ms/step (23.99 t/s)** | **-18.0 %**, token ids identical (`a14854397efa688c`) |
| the same with a DFlash2 draft at `--spec-draft-n-max 5` (the reporter's config) | 18.62 ms/step (53.71 t/s) | **14.83 ms/step (67.42 t/s)** | **-20.3 %**, token ids identical |
| single slot, 9.8K, nothing to skip | 41.62 / 41.53 ms/step | 41.30 / 41.34 ms/step | -0.6 % (the padded tail group is skipped too) |
| single slot at 105K depth, nothing to skip | 25.1 t/s | 25.1 t/s | neutral; text identical |
| `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` | 30.41 / 98.28 / 152.17 t/s | 30.32 / 99.09 / 152.29 t/s | within noise (the band is off for `B > 1`) |

Purity and accuracy:

| gate | result |
|---|---|
| `test-backend-ops -o FLASH_ATTN_EXT` (full matrix, incl. 12 new decode-width packed-hole cases: `nb` 1/2/3/8 x f16/bf16/q8_0 x `GGML_HIP_FA_BAND_WIDE=0/1` x `GGML_HIP_FA_BAND_FETCH_AHEAD=0/1`) | **6414/6414** |
| the same cases with `fattn_kq_group_masked` mutated to skip one *visible* group (coverage probe, not shipped) | **0/14 pass** -- the new cases really do exercise the skip |
| `test-logits-width-probe` (27B UD-Q4_K_XL, prose, `P=256`), `KV=q8_0/f16/bf16`, skip on vs off | **PASS, worst maxdiff 0**; row-0 hashes identical between arms (`710641958f1aa46e` / `b12bd89b74c301f6` / `26bbb7aab7ee2010`) |
| `plain == draft-mtp` (`--spec-draft-n-max 3`, `-n 3000`, prose) 1 GPU and 2 GPU `-sm tensor` | text identical (`778154646e23` / `e4d32ce80614`), skip on and off |
| four-axis MTP acceptance (`-n 3000`, draft-mtp n3, `-lv 4`) | identical to the last digit: R 0.52697 / P 0.80144 / C 0.86491 / K 0.99200 (1837/3486, 1332/1662, 2164/2502, 372/375) |
| `scripts/gate-prefill-logits.sh` (27B Q8_0) | **PASS, mean KLD 0.000000 / same-top-p 99.980 %** |

## Boundaries

* **Band only.**  The predicate is the band's, so models whose decode uses the tile kernel (head != 256,
  GQA <= 4, RDNA3/RDNA3_5, or `GGML_HIP_FA_BAND_WMMA=0`) still read the other slots' masked cells.  The
  2026-09-26 campaign implemented the tile path too and measured it break-even at a small foreign
  fraction; extending it now would need that family's own decode-width hole cases and a measurement of
  the r130 shape on it.  Not done here.
* **RDNA4 only**, as the band is.  On other architectures `ggml_cuda_fattn_band_wmma_applies` is false, so
  `kq_block_skip_packed` is computed exactly as before (no behaviour change, nothing to test there).
* **The prepass is per FA launch, i.e. per layer per step.**  Caching per-slot group visibility in
  `llama_kv_cache` (option (a) of the old campaign) would remove it; measured unnecessary here, since the
  tail group it also skips already pays for it.
* `-sm tensor` was not run by the reporter; it is covered here at the text level (2 GPU: identical text
  and throughput for both arms, plain and draft-mtp).

## Files

* `ggml/src/ggml-cuda/fattn-common.cuh` -- `ggml_cuda_fattn_band_skip_enabled()`, the `band_skip` term of
  `kq_block_skip_packed`, and the hoisted `band_wmma` the 2-D launch reads.
* `ggml/src/ggml-cuda/fattn-mma-f16.cuh` -- the band entry passes `kq_blocks_tile`.
* `tests/test-backend-ops.cpp` -- the decode-width packed-hole cases (added by the delivery; the PR had
  none, and without them the band skip had no oracle at all).
