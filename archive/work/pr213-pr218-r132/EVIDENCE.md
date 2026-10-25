# EVIDENCE — r132 promotion of PR #213 and PR #218

Everything here was run on gfx1201 (3x R9700), ROCm 10.1.0, `~/bin/build-llama-rocm`, sequential (never
two GPU jobs at once).  "base" = the r131 tree; "fold" = the r132 tree (the same changes folded into
blocks 06/08/09/10); `pt213` / `pt218` = the same changes applied at the r131 tip for a per-PR check.

## 1. Coherence (same-seed greedy, 12 configs)

`a` dense 4B Q8_0 1 GPU, `b` 4B Q8_0 3 GPU `-sm tensor`, `c` 27B UD-Q4_K_XL 3 GPU tensor, `d` 35B-A3B
UD-Q4_K_M `-ncmoe 40` 2 GPU tensor, `e` the same on 1 GPU `-ncmoe 10`, `f` 27B UD-IQ3_S 3 GPU tensor,
`g` Flash-Next GSQ IQ3_XXS + its MTP head 2 GPU tensor `-ncmoe 10 -lm none -lzm off`, `h` the same
`-sm layer` (no draft), `i` Ling-3.0-tiny IQ4_NL 2 GPU `-sm layer`, `j` Gemma4 E4B Q8_0, `k` 35B-A3B
`-ncmoe 40` + `--spec-type draft-mtp-adaptive --spec-draft-n-max 7`, `l` 27B + the same draft.

```
base  a 56fc83127293  b 56fc83127293  c 1c379f1fe639  d 7fcb1edb40d8  e 7fcb1edb40d8  f a740d40d6bbd
base  g 0afd593e2b1e  h 0afd593e2b1e  i a5df88f3045b  j c69a8b7221b6  k 7fcb1edb40d8  l 1c379f1fe639
pt213 a..l: identical to base in all 12
pt218 a..j: identical to base in all 10 (k/l later: identical as well)
fold  a..l: identical to base in all 12
```

## 2. MTP four axes (Protocol A, 27B UD-Q4_K_XL, 1 GPU, `-n 3000`, `--spec-type draft-mtp`, n-max 3)

```
base R 53.9 t/s acc 0.54420 (1859/3416)   fold R 53.9 t/s acc 0.54420 (1859/3416)
base P 70.3 t/s acc 0.83041 (1420/1710)   fold P 70.5 t/s acc 0.83041 (1420/1710)
base C 73.3 t/s acc 0.87435 (2171/2483)   fold C 73.4 t/s acc 0.87435 (2171/2483)
base K 80.4 t/s acc 0.99200 ( 372/ 375)   fold K 80.5 t/s acc 0.99200 ( 372/ 375)
```

The text hashes in the `--log-verbosity 4` runs differ only because the log records interleave into the
generated stream; re-running the P axis without `-v 4` gives **5975 chars sha=2928e6d6ea55 on both
builds**.

## 3. Prefill logits (the gate, both recorded points, fold build)

```
27B Q8_0       mean KLD 0.000000  same-top-p 99.980 %   PASS
Ling IQ4_NL    mean KLD 0.000000  same-top-p 100.000 %  PASS
```

## 4. Verify widths (`llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`, 27B UD-Q4_K_XL, q8_0 KV, 3 GPU
tensor, `-t 8`, two interleaved rounds)

```
TG t/s  B=1 / B=4 / B=8      base 44.99 / 145.15 / 244.47   fold 44.99 / 146.23 / 244.27
                             base 44.53 / 146.23 / 244.27   fold 44.47 / 145.31 / 244.18
```

## 4b. `GGML_META_SS_VERIFY=1` (the split-state stale-hit check)

Run on the folded build for configs `c` (27B UD-Q4_K_XL, 3 GPU `-sm tensor`) and `g` (Flash-Next GSQ + MTP,
2 GPU tensor): every outermost cache hit is recomputed and compared, and the run aborts on a mismatch.
Silent (no abort) and the text is identical to the unset run (`1c379f1fe639` / `0afd593e2b1e`).

## 5. `test-backend-ops` and the marginal-tolerance flake

* `-o MUL_MAT`: base **8/8 clean runs** of 2445/2445; fold 15 runs with **2** single-case marginal
  failures (`q4_1 m=16 n=1 k=4096` 0.000665, `q4_1 k=256` 0.000665, `q5_0 k=32` 0.000658 — all against
  the 0.000500 threshold), `-o MUL_MAT_ID` 931/931 on both.
* The tester's data is **random per run** (`init_tensor_uniform` seeds from `std::random_device` unless
  `LLAMA_TEST_SEED` is set), so a case sitting at the tolerance flips with the draw.  Proof that this is
  not the change: with `LLAMA_TEST_SEED=41` the `q4_1 m=16 n=1 k=256` case fails with
  **ERR = 0.000511759 on all four builds** (base, pt213, pt218, fold); with seeds 1-3 the full suite is
  2445/2445 on base *and* fold; 40 random-data runs of the case alone are clean on both.
  **Use `LLAMA_TEST_SEED` for any build-vs-build `test-backend-ops` comparison** — it exists for this.

## 6. Is PR #218 part 2 reachable here? (probe, temporary, not shipped)

Inserted next to each `mmvq` key walk (behind `Q8_1_KEY_PROBE=1`), then run over configs `b c d e f g i`:

```cpp
const ggml_tensor * r = src1; while (r->view_src) r = r->view_src;
if (r != src1_key) fprintf(stderr, "Q8_1_KEY_PROBE %s: key=%s(root=%s) ...\n", site, ...);
```

Result: **0 offset-view keys** in every run, e.g. `0 / 14336`, `0 / 60416`, `0 / 29568` walks (the count
is per (op, graph) and scales with tokens).  So on this model set the cache identity was already the root
in every case, which is why the fix is a no-op here and why the reporter's fault needs their topology.

## 7. PR #213 is codegen-neutral on our toolchain

`objdump -d --disassemble=<mangled symbol>` of `ggml_backend_meta_get_split_state` in
`libggml-base.so.0.27.0`:

```
base   1272 instructions, 3 memcpy calls, 0 rep movs
pt213  1271 instructions, 3 memcpy calls, 0 rep movs
```

Compiled standalone from the same TU with ROCm 7.14.1's clang 23 (`-O3`): base 1404 lines, patched 1403
lines of disassembly, no `rep movs` in either.  `perf record -e cpu-clock` over short follow-up turns on a
Flash-Next `-sm tensor` server: the function's self time moves 1.63-1.92 % -> 1.56-1.60 % across three
pairs (noise; the symbol is ~1.7 % of *all* process samples, decode included).

## 8. Per-block build and the delivery gates

```
scripts/gate-per-block-build.sh --changed-from 8a449543b   -> 5/5 built (blocks 06, 08, 09, 10 + the tip)
scripts/validate-set.sh                                    -> strict 14/14 on a fresh tarball, tree == release.json.tree
scripts/check-gpu-faults.sh --since "2 hours ago" --log …   -> clean (journal + 6 run logs)
```
