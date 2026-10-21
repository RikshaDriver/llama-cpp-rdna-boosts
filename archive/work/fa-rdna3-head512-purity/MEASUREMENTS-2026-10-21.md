# MEASUREMENTS 2026-10-21 — fingon (gfx1100), head-512 WMMA FA purity

Session 1 (the whole campaign).  Box `fingon`, 1x RX 7900 XTX (gfx1100, 24 GiB), ROCm 10.1.0 at
`/opt/rocm-10.1.0-gfx1100`, **one GPU job at a time**, `HIP_VISIBLE_DEVICES=0`,
**`ROCBLAS_USE_HIPBLASLT=0` on every run** (issue #67).

## 0. Trees, files, housekeeping

| what | value |
|---|---|
| reference tree | `~/llama-kld` = git worktree of `~/llama.cpp` @ `fe932de4d`, tree **`42679a04ff814600483326a05353d272227e186e`** = `release.json` tip/tree for **v12-097cf33d9-r102** |
| build | `BUILD_DIR=build-kld ~/bin/build-llama-rocm` (ROCm 10.1.0). A full ggml-hip rebuild is ~7 min cold, **~4.2 min** per kernel-variant edit with a warm 30 GB ccache |
| model | `/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` — 30 layers: 5 global (head 512, GQA 8) + 25 SWA (head 256). It is a **reasoning** model |
| corpus (default point) | `/llm/models/wikitext-2-raw/wiki.test.raw` |
| corpus (in-domain point) | `prompts/fa-family-switch-gemma4-26B-A4B.txt`, sha256 `f698f20a8171e9daecfcc40938868f2abf35f0ce06661167ef5a7ea9faf74724`, 64864 B |

Housekeeping done: `~/llama-fa512` worktree removed (superseded by r102), the truncated
`/tmp/b-tile-ub256.kld` deleted, `ccache --max-size=30G` (it was at 20 GB).

Base files (all in `$HOME`, `$HOME` has ~480 GB free; a base is 3.7-5.3 GB):

| file | arm recorded | point | sha256 |
|---|---|---|---|
| `b-fulltile.kld` | fully tile (`GGML_CUDA_FA_WMMA_256=0`) | wikitext, `-c 512 --chunks 40` | `d871235cd52f7b1c90cfe171e6e30280aaef2722fc496a19a6f1b7d9406d7439` (verified) |
| `b-r102-256only.kld` | `GGML_CUDA_FA_RDNA3_WMMA_512=0` | ditto | recorded this session |
| `b-tile-ub256.kld` | fully tile, `-ub 256` | ditto | (control) |
| `s-A-fulltile.kld`, `s-B-256only.kld`, `s-C-tile-ub256.kld` | as above | self-continuation `-c 512 --chunks 30` | (control) |
| `c-A-fulltile.kld`, `c-B-256only.kld`, `c-C-tile-ub256.kld` | as above | `prompts/fa-family-switch-…`, `-c 2048 --chunks 10` | (control) |
| `~/.cache/rdna-boosts/gates/…-familyswitch.kld` (+`-control-ub256`) | via `scripts/gate-prefill-logits.sh --record --family-switch` | `-c 2048 --chunks 10` | in the `.meta` sidecars |

**Base-file integrity was verified first**: the fully-tile arm compared against `b-fulltile.kld` reads
`-0.000000 / 100.000 %`, so the reusable base is exact and the instrument is deterministic.  Two traps
confirmed: `--kl-divergence-base` alone *writes* the file (add `--kl-divergence` to compare), and
`Mean PPL(base)` in KL mode is computed from `uint16`-stored log-probs, so it is **not** comparable to
`Mean PPL(Q)` (identical arm reads 543.07 vs 607.33 at KLD 0) — use plain runs for PPL, or only compare
`Mean PPL(Q)` across arms.

## 1. Reproduction (nothing changed) — the three numbers, and what the base file actually is

```bash
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1100/lib ROCBLAS_USE_HIPBLASLT=0 HIP_VISIBLE_DEVICES=0
M=/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
C=/llm/models/wikitext-2-raw/wiki.test.raw
./build-kld/bin/llama-bench -m $M -p 2048 -n 0 -d 32768 -r 2 -fa on -ctk q8_0 -ctv q8_0 -t 8
GGML_CUDA_FA_RDNA3_WMMA_512=0 ./build-kld/bin/llama-bench -m $M -p 2048 -n 0 -d 32768 -r 2 -fa on -ctk q8_0 -ctv q8_0 -t 8
./build-kld/bin/llama-perplexity -m $M -f $C -c 512 --chunks 40 -ngl 99 -fa on -t 8 \
    --kl-divergence --kl-divergence-base /home/stew675/b-fulltile.kld
./build-kld/bin/llama-perplexity -m $M -f $C -c 512 --chunks 40 -ngl 99 -fa on -t 8         # plain PPL
```

| arm | env | pp2048 @ d32768 | KLD vs fully tile | top-p | KLD vs 256-only | top-p |
|---|---|---|---|---|---|---|
| fully tile | `GGML_CUDA_FA_WMMA_256=0` | — | **-0.000000 ± 0.000000** | 100.000 % | — | — |
| head-256 WMMA only | `GGML_CUDA_FA_RDNA3_WMMA_512=0` | 1913.30 ± 17.59 | 0.044055 ± 0.001500 | 90.627 % | reference | |
| shipped default (256+512) | *(none)* | **2570.07 ± 27.14** | 0.046179 ± 0.001437 | 90.422 % | **0.029285 ± 0.001000** | **92.588 %** |

Throughput **+34.3 %** (1913.30 -> 2570.07).  tg128 @ d32768: 127.37 ± 1.93 (default) vs 127.19 ± 1.91
(512 rows off) — unchanged, decode still takes the tile kernel.

PPL (f16 K/V, plain run / `Mean PPL(Q)`): fully tile 607.3325 ± 31.8855, head-256-only 604.0583 ±
31.7047, default 607.1974 ± 31.8960 — all inside each other's ±5 %: PPL resolves nothing at this point.

**Findings (F1-F3, reported to the maintainer before any code change):**

* **F1 — `0.0293 / 92.59 %` is the *marginal* number, not a fully-tile one.**  The shipped default
  versus **fully tile** is `0.0462 / 90.42 %`; the delivery without the 512 rows already measures
  `0.0441 / 90.63 %` against fully tile (which reproduces the §46 row "head-256 WMMA vs fully tile" to
  the last digit).  `0.0293 / 92.588 %` reproduces **exactly** as default-vs-256-only.  `WORKLOG`
  261021.004 labels the marginal pair "head-512 WMMA vs tile".  Consequence: outcome A's "KLD ≤ 0.005
  against the base" is unreachable against a fully-tile base by *any* head-512 row set.
* **F2 — `Mean PPL(base)` is not comparable to `Mean PPL(Q)`** (see the trap above).  Use plain runs.
* **F3 — the point, not the kernel, is where the variance is.**  Measured control: fully tile at
  `-ub 512` versus fully tile at `-ub 256` — the same kernel, a pure scheduling difference —
  **0.104989 / 85.000 %** at the wikitext point.  A point where a benign ubatch change is 21x the KLD
  limit and 13 points below the top-p floor cannot decide a kernel question.

## 2. The instrument: three measurement points on the same model

The model is a reasoning model, so raw wikitext is far outside its confident range.  Two in-domain points
were built from the model's own greedy continuations (six task prompts — a London-theatre essay, a CPU
execution explainer, heirloom tomatoes, the Industrial Revolution, Linux scheduling, the Roman Republic —
`--temp 0 --seed 42`, `-n 4000` each, chat-templated, concatenated; the resulting corpus is shipped as
`prompts/fa-family-switch-gemma4-26B-A4B.txt`).

| point | base PPL | fully-tile self-test | **control: same kernel, `-ub 512` vs `-ub 256`** | head-256 rows vs fully tile | shipped default vs fully tile | **head-512 rows, marginal** |
|---|---:|---|---|---|---|---|
| wikitext `-c 512 --chunks 40` | 607 | 0.000000 / 100 % | **0.104989 / 85.000 %** | 0.044055 / 90.627 % | 0.046179 / 90.422 % | 0.029285 / 92.588 % |
| self-continuation `-c 512 --chunks 30` | 13.65 | — | 0.036654 / 93.660 % | 0.022562 / 96.000 % | 0.021852 / 95.869 % | 0.011303 / 97.451 % |
| `prompts/fa-family-switch-…`, `-c 2048 --chunks 10` | **2.65** | 0.000000 / 100 % | 0.013962 / 97.221 % | 0.005994 / 98.324 % | 0.006150 / 98.227 % | **0.001778 / 98.967 %** |

PPL parity at the calibrated point: fully tile 2.6530 ± 0.0685, 256-only 2.6460, default 2.6515 — within
0.3 %, i.e. identical within the error bar.

**Result: at a point inside the gate's calibrated domain the head-512 rows pass the absolute limits**
(KLD 0.001778 ≤ 0.005, top-p 98.967 % ≥ 98 %), and are 8x more faithful than the same-kernel control.
At every point they are *more* faithful than the head-256 rows shipped since r5 (those are the bigger
*total* contributor only because there are 25 head-256 layers against 5).

## 3. The decomposition — every variant costed at four bases

One dimension at a time on the row prefill actually uses (`ncols == 32`, i.e. ncols1 4 x ncols2 8; AMD
caps heads > 256 at 32 columns).  Path: `archive/work/fa-rdna3-head512-purity/{sweep.sh,measure2.sh}`, each variant
saved in `variants/`.  All in the same binary family, `llama-bench` r=2, one job at a time.

| arm | row (nthreads, occ, nbatch_fa, K2, V2, combine, stages, Q_in_reg) | pp2048 @ d32768 | wiki vs tile | wiki marginal | conf vs tile | conf marginal |
|---|---|---:|---|---|---|---|
| **V0 shipped** | `128,1,32,64,64,64,1,false` | **2570.07 ± 27.14** | 0.046179 / 90.422 % | 0.029285 / 92.588 % | 0.022917 / 95.686 % | 0.008717 / 97.425 % |
| V1 `nbatch_fa` 64 | `128,1,64,64,64,64,1,false` | 2077.58 ± 7.36 (**-19 %**) | 0.048238 / 90.157 % | 0.030865 / 92.412 % | 0.021085 / 95.974 % | 0.012032 / 97.268 % |
| V3 `nbatch_fa` 128 + QIR (= the tile kernel's strip length) | `128,1,128,64,64,64,1,true` | 2219.54 ± 6.40 (-14 %) | 0.046708 / 90.392 % | 0.029081 / 92.843 % | 0.021701 / 95.856 % | 0.009872 / 97.412 % |
| V4 `Q_in_reg=true` only | `128,1,32,64,64,64,1,true` | 1814.14 ± 7.62 (**-29 %**) | **0.046179 / 90.422 %** | **0.029285 / 92.588 %** | **0.022917 / 95.686 %** | **0.008717 / 97.425 %** |
| V5 `nthreads` 256 | `256,1,32,64,64,64,1,false` | — | **does not compile** (static asserts: `nbatch_fa % stride`, `zero-length array`) | | | |
| E1 fp32 PV accumulator (`T_C_VKQ = tile<16,16,float>`) | row unchanged | 1740.38 ± 6.98 (**-32 %**, below the tile arm) | 0.045722 / 90.245 % | 0.030489 / 92.382 % | 0.021852 / 95.869 % | 0.011303 / 97.451 % |
| V2 `nbatch_fa` 96 | — | not run: the axis is bracketed by V1 (64) and V3 (128) and both move the number by ≤ 6 % |

`conf` = the self-continuation `-c 512` point (base PPL 13.7) for the sweep arms, since the
`prompts/fa-family-switch` point was introduced later in the session.

**What the table says.**

* `Q_in_reg=true` is **bit-identical** to the shipped row — the KLD matches to every printed digit — at
  **-29 %** throughput.  The PR #199 win really is the VGPR-spill fix, and it costs nothing in accuracy.
* No `nbatch_fa` value moves the divergence beyond ±6 % (and the cadence is *not* the cause even in
  principle: the fp32 accumulator eliminated the rescale-precision hypothesis and changed nothing).
* The tile kernel's block size (`nthreads` 256) is not instantiable with these tiles.
* The fp32 PV accumulator — the one *structural* difference from the tile kernel, which accumulates VKQ
  in fp32 — buys nothing and violates the "not below the feature-disabled arm" floor (1740 < 1913 t/s).

## 4. Op-level accuracy reference (deterministic)

New wip instrument in `variants/T-oracle-instrument.patch`: `FA_DBG_NMSE=1` makes
`test-backend-ops` print the NMSE it already computes against the CPU backend for every case, and
`FA_DBG_SEED=1` pins the random input so two arms are comparable (**without it the data is redrawn per
run and NMSE moves by up to 13x — comparing unseeded runs is meaningless**).  Two head-512 prefill-shape
cases were added (`hsk=hsv=512, nh=2, nr23=[8,1], kv=32768, nb=512`, f16 and q8_0).

| shape | tile NMSE | WMMA NMSE | threshold |
|---|---|---|---|
| kv=4096, nb=24, q8_0 | 6.689e-06 | **2.413e-06** | 5e-4 |
| kv=4096, nb=64, f16, n_kv_max=512 | 1.295e-05 | **1.170e-05** | 5e-4 |
| kv=4096, nb=64, f16, n_kv_max=2048 | 5.123e-05 | **4.574e-05** | 5e-4 |
| **kv=32768, nb=512, f16 (the model's own prefill shape)** | **5.713e-05** | 5.986e-05 | 5e-4 |
| **kv=32768, nb=512, q8_0** | **1.639e-05** | 2.301e-05 | 5e-4 |

So the two families are within a factor 1.3 of each other at the model's shape (each wins some shapes),
both ~10x under the threshold: the switch is not an accuracy regression, and neither kernel is "the
wrong one".  This also refutes the natural first hypothesis — that the divergence is the WMMA kernel being
less accurate — and it is the deterministic leg of the family-switch criterion.

## 5. Acceptance on the shipped tree (nothing changed)

| check | result |
|---|---|
| `test-backend-ops -o FLASH_ATTN_EXT` | **6389/6389, 0 failed** (the r102 ledger says 6391; PR #199's own verification on r101 says 6389 — a 2-case count discrepancy in the ledger, not a failure) |
| `GGML_CUDA_FA_RDNA3_WMMA_512=0` | text identical to r102 by construction — **no shipped kernel was modified** in this campaign; the only library change ever built was the wip variants (all reverted), plus the latent one-line `VKQ_C` sizing fix in `variants/W0-…patch`, which is an exact no-op for every fp16 C tile the delivery instantiates |
| tg128 @ d32768 | 127.37 (default) vs 127.19 (512 off) — unchanged |
| gfx1201 / gfx1151 | not testable on this box; nothing in the delivery changed, and the sizing fix is a compile-time no-op for fp16 tiles (all gfx1201/gfx1151 rows) |
| prefill-logit, family-switch mode | **PASS: 0.001778 / 98.967 %** (control 0.012049 / 97.333 %, point PPL 2.646) |

## 6. Reproduce

```bash
# the calibrated point, end to end (records both bases, then compares)
scripts/gate-prefill-logits.sh --record --family-switch --ab "GGML_CUDA_FA_RDNA3_WMMA_512=0" \
    --model /llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
    --corpus prompts/fa-family-switch-gemma4-26B-A4B.txt --ctx 2048 --chunks 10
scripts/gate-prefill-logits.sh --family-switch \
    --model /llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
    --corpus prompts/fa-family-switch-gemma4-26B-A4B.txt --ctx 2048 --chunks 10

# a row variant, costed
wip/…/sweep.sh V1-fa64 "128, 1, 64, 64, 64, 64, 1, false"    # apply, build, measure, revert (moved to archive)
```
