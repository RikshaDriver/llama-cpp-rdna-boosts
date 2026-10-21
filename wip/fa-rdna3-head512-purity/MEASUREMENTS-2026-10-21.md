# MEASUREMENTS 2026-10-21 — fingon (gfx1100), r102 reference + reproduction

Session 1.  Box `fingon`, 1x RX 7900 XTX (gfx1100, 24 GiB), ROCm 10.1.0 at
`/opt/rocm-10.1.0-gfx1100`, one GPU job at a time, `HIP_VISIBLE_DEVICES=0`,
**`ROCBLAS_USE_HIPBLASLT=0` on every run** (issue #67).

## 0. Trees and files

| what | value |
|---|---|
| reference tree | `~/llama-kld` = git worktree of `~/llama.cpp` @ `fe932de4d`, tree **`42679a04ff814600483326a05353d272227e186e`** = `release.json.tip/tree` for **v12-097cf33d9-r102** |
| build | `BUILD_DIR=build-kld ~/bin/build-llama-rocm` (ROCm 10.1.0, full ggml-hip rebuild, ~7 min from a 30 GB ccache) |
| model | `/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (30 layers: 5 global head-512/GQA-8, 25 SWA head-256) |
| corpus | `/llm/models/wikitext-2-raw/wiki.test.raw` |
| base file A (fully tile) | `/home/stew675/b-fulltile.kld`, sha256 **`d871235cd52f7b1c90cfe171e6e30280aaef2722fc496a19a6f1b7d9406d7439`**, 5347901140 bytes — **verified** before use |
| base file B (delivery minus the 512 rows) | `/home/stew675/b-r102-256only.kld`, 5347901140 bytes, recorded this session with `GGML_CUDA_FA_RDNA3_WMMA_512=0` (`--kl-divergence-base` only) |

Housekeeping: `~/llama-fa512` worktree removed (superseded by r102), truncated
`/tmp/b-tile-ub256.kld` deleted, `ccache --max-size=30G` (it was at 20 GB).

## 1. Base-file integrity (do this first — it decides the arm definitions)

`GGML_CUDA_FA_WMMA_256=0` (= **fully tile**, the arm base file A was recorded with) compared against
base file A:

```
Mean KLD: -0.000000 ± 0.000000      Same top p: 100.000 ± 0.000 %
```

so base file A is exactly this build's fully-tile logits: it is reusable and it is *reproducible*.

**`--kl-divergence-base FILE` alone writes the file; the compare run needs `--kl-divergence` too** (trap
confirmed).

## 2. Arm definitions and the reproduction

Commands (each in a fresh process; every arm is the *same binary*, switched by env):

```bash
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1100/lib ROCBLAS_USE_HIPBLASLT=0 HIP_VISIBLE_DEVICES=0
M=/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
B=/home/stew675/b-fulltile.kld ; B2=/home/stew675/b-r102-256only.kld
C=/llm/models/wikitext-2-raw/wiki.test.raw

# throughput (q8_0 K/V)
./build-kld/bin/llama-bench -m $M -p 2048 -n 0 -d 32768 -r 2 -fa on -ctk q8_0 -ctv q8_0 -t 8
GGML_CUDA_FA_RDNA3_WMMA_512=0 ./build-kld/bin/llama-bench -m $M -p 2048 -n 0 -d 32768 -r 2 -fa on -ctk q8_0 -ctv q8_0 -t 8

# prefill logits, gate protocol (f16 K/V, 40 x 512)
./build-kld/bin/llama-perplexity -m $M -f $C -c 512 --chunks 40 -ngl 99 -fa on -t 8 \
    --kl-divergence --kl-divergence-base $B

# PPL, plain run (see the trap in §3)
./build-kld/bin/llama-perplexity -m $M -f $C -c 512 --chunks 40 -ngl 99 -fa on -t 8
```

| arm | env | pp2048 @ d32768 (t/s) | KLD vs A (fully tile) | top-p vs A | KLD vs B (256-only) | top-p vs B |
|---|---|---|---|---|---|---|
| **fully tile** | `GGML_CUDA_FA_WMMA_256=0` | — | **-0.000000 ± 0.000000** | **100.000 ± 0.000 %** | — | — |
| **head-256 WMMA only** (= delivery minus the 512 rows) | `GGML_CUDA_FA_RDNA3_WMMA_512=0` | 1913.30 ± 17.59 | 0.044055 ± 0.001500 | 90.627 ± 0.289 % | reference | reference |
| **shipped default (r102: 256 + 512)** | *(none)* | **2570.07 ± 27.14** | 0.046179 ± 0.001437 | 90.422 ± 0.291 % | **0.029285 ± 0.001000** | **92.588 ± 0.259 %** |

Throughput: **1913.30 -> 2570.07 t/s = +34.3 %** (rr 2 rounds each, `±` is llama-bench's own sd).

PPL (f16 K/V, `-c 512 --chunks 40`, wikitext-2):

| arm | PPL |
|---|---|
| fully tile (plain run) | 607.3325 ± 31.8855 |
| head-256 WMMA only (`Mean PPL(Q)` of the KLD run) | 604.0583 ± 31.7047 |
| shipped default (`Mean PPL(Q)` of the KLD run) | 607.1974 ± 31.8960 |

All three are inside each other's ±5 % error bar: **PPL does not resolve anything at this point.**

## 3. Findings from the reproduction (reported to the maintainer before any code change)

**F1 — the documented `0.0293 / 92.59 %` is a *marginal* number, not a fully-tile number.**
Against base file A (fully tile) the shipped default measures **0.0462 / 90.42 %**, and the delivery
without the 512 rows already measures **0.0441 / 90.63 %** (which reproduces the §46 "head-256 WMMA vs
fully tile = 0.0441 / 90.63 %" row to the last digit). The `0.0293 / 92.59 %` in `GREEDY-PURITY.md` §46,
`TODO.md` #81 and the r102 `WORKLOG` entry reproduces **exactly** as the shipped default against **base
file B** (the delivery with the 512 rows switched off) — i.e. it is the *marginal* cost of the head-512
rows, and `WORKLOG` 261021.004 labels it "head-512 WMMA vs tile".

Consequence for the mission: **outcome A's criterion "mean KLD <= 0.005 against the base" cannot be met
against base file A by *any* head-512 row set**, because the head-256 rows (shipped default-on since r5,
out of scope here) already cost 0.0441 against it. Measured against base file B, a passing variant is
decidable (it means: "adding the 512 rows moves the logits by <= 0.005 / >= 98 %"). Both instruments are
recorded for every variant below.

**F2 — `Mean PPL(base)` in `--kl-divergence` mode is not comparable to `Mean PPL(Q)`.** The base arm's
log-probs are stored as `uint16` (`tools/perplexity/perplexity.cpp`), so `PPL(base)` is quantized: the
identical arm reads 543.07 as base and 607.33 as Q while the KLD is exactly 0. Use **plain** runs for
PPL, or only `Mean PPL(Q)` across arms. (The §46 "f16 604.06 vs 607.20" pair is a `Mean PPL(Q)` pair and
is consistent with the table above.)

**F3 — against the CPU reference the WMMA head-512 kernel is the *more* accurate one.** New wip
instrument: `FA_DBG_NMSE=1 tests/test-backend-ops.cpp` prints every case's error magnitude (the
`nmse()` helper, GPU vs the CPU backend) instead of only pass/fail. Head-512 prefill-shaped cases,
one binary, arms switched by env:

| case (hsk=hsv=512, nr23=[8,1] = GQA 8) | tile NMSE | WMMA NMSE | threshold |
|---|---|---|---|
| kv=4096, nb=24, q8_0 | 7.515e-06 | **2.815e-06** | 5e-4 |
| kv=4096, nb=64, f16, n_kv_max=512 | 1.286e-05 | **1.153e-05** | 5e-4 |
| kv=4096, nb=64, f16, n_kv_max=2048 | 5.323e-05 | **4.520e-05** | 5e-4 |

Every case passes in both arms (6391/6391 unchanged), and the WMMA kernel is 1.2-2.7x closer to the CPU
reference. The 0.0293 therefore measures *how far the two kernels are apart*, not that the 512 rows are
worse — the direction of the change is toward the truth, which is the premise of outcome B.

## 4. Next (decomposition, one dimension at a time, measured on both bases)

Rows (`ggml_cuda_fattn_mma_get_config_rdna`, head-512 block, prefill uses `ncols=32` = ncols1 4 x ncols2 8
because AMD caps heads > 256 at 32 columns).  Current shipped row: `(nthreads 128, occ 1, nbatch_fa 32,
nbatch_K2 64, nbatch_V2 64, nbatch_combine 64, nstages 1, Q_in_reg false)`; the tile kernel's head-512
row at ncols 32 is `(256 threads, occ 2, nbatch_fa 128, nbatch_K 64)`.

| variant | change vs shipped | LDS (bytes) | purpose |
|---|---|---|---|
| V0 | — | 42,304 | reference (already measured above) |
| V1 | `nbatch_fa` 32 -> 64 | 51,264 | fewer fp16 accumulator rescales |
| V2 | `nbatch_fa` 32 -> 96 | 60,224 | ditto, at the LDS limit for `Q_in_reg=false` |
| V3 | `nbatch_fa` 32 -> 128 + `Q_in_reg=true` (needs it: 69,184 B otherwise, gfx11 LDS = 64 KiB) | 35,904 | **the tile kernel's strip length** |
| V4 | `Q_in_reg=true`, `nbatch_fa` 32 | 33,280 | isolates the price of putting Q back in registers (the "is the win really the spills" check) |
| V5 | `nthreads` 128 -> 256 | | the tile kernel's block size |

If V3 does not reach `<= 0.005` on base B, the residual is structural: on gfx11 the PV accumulator
(`T_C_VKQ`) is **fp16** (`tile<16,16,half2>`, `mma.cuh` `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32`)
and is rescaled in fp16 once per `nbatch_fa` block, while the tile kernel accumulates VKQ in **fp32** —
`nbatch_fa` only changes *how many* fp16 rescales happen, never that they are fp16. That is what the
next round then tests as a candidate: `T_C_VKQ = tile<16,16,float, DATA_LAYOUT_I_MAJOR>` for the 512
rows only (the float path already exists and is live for the RDNA3 head-80/112 rows), which costs
accumulator registers but no extra memory traffic.
