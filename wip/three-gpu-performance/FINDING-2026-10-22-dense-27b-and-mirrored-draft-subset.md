# FINDING 2026-10-22 — dense 27B at 2 vs 3 GPUs, depth-16384, and the mirrored MTP draft subset

**What this is.**  A same-day, single-build measurement set that landed while closing `TODO.md` #85 (the
`--spec-draft-mtp-vocab` validation arms; misfiled as #83 in the r107 record).  It adds two things this campaign did not have: a **dense**
2-vs-3-GPU four-axis table with a clean control, and the **first measured 3-GPU regression** attributable
to a specific mechanism (a *delivery feature*, not the all-reduce path).

- Build: `~/llama.cpp/build-rocm`, r109 tree (`933b964c7` / `8e67f977…`), ROCm 10.1.0, 3x R9700
  (gfx1201).  `ROCBLAS_USE_HIPBLASLT=0` for every run (fresh-process determinism, `GREEDY-PURITY.md` §44).
- Model: `Qwen3.8-27B-Q8_0.gguf` (vocab 248320), embedded MTP head, `-sm tensor -mg 0`, f16 KV,
  `-c 32768 -b/-ub 2048`, seed 42, temp 0, `-n 3000`.
- Speculation: `draft-mtp-adaptive,ngram-mod --spec-ngram-mod-n-match 45 --spec-draft-n-max 7`
  (**`--spec-draft-n-start` left at its default 3** — see the note at the end, it matters).
- Prompts: `prompts/{reasoning,prose-rdna-boosts,code-python,recall}.txt` (hashes in `prompts/README.md`);
  R runs `--reasoning on`, P/C/K `--reasoning off`.
- Harness + every raw number: [`tools/arms-2026-10-22/`](tools/arms-2026-10-22/) (`summary.csv` = the runs
  used here, `invalidated.csv` = the two setup mistakes kept for the record, the sweep scripts, and the
  depth-16k prompt).

## A. Four-axis, trim off — 3 GPU is *not* uniformly slower on a dense model

| axis | 2 GPU | 3 GPU | Δ (3 vs 2) | acceptance 2g → 3g | mean len 2g → 3g |
|---|---:|---:|---:|---|---|
| R (reasoning) | 68.8 | **75.1** | **+9.2 %** | 0.57160 → 0.57400 | 2.87 → 2.85 |
| P (prose) | 94.3 | **98.3** | **+4.2 %** | 0.67018 → 0.68878 | 4.72 → 4.85 |
| C (code) | 116.8 | 117.2 | +0.3 % | 0.75887 → 0.74085 | 6.03 → 5.89 |
| K (verbatim recall) | **348.2** | **298.1** | **−14.4 %** | 0.98361 → 0.98361 | **29.24 → 29.24** |
| aggregate (3000 tok/axis) | 109.4 | **113.1** | **+3.4 %** | | |

**The K axis is the clean datapoint of the whole campaign.**  Its acceptance, accepted/generated counts
(480 / 488) and mean accepted length (29.24) are **bit-identical** across 2 and 3 GPUs — ngram-mod
dominates it, so the two arms draft the same proposals over the same text.  The −14.4 % is therefore a
pure device-count effect, measured at **verify width ≈ 30**.

The shape is width-dependent, not direction-dependent:

- **small verify batches** (R, mean len 2.85; P, 4.85) — the 3rd device splits the weights and wins
  (+9.2 % / +4.2 %);
- **large verify batches** (K, mean len 29.2) — the 3-way collective is in the way and the 3rd device
  loses 14.4 %.

That is the same discriminator §5 predicts: on 3 devices the small-message path is the **host-staged
chunked kernel over the GTT wire ring** (both 2-device fast paths — copy-engine and P2P push — require
exactly 2 devices), and its cost grows with the message, while the per-token weight streaming it saves is
fixed.  On the *dense* model at 2-4 tokens drafted the arithmetic favours 3 GPUs; at 30 tokens it does not.
`TODO.md` #51's MoE finding (3-GPU *slower*) is the same mechanism at a much larger verify width, not a
contradiction of this table.

## B. Depth 16384 — 3 GPU is 8.5 % slower, and the shallow→deep penalty is 6x worse

Prompt = `prompts/prose-rdna-boosts.txt` ×3 → **15746 tokens**, `sha256 6fec9806b375…`, `-c 20480`, so
decode runs from depth ≈15746 to ≈18700 (the delivery's decode-validation depth).

| arm | 2 GPU | 3 GPU | Δ (3 vs 2) |
|---|---:|---:|---:|
| trim 0 | 92.2 | 84.4 | **−8.5 %** |
| trim 98304 | 92.3 | 83.2 | −9.9 % |
| shallow → deep (same width, trim 0) | 94.3 → 92.2 (**−2.2 %**) | 98.3 → 84.4 (**−14.1 %**) | |

At depth the 3rd GPU turns from a +4.2 % win (P, shallow) into an 8.5 % loss, while 2 GPUs give up only
2.2 %.  A deep KV widens every per-token collective payload (the attention layers read the whole cache)
and the 3-device host-staged ring is exactly what does not scale with it — a concrete, testable target for
the campaign's step-2 profile (`GGML_CUDA_AR_PROFILE=1`, and the `-sm layer` arm to separate the tensor
split from the device count).

## C. The 3-GPU regression: a **mirrored** MTP draft subset (block 13, r107)

`--spec-draft-mtp-vocab N` (opt-in) is the delivery's first feature whose sign **flips with device count**.
Same gate, same build, trim off vs on (98304, n-max 7):

| axis | 2 GPU off → on | Δ | 3 GPU off → on | Δ |
|---|---:|---:|---:|---:|
| R | 68.8 → 69.5 | +1.0 % | 75.1 → 73.8 | **−1.7 %** |
| P | 94.3 → 95.8 | +1.6 % | 98.3 → 96.0 | **−2.3 %** |
| C | 116.8 → 119.4 | +2.2 % | 117.2 → 114.1 | **−2.6 %** |
| K | 348.2 → 347.5 | −0.2 % | 298.1 → 291.9 | **−2.1 %** |
| aggregate | 109.4 → 110.9 | **+1.4 %** | 113.1 → 110.6 | **−2.2 %** |

### Mechanism, and the control that proves it

The subset head is allocated from the LM head's buffer type, so under `-sm tensor` it is **mirrored on
every device** while the untrimmed head is **split**.  The per-device draft-matmul width is therefore

```
trim off :  n_vocab / n_devices      (each device computes its vocabulary shard)
trim on  :  N                        (every device computes the whole subset)
```

so the trim is a win only when **`N · n_devices < n_vocab`** — *not* "N is smaller than the vocabulary"
(the basis the r107 record used: "still 2.5x smaller at 98304" compares against 248320, but the live
comparison on a tensor split is against the per-device share).

The width sweep on P is the control, because the 98304 and 131072 arms have **identical acceptance**
(so the workload is identical and only the matmul width differs):

| device share | trim 0 (split head) | N = 98304 | N = 131072 |
|---|---:|---:|---:|
| 2 GPU — share 124160 | 94.3 | **95.8** (+1.6 %) | 93.1 (−1.3 %) |
| 3 GPU — share 82773 | 98.3 | 96.0 (−2.3 %) | 93.3 (−5.1 %) |

Monotonic in the per-device column count in both directions, crossing over at the split share.  2 GPU
acceptance for the two trim arms: 0.67337 / mean len 4.67 for **both**; 3 GPU: 0.68827 / 4.83 for **both**.
A repeat of the 2-GPU P pair measured 94.3 → **96.1** t/s with the same acceptance (0.67018 → 0.67337) and
mean len (4.72 → 4.67), i.e. **identical text, ±0.3 % timing noise** — the deterministic-output property
holds and none of the deltas above is smaller than that noise floor.

**At 3 GPUs there is no profitable N on this payload.**  The width rule wants `N < 82773`, but 64000 costs
so much acceptance there (`0.68878 → 0.67260`, mean len `4.85 → 4.11`) that it measures **89.1 t/s
(−9.4 %)**.  The width threshold is below the acceptance threshold, so the honest 3-GPU advice for a
248320-vocab model is *leave the trim off*.  (At 2 GPUs 64000 is +0.6 % — acceptance loss nearly cancels
the width gain; 98304 remains the better cut.)

**Purity is intact at both widths and at depth** — the trim still only loses proposals:

| arm | text |
|---|---|
| 3 GPU shallow, `none` / trim 0 / trim 98304 | all `c5421068b4e0` (7465 chars) |
| 3 GPU depth ~15746, `none` / trim 0 / trim 98304 | all `12048e399854` (3561 chars) |

`scripts/check-gpu-faults.sh --since "90 min ago"` over the run logs and the kernel journal: **clean**.

## Reproducer

```bash
cd wip/three-gpu-performance/tools/arms-2026-10-22
ARMS_LOG_DIR=/tmp/arms22 ./gate4.sh        # A + C: four axes x trim {0,98304} x {2,3} GPU
ARMS_LOG_DIR=/tmp/arms22 ./arm2-depth.sh   # B: prose x3 (15746 tok), -c 20480
ARMS_LOG_DIR=/tmp/arms22 ./arm3-moe.sh     # the MoE/-ncmoe arm (35B-A3B Q8_0), see the block-13 record
```

`lib.sh` holds the single canonical command line; every run above is one invocation of it with `GPUS`,
`CTX` and the axis prompt swapped.  `summary.csv` is the set used by this finding (includes a repeat of the
2-GPU P pair, which confirms deterministic text at ±0.3 % timing noise); `invalidated.csv` keeps the two
setup mistakes (an `-n-start 7` batch and a Protocol-A-flag batch) so they are not re-run by hand.

## Note on the r107 campaign's recorded command (docs bug)

`archive/work/spec-draft-mtp-vocab/README.md` §3's preamble writes the invocation as
`--spec-draft-n-max <N> --spec-draft-n-start <N>`.  `--spec-draft-n-start` is the **initial** adaptive
depth, not a ceiling; setting it equal to n-max starts the controller at the ceiling, which reproduced as a
materially different trajectory (2-GPU P 94.3 → 87.6 t/s, acceptance 0.67018 → 0.62704).  The validated
configuration is **`--spec-draft-n-max 7` with `-n-start` left at its default 3**.  Recorded in
`archive/work/spec-draft-mtp-vocab/VALIDATION-2026-10-22.md`.
