# PR #206 promotion record: MTP prompt-tail readback skip (r118, 2026-10-23)

The submitter's `wip/mtp-tail-readback` patch (offered against r109) folded into **block 02** as the r118
amendment, with two changes of ours.  This file is the measurement record behind the `WORKLOG.md` 261023
entry; the decision record is `archive/work/mtp-tail-readback/RESOLUTION.md`.

Every fresh-process comparison ran with **`ROCBLAS_USE_HIPBLASLT=0`** (issue #67: hipBLASLt picks its
solution per process start, which has caused false causality here before).  One GPU job at a time.
Builds: `~/bin/build-llama-rocm` (ROCm 10.1.0), the *same build directory* for both arms, so only
`libllama.so` and `libllama-common.so` differ.

## Setup

| | |
|---|---|
| box | 3x Radeon AI PRO R9700 (gfx1201), 16-core 9950X3D; this run pinned `HIP_VISIBLE_DEVICES=0,1` |
| target | `/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (71 GiB, 2 parts) |
| drafter | `-md /llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (sidecar head) |
| server flags | `-ngl 99 -sm tensor -ncmoe 10 -c 32768 -b 2048 -ub 2048 --load-mode none --lazy-mode off -fa on -ctk f16 -ctv f16 -t 8 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 -np 1`, `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0` |
| prompt | `prompts/prose-rdna-boosts.txt` x6 = 31,476 tokens, `cache_prompt: false`, greedy, seed 42 |
| arms | r117 (base) vs r117 + the block-02 amendment, with `LLAMA_MTP_INGEST_TAIL=4096`, and the amendment with `LLAMA_MTP_TAIL_READBACK=0` |

The tail only has an effect in `llama-server` (only the server reports a prompt end), so every arm here is
a server arm.  Two requests per load; the first request of a load is not the steady state (graph capture
and expert-cache arena evictions land there), so the tables below use both requests but the per-request
spread is reported.

## Prefill (31,476-token prompt, `LLAMA_MTP_INGEST_TAIL=4096`)

| arm | request 0 | request 1 | mean |
|---|---:|---:|---:|
| r117 | 1480.04 / 1479.71 | 1496.59 / 1493.72 | **1488.3** |
| r118 (shipped) | 1498.22 / 1500.01 | 1513.54 / 1515.56 | **1506.8 (+1.24 %)** |
| r118, `LLAMA_MTP_TAIL_READBACK=0` | 1480.13 / 1482.59 | 1497.16 / 1500.11 | **1490.0 (+0.11 %)** |

An earlier series (3 alternating pairs, 6 requests per arm on the same setup) measured r117 1488.6 vs the
amendment 1509.8 t/s: **+1.42 % (95 % CI +0.92 to +1.93)**.  The submitter reported +1.9 % (CI +0.6 to
+3.2) on 2 GPU at 2180 t/s; our absolute t/s is lower because each R9700 here is behind a x4 host link,
and their figure sits inside our interval.  The kill-switch arm landing within 0.11 % of r117 is the
end-to-end proof that the switch restores the pre-r118 behaviour.

No tail (`LLAMA_MTP_INGEST_TAIL` unset): r117 1424.6 vs r118 1423.3 t/s, **-0.1 %** (noise; with every
threshold 0 the skip cannot trigger).  The tail's own prefill gain is +4.5 % on r117 and **+6.1 %** with
the patch.

## Where the gain comes from (2x2 on one diagnostic binary)

`llama_get_embeddings_nextn()` and `..._ith()` both call `ctx->synchronize()`, and r117's MTP `process()`
called the former after every prompt batch, so the host waited for both backends to go idle before it
built the next ubatch.  The patch removes that whole-path drain (the hook's early return) and the 21 MB
per 2048-row ubatch nextn copy (the target's skip).  A 2x2 with a diagnostic binary (one env-gated switch
per half, steady-state request, 2 loads per arm):

| arm | prefill t/s | vs both-off |
|---|---:|---:|
| both halves off | 1500.3 | - |
| readback skip only | 1502.3 | +0.1 % |
| hook skip only | 1509.6 | **+0.6 %** |
| both (shipped) | 1514.5 | **+0.9 %** |

The diagnostic build's both-off arm sat ~0.4 % above the r117 binary, so only the split is used: **most of
the win is the drain the hook's getter used to force; the readback copy is small on this box.**

## Correctness

300 greedy tokens after the 31.5k prompt, `LLAMA_MTP_INGEST_TAIL=4096`, 2 loads per arm (base and
amendment), plus the kill-switch arm:

| arm | text sha | draft acceptance | acc/pos | decode t/s |
|---|---|---|---|---|
| r117 | `98c4f67d6ed9` | 0.48626 (177 / 364) | 0.754 / 0.426 / 0.270 | 98.31 |
| r118 | `98c4f67d6ed9` | 0.48626 (177 / 364) | identical | 98.22 / 98.13 |
| r118, `TAIL_READBACK=0` | `98c4f67d6ed9` | 0.48626 (177 / 364) | identical | flat |

Re-run on the shipped form (one load per arm, 300 tokens): `98c4f67d6ed9` and 177 / 364 in all three arms
again, prefill 1484.71 (r117) / 1494.60 (r118) / 1471.74 (kill-switch).

Identical draft counts on every load means the draft saw exactly the same data, which is the invariant
the patch claims (an ingested row `p` reads `h(p - 1)`, and any ubatch holding a row at
`pos >= ingest_from - 1` is never skipped).

Mechanism wiring, per-ubatch debug logging on a 15,738-token prompt (8 chunks): skip on -> 5 ubatches
skipped / 10 read; `LLAMA_MTP_TAIL_READBACK=0` -> 0 skipped / 15 read.

## Gates

| check | result |
|---|---|
| prefill-logit gate (27B Q8_0, wikitext, `--chunks 40`, base recorded with the r117 binary) | mean KLD **0.000000**, same-top-p **100.000 %** (no compute change) |
| dense coherence (4B Q8_0, `-sm tensor`, 200 greedy tokens) | `ba301f021e66` for both builds |
| greedy text, every server arm above | identical (`fa78a9826423` at 8 tokens, `98c4f67d6ed9` at 300) |
| decode | flat (98.31 vs 98.22 / 98.13 t/s) |
| GPU faults | `scripts/check-gpu-faults.sh` clean |
| `scripts/validate-set.sh`, `scripts/gate-per-block-build.sh --changed-from` | green (see the WORKLOG entry) |

## Reproduce

```bash
# both arms from one build directory, so only the two libraries differ
cd ~/llama.cpp && git checkout <r117 tip> && cmake --build build-rocm --target llama-server -j 16
cp -a build-rocm/bin /tmp/bin-r117
# amend block 02, then:
cmake --build build-rocm --target llama-server -j 16 && cp -a build-rocm/bin /tmp/bin-r118

HIP_VISIBLE_DEVICES=0,1 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 ROCBLAS_USE_HIPBLASLT=0 \
LLAMA_MTP_INGEST_TAIL=4096 /tmp/bin-r118/llama-server -m <GSQ IQ3_XXS 1/2> -md <shared Q8_0 head> \
  -ngl 99 -sm tensor -ncmoe 10 -c 32768 -b 2048 -ub 2048 --load-mode none --lazy-mode off \
  -fa on -ctk f16 -ctv f16 -t 8 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 \
  -np 1 -lv 4
# then POST /completion with the x6 prose prompt, n_predict 8, temperature 0, cache_prompt false,
# and read timings.prompt_per_second (and the "draft acceptance" line in the server log).
```
