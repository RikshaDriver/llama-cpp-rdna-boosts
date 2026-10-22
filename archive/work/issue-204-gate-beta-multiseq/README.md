# issue #204 — the GDN gate/beta BF16 fusion was wrong for every sequence but the first

**Status: closed in `v13-097cf33d9-r108`** (block-10 amendment).  Doctrine: `GREEDY-PURITY.md` §49.  Shipped
regression gate: `tests/test-fusion-gate-beta-multiseq.cpp`.  Code: `r108-mmvf-dst-flat.patch` (the
`ggml/src/ggml-cuda/mmvf.cu` half; the test file is in the delivery tree).

Reported by **@briansp2020** in <https://github.com/stew675/llama-cpp-rdna-boosts/issues/204> with the one-line
fix, after it shipped in r102 (the r102 promotion of PR #200).

## The defect

`ggml_cuda_gate_beta_bf16_try_fuse` (`ggml/src/ggml-cuda/mmvf.cu`) collapses the GDN chain

```
alpha = MUL_MAT(alpha_w, y); a3 = RESHAPE3D(alpha, nr, n_seq_tokens, n_seqs); bias = a3 + dt;
sp = SOFTPLUS(bias); gate = sp * ssm_a; gate4 = RESHAPE4D(gate, 1, nr, n_seq_tokens, n_seqs)
beta = MUL_MAT(beta_w, y); beta4 = RESHAPE4D(beta, 1, nr, n_seq_tokens, n_seqs)
```

into one `mul_mat_vec_f` launch (`gb` rows: alpha rows through the gate chain, then beta rows).  The launch takes
its **column count from `dst->ne[1]`** and its **channel count from `dst->ne[2]`**, and it was passed the gate
`MUL` as `dst`:

| ubatch | gate MUL (dst) | `y` | launch |
|---|---|---|---|
| 1 sequence, n tokens | `[nr, n, 1]` | `[ne10, n]` | n cols, 1 channel — correct |
| 2 sequences, 1 token each | `[nr, 1, 2]` | `[ne10, 2]` | 1 col, 2 channels — **wrong** |
| 4 sequences, 2 tokens each | `[nr, 2, 4]` | `[ne10, 8]` | 2 cols, 4 channels — **wrong** |

The kernel maps a channel to the input with `channel_y = fastmodulo(channel_dst, nchannels_y)`, and `y` has one
channel, so **every** channel re-read `y`'s first `n_seq_tokens` columns.  The result: each sequence past the
first got sequence 0's alpha/beta gate values, and the beta rows were computed `n_seqs` times.

Note this is *not* an out-of-bounds read of `y` and the writes stay inside `gate_mul`/`beta_w` (both have
`n_seqs × n_seq_tokens` columns) — the failure mode is wrong values, not a crash, which is why the text stays
coherent.

## Why no gate saw it

* single-sequence decode/verify has identical shapes → the fusion was bit-identical there;
* the wrong gates are plausible decays, so the text stays fluent;
* it needs BF16 `ssm_alpha`/`ssm_beta` tensors — a BF16-tensor GGUF (Qwen3.8-27B-BF16, Flash-Next GSQ).  Q8_0
  models take the separate `GGML_CUDA_FUSE_GATE_BETA_VERIFY` arm (correct: it sizes itself from
  `ggml_nelements(dst) == nrows*ncols` and asserts it), and F32 gate tensors take neither.

**Exposure band is per projection** (found here, not in the report): the fusion only runs where the plain
matmul would be an MMVF, and `ggml_cuda_should_use_mmvf` caps BF16 at `ne11 <= 3` once `should_use_mmf` is
applicable.  A 48-row projection (27B, Flash-Next: `num_v_heads = 48`) keeps the whole 8-token band on gfx1201,
so 4 concurrent requests × 2 tokens is exposed on our box; a 32-row projection (Qwen3.5-4B) only reaches 3.

## The fix

```diff
-    ggml_cuda_mul_mat_vec_f(ctx, wa, y, nullptr, gate_mul, nullptr, &gb);
+    ggml_tensor dst_flat = *alpha_w;
+    dst_flat.data = gate_mul->data;
+    ggml_cuda_mul_mat_vec_f(ctx, wa, y, nullptr, &dst_flat, nullptr, &gb);
```

The projection's shape is the flat `[nr, n_tokens]`, which is the same memory (both tensors are contiguous with
the same element count — the matcher already checks both) — so the fused launch writes the flat layout the
`RESHAPE4D` consumer reads.  A single-sequence launch is byte-for-byte identical (ne/nb/data/op_params), so
single-stream numerics cannot move.

## Reproduction / evidence tools

`tools/` holds the two instruments used; both were written for this issue.

* `multiseq_probe.cpp` (+ `build_multiseq_probe.sh`) — decodes N distinct sequences in **lockstep** (one token
  per sequence in a single batch, so the graph gets a real multi-sequence ubatch) and prints the greedy token,
  argmax and an FNV-1a hash of the whole logit row per step per sequence.  Nothing else in the process varies,
  so two runs of the same arm are bit-identical and two arms (env switch on/off, or two builds) compare exactly.
  `SPLIT=tensor|layer`, `HIP_VISIBLE_DEVICES` select the model placement.
  `tools/multiseq_probe model.gguf <n_seqs> <n_steps>`.
* `sweep_fusions.sh` — the fusion-family sweep: for each switch, run the probe with it on and off, compare the
  logits **and** the fusion-skip signature (a changed signature proves the arm fired, so "identical" is not
  vacuous).  Needs a build with the temporary `GGML_CUDA_SKIP_DEBUG` print in
  `ggml_cuda_graph_compute` (removed from the delivery; re-add it, or read the arm's own switch effect).

Dummy models for the qwen4exp-only arms: `test-llama-archs -a qwen4exp -o DIR` and `-a qwen35 -o DIR`.

## Results

**Lockstep probe** (gfx1201; `ROCBLAS_USE_HIPBLASLT=0`):

| model | placement | n_seqs | r107 | r108 |
|---|---|---|---|---|
| Qwen3.5-4B-BF16 (derived: `llama-quantize --allow-requantize … bf16`, 32-row gate) | 1 GPU layer | 1 | fused == unfused | == r107 |
| " | " | 2, 3 | **fused != unfused at every step** | fused == unfused |
| Qwen3.8-27B-BF16 (48-row gate) | 3 GPU layer | 2 | **differ** | identical |
| " | 3 GPU tensor | 3 | **differ** | identical |
| Flash-Next GSQ IQ3_XXS (qwen4exp, 48-row gate) | 3 GPU layer | 3 | **differ** | identical |

In the r107 arms sequence 0 is unaffected and sequences ≥ 1 are wrong and equal to each other — the signature of
"every channel reads channel 0".  Post-fix the fused arm equals the pre-fix **unfused** reference exactly.

**Shipped test** `test-fusion-gate-beta-multiseq`: r107 library → exit 2, 7 failing rows, `maxdiff` 0.75-1.58;
r108 → PASS, 11/11 shapes fused, 0 failed.

**Single-stream unchanged**: 4B Q8_0 / 27B Q8_0 / 35B-A3B Q8_0 `-ncmoe 40` greedy text identical between the
r107 and r108 libraries; prefill-logit A/B (27B Q8_0, base recorded on the r107 library) mean KLD **0.000000**,
same-top-p 100.000 % at `-ub 8` **and** `-ub 512`.

**Throughput** (4B BF16, 1 GPU, np concurrent slots, n_predict 256, 2 runs each):

| np | r107 | r108 |
|---|---|---|
| 1 | 64.23 / 64.20 | 64.21 / 64.16 |
| 2 | 114.57 / 114.86 | 114.36 / 114.74 |
| 3 | 165.56 / 165.55 | 165.51 / 165.50 |

Unchanged.  The reporter's 4-stream −3.5 % was dominated by the draft-acceptance loss (0.85 → 0.82 with MTP),
which is why the correctness fix is the whole of it on our box.

## Fusion-family sweep (the siblings)

Switch on vs off, multi-sequence logits compared bit-exactly, skip signature compared to prove the arm fired:

| switch | fires on | logits (r108) | logits (r107) |
|---|---|---|---|
| `GGML_CUDA_FUSE_GATE_BETA_BF16` | 4B BF16, Flash-Next | identical | **DIFFERS** |
| `GGML_CUDA_FUSE_GATE_BETA_VERIFY` (Q8_0) | 27B Q8_0 (layer+tensor) | identical | identical |
| `GGML_CUDA_FUSE_GDN_GATE` | all | identical | identical |
| `GGML_CUDA_FUSE_GDN_BETA_SIGMOID` | all | identical | identical |
| `GGML_CUDA_FUSE_GDN_CONV_NORM` | dummy qwen4exp | identical | identical |
| `GGML_CUDA_FUSE_CONT_GATE_Q8_1` | 27B Q8_0, Flash-Next | identical | identical |
| `GGML_CUDA_FUSE_GDN_NORM_GATE` | never (guard bails on `xn->ne[3] != 1`) | n/a | n/a |
| `GGML_CUDA_DISABLE_GDN_CPY` (snapshot copies) | 4B BF16 | identical | — |

So the r102 family had exactly one multi-sequence defect.  The three other PR #200 arms are element/flat-based
(`ggml_nelements` + explicit strides); the Q8_0 sibling asserts its shape contract.

## Gates run for the release

`scripts/validate-set.sh` green (13/13 strict `git am` from a fresh tarball, applied tree ==
`release.json.tree`); `scripts/gate-per-block-build.sh --changed-from f2ee094df` 2/2 (block 10 + tip; the other
11 commits are patch-identical); the new test PASS on the release tree and exit 2 on the r107 library.
`test-backend-ops` not run: no op, instantiation or dispatch predicate changed.
