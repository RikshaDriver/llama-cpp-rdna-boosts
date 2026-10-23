# `qwen4exp-dflash2` — DFlash2 drafting on Qwen3.8-Flash-Next: the target must export **HC-averaged** layer inputs

**Status: OPEN (seeded 2026-10-23, r120).  Scoping + one known shape blocker, nothing attempted in code yet.**
Tracked as **TODO #94**.  **Nothing here is part of the delivery**: this campaign is a plan plus the evidence
gathered while answering "is there a DFlash2 head for Flash-Next?".

One line: a DFlash2 head for Flash-Next exists, converts cleanly and its shapes line up, but the qwen4exp
graph exports the **wide hyper-connection residual** (`hc * n_embd` = 10240) as its DFlash layer input while
the head wants the **mean over the 4 HC streams** (`n_embd` = 2560 per layer, 12800 total), so attaching the
head aborts in `llama_context::extract_layer_inp()`:

```
src/llama-context.cpp:3437: GGML_ASSERT(row_floats == model.hparams.n_embd) failed
```

`src/models/deepseek4.cpp` already exports the averaged state for its own DFlash path
(`build_hc_mean(inpL)`, lines 1239-1242 / 1329-1332) and is the model for the fix.

---

## 1. Why bother (the motivation, measured)

A one-off comparison on the same target (`Qwen3.8-27B-UD-Q4_K_XL`, 2 GPU `-sm tensor`, `-c 16384`, q8_0 KV,
`-n 2000`, reasoning pinned per axis, `prompts/*`, one run per cell) at n-max 5 for both, then each at its
own ceiling (DFlash2's head is `block_size 8`, so its maximum is 7):

| arm | R | P | C | K | mean |
|---|---:|---:|---:|---:|---:|
| plain | 48.8 | 48.0 | 48.8 | 48.9 | 48.6 |
| `draft-mtp` n5 | 97.7 | 125.2 | 141.0 | 160.7 | 131.2 |
| `draft-mtp-adaptive` n-max 5 | 96.2 | 122.3 | 138.4 | 150.3 | 126.8 |
| `draft-mtp-adaptive` n-max 8 | 96.1 | 120.0 | 133.2 | 157.6 | 126.7 |
| `draft-dflash` (DFlash2) n-max 5 | 119.3 | 119.1 | 160.4 | 178.5 | **144.3** |
| `draft-dflash` (DFlash2) n-max 7 | **119.7** | 112.6 | **173.9** | **206.2** | **153.1** |

DFlash2 wins on the mean by ~20 % against the best adaptive-MTP arm, and by 24-31 % on reasoning / code /
recall.  Adaptive MTP only wins on generic prose (~8 %), where its acceptance is higher.  Raising the
*adaptive* cap from 5 to 8 changes nothing on average (the controller self-limits to a mean accepted length
of 3.1-4.9); fixed MTP n8 collapses on reasoning (69.6 t/s at acceptance 0.29).

Two costs to weigh on the Flash-Next arm: DFlash2 prefill measured up to ~8 % lower on the 27B arm (P 1402
vs 1526, C 790 vs 829 t/s; the draft context consumes the injection passes), and the head is a separate model
(362 MiB at Q4_K_M), though Flash-Next has no embedded MTP head at all, so today the alternative for it is
plain decode.

A DFlash2 draft is also **lossless by design** (the target verifies every token), so the arm can be gated on
`plain == draft-dflash` greedy text, the same instrument the MTP purity work uses.

## 2. The heads that exist (2026-10-23)

| repo | what |
|---|---|
| `incoai/Qwen3.8-Flash-Next-DFlash2` | the head, safetensors (`DFlash2DraftModel`), created 2026-10-19, Apache-2.0.  Ships `patches/qwen38-dflash2-capture.patch` (see §3.3), saved here as `evidence/incoai-qwen38-dflash2-capture.patch` |
| `incoai/Qwen3.8-Flash-Next-DFlash2-GGUF` | GGUF conversions: BF16 1.2 GB, Q8_0 652 MB, Q6_K 506 MB, Q5_K_M 432 MB, **Q4_K_M 362 MB** |
| `skeggsguy/Qwen3.8-Flash-Next-DFlash2-GGUF` | a second conversion (Q8_0) |

For the record, the same family of heads for other targets: `z-lab/Qwen3.8-27B-DFlash2-GGUF` (the head this
box already has), `aminya/Qwen3.6-35B-A3B-DFlash2-GGUF` (MoE), `incoai/Muse-Glimmer-30B-DFlash2-GGUF`,
`kaivoss/Ornith-1.5-122B-DFlash2-GGUF`, `emwesoft/GLM-5.3-DFlash2-GGUF`.  `dflash-<tag>.gguf` siblings are
picked up automatically by `--spec-type draft-dflash` in the model's own HF repo
(`find_best_dflash()`, `common/download.cpp:614`).

Local copy used so far: `/tmp/df2/head/Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf` (fetch with
`tools/fetch-head.sh`; it is a scratch path, not under `/llm/models`).

## 3. What breaks, and why

### 3.1 The metadata lines up

Head (`incoai/...-Q4_K_M.gguf`), read with `llama-gguf` / `gguf-py`:

```
dflash.block_size     = 8        -> --spec-draft-n-max 7 is the ceiling (8 would be clamped)
dflash.selector_top_k = 16       -> DFlash2 (a selector lattice, not logits + top-k); bails this out of DFlash 1
dflash.target_layers  = [6, 16, 26, 36, 46]
dflash.embedding_length = 2560   -> the head's own width
fc.weight             = [2560, 12800]   -> 5 extracts x 2560 = the target's PLAIN hidden width
selector_predecessor  = [248320, 256]   -> over the target's vocab
```

Target (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS`):

```
qwen4exp.embedding_length      = 2560
qwen4exp.hyper_connection.count = 4
qwen4exp.block_count            = 48
output.weight                   = [248320, 2560]   -> the LM head is PLAIN-width, so the selector's use of
                                                      the target's output.weight is shape-correct
```

So: vocab matches, `target_layers` are in range, `n_embd_enc = 5 x 2560 = 12800` matches `fc`, and the
selector's projection matches.  The only mismatch is the HC reduction.

### 3.2 The abort

With the head attached the target loads, sizes the expert cache (all 432 tables `identity fast path ON`),
and then:

```
src/llama-context.cpp:3437: GGML_ASSERT(row_floats == model.hparams.n_embd) failed
```

`src/models/qwen4exp.cpp:950` exports the layer input as the **wide** residual:

```cpp
ggml_tensor * res_hc = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
        n_embd, hc, n_tokens, 1);          // hc = hparams.dsv4_hc_mult = 4
...
for (int il = 0; il < n_layer; ++il) {
    res->t_layer_inp[il] = res_hc;         // 10240 wide at il = 6
```

`extract_layer_inp()` divides the exported tensor by `n_tokens` and requires `n_embd` per row
(`llama-context.cpp:3434-3437`), so 10240 != 2560.  The F1 **device** half of the same path is width
agnostic (it allocates `t->ne[0] x n_batch`, `llama-context.cpp:3376`), so it needs no change once the
export is `n_embd` wide.

### 3.3 The reference implementation settles the convention

`evidence/incoai-qwen38-dflash2-capture.patch` (shipped with the head, sglang `qwen4_exp.py`) is explicit:

```python
if i in getattr(self, "_dflash_completed_layer_ids", ()):
    # DFlash2 was trained on completed, zero-based layer outputs,
    # averaged over the four hyper-connection streams.
    assert hidden_states.shape[-1] == self.hc_count * self.hidden_size
    aux_hidden_states.append(
        hidden_states.unflatten(-1, (self.hc_count, self.hidden_size)).mean(dim=-2)
    )
...
def set_dflash_layers_to_capture(self, layer_ids):
    assert list(layer_ids) == [5, 15, 25, 35, 45], layer_ids
    print("Q38_DFLASH_CAPTURE completed_layers=[5,15,25,35,45] hc_mean=True features=12800", flush=True)
```

Two things follow:

1. **The reduction is the mean over the `hc` streams**, exactly `build_hc_mean()`.
2. **The +1 in `dflash.target_layers` is already the conversion's own shift**: the reference captures
   *completed* layer `i` (so the state **after** layer 5), and the GGUF says layer **6**.  llama.cpp's
   `t_layer_inp[il]` is the state at the **top** of the loop, i.e. the input to layer `il`, so
   `target_layers = completed + 1` is the correct mapping and needs no change.  The 27B head is consistent
   with this (`dflash.target_layers = [6, 20, 34, 48, 62]` for 64 layers, `fc = [5120, 25600]`), and that
   head drafts at 0.85-1.00 acceptance on this box today.

## 4. Proposed change (block 11 owns qwen4exp)

Mirror what `src/models/deepseek4.cpp` already does, gated exactly like it is so **nothing changes for a
context that does not ask for layer inputs** (plain decode, MTP, and every non-DFlash run):

1. Add to `llama_model_qwen4exp::graph` (same 8 lines as `deepseek4.cpp:269-278`):

   ```cpp
   ggml_tensor * build_hc_mean(ggml_tensor * x) const {
       const int64_t hc = x->ne[1];
       ggml_tensor * acc = ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], 0);
       for (int64_t s = 1; s < hc; ++s) {
           acc = ggml_add(ctx0, acc, ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
       }
       return ggml_scale(ctx0, acc, 1.0f/hc);
   }
   ```

2. In the layer loop (`qwen4exp.cpp:950`), replace the unconditional pointer assignment with the gated
   export deepseek4 uses:

   ```cpp
   if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
       res->t_layer_inp[il] = build_hc_mean(res_hc);
       cb(res->t_layer_inp[il], "layer_inp", il);
       ggml_build_forward_expand(gf, res->t_layer_inp[il]);
   }
   ```

   The gate matters: the current line is a pointer assignment that adds no node, so an unconditional
   `build_hc_mean` would add `hc-1` adds + a scale to **every** qwen4exp graph (decode, MTP, QSA, all of
   them) for a tensor nobody reads.  `cparams.embeddings_layer_inp` is set from the drafter's
   `dflash.target_layers` (`llama-context.cpp:314-318`), so it is true only while a DFlash draft is attached.

3. Nothing else: no new env var, no new device-iface entry, no graph-params field (the exported tensor is
   already `GGML_TENSOR_FLAG_OUTPUT`-ed by `llama-graph.cpp:1518-1522` when requested).

Question to settle while implementing: whether `ggml_view_2d` + `ggml_add` + `ggml_scale` is the cheapest
form on the RDNA4 path for `n_tokens` up to the ubatch (a `GGML_OP_MEAN` over `ne[1]` may exist and be
better; deepseek4 chose adds+scale, so matching it first is the conservative choice, and a fused
`hc_mean` op is a later micro-optimisation, not part of the fix).

## 5. Open questions / expected next obstacles

* **Does the head's own encoder agree with the mean?**  The reference patch says yes (§3.3), and acceptance
  is the instrument: a wrong reduction shows up as low acceptance on predictable workloads (the 27B head
  reads 0.75-0.85 on code and 1.00 on recall; a broken encoder would collapse toward the ~0.3 the
  over-drafted fixed-depth arms show).  Gate on acceptance **and** `plain == draft-dflash` text.
* **`-sm tensor` + `-ncmoe` (the slab narrow-2 arm).**  This is the r106 shape: the DFlash2 selector's
  `ggml_top_k` / `get_rows` over the target's vocabulary-split `output.weight` needs `GGML_OP_ALL_GATHER`
  (r106, TODO #78).  Expect it to work, but it is the arm to watch, together with the draft-context
  `graph_reserve` narrow-2 yield.
* **VRAM on 3x 32 GB: not a blocker (measured).**  With clean VRAM the 75 GB target plus the derived draft
  reserve (1226 MiB/device at `-c 16384`) loads fine through the normal fit path and reaches the assert
  in §3.2, 432/432 identity tables sized.  `--fit off` is **not** needed.  Every "2257 MiB cudaMalloc" seen
  while seeding this campaign was a **stray llama process** from an interrupted run holding ~33 GB/GPU
  (the failure lands at 0.01 s, before the loader has done anything), not a fit or reserve problem.  Check
  VRAM first; do not read an OOM as a DFlash2 defect.
* **`embd_state` for qwen4exp drafts.**  `gemma4-assistant` style drafters assert on a missing
  `ubatch->embd_state`; a DFlash *draft* consumes the target's layer inputs, not `embd_state`, so this is
  expected to be fine, but it is the next assert to watch for.
* **Whether the same fix is needed for `deepseek4`'s MTP export** (it already `build_hc_mean`s) or for
  qwen4exp's **MTP** head path (which does not use `t_layer_inp` at all, so it should be unaffected).

## 6. Validation plan (when the code is written)

Delivery rules apply in full (`AGENTS.md`, `MANIFESTS.md`).  The change is block 11 and is inert without a
DFlash draft, so the gates are:

1. **Non-DFlash purity first (the regression risk)**: with no draft, `cparams.embeddings_layer_inp` is all
   false, so assert the graph is byte-identical.  Cheapest proof: `GGML_CUDA_WORK_ALLOC_DEBUG=1` graph node
   counts and the same-seed coherence hash on Flash-Next GSQ (3 GPU `-sm tensor`, `-ncmoe 48`) and on
   qwen4exp MTP against r120.
2. **The arm itself**: Flash-Next GSQ + the head, `--spec-type draft-dflash --spec-draft-n-max 7
   -ncmoe 48 -sm tensor -c 16384 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 -t 8`:
   * `plain == draft-dflash` greedy text (lossless by design) on prose + code + recall,
   * draft acceptance per axis (healthy = the 27B's 0.58-1.00 band; a broken encoder collapses toward ~0.3),
   * `-sm layer` acceptance baseline against `-sm tensor` (r106's rule),
   * 2 GPU and 3 GPU (the greedy text is a per-device-count reference on the MoE tensor split,
     `AGENTS.md`; do not compare text across counts),
   * the two-slot unified server shape (`tools/two_slot_check.py` in
     `archive/work/dflash2-draft-sched2/tools/` adapted to Flash-Next) so the r82/r84 and r120 slot
     behaviour is covered,
   * `-ncmoe 48` (host experts, slab + arena + DFlash draft) as its own arm.
3. **Prefill-logit gate** (`scripts/gate-prefill-logits.sh`): a graph change on the *prefill* path is
   exactly what this gate exists for, even though the added nodes only exist when a draft is attached.
   Run it with the head attached and without.
4. **Per-block build** (`gate-per-block-build.sh --changed-from <r120 tip>`) and `validate-set.sh`.
5. **GPU faults** (`scripts/check-gpu-faults.sh`).

## 7. Repro so far (exact commands, and the trap)

```bash
# head (362 MB), scratch path
bash tools/fetch-head.sh

# target alone loads and sizes the whole cache (432/432 identity fast path ON)
HIP_VISIBLE_DEVICES=0,1,2 llama-server \
  -m /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  -ngl 99 -sm tensor -mg 0 -ncmoe 48 -c 16384 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 -t 8 -np 1

# with the head: loads (432/432 identity fast path ON), then aborts in extract_layer_inp -- the blocker
HIP_VISIBLE_DEVICES=0,1,2 llama-server \
  -m <the GSQ shard 1> -md /tmp/df2/head/Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  -ngl 99 -sm tensor -mg 0 -ncmoe 48 -c 16384 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 -t 8 -np 1
```

```
0.19.6xx W alloc_table_locked: ... (432 tables) identity fast path ON
/home/stew675/llama.cpp/src/llama-context.cpp:3437: GGML_ASSERT(row_floats == model.hparams.n_embd) failed
```

One trap, hit repeatedly while gathering the evidence: **check for stray processes before believing an
OOM.**  A leftover server from an interrupted run holds ~33 GB per GPU, and the next run then fails with a
2257 MiB `cudaMalloc` at 0.01 s (before the loader does anything), which reads exactly like a fit problem.
`pgrep -a llama` (ignore `llama-slot-proxy`) and `rocm-smi --showmeminfo vram` must show the ~60 MB
baseline.  All three commands above were re-run with clean VRAM to confirm the assert is the only blocker;
`--fit off`, which this campaign first believed was required, changes nothing.

Harness that is worth adapting rather than rewriting: `archive/work/dflash2-draft-sched2/tools/` has
`run-arm.sh` (one arm, base/fix library swap), `run-instr.sh` (the `LLAMA_DRAFT_GRAPH_DEBUG` reuse trace,
temporary instrumentation) and `two_slot_check.py` (2-slot server, two concurrent requests, per-slot
acceptance).  For this campaign the base/fix library swap is replaced by the Flash-Next paths.

## 8. Files

| path | what |
|---|---|
| `evidence/incoai-qwen38-dflash2-capture.patch` | the head author's sglang capture patch: the mean-over-HC convention and the `[5,15,25,35,45]` completed-layer list (§3.3).  Provenance: `incoai/Qwen3.8-Flash-Next-DFlash2`, `patches/qwen38-dflash2-capture.patch`, Apache-2.0 |
| `tools/fetch-head.sh` | fetch the Q4_K_M head to `/tmp/df2/head` and print its metadata |
| `tools/repro.sh` | the target-only and with-head server runs from §7, with the VRAM pre-flight |
| `src/models/qwen4exp.cpp:928-950` | the wide `res_hc` export to change |
| `src/models/deepseek4.cpp:269-278`, `1239-1242`, `1329-1332` | `build_hc_mean` and the gated export to copy |
| `src/llama-context.cpp:3330-3445` | `extract_layer_inp()`: the host assert (3437) and the width-agnostic F1 device half |
| `src/llama-graph.cpp:1518-1522` | where a requested `t_layer_inp` becomes a graph output |
| `archive/work/dflash2-draft-sched2/` | the r120 DFlash2 scheduler fix and its harnesses (the 27B numbers in §1) |
| `patches/README.md` block 11 | where the qwen4exp amendment goes |

## 9. Exit criteria and promotion path

Promote into **block 11** when all of §6 is green, with the head's acceptance and the `plain == draft-dflash`
text recorded per device count, then: re-cut (`make-patches.sh` + `make-release.sh`), `validate-set.sh`,
`gate-per-block-build.sh --changed-from`, new `WORKLOG.md` entry citing TODO #94, `ENVIRONMENT.md` untouched
(no new knob), `patches/README.md` block 11 note, this directory moved to `archive/work/qwen4exp-dflash2/`
and the campaign row deleted from `wip/CAMPAIGNS.md`.

If the acceptance comes back in the broken-encoder band after the mean fix, the fallback hypotheses are, in
order: (a) the capture point is wrong for this head (try `completed` = `il` rather than `il+1`), (b) the
head wants the first HC stream or the pre-mix state rather than the mean, (c) the selector needs the target's
wide head after all (its `output.weight` is plain-width, so this is the least likely).
