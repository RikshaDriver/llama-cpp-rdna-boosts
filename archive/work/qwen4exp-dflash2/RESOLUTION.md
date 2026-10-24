# `qwen4exp-dflash2` — RESOLVED (r121, 2026-10-23): the HC-averaged layer-input export is in block 11

**Status: CLOSED / PROMOTED.**  **TODO #94 is retired**; the fix ships as the **block 11** amendment in
`v14-097cf33d9-r121`, plus a **block 10** fix found while validating it (below, §5).
Issue: [#211](https://github.com/stew675/llama-cpp-rdna-boosts/issues/211).  Release record:
`WORKLOG.md` 261023.006.

This file is the campaign handover that was written before the fix (`README.md` next to it, kept as the
seeded record).  Everything below is the resolution; the numbers in `README.md` §1 for the 27B target are
from that session and are **not** re-measured here.

---

## 1. What was wrong

`src/models/qwen4exp.cpp` published the target's DFlash/DSpark layer input as the **wide** hyper-connection
residual (`t_layer_inp[il] = res_hc`, `hc * n_embd` = 10240 on Flash-Next).  A drafter's feature tap expects
`n_embd` floats per row, so the target aborted in `llama_context::extract_layer_inp()`:

```
src/llama-context.cpp:3437: GGML_ASSERT(row_floats == model.hparams.n_embd) failed
```

(the host path; the F1 device path asserted `t->ne[1] == n_tokens` instead).  Reproduced on the r120 base
build with `incoai/Qwen3.8-Flash-Next-DFlash2-Q4_K_M` attached, so it was never a head problem: the head is
a real DFlash2 head (`selector_top_k 16`, `block_size 8`, `target_layers [6,16,26,36,46]`, `fc.weight
[2560, 12800]`) and every other shape lines up.

## 2. The fix (block 11, 28 lines, no new environment variable)

* `llama_model_qwen4exp::graph::build_hc_mean()` — the same 8 lines `deepseek4`/`glm5-next` already have:
  `[n_embd, hc, n_tokens] -> [n_embd, n_tokens]` by summing one 2-D view per stream and scaling by `1/hc`.
* The export is **gated on `cparams.embeddings_layer_inp[il]`**, exactly like deepseek4's.  The old line was
  an unconditional pointer store that added no node, so an ungated collapse would have added `hc-1` adds and a
  scale to *every* qwen4exp graph (plain decode, MTP, QSA) for a tensor nobody reads.
* Nothing else: no device-iface entry, no graph-params field, no env var.

**The convention was settled from source, not by search.**  The head author's sglang capture patch
(`evidence/incoai-qwen38-dflash2-capture.patch`) inserts the capture *after* the layer call
(`hidden_states = layer(...)` … `normalized_input = norms.output` … capture), so `completed layer i` is the
state at the **top of layer `i+1`** — which is exactly llama.cpp's `t_layer_inp[il]` and the GGUF's
`target_layers` (`completed + 1`).  No index shift, and the reduction is the mean over the streams
(`hidden_states.unflatten(-1, (hc, hidden)).mean(dim=-2)`).

## 3. Evidence

**Purity (all `ROCBLAS_USE_HIPBLASLT=0`; a fresh-process comparison is invalid without it).**
`plain == draft-dflash` greedy text is **body-identical on six arm pairs**:

| arm | config | body sha256 (16) |
|---|---|---|
| code, 3 GPU tensor + `-ncmoe 48` | plain / dflash n7 / dflash n3 | `a8b745381528c23e` |
| recall, 3 GPU tensor + `-ncmoe 48` | plain / dflash n7 / dflash n3 | `a6c82525d9dcb8a4` |
| code, 2 GPU tensor + `-ncmoe 48` | plain / dflash n7 | `8ef8f70ba2ccf775` |
| code, 3 GPU `-sm layer` (experts resident) | plain / dflash n7 | `1ecea50e04159f6a` |
| code, host injection (`GGML_LF_DFLASH_DEV=0`) | plain / dflash n7 | `a18094c6dbe5539d` |
| code, all experts device-resident | plain / dflash n7 | `a18094c6dbe5539d` |

Pre-fix vs post-fix **plain** text is also identical (`a8b74538`), i.e. the non-DFlash paths are untouched.

**Prefill logits on the changed architecture** (`llama-perplexity`, Flash-Next GSQ, ctx 512, 10 chunks,
3 GPU tensor, `-ncmoe 48`): base PPL 1.9404, candidate **mean KLD −0.000000, same-top-p 100.000 %**.
The delivery's `gate-prefill-logits.sh` (27B Q8_0 base): PASS, 0.000000 / 100.000 %.

**The mean is the right reduction** (the campaign's fallback hypothesis (b)).  An env-gated stream-0-only
export (temporary, removed before shipping) measured recall acceptance length **2.41** vs the mean's
**4.32** — the mean nearly doubles it.

**Acceptance is healthy, and the campaign's "~0.3 means a broken encoder" heuristic is a block-size
artifact.**  The acceptance *ratio* falls as the block grows while the mean accepted length is the metric:
DFlash2 n-max 7 gives code 0.293 / len 3.04 and recall 0.478 / len 4.32, against the head card's own
reference (block 4, acceptance length 2.83-3.26 on GB300/sglang).

## 4. Decode (gfx1201, 3x R9700, ROCm 10.1.0, Flash-Next GSQ IQ3_XXS + the Q4_K_M head; `-c 16384 -ub 2048
-b 2048 -fa on -ctk q8_0 -ctv q8_0 -t 8`, greedy, seed 42, 256 generated tokens, `ROCBLAS_USE_HIPBLASLT=0`)

| arm (`-sm tensor -ncmoe 48`) | code t/s | code len | recall t/s | recall len |
|---|---:|---:|---:|---:|
| plain | 69.43 | 1 | 69.64 | 1 |
| DFlash2 n-max 3 | 92.14 | 2.55 | 115.72 | 2.93 |
| **DFlash2 n-max 7** | **97.19** | **3.04** | **132.97** | **4.32** |

Other shapes (code, n-max 7): 2 GPU tensor 67.85 (plain 68.80, parity); `-sm tensor` with every expert
resident 66.16 (plain 76.91); 3 GPU `-sm layer` without `-ncmoe` 57.67 (plain 53.95).

## 5. A slab peer copy into an ungranted range — found while validating, fixed in block 10

`-sm layer` + `-ncmoe 48` on Flash-Next GSQ aborted at warm-up with `ROCm error: invalid argument` in
`ggml_backend_cuda_cpy_tensor_async` → `hipMemcpyPeerAsync`, **on the r120 base build with no drafter
attached** (so unrelated to this campaign, but it blocked the `-sm layer` arm of the validation plan).  Root
cause and fix are in `patches/README.md` (block 10) and `WORKLOG.md` 261023.006 §5; the short form is that
the slab's peer-access grant was silently downgraded to owner-only for the second and third ~29 GiB slab, and
a peer copy into an ungranted range either errors host-side or **faults the GPU**.  After the fix that arm
runs (265 t/s prefill, 48.4 t/s decode) and its greedy text equals its `GGML_CUDA_SLAB=0` arm.

## 6. Known limit (documented, not fixed)

qwen4exp does not export layer-input index `n_layer` (the "output of the last layer", which `deepseek4` also
supports): after the loop `res_hc` may be row-gathered, so it needs the pre-gather residual.  No DFlash2 head
taps the final layer (`[6,16,26,36,46]`/48, `[6,20,34,48,62]`/64), and a head that did would hit the
`layer input tensor is null` assert rather than mis-drafting.

## 7. Files

| path | what |
|---|---|
| `README.md` | the seeded campaign handover (kept as-is; its §1 27B table and §5 hypotheses are that session's record) |
| `evidence/incoai-qwen38-dflash2-capture.patch` | the head author's sglang capture patch: the mean-over-HC convention and the `[5,15,25,35,45]` completed-layer list |
| `tools/fetch-head.sh`, `tools/repro.sh` | fetch the head / reproduce the pre-fix abort |
