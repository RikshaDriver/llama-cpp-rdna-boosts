# decode-warmup-stall — TODO #89 is a one-time lazy code-object load, not a periodic decode stall

**Status: CLOSED (2026-10-22, no delivery change).**  `TODO #89` is retired: the "residual ~70 ms
per-device decode stall" does not exist in steady state.  What is left after r115 is **one lazy HIP
code-object load per (module, device)**, ~4-6 per device, paid **once per process** — for the decode
families inside `common_init_from_params`'s existing warm-up, and for the prefill-only families inside
the first real prefill.  It is ~300 ms of the *cold start* (dense 27B, 3 devices), **0 % of the decode**
and it does not recur at any run length tested (300 / 1000 / 1500 tokens, decay flat, depth 8 K and
16 K).  Nothing in `patches/` changed.

Two side findings came out of the same passes, and both decide something:

1. The old lead "`hipFuncGetAttributes` at ~1.5 ms/call is a code-object load" was **RCCL's**, not ours
   (§5) — it is NCCL init at context creation, ~0.5 s, invisible to decode.
2. The `hipHostMalloc`/`hipHostFree` churn (2.2 s + 1.1 s in one trace) is
   `ggml_backend_cuda_buffer_set_tensor_2d`'s per-tensor pinned staging during **model load**.  A
   grow-only per-device staging buffer removes it: **27B Q8_0 3-GPU load 12.7 s -> 9.5 s**, measured,
   text byte-identical.  The patch is prepared at
   [`tools/stage-pin-staging.patch`](tools/stage-pin-staging.patch) and **is not applied** — it is
   `TODO #90`, waiting on the maintainer's go/no-go (§7).

Origin: split out of `TODO #51` when r115 removed the dominant cause
([`../graph-key-churn/`](../graph-key-churn/README.md)).  Started from the scoping note that used to live
at `wip/decode-warmup-stall/` (deleted; no stub).

---

## 1. What was measured, and how

The r115 probe ([`../graph-key-churn/tools/LLAMA_STEP_TRACE-probe.patch`](../graph-key-churn/tools/LLAMA_STEP_TRACE-probe.patch))
was extended with exactly the instrument the handover asked for — a **per `(device, op)` first-use
counter plus that call's host time** — and with pinned-allocation accounting
([`tools/LLAMA_STEP_TRACE-probe-ext.patch`](tools/LLAMA_STEP_TRACE-probe-ext.patch), applies to the
`v14-097cf33d9-r115` tree).  It prints:

```
FIRSTUSE ts=.. dev=D op=OP us=H step=N      the first launch of that op family on that device
SLOWOP   ts=.. dev=D op=OP name=NODE us=H   any single op whose host call exceeds 1 ms
CUDAGC_SLOW dev=.. nodes=.. total=.. | top0..   unchanged (per child-graph launch, > 5 ms)
PINALLOC/PINFREE/PIN_TOT                     pinned allocation churn
```

Baseline command (the r115 reproducer, `-sm tensor`, 3x R9700, gfx1201):

```bash
LLAMA_STEP_TRACE=1 HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_ALLREDUCE=ce \
  ./llama-cli -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ngl 99 -sm tensor -mg 0 \
  -fa auto -ctk bf16 -ctv bf16 -c 8192 -b 1024 -ub 1024 -t 7 -n 300 --seed 42 --temp 0 \
  --single-turn --no-display-prompt --spec-type draft-mtp-adaptive,ngram-mod \
  --spec-draft-n-max 9 --spec-ngram-mod-n-match 45 -f <repo>/prompts/reasoning.txt
```

`--spec-draft-n-match` no longer exists (it is `--spec-ngram-mod-n-match` now); everything else is
unchanged.  One GPU job at a time, no other GPU load.

## 2. The stall inventory is a fixed start-up set (dense 27B Q8_0, 3 devices)

`-n 300`, 12 `CUDAGC_SLOW` entries, **sum 311 ms**.  Every one lands in the same four launches, and every
one coincides with a `FIRSTUSE` of that op family on that device:

| launch | devices | what | per device |
|---|---|---|---|
| warm-up, subgraph i=0 (main ctx) | 0, 1, 2 | `MUL_MAT` 57.6 / 56.2 / 55.5 ms (mmvq code object), `GET_ROWS` 5.0/4.2/4.1, `CONCAT` 1.7/1.3/1.3, `UNARY` (in the fuse pass) 6.1/6.1/6.1, `SCALE` 0.4/0.3/0.2 | 71-76 ms |
| warm-up, subgraph i=6 | 0, 1, 2 | `FLASH_ATTN_EXT` 5.3/5.1/5.2, `SET_ROWS` 1.2/0.9/0.9, and `RMS_NORM` 4.7/5.1/5.0 **in the same window** (not a first use — see §4) | 11 ms |
| MTP draft ctx warm-up (5-node graph) | 0, 1, 2 | `ADD` 13.1/12.4/12.2 (`mtp_post_ffn-64`) | 12-13 ms |
| first decode-width subgraph (87 nodes) | 0, 1, 2 | `MUL_MAT` 7.3/6.5/6.4 (`node_15`, a different template width) | 6-7 ms |

Then nothing.  **Not one op takes > 1 ms on any device for the remaining ~4 700 steps of that run**
(~24 000 steps in the `-n 1500` one); the only later `FIRSTUSE`s are 3-8 us (`ROPE`).

**It does not scale with tokens.** The identical 12 entries (same totals to within 3 %) appear in a
`-n 1500` run — 5x the tokens, 5x the decode, **zero** additional stalls:

| run | generated | `CUDAGC_SLOW` | sum |
|---|---:|---:|---:|
| 27B, 3 GPU | 300 tokens | **12** | 311 ms |
| 27B, 3 GPU | 1500 tokens | **12** | 301 ms |
| Flash-Next GSQ, 3 GPU | 300 tokens | **36** | 1149 ms (§6) |
| Flash-Next GSQ, 3 GPU | 1000 tokens | **36** | 1145 ms, the same set |

The previous "~3 events per device **per 300 tokens**" reading came from dividing a one-time start-up
cost by the run length: the events are per *device* and per *code object*, not per N tokens.  The
"~10 % of decode" figure follows from the same arithmetic — the 300 ms is spent **before** the prompt
prefill, i.e. it is cold-start time, not decode time (§3).

## 3. It is paid inside the existing warm-up, so it never touches a user step

`common_init_from_params` warms up with a BOS+EOS 2-token decode (`common/common.cpp`), and the MTP draft
context does the same.  In the STEP trace the real prompt prefill (5 142-node subgraph) starts *after* all
12 stalls; at depth 8 K the prefill and the whole decode are then clean (at 16 K the prefill pays two more
one-time loads, §4.6).  Confirmed by removing the warm-up:

| arm | `CUDAGC_SLOW` | sum | where |
|---|---:|---:|---|
| default (warm-up on) | 12 | 311 ms | both warm-ups + the first decode-width launch |
| `--no-warmup` | **9** | **309 ms** | the user's own first steps |

Same cost either way — the warm-up is what keeps it off the request path today.  For `llama-server`
(one load, one warm-up, many requests) it is therefore **0 % of steady state**; for a library consumer
that never warms up it is 300 ms on the first request.

The answer to the handover's question 3 ("does the warm-up run every op family per device?") is: **for the
decode families, yes** — the 2-token decode touches `GET_ROWS`/`MUL_MAT`/`FA`/`CONCAT`/`RMS_NORM`/`ROPE`/
`SET_ROWS`/`GLU`/`ADD`, which is exactly the set that then runs clean.  It cannot cover the *prefill*
families, because they are shape/depth dependent: at depth 16 K the 3rd prefill ubatch loads a new FA
prefill variant mid-prefill (65 ms device 0, 36 ms device 1, once).  Only running a real prefill can reach
those, so "warm up harder" has no bounded form.

## 4. The mechanism (evidence chain)

1. **Every stall is a first use** — §2's table, extended to Flash-Next (§6).  The 55-58 ms `MUL_MAT`
   figure is identical on a Q8_0 dense 27B and an IQ3_XXS MoE, i.e. it is not the kernel shape or the
   quant type: it is the load of the whole `mul_mat_q` **code object** (mmq and mmvq share one TU, which
   is also why the first *prefill* `MUL_MAT` after a decode-only warm-up is cheap).
2. **The time is inside one `hipLaunchKernel` call.**  `rocprofv3 --hip-trace`: the longest
   `hipLaunchKernel`s are 117-122 ms (under the profiler's overhead) and the kernel-level trace from the previous
   campaign (`rocprofv3 --kernel-trace`) shows the GPU dispatching nothing in the window.
   No module-load API appears in the trace — HIP performs the load inside the launch.
3. **It inflates the rest of the window.**  `RMS_NORM` at 4.7-5.1 ms in the FA subgraph is *not* a first
   use: once a code object is loading, further enqueues on that device block behind it.  That is the
   "every op in the window is inflated ~1000x" signature, and why one stall looks like five.
4. **Graphs are irrelevant** (as the previous campaign found): `GGML_CUDA_DISABLE_GRAPHS=1` gives the
   same 12 entries with the same totals (311 vs 301 ms) — the same loads happen through the direct
   launch path.
5. **It scales with devices because the load is per device**, and the Meta backend drives the devices in
   sequence: 1 GPU = one 81 ms warm-up stall + 13 + 16 ms; 3 GPUs = 3 x ~110 ms.
6. **A deeper KV reaches a new variant later in the run** (depth 16 K: 14 entries, 405 ms — +2 stalls at
   the 3rd prefill ubatch, one device each) — still one-time, and still a first use.

## 5. The two old leads, resolved

- **"`hipFuncGetAttributes` costs ~1.5 ms/call (576 calls / 857 ms)"** — with
  [`tools/hip-api-probe.c`](tools/hip-api-probe.c) (an `LD_PRELOAD` interposer that attributes a HIP call
  to its **caller**) *every* `hipFuncGetAttributes` in the run comes from **librccl** (288 calls /
  878 ms, 100 % of them), i.e. NCCL module loads during `ncclCommInit` at context creation.  Nothing in
  this tree calls it except the RDNA2 occupancy workaround in `fattn-common.cuh` (inert on gfx1201).  It
  is not the stall, and it is not ours.
- **"`hipHostMalloc`/`hipHostFree` at 2.9/1.5 ms per call, 3.4 s of a 17 s run — who allocates pinned
  memory during decode?"** — nobody, during decode.  All 678 pairs are
  `ggml_backend_cuda_buffer_set_tensor_2d` (`ggml/src/ggml-cuda/ggml-cuda.cu`), the `-sm tensor` splice
  upload of the model loader: one `cudaMallocHost`+`cudaFreeHost` per tensor.  Measured 2.2 s + 1.1 s of
  a ~10.7 s load.  That is a real, containable cost — but it is **load** time, not the decode stall, and
  it is `TODO #90` (§7).

## 6. Second model (MoE): same shape, more code objects

Flash-Next GSQ-IQ3_XXS + shared Q8_0 MTP head, `-sm tensor`, 3 devices, q8_0 KV, `-n 300` and `-n 1000`
(identical 36 entries both times):

| launch | per device | first use |
|---|---|---|
| main warm-up (92-node subgraph) | 150 / 128 / 130 ms | `REPEAT` 12 ms, `HC_MIX` 4/2/2 ms, `GET_ROWS` 4.7, `MUL_MAT` 54.7 (mmvq again), `CONCAT` 1.6 |
| first real prefill (98-node subgraph) | **172 / 146 / 143 ms** | `MUL_MAT` 151 ms (the prefill mmq/mmb objects) + `MUL_MAT` 10 ms + `MUL_MAT` 7.9 ms, `DSV4_HC_PRE` 3.2 |
| remaining `CUDAGC_SLOW` entries | 5-14 ms | ordinary 56-node host cost crossing the fixed 5 ms threshold (see the caveat below), not stalls |

So the MoE arm pays ~0.89 s of cold start in those two groups (more code objects: MoE/expert, HC, PLE),
and again none of it recurs; the remaining ~0.26 s of its 1.15 s total is ordinary 56-node steps crossing
the fixed 5 ms threshold (see the caveat below).  Note that the first-prefill block cannot be removed by a warm-up: only a real prefill reaches those
code paths.

**A caveat for whoever reads these counters next:** `CUDAGC_SLOW`'s threshold is an absolute 5 ms per
child-graph launch, so on 1 GPU (one 4 000-5 000-node graph per launch) ordinary steps report 5-8 ms and
the counter looks "stallier" the fewer devices there are.  Normalise per node (`total / nodes`, ~1-2 us)
or by `FIRSTUSE`, or the 1-GPU arm is unreadable.

## 7. What was NOT done, and the one open decision

- **No delivery change for #89.**  There is nothing to fix: the residual is a one-time OS/driver cost
  (loading a code object into the device), it is already absorbed by the warm-up, and the only ways to
  move it are worse than the cost (a warm-up cannot cover shape-dependent prefill variants; loading each
  code object eagerly by hand would need a kernel registry the runtime does not expose).
- **"Overlap the three devices' loads" was considered and declined unmeasured**: it would save ~200 ms of
  a ~10 s startup, and the only way to get it is running the warm-up subgraph per device from separate
  host threads — a Meta scheduler change for a cold-start number below the maintenance threshold.
  Re-open only if cold start becomes a product requirement.
- **`TODO #90` (awaiting go/no-go): the load-path pinned staging.**  Prepared, measured, **not applied**:
  [`tools/stage-pin-staging.patch`](tools/stage-pin-staging.patch) makes
  `ggml_backend_cuda_buffer_set_tensor_2d` reuse one grow-only pinned staging buffer per device
  (≤ 64 MiB retained, `GGML_CUDA_STAGE_PIN=0` restores the per-call allocation) instead of
  `cudaMallocHost`+`cudaFreeHost` per tensor:

  | arm (27B Q8_0, `-sm tensor`, 3 GPUs, `-n 8`, `GGML_CUDA_STAGE_PIN` 0/1 alternating) | wall |
  |---|---:|
  | `=0` (today) | 10.75 / 12.53 s |
  | default (staging cached) | **9.51 / 9.52 s** |

  (an earlier build of the same change measured 10.53 / 12.69 / 12.68 s against 9.87 / 9.88 / 9.91 s)

  `678 -> 5` pinned allocations per load.  Same-seed text byte-identical with
  `ROCBLAS_USE_HIPBLASLT=0` on 27B Q8_0 3 GPU (200 tokens, `draft-mtp-adaptive,ngram-mod`), 27B Q8_0
  2 GPU, and Flash-Next GSQ `-ncmoe 48` + MTP 3 GPU (80 tokens).  It touches no kernel and no numerics,
  so it needs the set gates, not the prefill-logit gate.  On the GSQ `-ncmoe 48` load the win is inside
  that arm's run-to-run spread (49.4-54.3 s both ways) — claim it only for the dense/resident load.

## 8. Tooling (all reusable, none of it part of the delivery)

| file | what |
|---|---|
| [`tools/LLAMA_STEP_TRACE-probe-ext.patch`](tools/LLAMA_STEP_TRACE-probe-ext.patch) | the probe + per-`(device,op)` first-use/first-cost accounting + pinned-allocation accounting.  `LLAMA_STEP_TRACE=1`.  Applies to the r115 tree on top of, or instead of, the original probe |
| [`tools/hip-api-probe.c`](tools/hip-api-probe.c) | `LD_PRELOAD` interposer: attributes `hipHostMalloc`/`hipHostFree`/`hipFuncGetAttributes`/`hipModuleLoad`/`hipMalloc` to their caller, aggregated, with backtraces for the slowest calls |
| [`tools/stage-pin-staging.patch`](tools/stage-pin-staging.patch) | the `TODO #90` candidate — **not applied** |

`rocprofv3 --hip-trace` was used as the cross-check; `scripts/check-gpu-faults.sh --since "2 hours ago"`
over the run logs was clean (no `page fault`/`slab` lines, no `HSA_STATUS_ERROR`).
