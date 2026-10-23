# decode-warmup-stall — the residual ~70 ms per-device decode stall (TODO #89)

**Status: OPEN / scoping, not started.**  This is the handover prompt for the next session.  Nothing
here is applied to the fork; nothing is part of the delivery.  Origin: `TODO.md` **#89**, split out of
#51 when r115 removed the dominant cause (see `archive/work/graph-key-churn/`).

---

## Give this to the next session

> **Task.**  Find and fix the residual per-device decode stall that survives r115 (TODO #89).
>
> **What is known.**  On `v14-097cf33d9-r115`, 3x R9700 (gfx1201), Qwen3.8-27B Q8_0, `-sm tensor`,
> `draft-mtp-adaptive,ngram-mod n-max 9`, a 300-token decode still contains ~3 stall **events per
> device** (`CUDAGC_SLOW` shows 9 entries > 10 ms at 3 devices; ~12 slow entries in total), each ≈ 70 ms,
> for ~300 ms of a ~3050 ms decode (~10 % of the wall clock, worst single-op 51 ms).  The user-visible
> stream after r115 has **no gaps > 200 ms** (worst 173 ms, p99 46 ms), so this is the next tier down —
> but it is 10 % of decode and it is *periodic*, so it is worth killing.
>
> **The signature (this is the valuable part — it constrains the hypothesis space).**
> - The stall lands in the **first subgraph of a step** and inflates **every op in that window**:
>   `MUL_MAT` 51 ms, `UNARY` 6 ms, `GET_ROWS` 5 ms, `FLASH_ATTN_EXT` 5.4 ms, `RMS_NORM` 4.7 ms,
>   `ADD` 12 ms — i.e. unrelated ops on the same device, all ~1000x their normal host cost.
> - During the window the GPU **dispatches nothing** (verified with `rocprofv3 --kernel-trace`: the hole
>   contains no kernel, and the kernels bracketing it run at normal speed), the GPU sits at ~55 W / 40 %
>   busy, and the decode thread spins in ROCr's `BusyWaitSignal` — so it is a *host* block, not a slow
>   kernel.
> - **It scales per device**: ~3 events per device per 300 tokens at 1, 2 and 3 GPUs (3 / 6 / 9 entries),
>   so it is not a cross-device wait; under `-sm tensor` the meta backend runs the devices in sequence,
>   which is why one step shows ~3x70 ms.
> - **Already ruled out** for this stall (not for #51 — see the record): HIP graph capture
>   (`GGML_CUDA_DISABLE_GRAPHS=1` leaves it unchanged), the graph-cache key (r115 fixed that and the
>   count did not move), the all-reduce (`GGML_CUDA_AR_PROFILE=1`: max 2 ms/call), device count,
>   `-sm tensor` vs `-sm layer` (present in both), memory reclaim / PSI / swap-in (zero), power or
>   thermal throttling (55 W against a 250 W cap, 46-57 °C), PCIe runtime PM (`active` in every sample).
>
> **The strongest current lead: lazy kernel module load, per device.**  The inflated ops are the ones
> first *used* on that device in the run, and in the same traces `hipFuncGetAttributes` costs ~1.5 ms
> **per call** (576 calls / 857 ms in one 17 s run) — that is a code-object load, not an attribute read.
> A second, separate per-call host cost worth measuring in the same pass: `hipHostMalloc` / `hipHostFree`
> measured 2.9 ms / 1.5 ms per call (769 / 763 calls, **3.4 s of a 17 s run**) — find who allocates
> pinned memory during decode and whether it need not.
>
> **Plan (in this order, each step cheap and decisive).**
> 1. Apply the probe patch from `archive/work/graph-key-churn/tools/LLAMA_STEP_TRACE-probe.patch` to a
>    canonical `scripts/apply-all.sh` tree, build with `~/bin/build-llama-rocm`, run with
>    `LLAMA_STEP_TRACE=1`.  Extend it minimally: per `(device, op)` first-use counter plus that call's
>    host time.  If every stall coincides with a first use, the module-load hypothesis is confirmed and
>    the fix space is a per-device warm-up (or an explicit `hipModuleLoad`/`hipFuncGetAttributes` at
>    context creation) — not a scheduler change.
> 2. Cross-check with `rocprofv3 --hip-trace`: are `hipFuncGetAttributes` / `hipModuleLoad` on the
>    stall's critical path, and are they **repeated per launch** rather than cached per `(kernel,
>    device)`?  If they are repeated, caching them is a contained fix (and would help every shape, not
>    just the first use).
> 3. Test whether a warm-up already covers it: the 27B **server** config has warmup on and still stalls;
>    does `common_params`' warm-up run *every* op family per device, and does `GGML_CUDA_DISABLE_GRAPHS=1`
>    change which kernels are touched?  If a one-step warm-up at context creation removes the stall, that
>    is the fix and it costs nothing at steady state.
> 4. Only if 1-3 exonerate module loading: treat it as a driver/scheduler stall and bisect with
>    `rocprofv3 --hip-trace` for the longest non-kernel API call in the window (the r115 record already
>    saw 119 ms `hipLaunchKernel` calls in the pre-fix build, which the fix removed).
>
> **Measurement discipline (do not skip).**  One GPU job at a time — two concurrent GPU jobs change the
> arena/band layout and invent stalls.  `ROCBLAS_USE_HIPBLASLT=0` for any fresh-process comparison
> (`GREEDY-PURITY.md` §44).  Decode claims at depth 16384, not shallow `llama-bench`.  Judge a fix on
> the *stall counters* (`CUDAGC_SLOW` count and sum, plus the streamed-gap distribution from
> `wip/three-gpu-performance/`-style harnesses), not on one t/s number — this stall is 10 % of decode
> and partly hides inside noise.
>
> **Gates.**  Same-seed text identity against `v14-097cf33d9-r115` (dense 27B and Flash-Next GSQ, 2 and
> 3 GPUs); the adaptive-MTP 4-axis gate at `-n 3000` (`benchmarks/mtp-adaptive-methodology.md`), where
> *acceptance, accepted/generated counts and mean accepted length must stay bit-identical* — that is the
> cheapest proof that a host-side change did not touch numerics; `scripts/gate-prefill-logits.sh` only if
> a prefill kernel is touched; `scripts/check-gpu-faults.sh --since "2 hours ago"` over the run logs.
>
> **Do not** re-derive the r115 conclusion or "fix" the graph key again — `GGML_CUDA_GRAPH_KEY_DATA=0`
> exists for A/B, and the key must stay a stable identity (`AGENTS.md`, "the HIP graph cache must be
> keyed on a stable identity").
>
> **Deliverable.**  A block amendment in the owning block (the CUDA backend graph/op launch path —
> likely block 10, check with `grep -l <the function> patches/*.patch`), an updated
> `archive/work/decode-warmup-stall/` record with the numbers, a `WORKLOG.md` entry, and `TODO.md` #89
> closed or narrowed.  Report findings to the maintainer *before* shipping (AGENTS.md).

---

## Why this is a `wip/` entry and not an `archive/` one

Work has not started, so this is a scoping note, not a finished campaign.  When it closes or is promoted
it moves to `archive/work/decode-warmup-stall/` with a `RESOLUTION.md` and this directory is deleted (no
stubs — see the WIP rule in `AGENTS.md`).

## Pointers

- `TODO.md` #89 and #51; `WORKLOG.md` 261022.010 (r115).
- [`../../archive/work/graph-key-churn/`](../../archive/work/graph-key-churn/README.md) — the #51
  record: elimination table, the probe patch, the reproducers.
- [`../three-gpu-performance/`](../three-gpu-performance/README.md) — the 2-vs-3-GPU campaign and its
  streamed-gap harness.
- `benchmarks/mtp-adaptive-methodology.md`, `GREEDY-PURITY.md` §44 (hipBLASLt per-start determinism).
