# graph-key-churn — the tensor-split HIP graph cache never warms, and that is the 3-GPU stutter

**Status: CLOSED (r115, 2026-10-22, block 09).**  Promoted into the delivery as the block-09 amendment
that keys `ggml_cuda_graph_get_key()` on the first node's **data pointer** instead of its `ggml_tensor`
object address (`GGML_CUDA_GRAPH_KEY_DATA=0` restores the old key).  This directory is the record; the
diagnostic instrumentation used to find it is [`tools/LLAMA_STEP_TRACE-probe.patch`](tools/LLAMA_STEP_TRACE-probe.patch)
(apply to a canonical `apply-all.sh` tree, build, run with `LLAMA_STEP_TRACE=1`).

Originating item: `TODO.md` **#51** (3-GPU decode slower than 2-GPU) and the field report that 3-GPU
decode *stutters* — 200-400 ms pauses at a ~1 s period on Qwen3.8-27B Q8_0, ~3 s on Flash-Next GSQ —
while 2 GPUs looks smooth.  Campaign home at the time: `archive/work/three-gpu-performance/`.

## 1. Symptom, measured

The user's own 27B server command (`-sm tensor`, `--parallel 3`, `-c 614400`, bf16 KV,
`draft-mtp-adaptive,ngram-mod`, `GGML_CUDA_ALLREDUCE=ce`), driven through the OpenAI streaming API and
timestamped per chunk: 2494 tokens at **86 t/s**, with ~20 gaps of **475-485 ms** (plus a cluster at
~285 ms) per 30 s of decoding.  The stalls are `submit`-side, not GPU-compute-side: the per-decode
instrument shows the whole 475 ms inside `ggml_backend_sched_graph_compute_async`.

## 2. Where the time goes (in order of elimination)

| probe | result |
|---|---|
| `GGML_CUDA_AR_PROFILE=1` | all-reduce healthy: 1386+4096 calls, **max 2 ms**, spin p99 243 us.  Not the AR — which also rules out the user's `ce`-vs-`hybrid` theory. |
| kernel timeline (`rocprofv3 --kernel-trace`) | at a stall the GPU dispatches **nothing** for 200-450 ms and kernels in the stalled windows run ~100x slow (a `fillBufferUnAligned` at 2.5 ms instead of 15 us). |
| `/proc/vmstat` + PSI during stalls | zero reclaim, zero swap-in, zero I/O or CPU pressure.  Not memory. |
| `power1_cap` 250 W, `power1_average` 55 W, 46-57 °C at a stall | not power/thermal throttling (the box runs a -85 mV undervolt and a 250 W cap from `~/bin/tune_r9700.sh`; the stall is *below* both). |
| PCIe runtime PM (`power/runtime_status`, three GPUs) | `active` for every one of 2157 samples.  Not D3 suspend/resume. |
| CPU accounting per thread | the decode thread burns ~65 % of a core *inside* the stall, in ROCr's `BusyWaitSignal::WaitRelaxed` (a spin, not a sleep). |
| per-thread `eu-stack` during stalls | `hipGraphInstantiate` / `hipGraphExecDestroy` / `hsa_amd_memory_pool_allocate` under `ggml_backend_cuda_graph_compute` → **HIP graph capture churn**. |

## 3. Root cause

`ggml_cuda_graph_get_key()` is `{ cgraph->nodes[0], n_tokens }` — the first node's **tensor object
address**.  Under the Meta (tensor-split) backend every per-device subgraph is built from *simple
tensors* that `ggml_backend_meta_graph_compute` **re-creates on every rebuild** (it resets each
buffer's `stc_compute` containers and maps every node to a fresh object), so the object address changes
on every scheduler reallocation even when the graph is structurally identical and points at the same
device buffer.  With a per-device subgraph count of ~129 and MTP-adaptive verify widths changing every
few dozen tokens, the cache therefore **never warms**: each rebuild creates ~387 fresh keys, and each
fresh key pays a full HIP graph capture plus a `hipGraphInstantiate` (406 us measured at 3 devices),
and the 5-second sweep then destroys them again (up to 638 ms of `hipGraphExecDestroy` in one run).

Measured with the same binary, same workload (Qwen3.8-27B Q8_0, `-sm tensor`, 300-token decode,
`draft-mtp-adaptive,ngram-mod n-max 9`):

| arm | graph keys | captures | `hipGraphInstantiate` | decode |
|---|---:|---:|---:|---:|
| 1 GPU (`-sm none`) | 33 | 8 | 5 ms | 5674 ms |
| 2 GPU tensor | 2636 | 804 | 120 ms | 3947 ms |
| 3 GPU tensor | **3954** | **1206** | **484 ms** | **4315 ms** |
| 3 GPU **layer** (no meta backend) | **67** | 83 | 9 ms | 6885 ms |

The meta backend is the whole of it: `-sm layer` at the same device count has 59x fewer keys.  Note the
2-vs-3 device inversion in that table (3947 ms at 2 devices vs 4315 ms at 3) — this is TODO #51's
"3 GPUs is slower", and it is *not* the all-reduce.

## 4. The fix (shipped, r115, block 09)

Key on `nodes[0]->data` (the per-device buffer address the captured graph actually bakes in) instead of
the tensor object address:

| arm | keys | captures | instantiate | decode | Δ |
|---|---:|---:|---:|---:|---:|
| 3 GPU tensor, object key (r114) | 3954 | 1206 | 484 ms | 4315 ms | – |
| 3 GPU tensor, **data key** (r115) | **144** | **45** | **15 ms** | **3055 ms** | **+41 %** |
| 2 GPU tensor, object → data | 2636 → 96 | 804 → 30 | 120 → 5 ms | 3947 → 3722 ms | +6 % |
| 1 GPU, object → data | 33 → 51 | 8 → 8 | 5 → 5 ms | 5674 → 5681 ms | 0 % |

**Why it is safe.**  `data` is stable across a meta rebuild and is what the captured graph contains.
Two *different* graphs can only collide if their first node resolves to the same data pointer *and* the
same token count; that is not trusted — `ggml_cuda_graph_update_required()` re-checks every node's
`op`/type/shape/`data`/`src` pointers and recaptures on any difference, and a meta rebuild assigns a
fresh `cgraph_uid` which forces that check.  A collision therefore costs a recapture, never a
mismatched replay.  (This is *not* the r25/PR-#106-0002 change that was dropped: that one **added**
node/source data pointers *to* the object key, which fragments a warm cache; this one **replaces** the
unstable object address with the stable data address.)

## 5. Validation

Same binary, `-sm tensor`, 3x R9700, Qwen3.8-27B Q8_0, `--seed 42 --temp 0`:

- **Same-seed text byte-identical** to r114 at 2 and 3 GPUs (sha `b7d60a30bd6bc686` after stripping the
  build line and the timing line).
- **Adaptive-MTP 4-axis gate** (`--spec-type draft-mtp`, `-n 3000`, R `--reasoning on`, P/C/K off):

  | axis | r114 | r115 | Δ | acceptance (both arms, bit-identical) |
  |---|---:|---:|---:|---|
  | R | 83.1 | 90.9 | +9.4 % | 0.60463 (1933/3197), mean len 2.81 |
  | P | 97.1 | 109.7 | +13.0 % | 0.81173 (1315/1620), mean len 3.44 |
  | C | 105.9 | 119.0 | +12.4 % | 0.89700 (2186/2437), mean len 3.69 |
  | K | 99.7 | 128.1 | +28.5 % | 0.99200 (372/375), mean len 3.98 |
  | aggregate | 96.5 | 111.9 | **+16.0 %** | identical counts and mean lengths |

  Acceptance, accepted/generated counts and mean accepted length are **bit-identical on every axis**,
  and all four axes' generated text is byte-identical — the change moves only the cache.
- **Flash-Next GSQ-IQ3_XXS** (MoE, 3 GPU, its own MTP head, q8_0 KV): 91.7 → 128.3 t/s (**+40 %**),
  same acceptance (0.69637, 211/303, mean 3.43), identical text.  With `-ncmoe 46` (host experts, the
  movable-boundary slab live): 79.4 → 119.5 t/s (**+50 %**), keys 4041 → 810 (the slab does still move
  some per-device buffers, so this arm keeps more keys than the resident one, but the win is larger
  because the base churns more).
- `scripts/validate-set.sh` green (strict 14/14 `git am`, applied tree == `release.json.tree`).

## 6. What this does *not* fix, and what is left

- **A residual per-device stall remains**: ~70 ms, ~3 times per 2500 tokens, in the *first* subgraph of
  a step, with every op in that window inflated (a `MUL_MAT` at 51 ms, `UNARY` 6 ms, `GET_ROWS` 5 ms)
  and the GPU idle.  It is **unchanged** by the key fix and by `GGML_CUDA_DISABLE_GRAPHS=1`, and it is
  ~2-3 % of wall time after the fix.  Its signature (slow *first* use of each op kernel per device,
  `hipFuncGetAttributes` at 1.5 ms/call, code-object load) points at lazy kernel module loading rather
  than a scheduling defect.  Filed as a TODO follow-up.
- The deeper fix, if the maintainer prefers cache stability over a stable *key*, is to make the meta
  backend stop re-creating its simple tensors on a rebuild (or give each child cgraph a **stable** uid
  so `ggml_cuda_graph_update_required()` takes its `uid == uid` fast path).  Both would also remove the
  remaining per-call node-property walk; they touch block 10/12 code and were not attempted here.

## 7. Reproducer

```bash
# the probe (per-decode split, per-op top-3, graph-cache/instantiate totals)
cd <canonical apply-all.sh tree> && git apply <this dir>/tools/LLAMA_STEP_TRACE-probe.patch
~/bin/build-llama-rocm          # or a local cmake with -DCMAKE_HIP_FLAGS=-isystem/opt/rocm-10.1.0-gfx120X/include

LLAMA_STEP_TRACE=1 HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_ALLREDUCE=ce \
  ./bin/llama-cli -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf -ngl 99 -sm tensor -mg 0 \
  -fa auto -ctk bf16 -ctv bf16 -c 8192 -b 1024 -ub 1024 -t 7 -n 300 --seed 42 --temp 0 \
  --single-turn --no-display-prompt --spec-type draft-mtp-adaptive,ngram-mod \
  --spec-draft-n-max 9 --spec-draft-n-match 45 -f <this dir>/../../../prompts/reasoning.txt 2>&1 | \
  grep -E "GRAPH_CACHE|GRAPH_TOT|CUDAGC_SLOW|STEP"
```

`GRAPH_CACHE new_keys=` is the metric: a healthy tensor-split decode keeps it in the low hundreds.
