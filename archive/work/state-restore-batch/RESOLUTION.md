# RESOLUTION -- state restore batching (PR #209) promoted into block 12, r119

**Status:** promoted 2026-10-23 in `v14-097cf33d9-r119` (block 12).  The submission is preserved
verbatim in this directory (`README.md`, the r115 format-patch).  The release record is
`WORKLOG.md` 261023.004.

## What was reviewed

The patch adds the restore counterpart of the batched state save folded in r96 (PR #189).  Code review:

* `llama_io_read_host` takes the context, packs the deferred pieces into the same pinned
  `state_stage`, queues each with `ggml_backend_tensor_set_async` on the backend that owns it
  (`state_backend`), and calls `llama_state_sync_all` once.  This is the save path in reverse and
  the same bytes/tensors are written.
* The fallback conditions are correct and conservative: no context, a tensor with no resolvable
  backend, a state above `LLAMA_STATE_ASYNC_MAX_MIB`, `LLAMA_STATE_ASYNC=0` and
  `LLAMA_STATE_ASYNC_READ=0` all keep the per-tensor synchronous path.  All failure points in
  `flush_async()` are before the first copy, so a fallback never leaves a partial batch.
* `discard()` still clears `rinfos`, so the error paths in `state_set_data` /
  `state_seq_set_data` do not flush a partial restore.
* The stage is one buffer per context and is now shared by saves and restores; the comment states
  that saves and restores must not run concurrently (the llama state API was already not
  thread-safe).
* The patch was cut against r115 and applied to the r118 tree with a 44-line offset; no source
  conflict.  The patch sha256 is `68bc1226...` as stated in the submission README.

No code change was requested.  The 1 GPU regression disclosed by the submitter is real and is
accepted as the cost of the extra staging copy on a single device; the feature's win is the
multi-device sync saving.

## Reproduced (2x R9700 gfx1201, ROCm 10.1, stock r118 vs r118 + the patch)

Server prompt-cache restore, 1571-token checkpoint, q8_0 KV, 4 new tokens after the restore,
`prompt_ms` medians of the timing samples (one correctness sample per arm excluded):

| config | stock | + patch | delta |
|---|---:|---:|---:|
| Flash-Next GSQ, 2 GPU `-sm tensor -ncmoe 10` | 88.6 ms | 75.5 ms | -14.8 % |
| 35B-A3B Q4_K_M, 2 GPU `-sm tensor -ncmoe 10` | 48.8 ms | 39.9 ms | -18.2 % |
| 35B-A3B Q4_K_M + DFlash2 draft, 2 GPU | 68.5 ms | 58.2 ms | -15.0 % |
| 35B-A3B Q4_K_M, 1 GPU | 39.1 ms | 40.9 ms | +4.6 % (about 2 ms) |

The submitter's G1 uses a shared Q8_0 Flash-Next MTP head that is not in this box's model set; the
no-draft Flash-Next arm and a DFlash2 draft on 35B-A3B exercise the same target/draft
`llama_state_seq_set_data_ext` path.

## Gates

* Restored checkpoint's 48 greedy tokens == a fresh `cache_prompt:false` run on 35B-A3B (plain and
  DFlash2) and Flash-Next GSQ, 2 GPU tensor.
* Four-axis MTP A/B, 35B-A3B Q4_K_M, 2 GPU tensor, `-n 3000`, `draft-mtp n3`: generated text
  identical stock vs patched; acceptance identical (R 0.77335, P 0.75685, C 0.88240, K 0.98148);
  generation within noise.
* Same-seed coherence (Qwen3.5 4B Q8_0, 3 GPU tensor): text identical.
* `scripts/gate-prefill-logits.sh`: mean KLD 0.000000, same-top-p 100.000 %.
* `test-save-load-state`, `test-state-restore-fragmented`: PASS (the latter restores a 52.4 MiB
  state through the new path).
* `scripts/check-gpu-faults.sh` over the run logs and the kernel journal: clean.
* `scripts/gate-per-block-build.sh --changed-from b0d8e4bf8`: block 12 and block 13 build `all`.
* `scripts/validate-set.sh`: green on the re-cut set (fresh-tarball strict apply, tree
  `64a7d1492`).

## Where it lives

Block 12 (`src/llama-context.cpp`), with `LLAMA_STATE_ASYNC_READ` in `ENVIRONMENT.md`, the block 12
note and the r119 verification row in `MANIFESTS.md`, and the shared-stage invariant in `AGENTS.md`.
