# wip/state-restore-batch

Batches the device writes of a context state restore, the counterpart of the batched save reads folded in r96 (#189). The server restores a context checkpoint before each follow-up turn of a hybrid model. A checkpoint is about 75 small ranges of the recurrent-state tensors, and the restore wrote them one synchronous `ggml_backend_tensor_set` at a time. Under `-sm tensor` each of those bounces through a pinned chunk and synchronizes every device.

One format-patch on `v14-097cf33d9-r115` (tree `4f7139ca`); applied tree `2f1aae2f`. Patch sha256 `68bc1226…`.

## What changes

`src/llama-context.cpp` only, +54 / −9:
- `llama_io_read_host` takes the context (all three construction sites pass `this`). When it is destroyed it packs the deferred pieces into the context's pinned stage (`state_stage`, the buffer the save path already keeps), queues each piece with `ggml_backend_tensor_set_async` on the backend that owns it (`state_backend`), and synchronizes each backend once.
- The per-tensor synchronous path stays for:
  - no context (a reader constructed without one);
  - a tensor no context backend owns;
  - a state larger than `LLAMA_STATE_ASYNC_MAX_MIB` (default 256);
  - `LLAMA_STATE_ASYNC=0` or `LLAMA_STATE_ASYNC_READ=0`.
- The pieces sit between small metadata fields in the caller's buffer, so they are packed one `memcpy` each.
- The comment above `llama_state_async_enabled` said restores measured no faster batched. It now describes both directions. That was our measurement on r91 for #189; on r115 the batched restore is faster (below). We have not pinned down what changed in between.
- The stage is now shared by restores as well as saves, so the "one buffer per context, no concurrent state copies" sentence in that comment now says saves and restores.

## Gates

Setup:
- 2x R9700 (gfx1201), native Linux.
- Stock r115 against r115 + the patch, same build dir; only `libllama` differs. Your default env (no local variables), `--host-experts` default.
- The #189 configs:

| | model | split | other |
|---|---|---|---|
| G1 | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` + shared Q8_0 MTP head | 2 GPU `-sm tensor` | `-ncmoe 10` |
| G1L | same | 2 GPU `-sm layer` | `-ncmoe 10` |
| G2 | Qwen3.6-35B-A3B Q4_K_M, no MTP | 1 GPU | `-ncmoe 10` |
| G3 | Qwen3.6-35B-A3B Q4_K_M, no MTP | 2 GPU `-sm tensor` | `-ncmoe 10` |

  G1's command is the one from #186:

  ```
  llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -c 262144 -ngl 99 -ctk q8_0 -ctv q8_0 \
    -cram 2048 -ub 2048 -b 2048 --reasoning-effort low --temp 0.7 --top-k 20 --top-p 0.95 --min-p 0 \
    --spec-type draft-mtp -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -ngld 99 --spec-draft-n-max 3 \
    --spec-draft-p-min 0 -ctkd q8_0 -ctvd q8_0 \
    -sm tensor --n-cpu-moe 10 -ot per_layer_token_embd.*=CPU -lm none -lzm off
  ```

  G1L uses `-sm layer`. G2 and G3 change `-m` and drop the MTP flags and the `-ot` / `-lm` / `-lzm` options; G2 sets `ROCR_VISIBLE_DEVICES=0`.

**Greedy hashes** (400 tokens, prose / code), identical to stock r115 on every config:

| config | prose / code |
|---|---|
| G1 | `a7a81f01b3bc` / `4bb43e3339c8` |
| G1L | `946ffc37b143` / `e2f7603679a0` |
| G2 | `047fd057911b` / `231119399578` |
| G3 | `c0cf0739bdab` / `930bcf1f4405` |

**Checkpoint restore** (as in #189):
- A ~2.2k-token prompt ends in " apple" (A), then the same prompt ends in " pear" (B), same slot, prompt cache on. B restores the checkpoint A left.
- B's 48 greedy tokens (`ignore_eos`) equal a fresh `cache_prompt: false` run of B in every arm, stock and patched, on all four configs.
- A's output differs from B's on G1, G1L and G2, so the check can tell them apart there.

**Faults:** none in 15 servers.

## Timing

**Restored request** (the restore check's B: 4 new tokens after the restore; server `prompt_ms`):

| config | stock r115 | + patch | samples per arm |
|---|---|---|---|
| G1, 2 GPU `-sm tensor` | 79.6 ms median (63.5-79.8) | **68.5 ms median (52.4-68.7)** | 6 (3 servers) |
| G1, patch with `LLAMA_STATE_ASYNC_READ=0` | | 79.4 / 79.7 ms | 2 (1 server) |
| G1L, 2 GPU `-sm layer` | 75.8 / 76.4 ms | 74.8 / 74.2 ms | 2 |
| G3, 2 GPU `-sm tensor` | 43.4 / 43.7 ms | 39.0 / 38.6 ms | 2 |
| G2, 1 GPU | 36.9 / 36.8 ms | 38.1 / 38.5 ms | 2 |

The control row puts the G1 difference on the restore path. As with the save side, one device has little to gain; G2 is about 1.5 ms slower here (means of 2).

**Time to first token** on G1, a 4-turn chat (as in #189), 3 servers per arm alternating: the 32-token follow-up turn (same prompt in both arms) went 348 → 329 ms server-side (means; 340 / 344 / 359 against 328 / 335 / 325). The other turns differ by less than their spread.

On our own build (r114 + our patches, IQ3_S `-ncmoe 47`, `-sm tensor`, MTP draft, a 113 MiB checkpoint with the draft context), a profile of one restore went 23 → 11.4 ms, with identical replies.
