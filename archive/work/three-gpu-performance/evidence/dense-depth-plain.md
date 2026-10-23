# dense 27B Q8_0 plain decode: depth arms

`llama-cli`, `ROCBLAS_USE_HIPBLASLT=0`, `-t 8`, r116, greedy (`--seed 42 --temp 0`).

**Deep** (`prompts/prose16k.txt` from `wip/…/tools/arms-2026-10-22/` = `prose-rdna-boosts.txt` x3,
15746 tok, sha256 `6fec9806b375b2b3c1516107bd146b6d0100fe7a883807c7c4ffabe7a73ad721`, `-c 20480`, `-n 300`):

| split | 2 GPU | 3 GPU | Δ | text |
|---|---:|---:|---:|---|
| `-sm layer` | 18.0 t/s | 16.9 t/s | **−6.1 %** | `88511ff8ae52` **both** |
| `-sm tensor` | 33.6 t/s | 42.2 t/s | **+25.6 %** | `d56f08cfbcf8` **both** |

**Shallow** (`prompts/prose-rdna-boosts.txt`, `-c 32768`, `-n 400`): layer 18.4 → 17.2 (−6.5 %),
tensor 34.3 → 43.2 (+25.9 %).

Two conclusions:

1. Greedy text is **identical at 2 and 3 devices** on a dense tensor split and in either layer split — the
   cross-device-count text difference is specific to the MoE tensor-split path (a reduction split), and its
   PPL is 0.16 % apart (`ppl-cross-count.txt`).
2. The MoE field config's layer-split deficit is a **generic layer-split property**: a single-token decode
   has no intra-token parallelism to gain, so the 3rd device only adds a pipeline stage.  The MoE
   plain-decode −2 % under tensor split is therefore MoE-specific, not a generic 3-device cost.
