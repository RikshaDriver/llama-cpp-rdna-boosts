# Controls — cache-off, all-reduce backend, AR block count

r116, MoE field config (`-ncmoe 48 -sm tensor`, `-c 40000`, mix30k 32358 tok, `-n 300`, plain decode).

| control | devices | prefill t/s | decode t/s | text | note |
|---|---|---:|---:|---|---|
| `MOE_EXPERT_CACHE_MIB=0` (streamed, no arena) | 2 | 1360.5 | 21.1 | `edae58fb75c5` | no cache is 3x slower than an identity arena |
| `MOE_EXPERT_CACHE_MIB=0` (streamed, no arena) | 3 | 1410.0 | 19.7 | `562063456051` | **−6.6 %** at 3 GPUs even with no cache |
| `GGML_CUDA_ALLREDUCE=internal` (forced for all sizes) | 3 | — | 67.5 | `7f185f379ff0` | +3.7 % vs hybrid 65.1, but it changes the *prefill* AR path → different text: smoke only |
| `GGML_CUDA_ALLREDUCE=nccl` | 3 | — | 59.3 | `7c899d234a02` | −9 % vs hybrid; not bit-identical, smoke only |
| `GGML_CUDA_AR_SMALL_BLOCKS=8` | 3 | — | 64.1 | — | no gain over the default 4 (65.1) |

AR per-call profile (`GGML_CUDA_AR_PROFILE=1`, same arm, plain decode, 4096 calls):

| devices | total | avg/call | p50 | p90 | decode-band share | spin (dev0/1/2 avg) |
|---|---:|---:|---:|---:|---:|---|
| 2 | 82.5 ms | 20.1 µs | 13.8 | 26.5 | **0.26 ms/token = 1.7 %** of 15.1 ms/token | 7.1 / 11.6 µs |
| 3 | 144.3 ms | 35.2 µs | 21.3 | 56.2 | **0.45 ms/token = 2.9 %** of 15.4 ms/token | 20.8 / 22.1 / 14.2 µs |

So the 3-device internal AR costs 0.19 ms/token more than the 2-device one = **1.2 % of decode**, which is
most of the −2.0 % plain-tensor gap and nothing more.  Suspect 1 is retired for *throughput*: the
all-reduce cannot explain a double-digit deficit, and neither `nccl` nor a different block count is
better at 3 ranks.
