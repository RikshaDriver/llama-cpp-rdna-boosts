# The identity cliff (r116) — why "3 GPUs is slower" was mostly "no table is 100 % resident"

MoE field config: Qwen3.8-Flash-Next GSQ-IQ3_XXS + shared Q8_0 MTP head, `-ncmoe 48 -sm tensor`,
`/tmp/srr/mixed30k.txt` (32358 tok, sha256 `69624f4d…62ae241`), `-c 40000`, **plain decode**, `-n 1000`,
`ROCBLAS_USE_HIPBLASLT=0`, `-t 8`.

| arm | MiB/device | arena total MiB | identity tables | prefill t/s | decode t/s |
|---|---:|---:|---:|---:|---:|
| explicit `MOE_EXPERT_CACHE_MIB=13800` | 13800 | 40845.1 | **0 / 432** | 1661.1 | **46.7** |
| explicit `MOE_EXPERT_CACHE_MIB=15000` | 15000 | 40925.0 | **432 / 432** | 1668.9 | **65.0** |
| auto (shipped default) | auto | 40925.0 | 432 / 432 | 1684.1 | 65.1 |

2 devices, same model: auto `-c 40000` = 40925.0 MiB, 294 identity tables, **66.4 t/s**;
auto `-c 204800` = 37808.0 MiB (92.4 %), **0** identity tables, **46.3 t/s**.

The shortfall is 80 MiB = 432 tables x 1 slot. So the cliff is not a device-count property at all: it is
"is *every* table 100 % resident". One slot short and the whole table drops off the identity fast path
(`alloc_table_locked: … is FULLY RESIDENT (512/512 slots): identity fast path ON (the decode band now
reads the arena directly; no per-token routing readback)`), which is worth **+40 %** decode here.

Sizing code: `ggml/src/ggml-cuda/moe-expert-cache.cu` `alloc_all_locked()` holds back
`rounding = dev_n_tables[d] * ggml_cuda_slab_arena_unit(d)` from the slab capacity, so a budget within one
arena unit per table of the host set lands just below 100 %.  **Resolved in r117** (block 09): the cache now prefers whole-table identity — 47 of 48 layers full and the
coldest one dropped to 0 slots on the `13800` arm above, still bit-identical text, **46.7 → 59.5 t/s**
(`MOE_EXPERT_CACHE_IDENTITY_PREF=0` restores the uniform placement).  An arena that is *percentages* short
(e.g. 8 %) is not fixed and still pays the readback: the drop is capped at ~4 % of the layers, because a
dropped layer serves every expert through the host copy (measured: 6 of 48 dropped = 65.7 t/s vs 93.4
uniform).
