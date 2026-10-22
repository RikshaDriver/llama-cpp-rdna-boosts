# RESOLUTION — `dflash-meta-split-impl`: promoted as `GGML_OP_ALL_GATHER` in r106 (TODO #78 closed)

**Status: RESOLVED (2026-10-21), shipped as block 11 of `v12-097cf33d9-r106`.**  Record:
`WORKLOG.md` 261021.016.  Evidence and the full as-validated change:
[`FINDINGS-2026-10-21.md`](FINDINGS-2026-10-21.md) + [`all-gather.patch`](all-gather.patch) (identical to the
folded block-11 hunks at the time of promotion).

## Route taken

The campaign's §1(b), **split→mirrored gather**, as a new general ggml collective `GGML_OP_ALL_GATHER`
(`ggml_all_gather()`): the input may be split along axis 0, the result is `MIRRORED`, so every device holds
the whole tensor and an op that reduces along the split axis -- the DFlash2 selector's `ggml_top_k` over the
draft logits and the `ggml_get_rows` that reads the candidate values -- runs locally.  `-sm layer` DFlash2 was
the only supported config before; it remains supported unchanged.

## Two routes refuted first (kept so nobody re-walks them)

* **Mirroring the DFlash2 drafter** (policy `MIRRORED` for every tensor it owns) does not help: the logits come
  from the *target's* `output.weight`, whose placement is resolved through the buffer's buft device (the
  target's policy), and it stayed `SPLIT_AXIS_0` with extents `[124160, 124160]`.
* **Demoting only the drafter to `-sm layer`** aborts at load: `pre-allocated tensor (output.weight) in a
  buffer (Meta()) that cannot run the operation (NONE)` -- a layer-split draft context cannot read the
  target's shared embedding/lm-head out of the target's Meta buffers, and materialising them would duplicate
  ~1.4 GB.

## What the work also fixed (both are delivery defects, not DFlash2-specific)

1. `extract_layer_inputs()`'s F1 device copy is one pointer range, which a Meta buffer cannot express: it
   aborted in `ggml_backend_tensor_set(NULL)` for **any** DFlash drafter requesting `dflash.target_layers`
   under `-sm tensor` (the r54 guard only covered selector models).  Now takes the host path.
2. The slab's **narrow-2** region only accepted the VA the creation reservation left unmapped, sized by the
   draft-context *estimate* (a measured 512 MiB `compute` tier).  A tensor-split drafter runs the whole draft
   graph per device (measured 1280 MiB), so the pin refused and the `cudaMalloc` fallback could not fit with
   the slab holding the VRAM.  The pin now yields from the arena (evicting the tables it covers, capping the
   arena top), preserving the unmapped-spare placement when it fits.

## Result

`-sm tensor` + DFlash2 (27B-Q4_K_XL + its DFlash2, 2 GPU, 16K ctx, q8_0 KV): **84.5 t/s** against **63.4**
for the old `-sm layer` config (+33 %), acceptance **identical** to `-sm layer` (0.67742, same per-position
rates); 3 GPU 61.1 (vs 60.8); `--parallel 2` green; 35B-A3B-Q8_0 + DFlash2 191.0 vs 161.9 t/s.  Gates:
`test-backend-ops` 3/3, prefill-logit **KLD 0.000000 / 100.000 %**, dense greedy text byte-identical to r105,
`validate-set.sh` green.  No new environment variable (`-sm layer` is the escape hatch).

## Pointers

* `patches/README.md` (block 11: the DFlash2 fold and the slab narrow-2 fix; "Facts that span blocks": the
  `ALL_GATHER` invariants).
* `patches/0011-rdna-boosts-block-11-campaign-remainder-CPU-wins-bac.patch`
* `ggml/src/ggml-backend-meta.cpp` (`handle_all_gather`, the subgraph cut, `all_gather`),
  `ggml/src/ggml-cuda/ggml-cuda-slab.cu` (`ggml_cuda_slab_narrow2_alloc` / `_pin_locked` / `_commit_locked`),
  `src/models/dflash.cpp` (`build_dflash2_selector`), `src/llama-context.cpp` (`extract_layer_inputs`).
