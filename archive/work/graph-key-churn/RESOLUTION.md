# RESOLUTION — graph-key-churn (TODO #51, the 3-GPU stutter)

**Shipped in `v14-097cf33d9-r115` (2026-10-22, block 09 amendment).**  One hunk in
`ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_graph_get_key`), 29 lines with the comment.  Kill switch
`GGML_CUDA_GRAPH_KEY_DATA=0` (A/B and bisect only).

**Diagnosis.**  Under `-sm tensor` the Meta backend rebuilds every per-device subgraph from freshly
allocated *simple tensors* on each scheduler reallocation, so `nodes[0]`'s object address — half of the
HIP graph cache key — changes even when the graph is identical.  The cache never warms: 3954 keys,
1206 captures and 484 ms of `hipGraphInstantiate` per 300-token decode at 3 devices, versus 67 keys /
9 ms with `-sm layer` on the same devices.  That host-side burst is the visible 200-450 ms stutter
(measured directly: the whole stall sits inside `ggml_backend_sched_graph_compute_async`, the GPU
dispatches nothing for its duration, and the decode thread spins in ROCr's `BusyWaitSignal`).

Keying on `nodes[0]->data` instead: 144 keys, 45 captures, 15 ms — **+41 % decode at 3 devices**
(+6 % at 2, unchanged at 1), byte-identical text, and an adaptive-MTP 4-axis aggregate of
**96.5 → 111.9 t/s (+16 %)** with acceptance, accepted counts and mean accepted length bit-identical.

**Verified.**  same-seed text identity at 2 and 3 GPUs (both dense 27B and Flash-Next GSQ, with and
without `-ncmoe`); 4-axis adaptive-MTP gate at `-n 3000`; `scripts/validate-set.sh` green; block bodies
00-08 and 13 unchanged by the regeneration (10-12 change only in `index` lines and hunk offsets).

**Not closed by this.**  A residual ~70 ms, ~3-per-2500-token per-device stall survives (unchanged with
graphs disabled; ~2-3 % of wall time) — see the README §6 and the TODO follow-up.  The deeper
alternative (stable simple tensors / a stable child-cgraph uid so the `uid == uid` fast path applies)
is recorded there for whoever picks it up.
