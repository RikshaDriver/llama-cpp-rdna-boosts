# Rung 7 (`d454a204e`) conflict survey — captured live, no resolution attempted

> **RESOLVED in session 2 (2026-10-24).**  All 13 hunks were resolved as the plan section describes
> (op-table unions, the op count recomputed to 108 at block 11, `qwen4exp.cpp` kept as CORDON = the
> delivery's derived-visibility design).  The chain went on to rungs 8 and 9 and is now final at
> `cd2c8f488` / base `3f03b1772`.  The actual decisions and evidence are in `DECISIONS.md` (session 2,
> rung 7) and the checkpoint summary in `STATE.md`.  This file is kept as the dated survey it was.


## `ggml/include/ggml-rpc.h` — 1 hunk(s)

- hunk 1: HEAD 4 lines / OURS 4 lines
  - HEAD: #define RPC_PROTO_PATCH_VERSION    1
  - OURS: #define RPC_PROTO_PATCH_VERSION    7

## `ggml/include/ggml.h` — 2 hunk(s)

- hunk 1: HEAD 1 lines / OURS 6 lines
  - HEAD: GGML_OP_TOP_K_MASK,
  - OURS: GGML_OP_FLASH_ATTN_QSA,
- hunk 2: HEAD 19 lines / OURS 41 lines
  - HEAD: // top-k sparse attention mask: keep the mask cells of the live picks and the extra cells, set the rest to -in
  - OURS: // hc_mix: fused hyper-connection mixer (qwen4exp / Flash-Next decode).

## `ggml/src/ggml-cpu/ggml-cpu.c` — 2 hunk(s)

- hunk 1: HEAD 4 lines / OURS 20 lines
  - HEAD: case GGML_OP_TOP_K_MASK:
  - OURS: case GGML_OP_HC_MIX:
- hunk 2: HEAD 1 lines / OURS 3 lines
  - HEAD: case GGML_OP_TOP_K_MASK:
  - OURS: case GGML_OP_HC_MIX:

## `ggml/src/ggml-cpu/ops.h` — 1 hunk(s)

- hunk 1: HEAD 1 lines / OURS 3 lines
  - HEAD: void ggml_compute_forward_top_k_mask(const struct ggml_compute_params * params, struct ggml_tensor * dst);
  - OURS: void ggml_compute_forward_hc_mix(const struct ggml_compute_params * params, struct ggml_tensor * dst);

## `ggml/src/ggml.c` — 5 hunk(s)

- hunk 1: HEAD 1 lines / OURS 6 lines
  - HEAD: "TOP_K_MASK",
  - OURS: "FLASH_ATTN_QSA",
- hunk 2: HEAD 1 lines / OURS 1 lines
  - HEAD: static_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT != 102");
  - OURS: static_assert(GGML_OP_COUNT == 107, "GGML_OP_COUNT != 108");
- hunk 3: HEAD 1 lines / OURS 5 lines
  - HEAD: "top_k_mask(mask, top_k, score, blocks, extra)",
  - OURS: "flash_attn_qsa(q, k, v, idx, mask)",
- hunk 4: HEAD 1 lines / OURS 1 lines
  - HEAD: static_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT != 102");
  - OURS: static_assert(GGML_OP_COUNT == 107, "GGML_OP_COUNT != 108");
- hunk 5: HEAD 38 lines / OURS 80 lines
  - HEAD: // ggml_top_k_mask
  - OURS: // ggml_hc_mix

## `src/models/qwen4exp.cpp` — 1 hunk(s)

- hunk 1: HEAD 17 lines / OURS 5 lines
  - HEAD: // keep the kq_mask cells of the live picked pools and the tail, -inf elsewhere
  - OURS: // fused expand + mask + top-k: value[c] = score[cell_blk[c]] + additive[c]

## `tests/test-backend-ops.cpp` — 1 hunk(s)

- hunk 1: HEAD 2 lines / OURS 4 lines
  - HEAD: for (ggml_type type : {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S}) {
  - OURS: // iq4_nl is a 32-value sub-block type: exercise row widths that are not a whole number of

## How the next session should resolve these (plan, not applied)

Rung 7's rebase was started and **aborted** (`git rebase --abort`) at the block-11 step; the chain is
still at `rebase-ckpt-6` (`bb4825231`, base `13786fbb3`).  Nothing was resolved, nothing was lost.

1. **Op-table unions are the bulk of it.**  Upstream adds `GGML_OP_TOP_K_MASK` (the fused QSA mask op);
   the delivery adds `GGML_OP_FLASH_ATTN_QSA`, `GGML_OP_HC_MIX`, `GGML_OP_HC_COMBINE`, ...  Keep both
   in every table (`ggml.h` enum + op-name string + the `dumped` case list, `ggml.c`'s
   `ggml_op_name`/`ggml_op_symbol`/execute switch, `ggml-cpu/ops.h` + `ggml-cpu.c` dispatch,
   `ggml-rpc.h` `RPC_PROTO_PATCH_VERSION`).
2. **The op-count static_asserts must be recomputed, not unioned**: HEAD asserts 102, ours asserts 107
   (with a mismatched message that says 108 — a pre-existing typo in block 11).  After the union the
   count is HEAD's + the delivery's added ops; count the enum entries and write both the number and the
   message consistently.  This is why the build should be run before anything else.
3. **`qwen4exp.cpp` (1 hunk) is the semantic one**: upstream now builds the QSA mask through the fused
   `TOP_K_MASK` op, while the delivery's QSA path derives visibility in-kernel (`cell_vis`/`q_vis`) and
   creates no mask at all.  Keep ours (CORDON; upstream's op stays available and is unused by qwen4exp),
   and keep the derived path's `kq_mask == nullptr` invariant.  Re-run
   `test-backend-ops -o FLASH_ATTN_QSA` (18/18 minimum) and the QSA perplexity-ratio check
   (`GREEDY-PURITY.md` §21).
4. **`tests/test-backend-ops.cpp`**: union again — upstream's IQ3_XXS/IQ3_S get_rows cases and our
   IQ4_NL sub-block row-width cases were inserted at the same anchor (the same pattern as rung 6).

After resolving: build, then the full rung gate set (coherence plain == draft-mtp, prefill-logit,
`FLASH_ATTN_EXT`, `FLASH_ATTN_QSA`, `GATED_DELTA_NET`, `HC_MIX` if present), then checkpoint 7.
