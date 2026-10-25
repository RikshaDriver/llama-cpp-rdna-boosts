# pr213-pr218-r132 — the two external PRs promoted in r132

Two contributor pull requests reviewed and folded into the delivery in **r132**
(`v14-3f03b1772-r132`).  Neither changes a kernel's arithmetic on the maintainer's fleet, and neither
produced the win its author measured — the review established *why* in each case, and that is the part
worth keeping.  Numbers and commands are in [`EVIDENCE.md`](EVIDENCE.md).

| PR | author | what it does | folded into | measured here |
|---|---|---|---|---|
| **#218** part 1 | @slavap | MMB's per-arch policy predicates key on the **running device's** cc instead of `devices[0].cc`, and `graph_optimize` selects its own context's device first | **block 06** | bit-for-bit identical on a single-arch box (by construction); 12/12 same-seed configs byte-identical, prefill-logit KLD 0.000000 on both recorded points |
| **#218** part 2 | @slavap | the Q8_1 quantize cache's key walk stops at the first pointer change (offset views no longer alias the root's entry) + `q8_1_src_range_ok()` validates the source read range before every quantize / fused gated-Q8_1 launch | **block 08** (mmvq + the unary impl site, and the guard itself) and **block 10** (the fused GLU site and the norm/GLU launch arms) | not reachable here: **0 offset-view keys** in 14336-60416 key walks per config across seven configs; 12/12 configs byte-identical, MTP acceptance identical on all four axes |
| **#213** | @briansp2020 | the Meta split-state cache hit returns `entry.get()` directly instead of copying the state out of a named local | **block 09** | **codegen-neutral**: clang 24 and clang 23 at `-O3` already build the state in the return slot (identical function size, one instruction's worth of difference); the function's sampled self time is 1.6-1.9 % before / 1.6 % after, within noise |

## Why the PR claims did not reproduce

* **#213's "two `rep movsq` -> one".**  The claim is about the *compiler*, not the source form: with our
  ROCm trees (clang 24 in 10.1.0, clang 23 in 7.14.1) `ggml_backend_meta_get_split_state` is
  emitted the same way before and after the patch.  The PR author's build must have failed the NRVO
  elision (it is optional for a function with several `return`s), so the patch was a real win *there*.
  We took it anyway: it states the intent at the source level and removes the dependency on that
  elision, and it is provably value-identical.  Recorded as such in the block-09 body so nobody re-reads
  it as a measured speedup.
* **#218 part 1's mixed-arch failure.**  Our fleet has no RDNA2 device beside an RDNA3/RDNA4 one, so the
  reporter's failure (PPL == vocab on the V620 secondary) cannot be reproduced here.  What *can* be
  checked is that the single-arch behaviour is unchanged, and it is: the current device's cc is
  `devices[0].cc` on a uniform box, so every predicate resolves the same way.
* **#218 part 2's fault.**  Probed at the four `mmvq` sites on seven configs: **no** q8_1 src1 reaches
  the cache as an offset view here, so the mis-hit and the stale-pointer fault are configurations we do
  not run (the reporter's V620 decode was one).  The change is still correct and cheap, and the guard
  turns a future stale pointer into a named abort instead of a GPU fault.

## What else the review added

* `ggml_backend_cuda_graph_optimize()` now selects its own context's device.  The MMB marking half
  consults the same per-device predicates as the launch path, and under the Meta backend the pass is
  forwarded per child from meta `graph_compute`, where the current device is still the previously
  computed one.  Without it, `#218` part 1 would be truthful for launches and approximate for the marks
  (`GGML_CUDA_MMB_DOWN16` / blk16 / res16 can elide an F32 output).  `ggml_cuda_set_device()` is a no-op
  when the device already matches, so a single-device run is unchanged.
* The block-10 fused GLU -> Q8_1 launch (which block 10 adds) got the same key walk and guard; the PR
  patched only the six sites that existed in the tree it was cut against.
* `mmb_cfg()`'s three magic statics are documented as per-architecture-*class*, which is exactly
  per-arch because `mmb_arch_defaults()` discriminates only RDNA4 and RDNA3_0 (add a branch if that
  changes).

## Where the rules now live

`GREEDY-PURITY.md` §55 (the Q8_1 cache identity + read-range rule) and §56 (per-device arch dispatch),
`AGENTS.md` (both as critical facts), `patches/README.md` (blocks 06/08/09/10), `MANIFESTS.md` (the four
per-block verification rows), `ENVIRONMENT.md` (the MMB per-device note and the "not a knob" note on the
Q8_1 guard), `WORKLOG.md` 261025.001.
