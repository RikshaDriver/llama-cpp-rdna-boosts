# RESOLUTION — fa-rdna3-head512-purity (closed 2026-10-21, r102 tree)

**Question.**  The r102 default-on gfx1100 head-512 WMMA FA rows (PR #199) measured mean KLD `0.0293` /
same-top-p `92.59 %` in the prefill-logit gate, against a limit of `0.005` / `98 %`.  Is there a row set
that passes, or is the divergence inherent to a tile→WMMA family switch and the criterion wrong?

**Answer: neither of those two.**  The rows are fine and the kernel is fine — **the measurement point was
the problem**.  At a point inside the gate's calibrated domain the same, unmodified rows measure **mean
KLD 0.001778 / same-top-p 98.967 %**, i.e. they pass both limits, and they are **8x more faithful than a
same-kernel ubatch reordering** measured at the same point.  No row parameter, no accumulator precision
and no block size moves the diverging number; what moves it is the corpus (base PPL 607 → 2.65).  The
criterion has been replaced with a decidable one (`--family-switch` mode) and `TODO` #81 is closed.

## The three findings

1. **The reported `0.0293` was the *marginal* number.**  Default vs **fully tile** is `0.046179 / 90.422 %`;
   the delivery *without* the 512 rows already measures `0.044055 / 90.627 %` against fully tile; `0.029285
   / 92.588 %` is default vs "delivery minus the 512 rows".  `WORKLOG` 261021.004 called the marginal pair
   "head-512 WMMA vs tile".  So the absolute limit was never reachable *against a fully-tile base* by any
   head-512 row set, because the head-256 rows (out of scope, shipped since r5) are already at 0.0441 there.

2. **At the wikitext point the gate cannot decide a family switch at all.**  Control: fully tile at
   `-ub 512` vs fully tile at `-ub 256` — same kernel, only the ubatch split differs — measures
   **0.104989 / 85.000 %**.  A benign scheduling change is 21x the KLD limit and 13 points below the top-p
   floor there, because gemma-4-26B-A4B is a *reasoning* model and raw wikitext is 45x outside its
   confident range (base PPL 607).  Any "failure" at that point is uninformative.

3. **At a calibrated point the switch passes, and the numbers order the ways you'd want.**

   | point (gemma-4-26B-A4B QAT, f16 K/V) | base PPL | same-kernel control | head-256 rows | **head-512 rows (marginal)** |
   |---|---:|---|---|---|
   | wikitext `-c 512 --chunks 40` | 607 | 0.104989 / 85.000 % | 0.044055 / 90.627 % | 0.029285 / 92.588 % |
   | self-continuation `-c 512 --chunks 30` | 13.65 | 0.036654 / 93.660 % | 0.022562 / 96.000 % | 0.011303 / 97.451 % |
   | `prompts/fa-family-switch-gemma4-26B-A4B.txt`, `-c 2048 --chunks 10` | **2.65** | 0.013962 / 97.221 % | 0.005994 / 98.324 % | **0.001778 / 98.967 %** |

   The head-512 rows are the *most* faithful of the three at every point; PPL parity is 2.6515 vs 2.6530
   (±0.3 %), and throughput is untouched (**+34.3 %** over the feature-off arm: 2570.07 vs 1913.30 t/s
   pp2048 @ d32768; tg128 127.37 vs 127.19).

## What was ruled out (so nobody re-runs it)

| hypothesis | measurement | verdict |
|---|---|---|
| the fp16 PV accumulator (`T_C_VKQ = tile<16,16,half2>`, the one structural difference from the tile kernel's fp32 accumulator) | `E1`: fp32 C tile (the path the head-80/112 rows use) → wikitext marginal 0.030489, confident marginal 0.011303, **-32 %** throughput (1740 t/s, *below* the tile arm) | not the cause, and it breaks the "not below the feature-disabled arm" floor |
| the online-softmax rescale cadence (`nbatch_fa` 32 vs the tile kernel's strip length 128) | `V1` 64 → 0.030865 at -19 %; `V3` 128+QIR → 0.029081 at -14 % | ±6 %, and both cost throughput |
| `Q_in_reg` (the spill fix itself) | `V4` `Q_in_reg=true` → **bit-identical** KLD to every printed digit, at **-29 %** | the win *is* the spills, and it is accuracy-neutral |
| the tile kernel's block size | `V5` `nthreads=256` | not instantiable (static asserts on the loop sizes) |
| the WMMA kernel being *less* accurate | op oracle vs the CPU backend at the model's own shape (kv 32768, nb 512, GQA 8): f16 5.99e-5 vs tile 5.71e-5, q8_0 2.30e-5 vs 1.64e-5 (threshold 5e-4); at nb 24/64 WMMA is 1.1-2.8x *closer* | both are equally accurate; neither is the wrong kernel |
| a genuinely broken kernel | `test-backend-ops -o FLASH_ATTN_EXT` 6389/6389; both arms within 1.3x of the CPU reference; the two arms' outputs agree at the op level | no defect |

## What landed (the deliverable)

* `scripts/gate-prefill-logits.sh --family-switch`: refuses a point outside the calibrated domain
  (`--family-max-ppl`, default 10), records a **same-kernel control base** next to the fallback base
  (same arm at `--control-ub`, default 256), always prints the control, and passes on the absolute limits
  or on "no worse than the control".  Verified end to end: `PASS 0.001778 / 98.967 %`, control
  `0.012049 / 97.333 %`, point PPL 2.646, and the record→compare pair is reproducible.
* `prompts/fa-family-switch-gemma4-26B-A4B.txt` — the hash-stable in-domain corpus (base PPL 2.65) the
  mode uses, with its generator recipe in `prompts/README.md`.
* `benchmarks/prefill-logit-methodology.md` — the family-switch protocol and the point-sensitivity table.
* `MANIFESTS.md` — gate 3 + the block-05 verification row.
* `GREEDY-PURITY.md` §46 — rewritten around the calibrated-point + control rule with the full number set,
  and the historical correction of the `0.0293` label.
* `TODO.md` #81 — closed (this file is its record).

**Nothing in `patches/` or `release.json` changed**: no block amendment, no new release.  The kernel the
campaign was about is byte-for-byte the r102 one.  The only code-level artefact worth keeping from the
experiments is a latent-bug fix found on the way (`variants/W0-vkq-c-size-fix.patch`): the RDNA3 `VKQ_C`
array is sized for the fp16 C tile's 32-column span, so any *fp32* C tile at `DV % 32 == 0` would index
out of bounds — unreachable in the delivery today (the only fp32 C rows are head 80/112, whose DV is not a
multiple of 32), but it is the road to any future fp32-accumulator experiment.  It is an exact no-op for
every row the delivery instantiates, so it can ride any future block-05 amendment.

## Files

| file | what |
|---|---|
| `README.md` | the original campaign brief (historical) |
| `MEASUREMENTS-2026-10-21.md` | the full session record: arms, bases, every number, reproduction commands |
| `NEXT-SESSION-PROMPT.md` | the handover prompt this session ran from (historical) |
| `variants/*.patch` | every experimental arm as a patch against the r102 tree (`V1`, `V3`, `V4`, `V5`, `E1`/`E1c`, the oracle instrument, the size fix) |
| `measure.sh`, `measure2.sh`, `sweep.sh` | the measurement drivers (`sweep.sh <tag> "<row args>"` applies, builds, measures at four bases and reverts) |
