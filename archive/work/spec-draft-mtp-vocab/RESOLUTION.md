# RESOLUTION — `spec-draft-mtp-vocab` promoted into **block 12** (r107, 2026-10-22)

**Shipped.**  `--spec-draft-mtp-vocab N` (opt-in, default `0` = off) is block 12 of the delivery as of
`v13-097cf33d9-r107`.  Record: `WORKLOG.md` 261022.001.  Block reference: `patches/README.md` → "Block 12".
Purity doctrine: `GREEDY-PURITY.md` §48.  Knob: `ENVIRONMENT.md` §3.  Open validation arms: `TODO.md` #83.

**What the campaign contributed, in one table** (this directory's README has the full record):

| step | outcome |
|---|---|
| source | discussion **#202** / **PR #205** (martino2k6); six format-patches cut on r101, authors Simon Teixidor, Laurent Zuijdwijk, Martino B.  PR #205 archived verbatim in [`../pr205-spec-draft-mtp-vocab/`](../pr205-spec-draft-mtp-vocab/IMPORT.md) |
| apply | all nine patches (six imported + three campaign) apply to the r106 tree with `git am -3`, no conflict; gated tree `91bcefc49ef42220ba97a937821fb5e1d4fda6a8` |
| blocker found | `-sm tensor` + trim aborted: a 2-D **vocabulary-split** LM head could only be read whole by `ggml_backend_meta_get_tensor_impl` (`GGML_ASSERT(size % chunk_size_full == 0)`) — fixed in the block (row-range splice, general, verified 18/18 on uneven/even/repeated/straddling cases) |
| coverage added | `gemma4-assistant` arm (own tied head) + a merge-order-guard fix (running-maximum comparison mislabelled gemma4 as "236336 of 236339 out of order"; it is 30/236339 = 0.013 %) |
| gemma4 verdict | supported, **not beneficial**: its useful cut is 245760 of 262144 rows (a no-op, 150.5 vs 150.3 t/s); below it acceptance halves (0.669 -> 0.339 at 64000) because its draft top-1 ids split across ids < 64000 and the 229376+ band |
| gate (the ship criterion) | Qwen3.8-27B Q8_0, 2 GPU `-sm tensor`, `draft-mtp-adaptive,ngram-mod`, `-n 3000`, four axes: R +1.3 %, P +1.7 %, C +2.4 %, K +0.8 % at `--spec-draft-n-max 7 --spec-draft-mtp-vocab 98304`; purity verified inside `n_max <= 7` with the trim on (three-way text identity vs `--spec-type none`) |
| ggml regression | `test-backend-ops` 21179/21179; `test-logits-width-probe SPLIT=tensor` W=1..8 maxdiff 0; prefill-logit KLD 0.000000 / 100.000 %; plain 27B tensor decode text- and t/s-identical to r106 |
| interpretation fixed | the campaign's first "purity divergence" was **outside** the documented band (adaptive n-max 9); with no campaign code the r106 build already gives three different texts for `draft-mtp` n3 / n9 / adaptive-n9.  Recorded as doctrine (§48) instead of a bug |

**Why n-max 7 is the recommended pairing.**  Per-position acceptance at 98304 is 0.906 / 0.771 / 0.670 / 0.584 /
0.512 / 0.308 / 0.090 / 0.011 / 0.007 — positions 8-10 contribute 0.7 %, so the ceiling of 9 buys nothing while
sitting outside the purity promise.  At n-max 7 every axis improves and the purity check is valid.

**Not carried into the block:** a two-band subset (keep `< N` **and** `>= n_vocab - M`) to catch gemma4's
space-piece band.  It measured 0.610 acceptance at 37 % of the rows but was 0.4 % *slower* than full vocab on
that payload, so it stayed a measurement (with a temporary env knob) and is filed under `TODO.md` #83 item 4.
