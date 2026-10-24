# deep-prefill-depth-scaling — issue #214's long-prompt prefill regression

**Status: OPEN (2026-10-24).  One cause found and fixed in prototype; the RDNA2 half is still open.**

The reporter's box never had an expert-cache arena, so the r123 band clamp is not their symptom; what is left
is a deep-context prefill regression.  This campaign reproduced the shape on gfx1201, separated our patch set
from upstream drift, and traced it to an RDNA4 FA config row: our decode/verify **band** row is also selected
by wide prefill (they share the `ncols = 64` instantiation), costing 26 % of the FA kernel at
`pp512 @ d98304`.

* Finding + numbers: [`RESULTS-2026-10-24.md`](RESULTS-2026-10-24.md)
* Repro harness: [`sweep.sh`](sweep.sh) (three builds x resident/streaming, one run at a time)
* Items: `TODO.md` **#97** (the regression, RDNA2 half open) and **#98** (the instantiation split, which owes
  `scripts/gate-prefill-logits.sh`)
* Doctrine: `GREEDY-PURITY.md` §51; block reference: `patches/README.md` block 05; record: `WORKLOG.md`
  261024.006.

Nothing here is part of the delivery.
