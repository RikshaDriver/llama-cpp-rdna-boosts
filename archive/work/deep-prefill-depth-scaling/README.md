# deep-prefill-depth-scaling — issue #214's long-prompt prefill regression

**Status: CLOSED (2026-10-24, r129) -- RDNA4 fixed in r126 and the RDNA2 half confirmed good by the
reporter.  Nothing left open here.**

The reporter's box never had an expert-cache arena, so the r123 band clamp is not their symptom; what is left
is a deep-context prefill regression.  This campaign reproduced the shape on gfx1201, separated our patch set
from upstream drift, and traced it to an RDNA4 FA config row: our decode/verify **band** row is also selected
by wide prefill (they share the `ncols = 64` instantiation), costing 26 % of the FA kernel at
`pp512 @ d98304`.

* Finding + numbers: [`RESULTS-2026-10-24.md`](RESULTS-2026-10-24.md)
* Repro harness: [`sweep.sh`](sweep.sh) (three builds x resident/streaming, one run at a time)
* Items: `TODO.md` **#97** (closed at r129: RDNA4 in r126, RDNA2 on the reporter's sweep) and **#98** (the
  instantiation split, shipped in r126)
* Doctrine: `GREEDY-PURITY.md` §51; block reference: `patches/README.md` block 05; record: `WORKLOG.md`
  261024.006.

Nothing here is part of the delivery.

## Resolution (r129, 2026-10-24) -- the RDNA2 half

The RDNA2 half was never measurable on this box (no gfx1030/gfx1031: the only RDNA2 silicon here is a
**gfx1036 iGPU with a 2 GiB BAR and ~1.8 GiB already in use by the desktop**, i.e. no room for a model, a KV
cache or an FA staging arena).  The reporter ran the sweep on their own 12 GiB gfx1031 card against the
current build and **reported it works well** (maintainer relay, 2026-10-24), so the deep-context prefill
shape that this campaign localised to the RDNA4 `ncols == 64` MMA row does not reproduce on RDNA2 through
block 05's RDNA2 tile table or the `prefill_stages = !RDNA3_5` staging gate.  `TODO.md` #97 is therefore
retired: the RDNA4 half shipped in r126 (`WORKLOG.md` 261024.007), the RDNA2 half closed on the reporter's
measurement.  No delivery code changed for the RDNA2 half.
