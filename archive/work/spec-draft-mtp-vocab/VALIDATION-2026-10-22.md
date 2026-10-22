# VALIDATION 2026-10-22 — the remaining `--spec-draft-mtp-vocab` arms (closes `TODO.md` #85)

The r107 ship gate (`README.md` §3) covered 2 GPU shallow.  This record runs the three arms that were open in
`TODO.md` **#85** — **3 GPU `-sm tensor`**, **depth-16384 decode**, and **MoE with `-ncmoe` / expert cache
active** — plus a reproduction check of the r107 table.  *(Handle note: the r107 entry filed this item as
#83, which is the r103 gemma-4 bf16 `rope.cu` item; #84 is the async-input ring, so the maintainer resolved it
as **#85** — `WORKLOG.md` 261022.004.)*  Build: r109 tree (`933b964c7` / `8e67f977…`),
ROCm 10.1.0, 3x R9700 gfx1201, `ROCBLAS_USE_HIPBLASLT=0`.

**Outcome in one line:** purity holds everywhere tested, MoE is neutral and depth is flat, but the **3-GPU
arm is negative (−2.2 % aggregate)** because the subset head is *mirrored* per device while the untrimmed
head is *split* — the feature is only profitable when `N · n_devices < n_vocab`, and at 3 GPUs on a
248320-vocab model it is not.  Detail + the width-sweep control:
[`../../wip/three-gpu-performance/FINDING-2026-10-22-dense-27b-and-mirrored-draft-subset.md`](../../wip/three-gpu-performance/FINDING-2026-10-22-dense-27b-and-mirrored-draft-subset.md)
(raw runs: `../../wip/three-gpu-performance/tools/arms-2026-10-22/summary.csv`).

## 0. Command — and a correction to §3 of the head record

Canonical 4-axis command (`benchmarks/2026-09-13-adaptive-mtp-4-axis.md` → "Reproducing"):
`-n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16
-fa auto -ngl 99 -lv 4`, default sampling, `--reasoning on` for R and `off` for P/C/K, plus
`--spec-type draft-mtp-adaptive,ngram-mod --spec-ngram-mod-n-match 45 --spec-draft-n-max 7
--spec-draft-mtp-vocab N`.

**§3 of the head record writes this as `--spec-draft-n-max <N> --spec-draft-n-start <N>`, which is wrong
to copy.**  `--spec-draft-n-start` is the *initial* adaptive depth (default 3), not a ceiling; setting it
equal to n-max starts the controller pinned at the ceiling and materially changes the trajectory.  Measured
on 2 GPU / P: `n-start 3` 90.6-94.3 t/s with acceptance 0.664-0.670 vs `n-start 7` 87.6-90.6 with 0.627 —
a different generated stream, i.e. not the same measurement.  **Use `--spec-draft-n-max 7` and leave
`-n-start` alone.**  (The reference `<N>` in that line means the ceiling, not both flags.)

Reproduction of the r107 table with the corrected command — Qwen3.8-27B Q8_0, 2 GPU, trim 0 → 98304:

| axis | r107 record | this run | r107 Δ | this Δ |
|---|---|---|---|---|
| R | 68.1 → 69.0 | 68.8 → 69.5 | +1.3 % | +1.0 % |
| P | 93.8 → 95.4 | 94.3 → 95.8 | +1.7 % | +1.6 % |
| C | 120.6 → 123.5 | 116.8 → 119.4 | +2.4 % | +2.2 % |
| K | 368.2 → 371.2 | 348.2 → 347.5 | +0.8 % | −0.2 % |
| aggregate | 110.1 → 111.8 | 109.4 → 110.9 | +1.6 % | +1.4 % |

Within 0.4 % per axis and 0.2 % aggregate, same sign on every axis except K (flat).  The r107 conclusion
for 2 GPU stands.

## 1. Arm 1 — 3 GPU `-sm tensor` (the one that changes the guidance)

| axis | trim 0 | trim 98304 | Δ | acceptance off → on | mean len |
|---|---:|---:|---:|---|---|
| R | 75.1 | 73.8 | **−1.7 %** | 0.57400 → 0.57240 | 2.85 → 2.85 |
| P | 98.3 | 96.0 | **−2.3 %** | 0.68878 → 0.68827 | 4.85 → 4.83 |
| C | 117.2 | 114.1 | **−2.6 %** | 0.74085 → 0.74085 | 5.89 → 5.89 |
| K | 298.1 | 291.9 | **−2.1 %** | 0.98361 → 0.98361 | 29.24 → 29.24 |
| aggregate | **113.1** | **110.6** | **−2.2 %** | | |

C and K have *identical* acceptance in both arms (the trim removes no proposal the draft would have made),
so their −2.6 % / −2.1 % is the pure cost of the wider per-device draft matmul: with the head **split**
each device computes `248320/3 = 82773` columns, with the subset **mirrored** each device computes
`98337`.  The 2-GPU arm has the opposite arithmetic (`124160` vs `98337`).  The width sweep that isolates
this (identical acceptance at 98304 vs 131072) is in the finding record §C:

| device share | trim 0 | 98304 | 131072 |
|---|---:|---:|---:|
| 2 GPU (share 124160) | 94.3 | **95.8** | 93.1 |
| 3 GPU (share 82773) | **98.3** | 96.0 | 93.3 |

and at 3 GPU the width rule (`N < 82773`) is below the acceptance floor — 64000 measures 89.1 t/s
(−9.4 %, acceptance 0.68878 → 0.67260) — so **there is no profitable N at 3 GPUs on this payload**.

Purity (prose, `-n 3000`, no `-lv` so the text is hashable): `none` / trim 0 / trim 98304 all
**`c5421068b4e0`** (7465 chars) at 3 GPU — the trim still only loses proposals.

## 2. Arm 2 — depth-16384 decode

Prompt `prompts/prose-rdna-boosts.txt` ×3 = **15746 tokens**, `sha256 6fec9806b375…`, `-c 20480` (decode
runs from depth ≈15746 to ≈18700).  P axis:

| arm | 2 GPU | 3 GPU |
|---|---:|---:|
| trim 0 | 92.2 | 84.4 |
| trim 98304 | 92.3 (+0.1 %) | 83.2 (−1.4 %) |

So the trim is **flat at depth** (2 GPU +0.1 %, 3 GPU −1.4 %, same sign as arm 1) — no depth-specific
effect on the trim itself, which is the question this arm asked.  Purity at depth, 3 GPU: `none` / trim 0 /
trim 98304 all **`12048e399854`** (3561 chars).  Bonus datapoint (belongs to `TODO.md` #51): the 3rd GPU
costs **−8.5 %** at depth (84.4 vs 92.2) while at shallow P it *gained* 4.2 %, and the shallow→deep penalty
is −14.1 % at 3 GPU vs −2.2 % at 2 GPU.

## 3. Arm 3 — MoE with `-ncmoe` / expert cache active

Qwen3.6-35B-A3B **Q8_0** (embedded MTP head, `qwen35moe.nextn_predict_layers = 1`; the UD-Q4_K_M on this
box carries **no** nextn block and cannot draft), 2 GPU `-sm tensor -ncmoe 40`, q8_0 KV, cache auto:

| axis | trim 0 | trim 98304 | Δ |
|---|---:|---:|---:|
| P (prose) | 196.1 | 197.9 | +0.9 % |
| K (recall) | 64.1 | 63.7 | −0.6 % (identical acceptance, 0.96951 / mean len 27.50) |

Load confirms the arm is what it claims: `ROCm_Host model buffer size = 32640.00 MiB` (host experts),
`MoE expert cache (auto): arena 32640.0 MiB of 32640.0 MiB host experts (100.0 % residency)`, 120 tables,
`arena alloc failures = 0`.  Neutral, as the fully-resident MoE arm was; no cache/residency interaction.
MoE carries no purity promise (`benchmarks/mtp-adaptive-methodology.md` rule 3); both arms produced sane
text and healthy acceptance.  (The 196.1 t/s vs the fully-resident arm's 233.5 is the host-expert cost,
not the trim.)

## 4. Gates

| gate | result |
|---|---|
| `scripts/check-gpu-faults.sh --since "90 min ago"` (kernel journal + all run logs) | **clean** |
| greedy purity, 2 GPU shallow (r107) / 3 GPU shallow / 3 GPU depth | pass (see hashes above) |
| `n_max <= 7` band | kept in every arm (`--spec-draft-n-max 7`, `-n-start` default) |
| MoE expert cache | 100 % residency, 0 alloc failures, no faults |

## 5. What this means for the block

1. **The 2-GPU guidance is unchanged**: `--spec-draft-n-max 7 --spec-draft-mtp-vocab 98304` on the dense
   27B is +1.4 % aggregate (r107: +1.6 %).
2. **Do not enable the trim under `-sm tensor` when `N · n_devices >= n_vocab`.**  At 3 GPUs on the 27B the
   shipped 98304 is above the per-device share (82773) and costs 2.2 % aggregate on a dense model with a
   full four-axis gate; on the MoE model it is neutral, not negative, because the experts dominate the
   token.  The rule is a width rule, not a device-count rule: `N < n_vocab / n_devices`.
3. **gemma4 corollary (not measured):** its only lossless cut is ≥ 245760 of 262144, i.e. above the 2-device
   share (131072), so under `-sm tensor` even the "no-op" cut would be a width loss.  It stays
   unsupported-in-practice for a different reason than the acceptance one already recorded.
4. **The two-band idea (old item 4) stays unshipped** — it measured 0.4 % slower on the payload it was
   built for, and with the width rule now known a *second* band only makes the mirrored matmul wider.

Open follow-ups (not failable gates, recorded so they are not re-derived): whether a per-device *slice* of
the subset (rather than a mirrored copy) would make the trim device-count neutral — that is a `Meta`
buffer-type change, not a knob; and the 3-GPU decode deficit itself, which is `TODO.md` #51 /
`wip/three-gpu-performance/`.
