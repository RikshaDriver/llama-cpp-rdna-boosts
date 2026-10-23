# Evidence for the `three-gpu-performance` / `TODO #51` resolution

r116 (`v14-097cf33d9-r116`), `~/llama.cpp/build-rocm` (ROCm 10.1.0), 3x R9700 gfx1201.
Every fresh-process arm ran with `ROCBLAS_USE_HIPBLASLT=0` (`GREEDY-PURITY.md` §44) and `-t 8`
(GPU-IRQ cores 13-15 left free); one GPU job at a time.  Field prompt = `mixed30k.txt`
(32358 tokens, sha256 `69624f4d207f40bdf7357e74ec978af5bfb4725a0968faf4eea4aab46c2ae241`, copied to
`/tmp/srr/`, never regenerated).

| file | what |
|---|---|
| `matrix-summary.csv` | **the 24-cell §1 matrix** (class, split, devices, spec, prefill, decode, acceptance, accepted/generated, mean len, arena, identity tables) |
| `raw-matrix-final.csv` | raw per-run rows for the final matrix (includes the discarded/earlier first pass, kept for audit) |
| `raw-matrix-first-pass.csv` | first pass: the PART class at `-c 40000` and the FULL class with an explicit budget (`-c 204800` runs and the 13800/device arm are described in the .md files) |
| `field-c204800.md` | **step 1**: the 2026-10-16 field/MoE arm re-measured (2 GPU 93.4 vs 3 GPU 131.2 t/s) |
| `identity-cliff.md` | the dominant residual: one arena unit per table short of 100 % drops *every* table off the identity fast path (46.7 vs 65.0 t/s) |
| `controls.md`, `ar-profile-2g.txt`, `ar-profile-3g.txt` | all-reduce per-call p50/p90/spin at 2 vs 3 devices, backend and block-count sweep, cache-off control |
| `dense-depth-plain.md`, `dense-depth-plain.txt`, `dense-3k-plain.txt` | dense 27B Q8_0 plain-decode control at depth 15746 and shallow, `-sm layer` vs `-sm tensor` (cross-count text identity holds) |
| `trim-block13.txt` | block-13 `--spec-draft-mtp-vocab` re-measured (+0.8 % at 2 GPU, −1.9 % at 3 GPU) |
| `ppl-cross-count.txt` | cross-device-count PPL on the MoE tensor split (10.8452 vs 10.8623) |

Scripts used (this directory's `../tools/`): `field116.sh` (server arm, one per line, prints the acceptance
line and the cache accounting), `cli116.sh` (llama-cli arm, hashes the cleaned generated text), and
`matrix116.sh` (the matrix driver).  `../tools/arms-2026-10-22/` is the r109 dense harness, unchanged.

Transient run logs lived in `/tmp/srr/<tag>.{out,err,txt,timing}`; the numbers above and the raw CSVs are
the record.  `scripts/check-gpu-faults.sh --since "150 min ago"` over the run logs + kernel journal: clean.
