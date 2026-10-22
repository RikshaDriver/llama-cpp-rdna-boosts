# RESOLUTION — `async-input-per-origin` (issue #201, TODO #84) + the RDNA3_5 `mmf32` default

**Status: RESOLVED (2026-10-21), promoted into `v12-097cf33d9-r105` as two block amendments — block 10 (the
ring) and block 06 (`mmf32` default extended to RDNA3_5).**  Release entry: `WORKLOG.md` 261021.015; the
record correction this campaign owed is its own entry, `WORKLOG.md` 261021.014.

## The defect (confirmed against the r104 tree, then measured)

`ggml_cuda_async_input_origin_locked()` keyed the per-origin map on `cudaStreamPerThread`, which is the
**sentinel** `((hipStream_t)2)` — the same value in every host thread (`hip_runtime_api.h:1160`).  The map
could therefore only ever hold **one** entry, so r101's per-origin ordering state was in fact shared per
device.  Three consequences, all fixed:

1. **Cross-thread drain.**  `wait()` iterated every origin and issued its queued uploads on the *caller's*
   stream, then cleared the state.  Thread B drained thread A's queue and A's graph then found nothing to
   wait for — while A's copy had been issued on B's stream.  Fixed by keying on `std::this_thread::get_id()`
   and by letting **only the owning thread consume** an origin's `pending`/`has_flush`.  A foreign wait may
   still *issue* a foreign queue: the loader legitimately uploads a graph input from another thread, and an
   upload must never stay queued when the graph that reads it runs.
2. **`o.q.empty() → continue` skipped the `has_flush` wait**, so a queue drained by another thread left the
   owner unordered behind that flush.  The `ev`/`flush_ev` waits now happen before the empty-queue early-out.
3. **`o.stream` was documented as an identity it never was.**  It stays `cudaStreamPerThread` (uploads are
   issued on the stream the synchronous path uses), and that is sound **because ordering is carried by the
   events**, recorded where the work was issued; a consumer waits on the event, never on a stream handle.
   The r101 comment block ("a stream is a `llama_context`") is replaced with this model — an origin is a host
   thread.

**A fourth gap, which only a correct key exposes.**  The ring BYTES are per device, so the head wrap must not
reuse a slot while *any* origin can still read it.  The old wrap drained only the caller's queue and synced
other origins' events — and a foreign origin's uploads that are still **queued** have no event yet, so their
bytes would be overwritten before the copy was ever issued.  The wrap now issues every origin's queue, then
`cudaDeviceSynchronize()`, then clears the per-origin guard flags and resets the head.  It is the only
ring-reuse site, happens once per 16 MiB of small uploads, and is host-blocking by design (maintainer
decision, 2026-10-21: the device sync, not per-origin event bookkeeping).

`GGML_CUDA_ASYNC_INPUT_DEBUG=1` was added as the instrument: per device it logs every origin created (map key,
calling thread, resulting origin count) and the first 16 waits (origin count, uploads issued by that call,
event/flush waits).  Diagnostic only.

## The proof (`repro/`)

Two host threads, each with its own CUDA backend (own context/stream) on **one device**, uploading a graph
input and running a graph per round:

| arm | origins | the owner's ordering after a foreign drain |
|---|---|---|
| r104 semantics (sentinel key, unconditional consume) | **1** (same key in both threads) | **lost** (`issued=0 ev=0 flush=0` on the second thread's first wait) |
| the fix (host-thread key, owner-only consume) | **2** (`key == thread` each) | **kept** (`flush=1` consumed by the owner) |

Re-verified on all three validated boxes: gfx1201 (`soar`), gfx1151 (`halo`), gfx1100 (`fingon`) — 2 origins on
each.  Logs: `repro/r104-arm.log`, `repro/fixed.log`; recipe in `repro/README.md`.

## Gates (gfx1201 unless stated; one GPU job at a time)

* **No arithmetic change.**  `GGML_CUDA_ASYNC_INPUT=0` A/B through the prefill-logit gate with the hipBLASLt
  solution pinned on both arms (`ROCBLAS_USE_HIPBLASLT=0`, `GREEDY-PURITY.md` §44): **0.000000 / 100.000 %** —
  the ring is byte-identical to the old path.  Standard point (dense 27B Q8_0, wikitext, `-c 512 --chunks 40`):
  base re-recorded on the shipping default (`arm=<default>`, dump sha `44d7605be6…`); the pre-ring-change
  build of this tree and the r99-era `build-rocm-hybrid` read the *same* 0.000707 / 98.755 % against the old
  2026-10-08 base, i.e. one class off that old build and identical to each other.
* **Ops the ring feeds**: `test-backend-ops -o SET,SET_ROWS,CPY` **422/422**, `-o MUL_MAT` **2444/2444**
  (`LLAMA_TEST_OP_NMSE=1 LLAMA_TEST_SEED=1`).
* **Determinism** (logits-dump hash, fresh process per start): 35B-A3B **12/12** one class at 1 GPU
  (`d598fa841cff`, the class r104 recorded) and **4/4** at 2 GPU `-sm tensor` (`e5179311f6d2`); dense 27B
  **4/4**.  On `halo` (gfx1151): 35B-A3B Q4_K_M **12/12** (`3f86d992a67f`, the class the r104 campaign
  recorded for `MMF32=1`).
* **Perf, no regression** (3 interleaved visits, ring on vs `GGML_CUDA_ASYNC_INPUT=0`): 35B-A3B 2 GPU
  `-sm tensor -ncmoe 40` → pp512 **+2.7 %**, pp2048 **+1.6 %**, tg128 **+4.3 %** (117.12 vs 112.28 t/s).
  Flash-Next GSQ-IQ3_XXS 2 GPU `-ncmoe 48 -ub 2048 --load-mode mmap --lazy-mode off` → pp2048 −0.3 % (inside a
  ±55-62 spread), tg128 **+4.9 %**.  The r90 +9.3 % / +5.2 % decode result survives.
* **Coherence**: 4B Q8_0 3 GPU `1c5d32ac537d`, identical to r93…r99.
* **`mmf32` for RDNA3_5** (`halo`, the arm that now ships by default): `test-backend-ops -o MUL_MAT`
  2444/2444; determinism 12/12 one class (**3f86d992a67f**); **+1.24 % pp512** (2238.4 vs 2211.0),
  **+1.42 % pp2048** (2233.5 vs 2202.3), decode flat; `--family-switch` at the shipped confident point
  (`prompts/kld-corpus-35b-in-domain.txt`, `-c 2048 --chunks 7`, base arm `GGML_CUDA_MMF32=0`, base PPL
  **1.4345**): candidate **0.013016 / 98.673 %** against a same-kernel control of **0.019125 / 97.975 %** →
  PASS with `--allow-control-pass` — the p5 cross-arch numbers reproduced exactly.
* **`mmf32` stays opt-in on RDNA3** (`fingon`, gfx1100): 0 intercepts with the default environment; the
  reproducer shows 2 origins there too.

## The record correction (do not inherit the r101 credit)

`WORKLOG.md` 261021.003 and `archive/work/fix-gemma-4/` credited this ring with taking the gemma4 MTP arm from
*unstable in every sample* to 8/10 "by removing a genuine cross-context ordering defect".  A patch that cannot
produce more than one origin cannot have done that, and the re-derivation refutes it: on the r105 tree the
arm is **10/10 one class with hipBLASLt on** (`05e32c4dee`, and 3/3 `867a09a093` at `-n 3000`), 10/10 with it
pinned, and **2 classes in 6** with `GGML_CUDA_MMF32=0` (`6f6923ef36` ×4, `51274cb0f2` ×2 — the exact two
classes §44 recorded).  The instability was the hipBLASLt solution lottery on the F32 router GEMM
(`m=128 n=375 k=2816`, inside `mmf32`'s window): pinned at r101 by `ROCBLAS_USE_HIPBLASLT=0`, and **removed
by r104's `mmf32`**.  Correcting entry: `WORKLOG.md` 261021.014; `archive/work/fix-gemma-4/RESOLUTION.md`
carries a pointer.

## Instruments and notes for the next session

* `GGML_CUDA_ASYNC_INPUT_DEBUG` (`ENVIRONMENT.md` §7.4, beside `GGML_CUDA_ASYNC_INPUT`) — the origin-count instrument.
* `scripts/gate-prefill-logits.sh` defaults `LLAMA_PPL` to `~/llama.cpp/build-rocm-hybrid`, which on `soar` was
  a **stale Oct-17 build**: always pass `--ppl` for a release gate.
* `--ab` **overwrites the standard base file** with the fallback arm's dump.  `--record` last on the command
  line resets the arm to `<default>` (`--ab` then `--record` = record the default arm); use that to restore it.
* A `--load-mode mtp` value does not exist in this tree (`auto|none|mmap|mlock|mmap+mlock|dio`); Flash-Next's
  field config here is `--load-mode mmap --lazy-mode off`.
* The origin map grows one entry per host thread that uploads (bounded by the process's threads, no erase).
  If pruning is ever added it must never drop an entry with `pending` or `has_flush` set.
