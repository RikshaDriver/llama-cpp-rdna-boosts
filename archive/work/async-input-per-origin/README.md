# wip/async-input-per-origin — make the async-input ring's "per-origin" state actually per-origin (issue #201)

**Status: OPEN (seeded 2026-10-21, r104).**  Nothing here is in `patches/`; nothing has been applied.  This
directory is the campaign handover: the finding, the fix design, the gates, and the **one promotion this
campaign also owns** (RDNA3_5 default-on for `mmf32`, deliberately bundled into the same release — see §6).

Subject: <https://github.com/stew675/llama-cpp-rdna-boosts/issues/201> — *"block 10: the async-input ring keys
its 'per-origin' state on the cudaStreamPerThread sentinel (constant `2`), so the map holds one origin and
r101's per-origin ordering does not exist"*.

## 1. The finding (re-verified against the r104 tree, not just the issue text)

`GGML_CUDA_ASYNC_INPUT` is default-on (r90) with `GGML_CUDA_ASYNC_INPUT_BATCH` default-on inside it (r89).  The
ring stages small graph-input copies through a per-device pinned buffer; r101 added per-**origin** ordering
state, keyed like this (`ggml/src/ggml-cuda/ggml-cuda.cu`, r104 anchors):

```cpp
// :1352  struct ggml_cuda_async_input_dev { ... std::map<cudaStream_t, ggml_cuda_async_input_origin> origins; };
// :1444
static ggml_cuda_async_input_origin & ggml_cuda_async_input_origin_locked(ggml_cuda_async_input_dev & d) {
    const cudaStream_t key = cudaStreamPerThread;      // <-- the defect
    ...
}
```

`hipStreamPerThread` is a **sentinel macro**, not a handle: `#define hipStreamPerThread ((hipStream_t)2)`
(`/opt/rocm-10.1.0-gfx120X/include/hip/hip_runtime_api.h:1160`; CUDA is `(cudaStream_t)0x2`).  The *runtime*
resolves it to the calling thread's stream per API call, but the value the code holds — and hashes on — is `2`
in **every** thread.  So `d.origins` can only ever hold one entry and every field of the "per-origin" state is
shared exactly as the pre-r101 per-device struct was.  Three consequences, all still open in r104:

1. **Cross-thread drain.**  `ggml_cuda_async_input_wait(device, stream)` (:1674) iterates *all* origins and
   issues their queued uploads **on the caller's stream**.  With one origin, thread B can drain the uploads
   thread A queued; A's later graph finds `o.q` empty and never waits, so A's graph may read a buffer before
   its copy lands — the exact failure the r101 comment says it fixed.
2. **Ring wrap can reuse a slot another thread is still reading** (:1587): the code syncs `o.stream` (which is
   "whatever thread is calling") and then loops "every OTHER origin" — which is empty.
3. **`o.stream` is not "this origin's stream".**  It stores the sentinel, i.e. "the calling thread's stream".
   The r101 comment block at :1353-1360 says *"A stream is a `llama_context`: MTP runs the draft and the target
   context concurrently on the same devices, each on its own host thread"* — the origin is a **host thread**,
   and a `llama_context`'s compute stream is a different object this state never keys on.  The comment must be
   corrected along with the code, or the next reader re-derives the same wrong model.

**Secondary gap that survives a correct key** (:1679): `if (o.q.empty()) { continue; }` skips the `has_flush`
wait below it, so a queue that *another* thread drained (which sets `A.has_flush`) leaves A's next graph
unordered behind that flush.  Keying the map alone does not fix the hazard.

**Impact.**  Latent in the default single-context path — this tree keeps one compute thread per device
(`server_queue::yield_to_queue()` runs the MTP draft decode on the calling thread), so today the collapse is
mostly a no-op.  It becomes real for two `llama_context`s decoded from two threads on one device (legitimate
llama.cpp usage, and the case the comment describes), for synchronous backend access from a second thread
during a decode (state save/restore through the HTTP threads), and for any future "draft on its own thread".

**It is also a record problem.**  r101 `WORKLOG.md` 261021.003 and `archive/work/fix-gemma-4/`
(`HANDOVER-2026-10-21b.md` §2, `RESOLUTION.md`, `MEASUREMENTS-2026-10-21d.md` §5) credit this patch with taking
the gemma4 MTP arm from *unstable in every sample* to 8/10 "by removing a genuine cross-context ordering
defect".  A patch that cannot produce more than one origin cannot be responsible for that step.  The residual
was then correctly attributed to the hipBLASLt lottery (`GREEDY-PURITY.md` §44, which already records that "the
async-input ring is not the residual"), which leaves the 0 → 8/10 step **unexplained**.  Re-derive it after the
fix; do not inherit the credit.  (Correcting a dated record means a new `WORKLOG` entry that states the
correction and points at the old one — never editing the old entry.)

## 2. What the fix is

1. **Key the map on a real per-thread key** — `std::this_thread::get_id()` (or a `thread_local` token) instead
   of `cudaStreamPerThread`.  Keep *issuing* on `cudaStreamPerThread`: the runtime resolves it per calling
   thread, so the issue target is already correct and only the keying is wrong.  Fix the :1353 comment.
   - `ggml_cuda_async_input_origin::stream` then means "the stream the issuing thread has", which for a
     foreign-thread drain is the **drainer's** stream.  That is fine *because* the ordering is carried by
     `o.ev` / `o.flush_ev` (recorded where the work was issued) and the owner waits on the **event**, never on
     a stream.  State that in the code comment; it is the property the whole fix rests on.
   - Lifetime: the map grows one entry per host thread that uploads; no erase is needed (bounded by the
     process's threads) — if the implementer adds pruning, it must never drop an entry with `pending` or
     `has_flush` set.
2. **Consume `flush_ev` even when the queue is empty** in `ggml_cuda_async_input_wait()`: wait on (and clear)
   `o.has_flush` before the `o.q.empty()` early-out.
3. The wrap path's "every other origin" sync then becomes meaningful — keep it, and keep the per-origin `ev`
   (per-copy path) / `flush_ev` (batched path) guards.
4. **Add a diagnostic** (`GGML_CUDA_ASYNC_INPUT_DEBUG=1`) that logs, per device, the origin count and thread id
   when an origin is created and on the first few `wait()`s — so a gate can *prove* >1 origin exists in a
   two-thread run instead of assuming it.  (House convention: `fprintf(stderr, …)` gated on the variable, first
   N prints only — see `GGML_CUDA_MMF32_LOG` in block 06.)

Owning block: **10** (qwen4exp: model, hyper-connections, lazy reader, QSA/indexer, the async graph-input
ring).  Amend it; never append.

## 3. Gates (all of them; this is an ordering change, so "no arithmetic change" is a claim to prove)

1. **The defect is provably gone**: a two-thread reproducer (or the §2.4 debug log) must show **2 origins** on
   one device in a two-context/two-thread run; today it shows 1.  Keep the reproducer in this directory.
2. **No arithmetic change**: `test-backend-ops` (full, or at least the ops the ring feeds: `SET`, `SET_ROWS`,
   `CPY`, `MUL_MAT`), plus `scripts/gate-prefill-logits.sh` at the delivery's standard point — expect
   `0.000000 / 100.000 %` (a base must be recorded on this build; the upload path must not move a logit).
3. **Determinism, per box**: 12 fresh `llama-perplexity -c 512 --chunks 1` starts = **1 class** on gfx1201
   (also 4 at 2 GPU `-sm tensor`), and the same on `halo` (gfx1151) once it carries the tree.
4. **The r101 gemma4 claim, re-derived**: 10 × `-n 300` (`--spec-type draft-mtp --spec-draft-n-max 3 -ncmoe 40
   -sm layer`, 2 devices) and 3 × `-n 3000`, **one arm with `ROCBLAS_USE_HIPBLASLT=0` and one with it on**;
   record the *classes*, never a single hash (`archive/work/fix-gemma-4/tools/fp.sh` is the instrument).  This
   is the number that decides whether the r101 credit stands or the record gets corrected.
5. **`GGML_CUDA_ASYNC_INPUT=0` A/B**: same-seed text and prefill logits must be identical (the kill switch must
   still be the old path).
6. **Perf**: this must not cost anything measurable — `llama-bench -p 512,2048 -n 128 -r 3` on 35B-A3B (3
   interleaved visits) and a Flash-Next `-ncmoe` run, both against `GGML_CUDA_ASYNC_INPUT=0`; the r90 record's
   +9.3 % / +5.2 % decode must survive.
7. `scripts/validate-set.sh` + `scripts/gate-per-block-build.sh` for the amended blocks, then the release
   mechanics in `MANIFESTS.md` / `BASELINE.md`.

## 4. Why it was not fixed in the session that filed it

It was filed from a code read at the end of the r104 session, with no code changes made, and the delivery's
rule is that a work item gets its own session, gates and record.  The r104 release (`mmf32`) is independent of
this and shipped without it.

## 5. The one thing that must NOT be inherited

The `0 → 8/10` gemma4 MTP step credited to r101 (§1).  Re-measure before repeating it.  If the numbers do not
reproduce, the r101 entry gets a correcting `WORKLOG` note and `archive/work/fix-gemma-4/RESOLUTION.md` gains a
pointer — the old record is not edited.

## 6. Co-deliverable: promote `mmf32` to default-on for RDNA3_5 in the SAME release (r105)

The maintainer's decision (2026-10-21): RDNA3_5 is validated, so it should be default-on — bundled with this
campaign's fix rather than cut as a release of its own.  Evidence and numbers:
`archive/work/hipblaslt-bypass/evidence/p5-cross-arch-validation.txt` and `WORKLOG.md` 261021.012.

**The change** (block 06, `ggml/src/ggml-cuda/mmf32.cu`, `ggml_cuda_mmf32_enabled`):

```cpp
    if (env >= 0) {
        return env == 1;
    }
    return GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc);   // was: IS_RDNA4 only
```

and update the comment above it (RDNA4 **and** RDNA3_5 are validated; RDNA3 stays opt-in because it measured
−1.3…−2.2 % at `pp2048`, one tie, on `fingon`).

**Why RDNA3_5 (halo, gfx1151) qualifies** — the same evidence shape that justified RDNA4 at r104:

| leg | gfx1151 result |
|---|---|
| correctness | `test-backend-ops -o MUL_MAT` 2444/2444; NMSE **6.11e-14 vs 6.83e-13** (rocBLAS) on the intercepted cases, threshold 5e-4 — 11x closer, and the *same* kernel NMSE as gfx1201 |
| determinism | 1 logits class in **12/12** fresh starts (`MMF32=1`); the box's default did not flip in that config |
| prefill | **+1.0 % pp512** (2225 vs 2203), **+1.6 % pp2048** (2227 vs 2192), decode flat, 3 interleaved visits |
| purity | `--family-switch` at `prompts/kld-corpus-35b-in-domain.txt` (base PPL 1.4345): **0.013016 / 98.673 %** vs same-kernel control **0.019125 / 97.975 %** → PASS with `--allow-control-pass` |

Docs to update with it: `ENVIRONMENT.md` §10.1 (the default column and the "validated device" sentence),
`patches/README.md` (block 06 scope/default), `GREEDY-PURITY.md` §47's scope paragraph, `TODO.md`'s accepted-
limitation bullet (which currently says "RDNA3/RDNA3_5 are opt-in"), and `MANIFESTS.md`'s block-06 row if it
names the arch.  **Re-record the gfx1151 prefill base on the new default** and log the family-switch magnitude
(`benchmarks/prefill-logit-methodology.md` §5b): the 0.013016 / 98.673 % above is that number.

RDNA3 (gfx1100, `fingon`) stays opt-in.  If it ever needs the kernel for determinism rather than speed, the
lever is a per-arch tile (`BN`/`ks` in `ggml_cuda_mul_mat_f32_skinny`), not the gate — it measured −1.3…−2.2 %
at `pp2048` (one visit a tie) with tight spreads, so a pinned-tile retune is plausible but unproven.

## 7. Release shape for the bundled r105

Two block amendments, one release:

| block | change |
|---|---|
| **10** | the async-input per-origin key + the `has_flush` gap + the debug log (this campaign) |
| **06** | `mmf32` default extended to RDNA3_5 (the one-line predicate) |

Then the usual: `scripts/make-patches.sh` / `make-release.sh`, `validate-set.sh`,
`gate-per-block-build.sh --changed-from <r104 tip>`, the gates in §3, the fork `rdna-boosts` refresh
(`--force-with-lease`, personal fork only), the tag, and the container worker picks it up.

## 8. Files

| file | what |
|---|---|
| `NEXT-SESSION-PROMPT.md` | the cold-start prompt for the session that does this work |
| `README.md` | this record |
| `repro/` | (to be created) the two-thread origin reproducer and its output |

Related records: `WORKLOG.md` 261021.002/.003 (r101 `fix-gemma-4` fold, the credit to re-derive), 261019.006/
.007 (r90: the ring is correct, the snapshot defect was `set_tensor_async`), `GREEDY-PURITY.md` §41/§44,
`archive/work/async-input-graph-ordering/` (the r90 resolution), `archive/work/fix-gemma-4/`,
`archive/work/hipblaslt-bypass/evidence/p5-cross-arch-validation.txt` (the RDNA3_5 numbers).
