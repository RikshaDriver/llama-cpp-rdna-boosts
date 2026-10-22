# repro — the async-input ring's origin count, measured (issue #201)

Two host threads, each with its **own CUDA backend** (its own context and compute stream) on **one device**,
each round uploading a small graph input (`ggml_backend_tensor_set` into a `COMPUTE` buffer, <= 1 MiB, i.e. the
async-input ring, which is default-on) and then running a graph (which calls `ggml_cuda_async_input_wait`).

`GGML_CUDA_ASYNC_INPUT_DEBUG=1` is the instrument the fix adds (README §2.4). It prints, per device:

* `AIN origin:` every origin **created** — the map key, the calling thread, and the resulting origin count;
* `AIN wait:`   the first 16 waits — the key this thread resolves to, the origin count, how many **queued
  uploads that call issued**, and how many event/flush waits it performed.

Both `key` and `thread` are printed through `std::hash<std::thread::id>` (stable within a process), so the two
arms are directly comparable.

## Build and run

```bash
g++ -O2 -std=c++17 -o ain_origin_repro ain_origin_repro.cpp \
    -I ~/llama.cpp/ggml/include -I ~/llama.cpp/include \
    -L ~/llama.cpp/build-rocm/bin -lggml-hip -lggml -lggml-base \
    -Wl,-rpath,$HOME/llama.cpp/build-rocm/bin

LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib GGML_CUDA_ASYNC_INPUT_DEBUG=1 \
  HIP_VISIBLE_DEVICES=0 ./ain_origin_repro 2>&1 | grep -E 'AIN |repro:'
```

## The two arms

The **r104 arm** is the fixed tree with exactly the two r104 semantics this issue reports restored, and nothing
else:

1. the map key is the `cudaStreamPerThread` sentinel instead of the host thread id
   (`ggml_cuda_async_input_origin_key()`), and
2. the shared origin's state is consumed unconditionally (`const bool owned = true;`) instead of only by the
   owning thread — which is what r104 did, because with one origin there was no owner.

(The debug log itself is identical in both arms.)

## `r104-arm.log` — one origin, and a foreign-thread drain

```
AIN origin: dev=0 key=785287a565e85510 thread=9bf81d9b72403918 origins=1
AIN wait: dev=0 key=785287a565e85510 thread=9bf81d9b72403918 origins=1 issued=2 ev=0 flush=1
AIN wait: dev=0 key=785287a565e85510 thread=19781a7c1b134b64 origins=1 issued=0 ev=0 flush=0
```

* **One** origin, and the key is the *same constant* in both threads (`785287a565e85510` == the hash of the
  sentinel `2`): the second thread joins the first thread's state. `origins=1` in every wait.
* Round 0: thread `9bf8…` issued **2** uploads — its own **and** the one `1978…` had queued — and waited on the
  flush. Thread `1978…` then found a drained queue and **nothing at all to wait for** (`issued=0 ev=0 flush=0`)
  while its copy had been issued on the *other* thread's stream. That is the foreign-thread drain of README
  §1.1, and it is the exact hole: its graph can read the buffer before the copy lands.

## `fixed.log` — two origins, and the owner keeps its ordering

```
AIN origin: dev=0 key=9cc0c00829b98feb thread=9cc0c00829b98feb origins=1
AIN origin: dev=0 key=faa596e929dc6d0b thread=faa596e929dc6d0b origins=2
AIN wait: dev=0 key=9cc0c00829b98feb thread=9cc0c00829b98feb origins=2 issued=2 ev=0 flush=2
AIN wait: dev=0 key=faa596e929dc6d0b thread=faa596e929dc6d0b origins=2 issued=0 ev=0 flush=1
```

* **Two** origins on one device, one per host thread, each keyed on its own thread (`key == thread`).
* Round 0: `9cc0…` still issues both queues (a foreign issue is legitimate — the loader uploads from another
  thread and a graph input must never stay queued), but it can no longer *consume* the other origin's state.
  `faa5…` therefore still has its flush to consume (`flush=1`) even though its queue is empty — this is the
  secondary gap of README §1 (`o.q.empty() -> continue` skipped the `has_flush` wait) closed: the owner waits on
  the event **recorded where the work was issued**, never on a stream handle.

## Result

| arm | origins on one device | owner's ordering after a foreign drain |
|---|---|---|
| r104 (sentinel key, unconditional consume) | **1** | lost — `issued=0 ev=0 flush=0` |
| the fix (host-thread key, owner-only consume) | **2** | kept — `flush=1` consumed |

The reproducer is diagnostic only and is **not** part of the delivery (it lives with this campaign and moves to
`archive/work/async-input-per-origin/repro/` with it).
