// ain_origin_repro -- prove the async-input ring's ordering state is per HOST THREAD (issue #201).
//
// Two host threads, each with its OWN CUDA backend (= its own llama_context-like context and compute
// stream) on ONE device.  Each thread repeatedly
//   1. uploads a small graph input with ggml_backend_tensor_set (COMPUTE buffer, size <= 1 MiB, i.e. it
//      takes the async-input ring when GGML_CUDA_ASYNC_INPUT is on -- the default), and
//   2. runs a graph, which calls ggml_cuda_async_input_wait() for that device.
//
// Run it with GGML_CUDA_ASYNC_INPUT_DEBUG=1.  The ring then logs every origin it CREATES (device, thread
// identity, resulting origin count) and every wait (origin count + how many queued uploads THAT call
// issued).  The two signatures to compare:
//
//   r104 (the defect) -- the map is keyed on cudaStreamPerThread, the sentinel `((hipStream_t)2)`, so:
//       * "AIN origin: ... origins=1" appears ONCE (the second thread joins the first thread's origin),
//       * one thread's wait issues 2 uploads and the other's issues 0 -- thread B drains thread A's queue,
//         and A's own graph then finds nothing to wait for.  That is the foreign-thread drain.
//
//   the fix -- the map is keyed on std::this_thread::get_id(), so:
//       * "AIN origin: ... origins=2" appears for the second thread,
//       * every wait issues exactly its own upload (1) and never another thread's (origins=2, issued=1).
//
// Build (both the r104 build and the fixed build work; the log is the instrument):
//   g++ -O2 -std=c++17 -o ain_origin_repro ain_origin_repro.cpp \
//       -I ~/llama.cpp/ggml/include -I ~/llama.cpp/include \
//       -L ~/llama.cpp/build-rocm/bin -lggml-hip -lggml -lggml-base \
//       -Wl,-rpath,$HOME/llama.cpp/build-rocm/bin
// Run:
//   LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib GGML_CUDA_ASYNC_INPUT_DEBUG=1 HIP_VISIBLE_DEVICES=0 \
//       ./ain_origin_repro

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static const int N_TOKENS  = 256;   // graph-input element count
static const int N_ROUNDS  = 3;     // upload + graph per thread

static void worker(int id) {
    // one backend per thread: a distinct context and compute stream, on the SAME device
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        fprintf(stderr, "worker %d: ggml_backend_cuda_init(0) failed\n", id);
        std::exit(1);
    }

    struct ggml_init_params ip = { /*mem_size*/ 4u << 20, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor  * t   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_TOKENS);
    ggml_set_name(t, "ain_repro_input");
    struct ggml_tensor  * r   = ggml_sqr(ctx, t);
    ggml_set_name(r, "ain_repro_out");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fprintf(stderr, "worker %d: alloc_ctx_tensors failed\n", id);
        std::exit(1);
    }
    // the async-input ring only takes COMPUTE-usage buffers (the scheduler's graph inputs)
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    std::vector<float> host(N_TOKENS);
    for (int i = 0; i < N_TOKENS; ++i) {
        host[i] = (float) (id*1000 + i)*1e-3f;
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);

    for (int round = 0; round < N_ROUNDS; ++round) {
        // 1. small host -> device graph-input upload: queued in the ring
        ggml_backend_tensor_set(t, host.data(), 0, host.size()*sizeof(float));
        // 2. a graph compute: calls ggml_cuda_async_input_wait() -> the queue is issued and ordered
        ggml_backend_graph_compute(backend, gf);
        fprintf(stderr, "repro: thread %d round %d done\n", id, round);
    }

    ggml_backend_synchronize(backend);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

int main(void) {
    fprintf(stderr, "repro: 2 threads x %d rounds, one device (graph-input upload + graph compute each)\n",
            N_ROUNDS);
    std::vector<std::thread> th;
    for (int i = 0; i < 2; ++i) {
        th.emplace_back(worker, i);
    }
    for (auto & t : th) {
        t.join();
    }
    fprintf(stderr, "repro: done\n");
    return 0;
}
