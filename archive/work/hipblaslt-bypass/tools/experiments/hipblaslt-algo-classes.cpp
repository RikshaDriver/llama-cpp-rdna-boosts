// wip/hipblaslt-bypass -- arm G probe: how many BIT-CLASSES does a hipBLASLt shape have, which one
// does the heuristic pick, and which is fastest?  (NOT delivery code, NOT part of patches/.)
//
// Why: the per-start logits lottery that closed TODO #75 is hipBLASLt choosing a solution per process.
// This answers the question that decides arm G (pin the algo ourselves): is the pick merely unstable,
// or is it unstable AND slower than what we could pin?  Usage:
//
//   hipcc -O2 -std=c++17 -o /tmp/hbl-classes hipblaslt-algo-classes.cpp \
//         "-isystem/opt/rocm-10.1.0-gfx120X/include" \
//         -L/opt/rocm-10.1.0-gfx120X/lib -lhipblaslt -lamdhip64
//   LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0 ./hbl-classes [m n k [bf16]]
//
// NOTE the -isystem: with a plain -I the Fedora ROCm 7.1 headers under /usr/include win and the build
// fails on __ocml_* -- same include-order trap the delivery's ~/bin/build-llama-rocm guards against.
//
// Defaults to the 35B-A3B GDN alpha/beta shape (m=32 n=512 k=2048 f32), which is the shape whose
// per-start kernel decides the logits class (FINDINGS-2026-10-21-todo-75.md).  Other shapes measured:
//   gemma4 26B-A4B router : 128 375 2816
//   Flash-Next BF16 prefill: 320 512 10240 bf16
//
// It prints (a) the heuristic's pick, (b) the min-index pick we would use for determinism, and (c) one
// line per solution: index, checksum of D, and a rough ms/launch, so the caller can see the bit-classes.
// On some shapes the (c) sweep stops early (a solution that crashes the driver-side path); the interesting
// part -- the heuristic's class vs the fastest class -- is printed first.
//
// Measured 2026-10-21 (ROCm 10.1.0, gfx1201), see evidence/p2-arm-g-algo-classes.txt:
//   m=32 n=512 k=2048 f32  : 3 classes; heuristic lands on the SLOWEST (0.19 ms) while 140219..140222
//                            are bit-identical to each other at 0.046 ms (4x faster);
//   gemma4 router          : 4 classes; heuristic 0.174-0.290 ms, idx 140232 is 0.066 ms alone;
//   Flash-Next BF16        : 3+ classes, heuristic's class is NOT the fastest class.
// => pinning is not just a determinism fix, it may also be a prefill win.  It does NOT remove hipBLASLt.
#include "/opt/rocm-10.1.0-gfx120X/include/hipblaslt/hipblaslt.h"
#include "/opt/rocm-10.1.0-gfx120X/include/hipblaslt/hipblaslt-ext.hpp"
#include "/opt/rocm-10.1.0-gfx120X/include/hip/hip_runtime.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#define BK(x) do { hipblasStatus_t s_ = (x); if (s_ != HIPBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "hipBLASLt status %d at %s:%d\n", (int) s_, __FILE__, __LINE__); exit(3); } } while (0)

static int M = 32, N = 512, K = 2048;
static bool BF16 = false;

static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p; uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

// deterministic pseudo-random fill so every process computes the same problem
static void fill(float * p, size_t n, uint32_t seed) {
    uint32_t s = seed ? seed : 1u;
    for (size_t i = 0; i < n; ++i) { s = s * 1664525u + 1013904223u; p[i] = (float) ((int32_t)(s >> 8) % 2000 - 1000) / 1000.0f; }
}

struct Arm {
    hipblasLtMatmulDesc_t     op   = nullptr;
    hipblasLtMatrixLayout_t   A = nullptr, B = nullptr, C = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
};

static Arm make_arm() {
    Arm a;
    BK(hipblasLtMatmulDescCreate(&a.op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t tA = HIPBLAS_OP_T, tB = HIPBLAS_OP_N;
    BK(hipblasLtMatmulDescSetAttribute(a.op, HIPBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
    BK(hipblasLtMatmulDescSetAttribute(a.op, HIPBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
    // A: k x m, B: k x n (transB=N), C/D: m x n -- all row-major-agnostic here, we only need ONE fixed
    // problem to compare algorithm selection, not a bit-exact match with ggml's layout.
    BK(hipblasLtMatrixLayoutCreate(&a.A, BF16 ? HIP_R_16BF : HIP_R_32F, K, M, K));
    BK(hipblasLtMatrixLayoutCreate(&a.B, BF16 ? HIP_R_16BF : HIP_R_32F, K, N, K));
    BK(hipblasLtMatrixLayoutCreate(&a.C, HIP_R_32F, M, N, M));
    BK(hipblasLtMatmulPreferenceCreate(&a.pref));
    return a;
}

static void destroy_arm(Arm & a) {
    if (a.pref) hipblasLtMatmulPreferenceDestroy(a.pref);
    if (a.C)    hipblasLtMatrixLayoutDestroy(a.C);
    if (a.B)    hipblasLtMatrixLayoutDestroy(a.B);
    if (a.A)    hipblasLtMatrixLayoutDestroy(a.A);
    if (a.op)   hipblasLtMatmulDescDestroy(a.op);
}

int main(int argc, char ** argv) {
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    if (argc >= 5 && !strcmp(argv[4], "bf16")) BF16 = true;
    printf("== shape m=%d n=%d k=%d %s\n", M, N, K, BF16 ? "bf16" : "f32");
    int dev = 0; CK(hipSetDevice(dev));
    hipblasLtHandle_t handle = nullptr;
    BK(hipblasLtCreate(&handle));

    const size_t szA = (size_t) K * M, szB = (size_t) K * N, szC = (size_t) M * N;
    std::vector<float> hA(szA), hB(szB), hC(szC, 0.0f), hDout(szC);
    fill(hA.data(), szA, 12345); fill(hB.data(), szB, 999);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dW = nullptr;
    CK(hipMalloc(&dA, szA * sizeof(float))); CK(hipMalloc(&dB, szB * sizeof(float)));
    CK(hipMalloc(&dC, szC * sizeof(float))); CK(hipMalloc(&dD, szC * sizeof(float)));
    CK(hipMalloc(&dW, (size_t) 64 << 20));
    CK(hipMemcpy(dA, hA.data(), szA * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(dB, hB.data(), szB * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(dC, hC.data(), szC * sizeof(float), hipMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;

    // ---- ARM 1: the heuristic, exactly how rocBLAS asks for a solution -----------------------------
    {
        Arm a = make_arm();
        std::vector<hipblasLtMatmulHeuristicResult_t> res(1);
        int n_res = 0;
        hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(handle, a.op, a.A, a.B, a.C, a.C, a.pref, 1, res.data(), &n_res);
        if (st != HIPBLAS_STATUS_SUCCESS || n_res < 1) {
            printf("heur: no heuristic result (status %d, n=%d)\n", (int) st, n_res);
        } else {
            int idx = hipblaslt_ext::getIndexFromAlgo(res[0].algo);
            BK(hipblasLtMatmul(handle, a.op, &alpha, dA, a.A, dB, a.B, &beta, dC, a.C, dD, a.C,
                               &res[0].algo, dW, res[0].workspaceSize, nullptr));
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(hDout.data(), dD, szC * sizeof(float), hipMemcpyDeviceToHost));
            printf("heur   solution_index=%-7d ws=%-9zu D=%016" PRIx64 "\n", idx, res[0].workspaceSize,
                   (unsigned long long) fnv1a(hDout.data(), szC * sizeof(float)));
        }
        destroy_arm(a);
    }

    // ---- ARM 2: our own pick from getAllAlgos: the smallest stable algo index ----------------------
    {
        Arm a = make_arm();
        std::vector<hipblasLtMatmulHeuristicResult_t> all;
        hipblasStatus_t st = hipblaslt_ext::getAllAlgos(handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                       HIPBLAS_OP_T, HIPBLAS_OP_N, BF16 ? HIP_R_16BF : HIP_R_32F, BF16 ? HIP_R_16BF : HIP_R_32F, HIP_R_32F, HIP_R_32F,
                                                       HIPBLAS_COMPUTE_32F, all);
        if (st != HIPBLAS_STATUS_SUCCESS || all.empty()) {
            printf("pinned: getAllAlgos failed (status %d, n=%zu)\n", (int) st, all.size());
        } else {
            int best = -1; size_t best_i = 0;
            for (size_t i = 0; i < all.size(); ++i) {
                int idx = hipblaslt_ext::getIndexFromAlgo(all[i].algo);
                if (best < 0 || idx < best) { best = idx; best_i = i; }
            }
            BK(hipblasLtMatmul(handle, a.op, &alpha, dA, a.A, dB, a.B, &beta, dC, a.C, dD, a.C,
                               &all[best_i].algo, dW, all[best_i].workspaceSize, nullptr));
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(hDout.data(), dD, szC * sizeof(float), hipMemcpyDeviceToHost));
            printf("pinned solution_index=%-7d ws=%-9zu D=%016" PRIx64 "  (n_algos=%zu)\n", best,
                   all[best_i].workspaceSize, (unsigned long long) fnv1a(hDout.data(), szC * sizeof(float)), all.size());
        }
        destroy_arm(a);
    }


    // ---- ARM 3: every algo, its bits and a rough timing ------------------------------------------
    {
        Arm a = make_arm();
        std::vector<hipblasLtMatmulHeuristicResult_t> all;
        hipblasStatus_t st = hipblaslt_ext::getAllAlgos(handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                       HIPBLAS_OP_T, HIPBLAS_OP_N, BF16 ? HIP_R_16BF : HIP_R_32F, BF16 ? HIP_R_16BF : HIP_R_32F, HIP_R_32F, HIP_R_32F,
                                                       HIPBLAS_COMPUTE_32F, all);
        if (st != HIPBLAS_STATUS_SUCCESS || all.empty()) {
            printf("all: getAllAlgos failed (status %d)\n", (int) st);
        } else {
            printf("all: %zu algos\n", all.size());
            for (size_t i = 0; i < all.size(); ++i) {
                int idx = hipblaslt_ext::getIndexFromAlgo(all[i].algo);
                if (hipblasLtMatmul(handle, a.op, &alpha, dA, a.A, dB, a.B, &beta, dC, a.C, dD, a.C,
                                    &all[i].algo, dW, all[i].workspaceSize, nullptr) != HIPBLAS_STATUS_SUCCESS) {
                    printf("  idx=%-7d UNSUPPORTED\n", idx);
                    continue;
                }
                CK(hipDeviceSynchronize());
                CK(hipMemcpy(hDout.data(), dD, szC * sizeof(float), hipMemcpyDeviceToHost));
                // rough timing: 20 launches
                hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
                CK(hipEventRecord(e0));
                for (int r = 0; r < 20; ++r)
                    hipblasLtMatmul(handle, a.op, &alpha, dA, a.A, dB, a.B, &beta, dC, a.C, dD, a.C,
                                     &all[i].algo, dW, all[i].workspaceSize, nullptr);
                CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1));
                CK(hipEventDestroy(e0)); CK(hipEventDestroy(e1));
                printf("  idx=%-7d D=%016" PRIx64 "  %.3f ms/launch\n", idx,
                       (unsigned long long) fnv1a(hDout.data(), szC * sizeof(float)), ms / 20.0f);
            }
        }
        destroy_arm(a);
    }

    hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dD); hipFree(dW);
    hipblasLtDestroy(handle);
        return 0;
}
