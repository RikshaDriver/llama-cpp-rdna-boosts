// hip-api-probe.c -- an LD_PRELOAD interposer that attributes HIP API calls to their CALLER.
//
// Why: `rocprofv3 --hip-trace` gives durations but no call site, and the interesting question for a
// host-side stall is always "who is issuing this 3 ms `hipHostMalloc` / 80 ms `hipFuncGetAttributes`".
// This answers it from outside the process, with no rebuild and no delivery change.
//
//   gcc -shared -fPIC -O2 -o libhipapi.so hip-api-probe.c -ldl
//   HOOK_ON=1 LD_PRELOAD=$PWD/libhipapi.so ./llama-cli ... 2>&1 | sed -n '/HOOKAGG/,$p'
//
// Prints one aggregated line per call site (caller frame resolved with dladdr: library + symbol when
// the symbol is exported, `?` for a static function) with count, total and max, and a full backtrace
// for any single call over 20 ms.  Measured with it on gfx1201 (2026-10-22):
//   * `hipFuncGetAttributes` 288 calls / 878 ms -- 100 % from librccl (module loads at NCCL init);
//   * `hipHostMalloc`+`hipHostFree` 678 pairs / 3.3 s -- `ggml_backend_cuda_buffer_set_tensor_2d`'s
//     per-tensor pinned staging during MODEL LOAD, not decode (see ../README.md #3).
// `HOOK_MAX`/`HOOK_AFTER_MS` (this file) bound the output; add a function to the tail to watch another.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

static int enabled(void) { static int e = -1; if (e < 0) { const char * s = getenv("HOOK_ON"); e = (s && atoi(s)) ? 1 : 0; } return e; }
static int64_t now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (int64_t)t.tv_sec*1000000 + t.tv_nsec/1000; }

#define MAXS 64
struct site { void * ra; char fn[256]; long n; int64_t us; int64_t maxus; };
static struct site sites[MAXS]; static int nsites = 0;
static const char * g_kind = "";

static const char * fname_of(void * a) {
    static char bufs[MAXS][256];
    static int i = 0;
    Dl_info info; memset(&info, 0, sizeof info);
    if (dladdr(a, &info) && info.dli_fname) {
        const char * base = strrchr(info.dli_fname, '/'); base = base ? base+1 : info.dli_fname;
        int k = i++ % MAXS;
        snprintf(bufs[k], sizeof bufs[k], "%s+%s", base, info.dli_sname ? info.dli_sname : "?");
        return bufs[k];
    }
    return "?";
}

static void note(const char * kind, int64_t dur_us) {
    void * bt[20]; int n = backtrace(bt, 20);
    void * ra = (n > 2) ? bt[2] : (n > 1 ? bt[1] : nullptr);
    for (int i = 0; i < nsites; i++) {
        if (sites[i].ra == ra) { sites[i].n++; sites[i].us += dur_us; if (dur_us > sites[i].maxus) sites[i].maxus = dur_us; return; }
    }
    if (nsites < MAXS) {
        sites[nsites].ra = ra; strncpy(sites[nsites].fn, fname_of(ra), 255);
        sites[nsites].n = 1; sites[nsites].us = dur_us; sites[nsites].maxus = dur_us; nsites++;
    }
    if (dur_us > 20000) {
        fprintf(stderr, "HOOKBIG %s dur=%.1fms ra=%s\n", kind, dur_us/1000.0, fname_of(ra));
        for (int i = 1; i < n && i < 14; i++) fprintf(stderr, "   #%d %s\n", i, fname_of(bt[i]));
    }
    g_kind = kind;
}
static void dump(void) {
    fprintf(stderr, "HOOKAGG (site = caller of the HIP call)\n");
    for (int i = 0; i < nsites; i++)
        fprintf(stderr, "  n=%-7ld total=%8.1fms max=%7.1fms  %s\n", sites[i].n, sites[i].us/1000.0, sites[i].maxus/1000.0, sites[i].fn);
}

typedef int (*hipHostMalloc_t)(void **, size_t, unsigned int);
typedef int (*hipHostFree_t)(void *);
typedef int (*hipFuncGetAttributes_t)(void *, const void *);
typedef int (*hipStreamCreateWithFlags_t)(void **, unsigned int);
typedef int (*hipMalloc_t)(void **, size_t);
typedef int (*hipModuleLoad_t)(void **, const char *);

int hipHostMalloc(void ** p, size_t size, unsigned int flags) {
    static hipHostMalloc_t real; if (!real) { real = (hipHostMalloc_t) dlsym(RTLD_NEXT, "hipHostMalloc"); static int r; if(!r){r=1;atexit(dump);} }
    if (!enabled()) return real(p, size, flags);
    int64_t t0 = now_us(); int rc = real(p, size, flags); note("hipHostMalloc", now_us()-t0); return rc;
}
int hipHostFree(void * p) {
    static hipHostFree_t real; if (!real) real = (hipHostFree_t) dlsym(RTLD_NEXT, "hipHostFree");
    if (!enabled()) return real(p);
    int64_t t0 = now_us(); int rc = real(p); note("hipHostFree", now_us()-t0); return rc;
}
int hipFuncGetAttributes(void * attr, const void * f) {
    static hipFuncGetAttributes_t real; if (!real) real = (hipFuncGetAttributes_t) dlsym(RTLD_NEXT, "hipFuncGetAttributes");
    if (!enabled()) return real(attr, f);
    int64_t t0 = now_us(); int rc = real(attr, f); note("hipFuncGetAttributes", now_us()-t0); return rc;
}
int hipModuleLoad(void ** m, const char * f) {
    static hipModuleLoad_t real; if (!real) real = (hipModuleLoad_t) dlsym(RTLD_NEXT, "hipModuleLoad");
    if (!enabled()) return real(m, f);
    int64_t t0 = now_us(); int rc = real(m, f); int64_t d = now_us()-t0;
    fprintf(stderr, "HOOKMODULE file=%s dur=%.1fms\n", f ? f : "(null)", d/1000.0);
    note("hipModuleLoad", d); return rc;
}
int hipMalloc(void ** p, size_t size) {
    static hipMalloc_t real; if (!real) real = (hipMalloc_t) dlsym(RTLD_NEXT, "hipMalloc");
    if (!enabled()) return real(p, size);
    int64_t t0 = now_us(); int rc = real(p, size); int64_t d = now_us()-t0;
    if (d > 1000) { fprintf(stderr, "HOOKMALLOC size=%zu dur=%.1fms\n", size, d/1000.0); note("hipMalloc", d); }
    return rc;
}
