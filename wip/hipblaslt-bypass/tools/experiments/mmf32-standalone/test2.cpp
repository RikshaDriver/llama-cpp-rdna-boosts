
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#define __launch_bounds__(x)
#include "kernel2.inc"

static void ref_gemm(const float * x, const float * y, float * d, int m, int n, int k, int ldc) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double acc = 0;
            for (int t = 0; t < k; ++t) acc += (double) x[(size_t) i*k + t] * (double) y[(size_t) j*k + t];
            d[(size_t) j*ldc + i] = (float) acc;
        }
}
static float frnd(uint32_t & s) { s = s*1664525u + 1013904223u; return (float)((int32_t)(s >> 8) % 2000 - 1000)/1000.0f; }

template<int BM,int BN,int BK,int TMR,int TNC>
static int run(int m, int n, int k, int ldc, int ks) {
    const size_t per = (size_t) ldc*n;
    std::vector<float> hx((size_t)m*k), hy((size_t)n*k), hd(per + 64, 0.f), hr(per + 64, 0.f);
    uint32_t s = 999; for (auto & v : hx) v = frnd(s);
    for (auto & v : hy) v = frnd(s);
    float *dx,*dy,*dd,*dt;
    hipMalloc(&dx, hx.size()*4); hipMalloc(&dy, hy.size()*4); hipMalloc(&dd, hd.size()*4); hipMalloc(&dt, per*ks*4);
    hipMemcpy(dx, hx.data(), hx.size()*4, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), hy.size()*4, hipMemcpyHostToDevice);
    hipMemset(dd, 0x7f, hd.size()*4); hipMemset(dt, 0x7f, per*ks*4);
    const int kchunk = (k + ks - 1)/ks;
    dim3 block(BM/TMR, BN/TNC, 1), grid((m+BM-1)/BM, (n+BN-1)/BN, ks);
    mul_mat_f32_skinny_kernel<BM,BN,BK,TMR,TNC><<<grid, block>>>(dx, dy, dt, m, n, k, ldc, kchunk);
    if (ks > 1) mmf32_reduce_kernel<<<((int)per+255)/256, 256>>>(dt, dd, (int)per, (int)per, ks);
    else { hipMemcpy(dd, dt, per*4, hipMemcpyDeviceToDevice); }
    auto e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("  %dx%dx%d ld=%d ks=%d: LAUNCH ERROR %s\n", m,n,k,ldc,ks, hipGetErrorString(e)); return 1; }
    hipMemcpy(hd.data(), dd, hd.size()*4, hipMemcpyDeviceToHost);
    ref_gemm(hx.data(), hy.data(), hr.data(), m, n, k, ldc);
    double num = 0, den = 0;
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
        const double a = hd[(size_t) j*ldc + i], b = hr[(size_t) j*ldc + i];
        num += (a-b)*(a-b); den += b*b;
    }
    const double nmse = num/den;
    const int bad = !(nmse <= 1e-7);
    printf("  %-22s m=%3d n=%5d k=%5d ldc=%3d ks=%-2d: %s (NMSE=%.2g)\n", bad ? "FAIL" : "OK", m,n,k,ldc,ks, bad ? "FAIL":"OK", nmse);
    hipFree(dx); hipFree(dy); hipFree(dd); hipFree(dt);
    return bad;
}
int main() {
    int rc = 0;
    // the production picks: m<=64 -> BN 256, else BN 64; tmr 4, tnc 8
    int ts[][5] = {{1,17,256,1,16},{7,43,300,7,16},{16,512,2048,16,16},{31,509,2051,31,16},
                   {32,512,2048,32,16},{32,44,2048,32,16},{33,100,512,33,16},{63,257,2816,63,16},
                   {64,512,2048,64,16},{100,128,2048,100,8},{128,256,2816,128,8},{255,300,1000,255,4},
                   {256,256,2048,256,8},{32,17,256,32,16},{256,17,2048,256,8},{40,8192,2048,40,4}};
    for (auto & t : ts) {
        if (t[0] <= 64) rc |= run<32,256,32,4,8>(t[0],t[1],t[2],t[3],t[4]);
        else            rc |= run<32, 64,32,4,8>(t[0],t[1],t[2],t[3],t[4]);
    }
    printf("%s\n", rc ? "SOME CASES FAIL" : "ALL OK");
    return rc;
}
