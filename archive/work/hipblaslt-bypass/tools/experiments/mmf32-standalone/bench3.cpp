#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#define __launch_bounds__(x)
#include "kernel2.inc"
static void fill(float * p, size_t n, uint32_t s) { for (size_t i = 0; i < n; ++i) { s = s*1664525u+1013904223u; p[i] = (float)((int32_t)(s>>8)%2000-1000)/1000.0f; } }
template<int BM,int BN,int BK,int TMR,int TNC>
static void bench(const char * tag, int m, int n, int k, int ldc, int ks, int reps) {
    std::vector<float> hx((size_t)m*k), hy((size_t)n*k); fill(hx.data(), hx.size(), 1); fill(hy.data(), hy.size(), 2);
    float *dx,*dy,*dd,*dt; hipMalloc(&dx,hx.size()*4); hipMalloc(&dy,hy.size()*4);
    const size_t per = (size_t)ldc*std::max(m,n);
    hipMalloc(&dd,per*4); hipMalloc(&dt,per*ks*4);
    hipMemcpy(dx,hx.data(),hx.size()*4,hipMemcpyHostToDevice); hipMemcpy(dy,hy.data(),hy.size()*4,hipMemcpyHostToDevice);
    dim3 block(BM/TMR, BN/TNC, 1), grid((m+BM-1)/BM, (n+BN-1)/BN, ks);
    const int kchunk = (k + ks - 1)/ks;
    for (int i = 0; i < 3; ++i) {
        mm_f32_splitk_kernel<BM,BN,BK,TMR,TNC><<<grid,block>>>(dx,dy,dt,m,n,k,ldc,kchunk);
        mm_f32_reduce_kernel<<<((int)per+255)/256,256>>>(dt,dd,(int)per,(int)per,ks);
    }
    hipDeviceSynchronize();
    hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1); hipEventRecord(e0);
    for (int r = 0; r < reps; ++r) {
        mm_f32_splitk_kernel<BM,BN,BK,TMR,TNC><<<grid,block>>>(dx,dy,dt,m,n,k,ldc,kchunk);
        mm_f32_reduce_kernel<<<((int)per+255)/256,256>>>(dt,dd,(int)per,(int)per,ks);
    }
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1);
    printf("  %-24s ks=%-2d %dx%dx%d: %7.4f ms  %6.2f TFLOP/s (blocks=%3d thr=%3d)\n", tag, ks, m,n,k, ms/reps,
           2.0*m*n*k/(ms/reps*1e-3)/1e12, grid.x*grid.y*grid.z, block.x*block.y);
    hipFree(dx);hipFree(dy);hipFree(dd);hipFree(dt);
}
int main() {
    printf("ssm 32x512x2048 (hipBLASLt best 0.046 ms / 1.46 TFLOP/s)\n");
    bench<32,128,32,4,8>("32x128x32 4x8", 32,512,2048,32,8,200);
    bench<32,128,32,4,8>("32x128x32 4x8", 32,512,2048,32,16,200);
    bench<32,256,32,4,8>("32x256x32 4x8", 32,512,2048,32,8,200);
    bench<32,256,32,4,8>("32x256x32 4x8", 32,512,2048,32,16,200);
    bench<32,128,64,4,8>("32x128x64 4x8", 32,512,2048,32,8,200);
    bench<32,64,32,4,8> ("32x64x32 4x8",  32,512,2048,32,16,200);
    bench<32,128,32,4,4>("32x128x32 4x4", 32,512,2048,32,8,200);
    bench<32,128,32,8,8>("32x128x32 8x8", 32,512,2048,32,16,200);
    printf("router 256x256x2048 (hipBLASLt best 0.047 ms / 5.7 TFLOP/s)\n");
    bench<32,64,32,4,8> ("32x64x32 4x8",  256,256,2048,256,4,200);
    bench<32,64,32,4,8> ("32x64x32 4x8",  256,256,2048,256,8,200);
    bench<32,128,32,4,8>("32x128x32 4x8", 256,256,2048,256,4,200);
    return 0;
}
