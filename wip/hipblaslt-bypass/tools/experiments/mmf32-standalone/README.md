# mmf32 standalone harness (kernel oracle + tile bench)

`test2.cpp` is the numerical oracle: it compiles the kernel straight out of
`ggml/src/ggml-cuda/mmf32.cu` (the `kernel.inc` extraction below) and checks 16 shapes against a
double-precision reference — NMSE is ~1e-13 (F32 rounding), covering odd k (2051), odd m/ldc, n < m,
n > m and n = 17.  `bench3.cpp` times a tile/ks sweep.

```bash
# extract the two kernels from the working tree's mmf32.cu (keeps the test honest)
python3 - <<'PY'
src = open('/home/stew675/llama.cpp/ggml/src/ggml-cuda/mmf32.cu').read()
start = src.index('template<int BM, int BN, int BK, int TMR, int TNC>\nstatic __global__')
end   = src.index('template<int BM, int BN, int BK, int TMR, int TNC>\nstatic void mmf32_launch')
open('kernel2.inc','w').write(src[start:end])
PY
hipcc -O3 -std=c++17 -o t2 test2.cpp "-isystem/opt/rocm-10.1.0-gfx120X/include" \
      -L/opt/rocm-10.1.0-gfx120X/lib -lamdhip64
LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0 ./t2
```

Both `#define __launch_bounds__(x)` away the launch-bounds attribute (the harness has no arch macro).
