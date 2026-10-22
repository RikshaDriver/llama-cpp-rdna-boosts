#!/bin/bash
set -e
SRC=${SRC:-$HOME/llama.cpp}
LIBS=/opt/rocm-10.1.0-gfx120X/lib
export LD_LIBRARY_PATH=$LIBS:$SRC/build-rocm/bin
g++ -O2 -std=c++17 -o "$(dirname "$0")/multiseq_probe" "$(dirname "$0")/multiseq_probe.cpp" \
    -I "$SRC/include" -I "$SRC/ggml/include" \
    -L "$SRC/build-rocm/bin" -Wl,-rpath,"$SRC/build-rocm/bin" \
    -lllama -lggml -lggml-base -lggml-cpu -lggml-hip -lpthread
