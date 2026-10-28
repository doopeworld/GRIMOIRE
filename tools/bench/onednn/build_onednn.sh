#!/bin/bash
# Builds oneDNN (SYCL GPU runtime, no CPU primitives) + benchdnn with the GRIMOIRE compiler.
# Usage (inside the grimoire-dnnl-build image = grimoire-dev + cmake/ninja, Dockerfile.dnnlbuild):
#   git clone --depth 1 --branch v3.13.4 https://github.com/uxlfoundation/oneDNN.git oneDNN-3.13.4
#   bash build_onednn.sh /src/oneDNN-3.13.4 /src/build-3.13.4 /src/install-3.13.4
# Then copy install-3.13.4/{include,lib} to the repo's third_party/onednn; the GRIMOIRE build
# scripts pick it up from there (gitignored).  ~20 min on the Tower's 16 cores.
set -e
SRC=$1; BLD=$2; INST=$3
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
export CC=icx CXX=icpx
cmake -S "$SRC" -B "$BLD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INST" \
  -DONEDNN_CPU_RUNTIME=NONE \
  -DONEDNN_GPU_RUNTIME=SYCL \
  -DONEDNN_GPU_VENDOR=INTEL \
  -DONEDNN_BUILD_EXAMPLES=OFF \
  -DONEDNN_BUILD_TESTS=ON \
  -DONEDNN_BUILD_GRAPH=ON
cmake --build "$BLD" -j 14 --target dnnl benchdnn
cmake --install "$BLD" >/dev/null
ls -la "$INST/lib" | head
echo BUILD_DONE
