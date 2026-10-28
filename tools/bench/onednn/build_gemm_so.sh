#!/bin/bash
# Rebuild only bin/libgrimoire_gemm.so (gemm_fast.cpp + onednn_gemm.cpp), the same command as
# tools/build_grimoire_only_b70.sh, for a change that touches nothing else (~3 min, no GPU).
set -euo pipefail
REPO=/mnt/storage/isos/grimoire-fuse
docker run --rm -v $REPO:/grimoire -w /grimoire grimoire-dev:latest bash -lc '
  set -e
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
  T=intel_gpu_bmg_g31
  icpx -fsycl -fsycl-targets=$T -Xsycl-target-backend=$T "-options \"-cl-intel-256-GRF-per-thread -doubleGRF\"" \
    -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
    -fPIC -shared -I include -I src src/gemm_fast.cpp \
    -DGRIMOIRE_WITH_ONEDNN -I third_party/onednn/include src/onednn_gemm.cpp -L third_party/onednn/lib -ldnnl \
    "-Wl,-rpath,\$ORIGIN" -o bin/libgrimoire_gemm.so.new
  mv bin/libgrimoire_gemm.so.new bin/libgrimoire_gemm.so
  echo SO_BUILT'
