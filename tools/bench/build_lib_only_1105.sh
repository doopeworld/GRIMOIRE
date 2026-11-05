#!/bin/bash
# 11-05: rebuild only libgrimoire_gemm.so in a tree (CPU only, no devices).
#   build_lib_only_1105.sh TREE TARGETS CPUS LOGNAME     e.g. /mnt/storage/isos/grimoire-mx-lib-1105 intel_gpu_bmg_g31 0-5 lib-mx
set -u
R=$1; TG=$2; CPUS=$3; NAME=$4
B=/mnt/storage/isos/grimoire-runs/bench-1022
exec >"$B/logs/build-$NAME.out" 2>&1
echo "LIB_BUILD_START=$(date -Iseconds) tree=$R source=$(git -C "$R" rev-parse --short HEAD) targets=$TG"
docker run --rm --name "grim-build-$NAME" --cpuset-cpus "$CPUS" -e TG="$TG" -v "$R":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    export OCL_ICD_FILENAMES=
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
    GRF=(); IFS=, read -r -a TL <<< "$TG"
    for t in "${TL[@]}"; do GRF+=("-Xsycl-target-backend=$t" "-options \"-cl-intel-256-GRF-per-thread -doubleGRF\""); done
    icpx -fsycl -fsycl-targets=$TG "${GRF[@]}" -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
      -fPIC -shared -I include -I src src/gemm_fast.cpp -DGRIMOIRE_WITH_ONEDNN -I third_party/onednn/include \
      src/onednn_gemm.cpp -L third_party/onednn/lib -ldnnl "-Wl,-rpath,\$ORIGIN" -o bin/libgrimoire_gemm.so.new
    mv bin/libgrimoire_gemm.so.new bin/libgrimoire_gemm.so
  '
rc=$?
echo "LIB_BUILD_END=$(date -Iseconds) rc=$rc"
[ "$rc" = 0 ] && { sha256sum "$R/bin/libgrimoire_gemm.so"; echo LIB_BUILD_PASS; } || echo LIB_BUILD_FAIL
