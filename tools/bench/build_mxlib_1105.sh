#!/bin/bash
# 11-05: libgrimoire_gemm.so (G31 only, test) from grimoire-mx-lib-1105 (mxfp4-smallm-1105: MXFP4 small-M past 8 rows
# with nibble pairs (k, k+4)), then tools/mx4rows_probe against it. CPU only, no devices; waits for the v1.10.4 release
# build to free cores 6-13.
set -u
L=/mnt/storage/isos/grimoire-mx-lib-1105
B=/mnt/storage/isos/grimoire-runs/bench-1022
exec >"$B/logs/build-mxlib-1105.out" 2>&1
until [ -z "$(docker ps -q --filter name=grim-release-build-1102b)" ]; do sleep 15; done
echo "MXLIB_BUILD_START=$(date -Iseconds) source=$(git -C "$L" rev-parse --short HEAD)"
docker run --rm --name grim-build-mxlib-1105 --cpuset-cpus "${CPUS:-6-13}" -v "$L":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    export OCL_ICD_FILENAMES=
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
    T=intel_gpu_bmg_g31
    cp -P third_party/onednn/lib/libdnnl.so* bin/
    icpx -fsycl -fsycl-targets=$T -Xsycl-target-backend=$T "-options \"-cl-intel-256-GRF-per-thread -doubleGRF\"" \
      -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
      -fPIC -shared -I include -I src src/gemm_fast.cpp -DGRIMOIRE_WITH_ONEDNN -I third_party/onednn/include \
      src/onednn_gemm.cpp -L third_party/onednn/lib -ldnnl "-Wl,-rpath,\$ORIGIN" -o bin/libgrimoire_gemm.so
    echo MXLIB_LIB_BUILT $(date +%T)
    icpx -fsycl -fsycl-targets=$T -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
      -fsycl-device-code-split=per_kernel -I include -I src tools/mx4rows_probe.cpp src/gemv_decode.cpp \
      -Lbin -lgrimoire_gemm "-Wl,-rpath,\$ORIGIN" -o bin/mx4rows_probe
    echo MXLIB_PROBE_BUILT $(date +%T)
  '
rc=$?
echo "MXLIB_BUILD_END=$(date -Iseconds) rc=$rc"
[ "$rc" = 0 ] && echo MXLIB_BUILD_PASS || echo MXLIB_BUILD_FAIL
