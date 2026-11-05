#!/bin/bash
# 11-05: G31 library (with the INT4 9-16-row plan override) + tools/i4sweep_probe in grimoire-mx-lib-1105, CPU only.
# Waits for the v1.10.4 gate's regression phase (text-only checks there; speeds of the release table come later from
# the image validation, on a quiet machine again: this build is ~8 minutes).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
L=/mnt/storage/isos/grimoire-mx-lib-1105
C=${1:-f18f4318ff6ce52475d200685f6a1c8ea3d559d2}
exec >"$B/logs/build-i4probe-1105.out" 2>&1
until grep -q "^results: " "$B/logs/regress-v1104b.out" 2>/dev/null || grep -qE "^V1104B_ALL_(DONE|STOP)" "$B/logs/v1104b-all.out"; do sleep 30; done
git -C "$L" checkout -q --detach "$C" || { echo "I4PROBE_BUILD_FAIL checkout"; exit 1; }
echo "I4PROBE_BUILD_START=$(date -Iseconds) source=$(git -C "$L" rev-parse --short HEAD)"
bash "$B/build_lib_only_1105.sh" "$L" intel_gpu_bmg_g31 6-13 lib-i4probe
grep -qx LIB_BUILD_PASS "$B/logs/build-lib-i4probe.out" || { echo "I4PROBE_BUILD_FAIL library"; exit 1; }
docker run --rm --name grim-build-i4probe --cpuset-cpus 6-13 -v "$L":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    export OCL_ICD_FILENAMES=
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
    icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
      -fsycl-device-code-split=per_kernel -I include -I src tools/i4sweep_probe.cpp src/gemv_decode.cpp \
      -Lbin -lgrimoire_gemm "-Wl,-rpath,\$ORIGIN" -o bin/i4sweep_probe
  '
rc=$?
echo "I4PROBE_BUILD_END=$(date -Iseconds) rc=$rc"
[ "$rc" = 0 ] && echo I4PROBE_BUILD_PASS || echo I4PROBE_BUILD_FAIL
