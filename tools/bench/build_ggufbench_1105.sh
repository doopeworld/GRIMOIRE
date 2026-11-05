#!/bin/bash
# 11-05: tools/bench_native_smallm (GGUF native small-M, with the GRIMOIRE_GGUF_SMALLM_KS override) in grimoire-mx-lib-1105,
# G31, CPU only; runs after build_i4probe_1105.sh (the gate's regression phase).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
L=/mnt/storage/isos/grimoire-mx-lib-1105
C=${1:-}
exec >"$B/logs/build-ggufbench-1105.out" 2>&1
until grep -qE "^I4PROBE_BUILD_(PASS|FAIL)" "$B/logs/build-i4probe-1105.out" 2>/dev/null; do sleep 30; done
[ -n "$C" ] && { git -C "$L" checkout -q --detach "$C" || { echo "GGUFBENCH_BUILD_FAIL checkout"; exit 1; }; }
echo "GGUFBENCH_BUILD_START=$(date -Iseconds) source=$(git -C "$L" rev-parse --short HEAD)"
docker run --rm --name grim-build-ggufbench --cpuset-cpus 6-13 -v "$L":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    export OCL_ICD_FILENAMES=
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
    icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -I include -I src \
      tools/bench_native_smallm.cpp -o bin/bench_native_smallm
  '
rc=$?
echo "GGUFBENCH_BUILD_END=$(date -Iseconds) rc=$rc"
[ "$rc" = 0 ] && echo GGUFBENCH_BUILD_PASS || echo GGUFBENCH_BUILD_FAIL
