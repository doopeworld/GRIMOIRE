#!/bin/bash
# Probe again with its own kernel names. The first build compiled int4_smallm_fdq_impl<2,GS> under the
# same SYCL kernel name as libgrimoire_gemm.so, so its "256" and "128" columns may be one kernel.
# Renaming the templates in this TU only makes the name unique; the arithmetic is the same header.
# CPU only: no devices, not privileged, spare cores 14-15. Not run here.
set -euo pipefail
W=/mnt/storage/isos/grimoire-verify-grf-1102
L=/mnt/storage/isos/grimoire-runs/bench-1022/logs/verify-probe2-build-1102.out
exec >"$L" 2>&1
echo "PROBE2_BUILD_START=$(date -Iseconds)"
docker run --rm --name grim-verify-probe2-build --cpuset-cpus 14-15 \
  -v "$W":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    set +u
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
    set -u
    icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 \
      -fno-fast-math -ffp-contract=fast -fno-math-errno -I include -I src \
      -Dint4_smallm_fdq_impl=int4_smallm_fdq_impl_probe128 -Dint4_smallm_impl=int4_smallm_impl_probe128 \
      tools/bench/onednn/i4wide_grf_probe.cpp -Lbin -lgrimoire_gemm -lgrimoire_attn \
      '\''-Wl,-rpath,$ORIGIN'\'' -o bin/i4wide_grf_probe2.new
    mv bin/i4wide_grf_probe2.new bin/i4wide_grf_probe2
  '
echo "own-name kernels in the new probe: $(grep -a -c int4_smallm_fdq_impl_probe128 "$W/bin/i4wide_grf_probe2")"
echo "PROBE2_BUILD_PASS $(date -Iseconds)"
