#!/bin/bash
# Third probe build: the local kernels compiled with the production library flags (256 registers) and their own
# names, so the plan comparison runs where production runs. Columns: library / local same plan / local narrow plan.
# "local same plan" should equal "library": that row checks the method itself.
# CPU only: no devices, not privileged, spare cores 14-15.
set -euo pipefail
W=/mnt/storage/isos/grimoire-verify-grf-1102
L=/mnt/storage/isos/grimoire-runs/bench-1022/logs/verify-probe3-build-1102.out
exec >"$L" 2>&1
echo "PROBE3_BUILD_START=$(date -Iseconds)"
docker run --rm --name grim-verify-probe3-build --cpuset-cpus 14-15 \
  -v "$W":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    set +u
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
    set -u
    icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 \
      -Xsycl-target-backend=intel_gpu_bmg_g31 "-options \"-cl-intel-256-GRF-per-thread -doubleGRF\"" \
      -fno-fast-math -ffp-contract=fast -fno-math-errno -I include -I src \
      -DPROBE_LOCAL_GRF=256 \
      -Dint4_smallm_fdq_impl=int4_smallm_fdq_impl_probe256 -Dint4_smallm_impl=int4_smallm_impl_probe256 \
      tools/bench/onednn/i4wide_grf_probe.cpp -Lbin -lgrimoire_gemm -lgrimoire_attn \
      "-Wl,-rpath,\$ORIGIN" -o bin/i4wide_grf_probe3.new
    mv bin/i4wide_grf_probe3.new bin/i4wide_grf_probe3
  '
echo "own-name kernels in the new probe: $(grep -a -c int4_smallm_fdq_impl_probe256 "$W/bin/i4wide_grf_probe3")"
echo "PROBE3_BUILD_PASS $(date -Iseconds)"
