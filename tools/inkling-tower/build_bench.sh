set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
cd /work/src
T=${T:-intel_gpu_bmg_g31}
icpx -fsycl -fsycl-targets=$T -O3 -std=c++20 -fno-fast-math -ffp-contract=off -Wno-deprecated-declarations -I include -I src \
  tools/bench_ink_sm.cpp -L bin -link_xexp '-Wl,-rpath,$ORIGIN' -o bin/bench_ink_sm
echo BENCH_OK
