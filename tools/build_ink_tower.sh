set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
cd /work/src
test ! -e /dev/dri
mkdir -p bin
T=${T:-intel_gpu_bmg_g31}
date -Iseconds
if [ ! -e bin/libink_moe.so ] || [ tools/inkling2_moe_gemm.cpp -nt bin/libink_moe.so ]; then
  icpx -fsycl -fsycl-targets=$T -O3 -std=c++20 -fno-fast-math -ffp-contract=off -Wno-deprecated-declarations -I include -I src \
    -Xsycl-target-backend=$T "-options \"-cl-intel-256-GRF-per-thread -doubleGRF\"" \
    -fPIC -shared tools/inkling2_moe_gemm.cpp -o bin/libink_moe.so
  echo MOE_LIB_OK
fi
if [ ! -e bin/libink_xexp.so ] || [ tools/inkling2_xexp.cpp -nt bin/libink_xexp.so ]; then
  icpx -fsycl -fsycl-targets=$T -O3 -std=c++20 -fno-fast-math -ffp-contract=off -Wno-deprecated-declarations -I include -I src \
    -fPIC -shared tools/inkling2_xexp.cpp -o bin/libink_xexp.so
  echo XEXP_LIB_OK
fi
if [ ! -e bin/libink_gd.so ] || [ src/gemv_decode.cpp -nt bin/libink_gd.so ]; then
  icpx -fsycl -fsycl-targets=$T -O3 -std=c++20 -fno-fast-math -ffp-contract=off -Wno-deprecated-declarations -I include -I src \
    -fPIC -shared src/gemv_decode.cpp -L bin -lgrimoire_gemm -o bin/libink_gd.so
  echo GD_LIB_OK
fi
icpx -fsycl -fsycl-targets=$T -O3 -std=c++20 -fno-fast-math -ffp-contract=off -Wno-deprecated-declarations -I include -I src \
  tools/inkling2_gpu.cpp src/inkling2_checkpoint.cpp src/safetensors.cpp src/quantize.cpp \
  -L bin -lgrimoire_gemm -link_moe -link_xexp -link_gd '-Wl,-rpath,$ORIGIN' -o bin/inkling2_gpu
echo BUILD_OK
date -Iseconds
