#!/usr/bin/env bash
set -euo pipefail
export OCL_ICD_FILENAMES="${OCL_ICD_FILENAMES:-}"
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
cd "${1:-/grimoire}"
# GPU targets for the AOT kernels.  Default: the Arc Pro B70 (BMG G31) only.
# GRIM_TARGETS=intel_gpu_bmg_g31,intel_gpu_bmg_g21 adds the Arc B580 (BMG G21),
# e.g. for a pipeline stage on a B580 (twice the device-compile time).
TARGETS="${GRIM_TARGETS:-intel_gpu_bmg_g31}"
GRF=()
IFS=',' read -r -a TLIST <<< "$TARGETS"
for t in "${TLIST[@]}"; do GRF+=("-Xsycl-target-backend=$t" "-options -cl-intel-256-GRF-per-thread"); done
# The large-M prompt GEMM lives in its own library so it alone gets 256
# registers (see src/gemm_fast.cpp); rebuild it with the engine.
icpx -fsycl -fsycl-targets="$TARGETS" "${GRF[@]}" \
  -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
  -fPIC -shared -I include -I src src/gemm_fast.cpp -o bin/libgrimoire_gemm.so
icpx -fsycl -fsycl-targets="$TARGETS" -O3 -std=c++20 \
     -fno-fast-math -ffp-contract=fast -fno-math-errno \
     -fsycl-device-code-split=off -I include -I src \
     tools/grimoire_server.cpp src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp \
     src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp \
     src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
     src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
     src/tokenizer.cpp -lpthread -Lbin -lgrimoire_gemm '-Wl,-rpath,$ORIGIN' -o bin/grimoire-server.new
mv bin/grimoire-server.new bin/grimoire-server
echo "built: bin/grimoire-server"
