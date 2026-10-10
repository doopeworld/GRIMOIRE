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
DGRF=()
IFS=',' read -r -a TLIST <<< "$TARGETS"
for t in "${TLIST[@]}"; do
  GRF+=("-Xsycl-target-backend=$t" "-options \"-cl-intel-256-GRF-per-thread -doubleGRF\"")
  DGRF+=("-Xsycl-target-backend=$t" "-options -doubleGRF")
done
# The large-M prompt GEMM lives in its own library so it alone gets 256
# registers (see src/gemm_fast.cpp); rebuild it with the engine. Both options
# are required: cl-256 for joint_matrix, doubleGRF for ESIMD flash prefill.
icpx -fsycl -fsycl-targets="$TARGETS" "${GRF[@]}" \
  -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
  -fPIC -shared -I include -I src src/gemm_fast.cpp -o bin/libgrimoire_gemm.so
# Speculative-verify attention: ESIMD kernels get the 256-register file only from
# -doubleGRF (src/attn_verify_dpas.cpp), so it is its own library as well.
icpx -fsycl -fsycl-targets="$TARGETS" "${DGRF[@]}" \
  -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
  -fPIC -shared -I include -I src src/attn_verify_dpas.cpp -o bin/libgrimoire_attn.so
icpx -fsycl -fsycl-targets="$TARGETS" -O3 -std=c++20 \
  -fno-fast-math -ffp-contract=fast -fno-math-errno \
  -fsycl-device-code-split=per_kernel -I include -I src \
  tools/grimoire_main.cpp src/grimoire.cpp src/qwen35_loader.cpp \
  src/native_model.cpp src/safetensors.cpp src/quantize.cpp src/gptq.cpp \
  src/gemv_decode.cpp src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
  src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
  src/tokenizer.cpp -Lbin -lgrimoire_gemm -lgrimoire_attn '-Wl,-rpath,$ORIGIN' -o bin/grimoire.new
mv bin/grimoire.new bin/grimoire
