#!/usr/bin/env bash
set -euo pipefail
export OCL_ICD_FILENAMES="${OCL_ICD_FILENAMES:-}"
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
cd "${1:-/grimoire}"
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 \
     -fno-fast-math -ffp-contract=fast -fno-math-errno \
     -fsycl-device-code-split=off -I include -I src \
     tools/grimoire_server.cpp src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp \
     src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp \
     src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
     src/moe_kernels.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
     src/tokenizer.cpp -lpthread -o bin/grimoire-server.new
mv bin/grimoire-server.new bin/grimoire-server
echo "built: bin/grimoire-server"
