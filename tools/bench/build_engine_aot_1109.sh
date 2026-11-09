#!/bin/bash
# Isolated B70 engine build; reuse unchanged release GEMM/attention libraries.
# Same compiler flags and code-splitting setting as the validated release.
set -eo pipefail
export OCL_ICD_FILENAMES="${OCL_ICD_FILENAMES:-}"
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
cd /work
mkdir -p bin
cp -a /release/bin/libgrimoire_gemm.so /release/bin/libgrimoire_attn.so /release/bin/libdnnl.so* bin/
SOURCES=(src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp src/gguf.cpp src/gguf_decode.cpp src/gguf_qwen35.cpp src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp src/tokenizer.cpp)
echo "AOT_START $(date -Iseconds)"
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 \
  -fno-fast-math -ffp-contract=fast -fno-math-errno \
  -fsycl-device-code-split=off -Wmisleading-indentation -I include -I src \
  tools/grimoire_server.cpp "${SOURCES[@]}" -lpthread -ldl -Lbin \
  -lgrimoire_gemm -lgrimoire_attn -ldnnl '-Wl,-rpath,$ORIGIN' -o bin/grimoire-server.aot-new
mv bin/grimoire-server.aot-new bin/grimoire-server
sha256sum bin/grimoire-server
echo "AOT_SERVER_PASS $(date -Iseconds)"
