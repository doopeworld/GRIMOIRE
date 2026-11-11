#!/bin/bash
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
cd /work
test ! -e /dev/dri
mkdir -p bin
cp -a /release/bin/libgrimoire_gemm.so /release/bin/libgrimoire_attn.so /release/bin/libdnnl.so* bin/
icpx -O2 -std=c++17 -I include tests/test_http.cpp -o bin/test_http
./bin/test_http
icpx -O2 -std=c++17 -I include tests/test_speculative_sampling.cpp -o bin/test_speculative_sampling
./bin/test_speculative_sampling
icpx -O2 -std=c++17 -I include tests/test_generation.cpp -o bin/test_generation
./bin/test_generation
icpx -O2 -std=c++17 -I include tests/test_chat_effort.cpp src/tokenizer.cpp src/gguf.cpp -o bin/test_chat_effort
./bin/test_chat_effort
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -I include -I src tools/test_sampling.cpp -o bin/test_sampling
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -I include -I src tools/test_coupled_sampling.cpp -o bin/test_coupled_sampling
SOURCES=(src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp src/gguf.cpp src/gguf_decode.cpp src/gguf_qwen35.cpp src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp src/tokenizer.cpp)
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno -fsycl-device-code-split=off -I include -I src \
 tools/grimoire_server.cpp "${SOURCES[@]}" -lpthread -ldl -Lbin -lgrimoire_gemm -lgrimoire_attn -ldnnl '-Wl,-rpath,$ORIGIN' -o bin/grimoire-server.new
mv bin/grimoire-server.new bin/grimoire-server
sha256sum bin/grimoire-server bin/test_sampling
printf '%s\n' "$GRIM_SOURCE_COMMIT" > SOURCE_REVISION
echo COUPLED_BUILD_PASS
