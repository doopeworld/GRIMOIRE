#!/bin/bash
# CPU only, no devices. G31 CLI and server with the CPU-expert pool change, on cores the release build does not use.
# Libraries are copied from the integration tree: same source, and the change touches neither.
set -u
W=/mnt/storage/isos/grimoire-cxpool-1102/tree
E=/mnt/storage/isos/grimoire-engine-1101
L=/mnt/storage/isos/grimoire-runs/bench-1022/logs/cxpool-build-1102.out
exec >"$L" 2>&1
echo "CXPOOL_BUILD_START=$(date -Iseconds) base=$(git -C "$W" rev-parse --short HEAD)"
git -C "$W" status --short
cp -P "$E"/bin/libdnnl.so* "$E"/bin/libgrimoire_gemm.so "$E"/bin/libgrimoire_attn.so "$W"/bin/
docker run --rm --name grim-cxpool-build --cpuset-cpus 1-5 -v "$W":/grimoire -w /grimoire grimoire-dev:latest bash -lc "
  set -e
  export OCL_ICD_FILENAMES=
  set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
  T=intel_gpu_bmg_g31
  icpx -fsycl -fsycl-targets=\$T -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
    -fsycl-device-code-split=off -I include -I src \
    tools/grimoire_server.cpp src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp \
    src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp \
    src/gguf.cpp src/gguf_decode.cpp src/gguf_qwen35.cpp \
    src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
    src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
    src/tokenizer.cpp -lpthread -Lbin -lgrimoire_gemm -lgrimoire_attn \"-Wl,-rpath,\\\$ORIGIN\" -o bin/grimoire-server.new
  mv bin/grimoire-server.new bin/grimoire-server
  echo CXPOOL_SERVER_BUILT \$(date +%T)
  icpx -fsycl -fsycl-targets=\$T -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
    -fsycl-device-code-split=per_kernel -I include -I src \
    tools/grimoire_main.cpp src/grimoire.cpp src/qwen35_loader.cpp \
    src/native_model.cpp src/safetensors.cpp src/quantize.cpp src/gptq.cpp \
    src/gguf.cpp src/gguf_decode.cpp src/gguf_qwen35.cpp \
    src/gemv_decode.cpp src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
    src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
    src/tokenizer.cpp -Lbin -lgrimoire_gemm -lgrimoire_attn \"-Wl,-rpath,\\\$ORIGIN\" -o bin/grimoire.new
  mv bin/grimoire.new bin/grimoire
  echo CXPOOL_CLI_BUILT \$(date +%T)
"
rc=$?
echo "CXPOOL_BUILD_END=$(date -Iseconds) rc=$rc"
[ "$rc" -eq 0 ] && echo CXPOOL_BUILD_PASS || echo CXPOOL_BUILD_FAIL
