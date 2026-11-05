#!/bin/bash
# Test build of mxfp4-smallm-1105 (grimoire-mx-smallm-1105): server only, G31 only (the B70s), CPU only, no devices.
# The two libraries and oneDNN come from the v1.10.3 build, as for v1.10.4.
set -u
R=/mnt/storage/isos/grimoire-mx-smallm-1105
SRC=/mnt/storage/isos/grimoire-release-1101
B=/mnt/storage/isos/grimoire-runs/bench-1022
L=$B/logs/build-mx-1105.out
exec >"$L" 2>&1
echo "BUILD_START=$(date -Iseconds) source=$(git -C "$R" rev-parse --short HEAD)"
mkdir -p "$R/bin"
cp -P "$SRC"/bin/libdnnl.so* "$SRC"/bin/libgrimoire_gemm.so "$SRC"/bin/libgrimoire_attn.so "$R"/bin/
[ -d "$R/third_party" ] || cp -a /mnt/storage/isos/grimoire-release-1102b/third_party "$R"/
docker run --rm --name grim-build-mx-1105 --cpuset-cpus ${CPUS:-0-13} \
  -v "$R":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    export OCL_ICD_FILENAMES=
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
    icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
      -fsycl-device-code-split=off -Wmisleading-indentation -I include -I src \
      tools/grimoire_server.cpp src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp \
      src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp \
      src/gguf.cpp src/gguf_decode.cpp src/gguf_qwen35.cpp \
      src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
      src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
      src/tokenizer.cpp -lpthread -Lbin -lgrimoire_gemm -lgrimoire_attn "-Wl,-rpath,\$ORIGIN" -o bin/grimoire-server.new
    mv bin/grimoire-server.new bin/grimoire-server
  '
rc=$?
echo "BUILD_END=$(date -Iseconds) rc=$rc"
[ "$rc" = 0 ] && { sha256sum "$R/bin/grimoire-server"; echo BUILD_PASS; } || echo BUILD_FAIL
