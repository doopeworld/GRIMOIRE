#!/bin/bash
# CPU-only G31+G21 build of the v1.10.5 candidate (mxfp4-smallm-1105 on top of v1.10.4 = 35201c2): no devices, not
# privileged. Unlike v1.10.4 the gemm library is rebuilt (gemm_fast.cpp: MXFP4 small-M past 8 rows); oneDNN and
# libgrimoire_attn.so come from the v1.10.3 build. Refuses any other engine source change.
#   run_release_build_v1105.sh COMMIT [CPUS]
set -u
COMMIT=$1
CPUS=${2:-6-13}
R=/mnt/storage/isos/grimoire-release-1105
SRC=/mnt/storage/isos/grimoire-release-1101
B=/mnt/storage/isos/grimoire-runs/bench-1022
L=$B/logs/release-build-1105.out
exec >"$L" 2>&1
[ -d "$R" ] || git -C /mnt/storage/isos/grimoire-fuse worktree add --detach "$R" "$COMMIT" || { echo "RELEASE_BUILD_FAIL worktree"; exit 1; }
git -C "$R" checkout -q --detach "$COMMIT" || { echo "RELEASE_BUILD_FAIL checkout"; exit 1; }
echo "RELEASE_BUILD_START=$(date -Iseconds) source=$(git -C "$R" rev-parse --short HEAD)"
if [ -n "$(git -C "$R" status --porcelain)" ]; then echo "RELEASE_BUILD_FAIL dirty source tree"; exit 1; fi
changed=$(git -C "$R" diff --name-only 35201c2 HEAD -- src include CMakeLists.txt tools/build_grimoire_only_b70.sh tools/build_server_only_b70.sh | sort | tr '\n' ' ')
echo "engine files changed since 35201c2 (v1.10.4): $changed"
[ "$changed" = "src/gemm_fast.cpp src/gemv_decode.cpp src/grimoire.cpp src/int4_smallm.hpp src/kernels.hpp " ] ||
  { echo "RELEASE_BUILD_FAIL unexpected source change"; exit 1; }
mkdir -p "$R/bin"
cp -P "$SRC"/bin/libdnnl.so* "$SRC"/bin/libgrimoire_attn.so "$R"/bin/
[ -d "$R/third_party" ] || cp -a /mnt/storage/isos/grimoire-release-1102b/third_party "$R"/
docker run --rm --name grim-release-build-1105 --cpuset-cpus "$CPUS" \
  -v "$R":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    export OCL_ICD_FILENAMES=
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
    T=intel_gpu_bmg_g31,intel_gpu_bmg_g21
    GRF=(); for t in intel_gpu_bmg_g31 intel_gpu_bmg_g21; do
      GRF+=("-Xsycl-target-backend=$t" "-options \"-cl-intel-256-GRF-per-thread -doubleGRF\""); done
    icpx -fsycl -fsycl-targets=$T "${GRF[@]}" -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
      -fPIC -shared -I include -I src src/gemm_fast.cpp -DGRIMOIRE_WITH_ONEDNN -I third_party/onednn/include \
      src/onednn_gemm.cpp -L third_party/onednn/lib -ldnnl "-Wl,-rpath,\$ORIGIN" -o bin/libgrimoire_gemm.so
    echo RELEASE_LIB_BUILT $(date +%T)
    icpx -fsycl -fsycl-targets=$T -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
      -fsycl-device-code-split=off -I include -I src \
      tools/grimoire_server.cpp src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp \
      src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp \
      src/gguf.cpp src/gguf_decode.cpp src/gguf_qwen35.cpp \
      src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
      src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
      src/tokenizer.cpp -lpthread -Lbin -lgrimoire_gemm -lgrimoire_attn "-Wl,-rpath,\$ORIGIN" -o bin/grimoire-server.new &
    S=$!
    icpx -fsycl -fsycl-targets=$T -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
      -fsycl-device-code-split=per_kernel -I include -I src \
      tools/grimoire_main.cpp src/grimoire.cpp src/qwen35_loader.cpp \
      src/native_model.cpp src/safetensors.cpp src/quantize.cpp src/gptq.cpp \
      src/gguf.cpp src/gguf_decode.cpp src/gguf_qwen35.cpp \
      src/gemv_decode.cpp src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
      src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
      src/tokenizer.cpp -Lbin -lgrimoire_gemm -lgrimoire_attn "-Wl,-rpath,\$ORIGIN" -o bin/grimoire.new &
    C=$!
    wait $S; echo RELEASE_SERVER_BUILT $(date +%T)
    wait $C; echo RELEASE_CLI_BUILT $(date +%T)
    mv bin/grimoire-server.new bin/grimoire-server
    mv bin/grimoire.new bin/grimoire
    strip --strip-unneeded bin/grimoire bin/grimoire-server
  '
rc=$?
echo "RELEASE_BUILD_END=$(date -Iseconds) rc=$rc"
if [ "$rc" -ne 0 ]; then echo "RELEASE_BUILD_FAIL rc=$rc"; exit "$rc"; fi
sha256sum "$R/bin/grimoire" "$R/bin/grimoire-server" "$R/bin/libgrimoire_gemm.so" "$R/bin/libgrimoire_attn.so"
echo RELEASE_BUILD_PASS
