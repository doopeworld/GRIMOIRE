#!/bin/bash
# 11-06: G31 server of radiance-heat-1106 (fnmtp-verify-1106 + the heat engine, GRIMOIRE_EXPERT_HEAT) in its own
# worktree grimoire-runs/radiance-heat-1106b (the first tree keeps serving its runs), CPU only (cores CPUS, default 0-5). Libraries from the 11-05 candidate tree
# (grimoire-release-1105b). First builds and runs the host tests in the build container (test_cpu_ffn_rows,
# test_expert_heat: must print PASS), so a wrong pool or policy change never reaches a GPU run.
#   run_heat_build2_1106.sh COMMIT [CPUS]
set -u
COMMIT=$1
CPUS=${2:-0-5}
X=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
B=/mnt/storage/isos/grimoire-runs/bench-1022
LIB=/mnt/storage/isos/grimoire-release-1105b/bin
exec >"$B/logs/heat-build2-1106.out" 2>&1
[ -d "$X" ] || git -C /mnt/storage/isos/grimoire-fuse worktree add --detach "$X" "$COMMIT" || { echo "HEAT2_BUILD_FAIL worktree"; exit 1; }
git -C "$X" checkout -q --detach "$COMMIT" || { echo "HEAT2_BUILD_FAIL checkout"; exit 1; }
echo "HEAT2_BUILD_START=$(date -Iseconds) source=$(git -C "$X" rev-parse --short HEAD)"
[ -f "$LIB/libgrimoire_gemm.so" ] || { echo "HEAT2_BUILD_FAIL no $LIB/libgrimoire_gemm.so"; exit 1; }
mkdir -p "$X/bin"
cp -P "$LIB"/lib* "$X/bin/"
docker run --rm --name grim-build-heat2-1106 --cpuset-cpus "$CPUS" --cpu-shares 64 -v "$X":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    export OCL_ICD_FILENAMES=
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
    icpx -fsycl -O2 -std=c++20 -I include tools/test_cpu_ffn_rows.cpp src/cpu_experts.cpp -lpthread \
      -o bin/test_cpu_ffn_rows
    ./bin/test_cpu_ffn_rows 6 5
    icpx -O2 -std=c++17 -I include tools/test_expert_heat.cpp -o bin/test_expert_heat
    ./bin/test_expert_heat
    icpx -O2 -std=c++17 -I include tools/route_sim.cpp -o bin/route_sim
    icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
      -fsycl-device-code-split=off -Wmisleading-indentation -I include -I src \
      tools/grimoire_server.cpp src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp \
      src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp \
      src/gguf.cpp src/gguf_decode.cpp src/gguf_qwen35.cpp \
      src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
      src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
      src/tokenizer.cpp -lpthread -Lbin -lgrimoire_gemm -lgrimoire_attn "-Wl,-rpath,\$ORIGIN" -o bin/grimoire-server
  '
rc=$?
echo "HEAT2_BUILD_END=$(date -Iseconds) rc=$rc"
[ "$rc" = 0 ] || { echo HEAT2_BUILD_FAIL; exit 1; }
ls -la "$X/bin"
echo HEAT2_BUILD_PASS
