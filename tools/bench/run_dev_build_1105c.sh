#!/bin/bash
# 11-05: G31 dev server of mxfp4-smallm-1105 at COMMIT (after the v1.10.5 candidate: the grouped speculative commit)
# in its own worktree grimoire-runs/dev-1105c, CPU only (cores CPUS, default 0-5, while the release build has 6-13 and
# no GPU job runs). Links and runs against the candidate's libraries (grimoire-release-1105b: same gemm_fast.cpp).
#   run_dev_build_1105c.sh COMMIT [CPUS]
set -u
COMMIT=$1
CPUS=${2:-0-5}
X=/mnt/storage/isos/grimoire-runs/dev-1105c
B=/mnt/storage/isos/grimoire-runs/bench-1022
exec >"$B/logs/dev-build-1105c.out" 2>&1
[ -d "$X" ] || git -C /mnt/storage/isos/grimoire-fuse worktree add --detach "$X" "$COMMIT" || { echo "DEV_BUILD_FAIL worktree"; exit 1; }
git -C "$X" checkout -q --detach "$COMMIT" || { echo "DEV_BUILD_FAIL checkout"; exit 1; }
echo "DEV_BUILD_START=$(date -Iseconds) source=$(git -C "$X" rev-parse --short HEAD)"
until [ -f /mnt/storage/isos/grimoire-release-1105b/bin/libgrimoire_gemm.so ]; do sleep 20; done
mkdir -p "$X/bin"
cp -P /mnt/storage/isos/grimoire-release-1105b/bin/lib* "$X/bin/"
docker run --rm --name grim-build-dev-1105c --cpuset-cpus "$CPUS" -v "$X":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    export OCL_ICD_FILENAMES=
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
    icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
      -fsycl-device-code-split=off -I include -I src \
      tools/grimoire_server.cpp src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp \
      src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp \
      src/gguf.cpp src/gguf_decode.cpp src/gguf_qwen35.cpp \
      src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
      src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
      src/tokenizer.cpp -lpthread -Lbin -lgrimoire_gemm -lgrimoire_attn "-Wl,-rpath,\$ORIGIN" -o bin/grimoire-server
  '
rc=$?
echo "DEV_BUILD_END=$(date -Iseconds) rc=$rc"
[ "$rc" = 0 ] || { echo DEV_BUILD_FAIL; exit 1; }
ls -la "$X/bin"
echo DEV_BUILD_PASS
