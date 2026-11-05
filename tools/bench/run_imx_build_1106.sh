#!/bin/bash
# 11-06: G31 dev build of mxfp4-smallm-1105 at COMMIT (imatrix INT4 rounding, the input-norm fold,
# GRIMOIRE_INTERLEAVE_SHARE) in its own worktree grimoire-runs/dev-1106, CPU only (cores CPUS, default
# 6-13; no GPU job). bin/grimoire (the CLI: --score-ids) and bin/grimoire-server, linked against the
# candidate's libraries (grimoire-release-1105b: gemm_fast.cpp / attn_verify_dpas.cpp unchanged since).
#   run_imx_build_1106.sh COMMIT [CPUS]
set -u
COMMIT=$1
CPUS=${2:-6-13}
X=/mnt/storage/isos/grimoire-runs/dev-1106
B=/mnt/storage/isos/grimoire-runs/bench-1022
exec >"$B/logs/imx-build-1106.out" 2>&1
[ -d "$X" ] || git -C /mnt/storage/isos/grimoire-fuse worktree add --detach "$X" "$COMMIT" || { echo "IMX_BUILD_FAIL worktree"; exit 1; }
git -C "$X" checkout -q --detach "$COMMIT" || { echo "IMX_BUILD_FAIL checkout"; exit 1; }
echo "IMX_BUILD_START=$(date -Iseconds) source=$(git -C "$X" rev-parse --short HEAD)"
mkdir -p "$X/bin"
cp -P /mnt/storage/isos/grimoire-release-1105b/bin/lib* "$X/bin/" || { echo "IMX_BUILD_FAIL libs"; exit 1; }
SRCS='src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp src/safetensors.cpp src/quantize.cpp src/gptq.cpp
      src/gguf.cpp src/gguf_decode.cpp src/gguf_qwen35.cpp src/gemv_decode.cpp src/gemm_xmx.cpp src/attention.cpp
      src/deltanet.cpp src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp
      src/prefill.cpp src/tokenizer.cpp'
docker run --rm --name grim-build-imx-1106 --cpuset-cpus "$CPUS" -e SRCS="$SRCS" -v "$X":/grimoire -w /grimoire \
    grimoire-dev:latest bash -lc '
    set -e
    export OCL_ICD_FILENAMES=
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
    F="-fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno -Wmisleading-indentation -I include -I src"
    # host test first: the imatrix reader and the importance-weighted INT4 rounding
    icpx -O2 -std=c++17 -pthread -I include tools/test_imatrix.cpp src/quantize.cpp -o bin/test_imatrix
    ./bin/test_imatrix
    icpx $F -fsycl-device-code-split=per_kernel tools/grimoire_main.cpp $SRCS \
      -lpthread -Lbin -lgrimoire_gemm -lgrimoire_attn "-Wl,-rpath,\$ORIGIN" -o bin/grimoire
    icpx $F -fsycl-device-code-split=off tools/grimoire_server.cpp $SRCS \
      -lpthread -Lbin -lgrimoire_gemm -lgrimoire_attn "-Wl,-rpath,\$ORIGIN" -o bin/grimoire-server
  '
rc=$?
echo "IMX_BUILD_END=$(date -Iseconds) rc=$rc"
[ "$rc" = 0 ] || { echo IMX_BUILD_FAIL; exit 1; }
ls -la "$X/bin"
echo IMX_BUILD_PASS
