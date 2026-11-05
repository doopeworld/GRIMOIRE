#!/bin/bash
# 11-05: G21 (B580 / B60) server of mxfp4-smallm-1105 with the tiered-expert CPU split in forward(), CPU only, for the B580
# A/B. Waits for the B580 comparison to end (llama.cpp's CPU experts must not share the cores with a compile). Output tree
# /mnt/storage/isos/grimoire-runs/split-tree-1105: the v1.10.4 libraries (G31+G21) + this server + the source's tools/.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
R=/mnt/storage/isos/grimoire-mx-smallm-1105
X=/mnt/storage/isos/grimoire-runs/split-tree-1105
exec >"$B/logs/build-split-g21-1105.out" 2>&1
until grep -qE "^B580_(DONE|STOP)" "$B/logs/after-gate-1104b.out" 2>/dev/null; do sleep 20; done
echo "SPLIT_BUILD_START=$(date -Iseconds) source=$(git -C "$R" rev-parse --short HEAD)"
mkdir -p "$R/bin-g21"
docker run --rm --name grim-build-split-g21 --cpuset-cpus 6-13 -v "$R":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    export OCL_ICD_FILENAMES=
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
    icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
      -fsycl-device-code-split=off -I include -I src \
      tools/grimoire_server.cpp src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp \
      src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp \
      src/gguf.cpp src/gguf_decode.cpp src/gguf_qwen35.cpp \
      src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
      src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
      src/tokenizer.cpp -lpthread -Lbin -lgrimoire_gemm -lgrimoire_attn "-Wl,-rpath,\$ORIGIN" -o bin-g21/grimoire-server
  '
rc=$?
echo "SPLIT_BUILD_END=$(date -Iseconds) rc=$rc"
[ "$rc" = 0 ] || { echo SPLIT_BUILD_FAIL; exit 1; }
rm -rf "$X"; mkdir -p "$X/bin"
cp -P /mnt/storage/isos/grimoire-release-1105/bin/lib* "$X/bin/"
cp "$R/bin-g21/grimoire-server" "$X/bin/"
cp -a "$R/tools" "$X/"
ls -la "$X/bin"
echo SPLIT_BUILD_PASS
