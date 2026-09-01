#!/bin/bash
# =====================================================================
#  build_b70.sh  --  Ahead-of-Time build for Intel Arc Pro B70
#
#  The target string in the source blueprint was wrong and would have
#  silently produced a binary that JIT-recompiles on first launch, which
#  is the exact thing AOT is meant to prevent:
#
#    WRONG   -fsycl-targets=intel_gpu_xe_hpg      <- Alchemist (Xe-HPG, A-series)
#    WRONG   -Xsycl-target-backend "-device bmg"  <- redundant AND conflicting
#                                                    with the triple above
#    RIGHT   -fsycl-targets=intel_gpu_bmg_g31     <- Battlemage G31 = Arc Pro B70/B65
#
#  intel_gpu_bmg_g21 is the SMALLER Battlemage die (B580, Arc Pro B50/B60).
#  Building for g21 and running on a B70 works, but you lose the tuning.
# =====================================================================
set -eo pipefail

: "${ONEAPI_ROOT:=/opt/intel/oneapi}"
# Intel's vars.sh dereferences OCL_ICD_FILENAMES before assigning it, so
# `set -u` kills the whole build at source time. Pre-seed it and keep -u
# off while sourcing.
export OCL_ICD_FILENAMES="${OCL_ICD_FILENAMES:-}"
if [ -f "$ONEAPI_ROOT/setvars.sh" ]; then
    set +u
    # shellcheck disable=SC1091
    source "$ONEAPI_ROOT/setvars.sh" --force >/dev/null 2>&1 || true
    set +u
fi
if ! command -v icpx >/dev/null; then
    for d in "$ONEAPI_ROOT"/compiler/*/bin; do
        [ -d "$d" ] && export PATH="$d:$PATH"
    done
fi

command -v icpx >/dev/null || { echo "icpx not found; install the oneAPI DPC++ compiler"; exit 1; }
command -v ocloc >/dev/null || echo "warning: ocloc not on PATH, AOT will fail"

TARGET="${B70_TARGET:-intel_gpu_bmg_g31}"
OUT="${B70_OUT:-b70_native_inference}"

echo "target : $TARGET"
icpx --version | head -1

# ---------------------------------------------------------------------
#  -ffast-math is deliberately NOT used.
#
#  It implies -ffinite-math-only, which lets the compiler assume no
#  infinities exist. The FlashDecoding kernel initializes its running
#  maximum to -inf and relies on exp(-inf) == 0 for masked lanes. Under
#  fast-math those become undefined and the attention output degrades in
#  a way that looks like a model quality problem rather than a compiler
#  flag. Use the targeted relaxations instead.
# ---------------------------------------------------------------------
icpx -fsycl \
     -fsycl-targets="$TARGET" \
     -O3 \
     -ferror-limit=0 \
     -std=c++20 \
     -fno-fast-math \
     -ffp-contract=fast \
     -fno-math-errno \
     -fsycl-device-code-split=off \
     -Xsycl-target-backend="$TARGET" "-options -cl-intel-256-GRF-per-thread" \
     -I include -I src \
     src/main.cpp src/quantize.cpp src/gemv_decode.cpp \
     src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
     src/moe_kernels.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
     src/tokenizer.cpp \
     -o "$OUT"

echo "built  : $OUT"
echo
echo "verify the AOT image really contains native code (no SPIR-V fallback):"
echo "  SYCL_UR_TRACE=1 ./$OUT 2>&1 | grep -i 'jit\\|compile'"
echo "an AOT binary should show no device-side compile at first kernel launch."

# ---------------------------------------------------------------------
# Host-only tools. No SYCL, so a plain compiler is enough and they build
# even when the GPU kernels do not.
# ---------------------------------------------------------------------
mkdir -p bin
icpx -O2 -std=c++17 -I include \
     tools/inspect.cpp src/safetensors.cpp -o bin/b70-inspect \
  && echo "built  : bin/b70-inspect" || echo "warn: b70-inspect failed"

icpx -O2 -std=c++17 -I include \
     tools/load.cpp src/qwen35_loader.cpp src/safetensors.cpp src/native_model.cpp -o bin/b70-load \
  && echo "built  : bin/b70-load" || echo "warn: b70-load failed"

# ---------------------------------------------------------------------
#  grimoire -- the inference CLI
# ---------------------------------------------------------------------
icpx -fsycl -fsycl-targets="$TARGET" -O3 -std=c++20 \
     -fno-fast-math -ffp-contract=fast -fno-math-errno \
     -fsycl-device-code-split=off \
     -I include -I src \
     tools/grimoire_main.cpp src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp \
     src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp \
     src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
     src/moe_kernels.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
     src/tokenizer.cpp \
     -o bin/grimoire \
  && echo "built  : bin/grimoire" \
  || { echo "=== GRIMOIRE BUILD FAILED ==="; }

# ---------------------------------------------------------------------
#  grimoire-server -- OpenAI-compatible HTTP front end (open-webui,
#  llama-benchy). Same model/engine code as bin/grimoire; see
#  tools/grimoire_server.cpp for the launch-flag shape (mirrors
#  `vllm serve <model> --quantization ...`).
# ---------------------------------------------------------------------
icpx -fsycl -fsycl-targets="$TARGET" -O3 -std=c++20 \
     -fno-fast-math -ffp-contract=fast -fno-math-errno \
     -fsycl-device-code-split=off \
     -I include -I src \
     tools/grimoire_server.cpp src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp \
     src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp \
     src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
     src/moe_kernels.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
     src/tokenizer.cpp \
     -lpthread \
     -o bin/grimoire-server \
  && echo "built  : bin/grimoire-server" \
  || { echo "=== GRIMOIRE-SERVER BUILD FAILED ==="; }

icpx -O2 -std=c++17 -I include \
     tools/verify_tokenizer.cpp src/tokenizer.cpp -o bin/b70-verify-tok \
  && echo "built  : bin/b70-verify-tok" || echo "warn: verify-tok failed"

# Offline BF16 -> B70-native execution-format compiler. Host-only: model
# conversion never needs a GPU framework or device runtime.
icpx -O3 -std=c++20 -I include \
     tools/b70_compile_model.cpp src/quantize.cpp src/qwen35_loader.cpp \
     src/safetensors.cpp src/native_model.cpp -o bin/b70-compile-model \
  && echo "built  : bin/b70-compile-model" || echo "warn: native model compiler failed"

icpx -O2 -std=c++20 -I include tools/inspect_native_model.cpp \
     src/native_model.cpp -o bin/b70-inspect-native \
  && echo "built  : bin/b70-inspect-native" || echo "warn: native inspector failed"

# Numeric parity gate for the NInfer Build-2 Q4G64 proposal head. Standalone:
# it dlopens the grouped bridge and checks the head kernel against a CPU
# reference, so it needs neither the target model nor a loaded engine.
#   bin/test_dflash_head HEAD.q4g64 IDS.i32 src/libgrimoire_xe2_grouped.so
icpx -fsycl -fsycl-targets="$TARGET" \
     -O2 -std=c++20 -fno-fast-math -I include -I src \
     tools/test_dflash_head.cpp -o bin/test_dflash_head -ldl \
  && echo "built  : bin/test_dflash_head" || echo "warn: dflash head test failed"
