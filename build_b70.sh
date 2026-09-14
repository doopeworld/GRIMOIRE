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
#  Both are built by default now: an AOT image for one die does not run on
#  the other at all, and a box can hold a mix (two B70s plus a B580).  A
#  single-die image is still one env var away -- see B70_TARGET below.
# =====================================================================
set -eo pipefail

# A failed REQUIRED target must reach the exit status.  `set -e` does not
# fire when a failure is caught by `|| { ... }`, so before this the script
# printed "BUILD FAILED", carried on, and exited 0 -- which means a later
# run tests whatever stale binary is still in bin/ and believes it tested
# the source change.  Auxiliary targets stay warn-only.
REQUIRED_FAILED=0

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
# ...and a PRESENT ocloc is not the same as a USABLE one.  MEASURED
# 2026-09-13: Ubuntu 24.04 ships intel-ocloc 23.43, which predates
# Battlemage entirely -- its device list stops at pvc/mtl/xe-lpg, and
# `ocloc compile -device bmg_g31` answers "Cannot get HW Info".  Checking
# only that the binary exists would let that build get all the way to the
# device compile before failing, with an error that names the die rather
# than the toolchain.  Battlemage wants compute-runtime 24.35 or newer,
# from Intel's own repo, not the distro's.
#
# MEASURED: on this build `ocloc ids bmg_g31` exits 226, and an earlier
# note here claiming it exits 0 either way was wrong -- that reading had
# captured a pipeline's status, not ocloc's.  The probe below therefore
# requires a POSITIVE answer (exit 0 AND "Matched ids") and does not care
# which way a failure is expressed.
# A probe that only looked for "Unknown acronym" accepted every OTHER way
# ocloc can fail to answer: a build without the `ids` subcommand, a broken
# install that cannot load its libraries, anything non-zero.  Require a
# POSITIVE answer -- exit 0 and a "Matched ids" line -- and report anything
# else, with the reason, rather than letting it through to the compiler.
ocloc_probe() {   # $1 = intel_gpu_<die>; echoes "<die>|<reason>" on failure
    local die="${1#intel_gpu_}" out rc
    out="$(ocloc ids "$die" 2>&1)"; rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "$die|ocloc ids exited $rc"
        return
    fi
    case "$out" in
        *"Matched ids"*) ;;                          # this ocloc knows it
        *"Unknown acronym"*) echo "$die|not a device this ocloc knows" ;;
        *) echo "$die|ocloc gave no usable answer: $(echo "$out" | head -1)" ;;
    esac
}

# Every Battlemage die, in one image.  AOT code compiled for g31 does NOT
# run on a g21: the binary has no device image for it and the card is
# simply not usable.  GRIMOIRE has to work on any Battlemage card --
# B70/B65 are g31, B580/B60/B50 are g21 -- and a machine can now hold a
# mix of both at once, so the default builds for both dies and the
# runtime picks the matching image per device.
#
# The cost is compile time: every kernel is compiled once per die.
# B70_TARGET=intel_gpu_bmg_g31 builds a single die when iterating.
TARGET="${B70_TARGET:-intel_gpu_bmg_g31,intel_gpu_bmg_g21}"
OUT="${B70_OUT:-b70_native_inference}"

# -Xsycl-target-backend takes ONE target, so a multi-die build needs the
# flag repeated per die.  Passing the comma list to it builds a binary
# that silently has the option on NEITHER.
IFS=',' read -r -a TARGET_LIST <<< "$TARGET"
BACKEND_OPTS=()
for _t in "${TARGET_LIST[@]}"; do
    BACKEND_OPTS+=( -Xsycl-target-backend="$_t" "-options -cl-intel-256-GRF-per-thread" )
done

if command -v ocloc >/dev/null; then
    BAD=""
    for _t in "${TARGET_LIST[@]}"; do
        _r="$(ocloc_probe "$_t")"
        [ -n "$_r" ] && BAD="$BAD
       ${_r%%|*}: ${_r#*|}"
    done
    if [ -n "$BAD" ]; then
        echo "ERROR: this ocloc cannot target what is being built:$BAD" >&2
        echo "       Battlemage needs Intel's compute-runtime 24.35+." >&2
        echo "       Ubuntu's intel-ocloc is 23.43 and predates it -- it" >&2
        echo "       installs cleanly and then cannot compile for bmg_g31." >&2
        echo "       B70_TARGET=... narrows the dies if that is what you want." >&2
        exit 1
    fi
fi
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
# -Wmisleading-indentation is here for a specific reason: a braceless
# `if` whose body LOOKED like two statements set a validity flag
# unconditionally while the write it described stayed conditional, and the
# result was a drafter reading a buffer nothing had filled.  The compiler
# could see that and was not asked.  It is a warning, not an error, so it
# cannot break a build that is otherwise fine.
icpx -fsycl \
     -fsycl-targets="$TARGET" \
     -O3 \
     -ferror-limit=0 \
     -Wmisleading-indentation \
     -std=c++20 \
     -fno-fast-math \
     -ffp-contract=fast \
     -fno-math-errno \
     -fsycl-device-code-split=off \
     "${BACKEND_OPTS[@]}" \
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
  || { echo "=== GRIMOIRE BUILD FAILED ==="; REQUIRED_FAILED=1; }

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
  || { echo "=== GRIMOIRE-SERVER BUILD FAILED ==="; REQUIRED_FAILED=1; }

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

# Numeric parity gate for the K2 and DFlash2-selector kernels, RUN on a
# real device rather than replayed on the host.  The host suites in tests/
# pin the math; a kernel that never launches still passes them.  This one
# submits the kernels from src/ops.cpp and src/prefill.cpp and diffs them
# against include/b70/k2_horizon.hpp, and it also checks that prefill and
# decode are on the SAME norm convention -- a divergence there would be
# completely silent.  It takes no model and runs in seconds.
#   ./bin/test_k2_kernels
icpx -fsycl -fsycl-targets=spir64 \
     -O2 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
     -fsycl-device-code-split=per_kernel \
     -I include -I src \
     tools/test_k2_kernels_device.cpp src/ops.cpp src/prefill.cpp \
     src/quantize.cpp \
     -o bin/test_k2_kernels \
  && echo "built  : bin/test_k2_kernels" || echo "warn: k2 kernel test failed"

# Engine gates.  All three write their own synthetic checkpoints, so
# they need no model, no tokenizer and no network, and they run in
# seconds:
#   bin/test_k2_e2e        the K2 path, loaded and generating
#   bin/test_model_matrix  every architecture x every projection format
#   bin/test_parallel_e2e  PP and TP must equal one process, token for token
#   bin/test_spec_e2e      speculation must equal plain decode, incl. TP and PP
ENGINE_SRC=(src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp
            src/safetensors.cpp src/quantize.cpp src/gptq.cpp
            src/gemv_decode.cpp src/gemm_xmx.cpp src/attention.cpp
            src/deltanet.cpp src/moe_kernels.cpp src/moe_ref.cpp
            src/ops.cpp src/prefill.cpp src/tokenizer.cpp)
# These build for spir64 (JIT), not the AOT "$TARGET".  They are
# CORRECTNESS gates: the kernel source is identical either way, and an
# AOT device compile of the whole engine three more times would add
# fifteen minutes to every build for no extra coverage.  They pay a
# one-time JIT cost at launch instead, which nobody is timing.
# GRIMOIRE_SKIP_GATES=1 skips them entirely.
if [[ -z "${GRIMOIRE_SKIP_GATES:-}" ]]; then
  for gate in test_k2_e2e_device:test_k2_e2e \
              test_model_matrix:test_model_matrix \
              test_parallel_e2e:test_parallel_e2e \
              test_spec_e2e:test_spec_e2e ; do
    gsrc="${gate%%:*}"; gbin="${gate##*:}"
    icpx -fsycl -fsycl-targets=spir64 \
         -O2 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
         -fsycl-device-code-split=per_kernel \
         -I include -I src \
         "tools/${gsrc}.cpp" "${ENGINE_SRC[@]}" -o "bin/${gbin}" \
      && echo "built  : bin/${gbin}" || echo "warn: ${gbin} failed to build"
  done
else
  echo "note : GRIMOIRE_SKIP_GATES set -- engine gates not built"
fi

# ---------------------------------------------------------------------
if [[ "$REQUIRED_FAILED" -ne 0 ]]; then
  echo
  echo "=== BUILD FAILED: a required target did not build ==="
  echo "    bin/ may still hold an OLDER binary.  Do not run or benchmark"
  echo "    it: it does not contain this source tree."
  exit 1
fi

# Rule 3: this script does NOT rebuild the bridge .so files.  Running it
# alone after touching a bridge leaves a stale .so, which has previously
# looked like a kernel bug and taken a card off the bus.
if [[ -z "${GRIMOIRE_BRIDGES_BUILT:-}" ]]; then
  echo
  echo "note: bridges were NOT rebuilt by this script."
  echo "      If you changed src/*_bridge.cpp, run tools/build_bridges_b70.sh"
  echo "      instead -- it rebuilds the bridges and then calls this script."
fi
