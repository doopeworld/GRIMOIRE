#!/usr/bin/env bash
set -euo pipefail
export OCL_ICD_FILENAMES="${OCL_ICD_FILENAMES:-}"
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
cd "${1:-/grimoire}"
COMMON=(
 -std=gnu++17 -O3 -fPIC -shared -fsycl -include /src/csrc/sycl_first.h
 -DVLLM_XPU_ENABLE_XE2 -DGRIMOIRE_ENABLE_AUTOTUNE
 -DCUTLASS_ENABLE_HEADERS_ONLY -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET
 -DCUTLASS_VERSIONS_GENERATED -Isrc -I/src -I/src/csrc
 -I/src/csrc/xpu/grouped_gemm/xe_2 -I/src/.deps/cutlass-sycl-src/include
 -I/src/.deps/cutlass-sycl-src/tools/util/include -fsycl-targets=spir64_gen
 -Xsycl-target-backend=spir64_gen
 "-device bmg-g31-a0 -internal_options -cl-intel-256-GRF-per-thread"
 -Xspirv-translator
 -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
)
icpx "${COMMON[@]}" src/xe2_grouped_bridge.cpp \
 -o src/libdense_tune_f32.so -lze_loader
icpx -O3 -std=c++17 -fsycl tools/autotune_b70_dense_qwen_f32.cpp \
 src/libdense_tune_f32.so -Wl,-rpath,/grimoire/src \
 -o bin/autotune_qwen_f32
