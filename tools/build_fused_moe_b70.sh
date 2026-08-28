#!/usr/bin/env bash
set -euo pipefail
export OCL_ICD_FILENAMES="${OCL_ICD_FILENAMES:-}"
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
COMMON=(
 -std=gnu++17 -O3 -fPIC -shared -fsycl -include /src/csrc/sycl_first.h
 -DVLLM_XPU_ENABLE_XE2 -DCUTLASS_ENABLE_HEADERS_ONLY -DCUTLASS_ENABLE_SYCL
 -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED
 -I/grimoire/src -I/fusedref -I/fusedref/csrc/xpu -I/src -I/src/csrc
 -I/src/.deps/cutlass-sycl-src/include
 -I/src/.deps/cutlass-sycl-src/tools/util/include
 -I/opt/venv/lib/python3.12/site-packages/torch/include
 -I/opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include
 -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen
 "-device bmg-g31-a0 -internal_options -cl-intel-256-GRF-per-thread"
 -Xspirv-translator
 -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate)
icpx "${COMMON[@]}" /grimoire/src/xe2_fused_moe_bridge.cpp \
 -o /grimoire/src/libgrimoire_xe2_fused_moe.so.new -lze_loader
mv /grimoire/src/libgrimoire_xe2_fused_moe.so.new \
 /grimoire/src/libgrimoire_xe2_fused_moe.so
ldd /grimoire/src/libgrimoire_xe2_fused_moe.so
