#!/usr/bin/env bash
set -euo pipefail

cd "${1:-/grimoire}"

: "${ONEAPI_ROOT:=/opt/intel/oneapi}"
export OCL_ICD_FILENAMES="${OCL_ICD_FILENAMES:-}"
if [[ -f "${ONEAPI_ROOT}/setvars.sh" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${ONEAPI_ROOT}/setvars.sh" --force >/dev/null 2>&1 || true
  set -u
fi
if ! command -v icpx >/dev/null; then
  for compiler_bin in "${ONEAPI_ROOT}"/compiler/*/bin; do
    [[ -d "${compiler_bin}" ]] && export PATH="${compiler_bin}:${PATH}"
  done
fi
command -v icpx >/dev/null || { echo "icpx not found" >&2; exit 1; }

COMMON=(
  -std=gnu++20 -O3 -fPIC -shared -fsycl
  -include /src/csrc/sycl_first.h
  -DVLLM_XPU_ENABLE_XE2 -DGRIMOIRE_ENABLE_AUTOTUNE
  -DCUTLASS_ENABLE_HEADERS_ONLY -DCUTLASS_ENABLE_SYCL
  -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED
  -Isrc -I/src -I/src/csrc
  -I/src/csrc/xpu/grouped_gemm/xe_2
  -I/src/csrc/xpu/attn/xe_2
  -I/src/.deps/cutlass-sycl-src/include
  -I/src/.deps/cutlass-sycl-src/applications
  -I/src/.deps/cutlass-sycl-src/tools/util/include
  # Upstream paged_kv_utils.h includes torch/all.h even though the raw chunk
  # prefill instantiation below uses no ATen symbols.  Headers are builder-only;
  # ldd validation guarantees that the produced raw bridge does not link Torch.
  -I/opt/venv/lib/python3.12/site-packages/torch/include
  -I/opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include
  -fsycl-targets=spir64_gen
  -Xsycl-target-backend=spir64_gen
  "-device bmg-g31-a0 -internal_options -cl-intel-256-GRF-per-thread"
  -Xspirv-translator
  -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
)

if [[ "${GRIMOIRE_BRIDGE_ONLY:-all}" == "all" ||
      "${GRIMOIRE_BRIDGE_ONLY:-all}" == "grouped" ]]; then
  icpx "${COMMON[@]}" src/xe2_grouped_bridge.cpp \
    -o src/libgrimoire_xe2_grouped.so.new -lze_loader
fi
if [[ "${GRIMOIRE_BRIDGE_ONLY:-all}" == "all" ||
      "${GRIMOIRE_BRIDGE_ONLY:-all}" == "attention" ]]; then
  icpx "${COMMON[@]}" src/xe2_attention_raw_bridge.cpp \
    -o src/libgrimoire_xe2_attention_raw.so.new -lze_loader
fi
if [[ "${GRIMOIRE_BRIDGE_ONLY:-all}" == "all" ||
      "${GRIMOIRE_BRIDGE_ONLY:-all}" == "gdn" ]]; then
  icpx "${COMMON[@]}" src/xe2_gdn_raw_bridge.cpp \
    -o src/libgrimoire_xe2_gdn_raw.so.new -lze_loader
fi
if [[ "${GRIMOIRE_BRIDGE_ONLY:-all}" == "all" ||
      "${GRIMOIRE_BRIDGE_ONLY:-all}" == "onednn" ]]; then
  icpx -std=gnu++20 -O3 -fPIC -shared -fsycl \
    -fsycl-targets=spir64_gen \
    -Xsycl-target-backend=spir64_gen \
    "-device bmg-g31-a0" \
    -Isrc -I/opt/intel/oneapi/dnnl/2026.0/include \
    src/xe2_onednn_bridge.cpp \
    -L/opt/intel/oneapi/dnnl/2026.0/lib \
    -Wl,-rpath,/opt/grimoire/lib \
    -ldnnl -lze_loader -o src/libgrimoire_onednn.so.new
fi

for bridge in grouped attention_raw gdn_raw; do
  [[ -f "src/libgrimoire_xe2_${bridge}.so.new" ]] || continue
  mv "src/libgrimoire_xe2_${bridge}.so.new" \
     "src/libgrimoire_xe2_${bridge}.so"
done
if [[ -f src/libgrimoire_onednn.so.new ]]; then
  mv src/libgrimoire_onednn.so.new src/libgrimoire_onednn.so
fi
if [[ "${GRIMOIRE_BRIDGE_ONLY:-all}" == "all" ]]; then
  bash build_b70.sh
fi
