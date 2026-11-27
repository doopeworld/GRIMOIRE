#!/bin/bash
# Inkling release build (branch inkling-df2-1126) for G31 + G21: inkling2_gpu + libink_xexp.so rebuilt; libink_moe.so,
# libink_gd.so, libgrimoire_gemm.so as shipped in v1.10.12 (ink-rel-1124b, sources unchanged since).  No GPU, no network.
set -eo pipefail
R=/mnt/storage/isos/grimoire-runs/ink-rel-1126
exec > "$R/build.out" 2>&1
mkdir -p "$R/src/bin"
tar -xf "$R/ink-rel-src.tar" -C "$R/src"
cp -a /mnt/storage/isos/grimoire-runs/ink-rel-1124b/src/bin/libgrimoire_gemm.so /mnt/storage/isos/grimoire-runs/ink-rel-1124b/src/bin/libink_moe.so /mnt/storage/isos/grimoire-runs/ink-rel-1124b/src/bin/libink_gd.so /mnt/storage/isos/grimoire-runs/ink-rel-1124b/src/bin/libdnnl.so* "$R/src/bin/"
nice -n 19 docker run --rm --name ink-rel1126-builder --network none --cpuset-cpus 2-13 --cpu-shares 128 -v "$R:/work" -w /work/src grimoire-dev:latest \
  bash -c 'test ! -e /dev/dri || exit 2; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
    set -e
    C="icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31,intel_gpu_bmg_g21 -O3 -std=c++20 -fno-fast-math -ffp-contract=off -Wno-deprecated-declarations -I include -I src"
    date -Iseconds
    $C -fPIC -shared tools/inkling2_xexp.cpp -o bin/libink_xexp.so; echo XEXP_OK; date -Iseconds
    $C tools/inkling2_gpu.cpp src/inkling2_checkpoint.cpp src/safetensors.cpp src/quantize.cpp -L bin -lgrimoire_gemm -link_moe -link_xexp -link_gd "-Wl,-rpath,\$ORIGIN" -o bin/inkling2_gpu; echo GPU_OK; date -Iseconds
    sha256sum bin/*'
echo INK_REL_BUILD_PASS
