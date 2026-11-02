#!/bin/bash
# Compile the wide-INT4 register probe. CPU only: no devices, not privileged, spare cores 14-15.
# Isolated worktree from verify-grf-probe-1101, linked against the combined build's own G31 libraries.
# The probe is NOT run here; that needs a free GPU0 and the guarded launcher.
set -euo pipefail
P=/mnt/storage/isos/grimoire-fuse
W=/mnt/storage/isos/grimoire-verify-grf-1102
E=/mnt/storage/isos/grimoire-engine-1101
L=/mnt/storage/isos/grimoire-runs/bench-1022/logs/verify-probe-build-1102.out
exec >"$L" 2>&1
echo "PROBE_BUILD_START=$(date -Iseconds)"
[ -e "$W" ] || git -C "$P" worktree add --detach "$W" b0101539816cce36958ffc7aa5a50fa740f80697
[ "$(git -C "$W" rev-parse HEAD)" = b0101539816cce36958ffc7aa5a50fa740f80697 ]
mkdir -p "$W/bin"
cp -P "$E"/bin/libgrimoire_gemm.so "$E"/bin/libgrimoire_attn.so "$E"/bin/libdnnl.so* "$W/bin/"
docker run --rm --name grim-verify-probe-build --cpuset-cpus 14-15 \
  -v "$W":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    set +u
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
    set -u
    icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 \
      -fno-fast-math -ffp-contract=fast -fno-math-errno -I include -I src \
      tools/bench/onednn/i4wide_grf_probe.cpp -Lbin -lgrimoire_gemm -lgrimoire_attn \
      '\''-Wl,-rpath,$ORIGIN'\'' -o bin/i4wide_grf_probe.new
    mv bin/i4wide_grf_probe.new bin/i4wide_grf_probe
  '
ls -l --time-style=long-iso "$W/bin/i4wide_grf_probe"
echo "PROBE_BUILD_PASS $(date -Iseconds)"
