#!/bin/bash
# CPU-only G31+G21 release build of the frozen source: no devices, not privileged, cores 6-13.
# Same steps and strip as the release Dockerfile. assemble-image-1101.sh requires the exact
# RELEASE_BUILD_PASS line in release-build-1101.out, so the interrupted log is kept beside it.
set -u
R=/mnt/storage/isos/grimoire-release-1101
B=/mnt/storage/isos/grimoire-runs/bench-1022
L=$B/logs/release-build-1101.out
[ -e "$L" ] && mv "$L" "$B/logs/release-build-1101.interrupted-1101.out"
exec >"$L" 2>&1
echo "RELEASE_BUILD_START=$(date -Iseconds) source=$(git -C "$R" rev-parse --short HEAD)"
if [ -n "$(git -C "$R" status --porcelain)" ]; then
  echo "RELEASE_BUILD_FAIL dirty source tree"; exit 1
fi
docker run --rm --name grim-release-build-1102 --cpuset-cpus 6-13 \
  -e GRIM_TARGETS=intel_gpu_bmg_g31,intel_gpu_bmg_g21 \
  -v "$R":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
    bash tools/build_grimoire_only_b70.sh /grimoire
    echo RELEASE_CLI_BUILT
    bash tools/build_server_only_b70.sh /grimoire
    echo RELEASE_SERVER_BUILT
    strip --strip-unneeded bin/grimoire bin/grimoire-server bin/libgrimoire_gemm.so bin/libgrimoire_attn.so
  '
rc=$?
echo "RELEASE_BUILD_END=$(date -Iseconds) rc=$rc"
if [ "$rc" -ne 0 ]; then
  echo "RELEASE_BUILD_FAIL rc=$rc"; exit "$rc"
fi
sha256sum "$R/bin/grimoire" "$R/bin/grimoire-server" "$R/bin/libgrimoire_gemm.so" "$R/bin/libgrimoire_attn.so"
echo RELEASE_BUILD_PASS
