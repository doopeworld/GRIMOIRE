#!/bin/bash
# Finish the CPU-only G31+G21 release build after the 11-02 stop: the CLI finished at 17:43, the server did not.
# Same source, image and cores as run_release_build_1102.sh. The stopped log is kept beside the new one.
# Refuses unless the CLI from that run is present. Start only after an explicit resume.
set -u
R=/mnt/storage/isos/grimoire-release-1101
B=/mnt/storage/isos/grimoire-runs/bench-1022
L=$B/logs/release-build-1101.out
[ -e "$L" ] && mv "$L" "$B/logs/release-build-1101.stopped-1102.out"
exec >"$L" 2>&1
echo "RELEASE_BUILD_START=$(date -Iseconds) source=$(git -C "$R" rev-parse --short HEAD) server only; CLI from the 11-02 17:43 run"
if [ -n "$(git -C "$R" status --porcelain)" ]; then
  echo "RELEASE_BUILD_FAIL dirty source tree"; exit 1
fi
[ "$(git -C "$R" rev-parse HEAD)" = 546ce890beaba05e025dac6434decb8e03c42450 ] || { echo "RELEASE_BUILD_FAIL wrong source"; exit 1; }
[ -x "$R/bin/grimoire" ] || { echo "RELEASE_BUILD_FAIL no CLI; run run_release_build_1102.sh instead"; exit 1; }
docker run --rm --name grim-release-build-1102 --cpuset-cpus 6-13 \
  -e GRIM_TARGETS=intel_gpu_bmg_g31,intel_gpu_bmg_g21 \
  -v "$R":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set -e
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
