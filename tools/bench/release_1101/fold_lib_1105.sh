#!/bin/bash
# 11-05: fold the 9-16-row MXFP4 plan rule (mxfp4-smallm-1105 2ed7142, only src/gemm_fast.cpp changed since the dd33cbe
# build) into the combined v1.10.4 candidate: after RELEASE_BUILD_PASS, move grimoire-release-1105 to 315167c and rebuild
# ONLY libgrimoire_gemm.so (G31+G21). The executables built from dd33cbe are unaffected (no other engine source differs).
# Holds logs/HOLD-fold so the gate waits; leaves it in place (and FOLD_FAIL) if anything goes wrong.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
R=/mnt/storage/isos/grimoire-release-1105
NEW=315167c9f8e8512ef2dbacc123ca04adfa9d7918
exec >"$B/logs/fold-lib-1105.out" 2>&1
touch "$B/logs/HOLD-fold"
until grep -qE "^RELEASE_BUILD_(PASS|FAIL)" "$B/logs/release-build-1105.out"; do sleep 15; done
grep -qx RELEASE_BUILD_PASS "$B/logs/release-build-1105.out" || { echo "FOLD_FAIL build failed"; exit 1; }
until [ -z "$(docker ps -q --filter name=grim-release-build-1105)" ]; do sleep 5; done
[ -z "$(git -C "$R" status --porcelain)" ] || { echo "FOLD_FAIL dirty tree"; exit 1; }
changed=$(git -C "$R" diff --name-only HEAD "$NEW" -- src include CMakeLists.txt | tr '\n' ' ')
[ "$changed" = "src/gemm_fast.cpp " ] || { echo "FOLD_FAIL unexpected change: $changed"; exit 1; }
git -C "$R" checkout -q --detach "$NEW" || { echo "FOLD_FAIL checkout"; exit 1; }
echo "tree at $(git -C "$R" rev-parse --short HEAD) $(date +%T)"
sha256sum "$R/bin/libgrimoire_gemm.so"
bash "$B/build_lib_only_1105.sh" "$R" intel_gpu_bmg_g31,intel_gpu_bmg_g21 6-13 lib-release-1105
cat "$B/logs/build-lib-release-1105.out"
grep -qx LIB_BUILD_PASS "$B/logs/build-lib-release-1105.out" || { echo "FOLD_FAIL library build"; exit 1; }
rm -f "$B/logs/HOLD-fold"
echo "FOLD_PASS $(date +%T)"
