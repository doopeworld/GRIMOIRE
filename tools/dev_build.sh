#!/bin/bash
# dev_build.sh [grimoire|server|all] -- compile bin/ in grimoire-dev: the oneAPI-only
# builder stage of the release Dockerfile.  Same compiler and IGC as the release image,
# and nothing else: no Python, no vLLM, no torch.  Run the result in grimoire-b70.
#   create the image once:  docker build --target builder -t grimoire-dev .
set -euo pipefail
REPO=/mnt/storage/isos/grimoire-fuse
IMG="${GRIM_BUILD_IMAGE:-grimoire-dev:latest}"
case "${1:-all}" in
  grimoire) cmd="bash tools/build_grimoire_only_b70.sh /grimoire" ;;
  server)   cmd="bash tools/build_server_only_b70.sh /grimoire" ;;
  all)      cmd="bash tools/build_grimoire_only_b70.sh /grimoire && bash tools/build_server_only_b70.sh /grimoire" ;;
  *) echo "usage: dev_build.sh [grimoire|server|all]" >&2; exit 2 ;;
esac
docker image inspect "$IMG" >/dev/null 2>&1 || docker build --target builder -t "$IMG" -f "$REPO/Dockerfile" "$REPO"
# CPUs the compile may use: the eight E-cores 6-13 by default, so a build never takes the
# whole Tower (P-cores 0-5 and low-power cores 14-15 stay free); GRIM_BUILD_CPUS=all lifts it.
CPUS="${GRIM_BUILD_CPUS:-6-13}"
CPUSET=()
[ "$CPUS" = all ] || CPUSET=(--cpuset-cpus "$CPUS")
exec docker run --rm ${CPUSET[@]+"${CPUSET[@]}"} -e GRIM_TARGETS="${GRIM_TARGETS:-}" -v "$REPO":/grimoire -w /grimoire "$IMG" bash -lc "$cmd"
