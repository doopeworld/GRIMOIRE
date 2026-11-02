#!/bin/bash
# CPU only, no devices. Build the CPU-expert pool bench from one tree into its own file. build_cxb_1102.sh TAG
set -euo pipefail
TAG=$1
W=/mnt/storage/isos/grimoire-cxpool-1102/tree
docker run --rm --name grim-cxb-build-$TAG --cpuset-cpus 14-15 -v "$W":/grimoire -w /grimoire grimoire-dev:latest bash -lc "
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  icpx -fsycl -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno -I include -I src tools/cpu_expert_bench.cpp src/cpu_experts.cpp -lpthread -o bin/cpu_expert_bench.$TAG"
ls -la "$W/bin/cpu_expert_bench.$TAG"
