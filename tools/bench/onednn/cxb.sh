#!/bin/bash
# 10-27: the CPU-expert pool on its own (tools/cpu_expert_bench.cpp), CPU only, no GPU device.
# Built now; it RUNS only after the GPU test chain (pin_chain.sh) has finished, so the two never
# share the cores.
S=/mnt/user/isos/grimoire-runs/bench-1022
R=/mnt/storage/isos/grimoire-fuse
O=$S/logs/cxb.out
docker run --rm -v $R:/grimoire -w /grimoire grimoire-dev:latest bash -lc \
  "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; icpx -fsycl -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno -I include -I src tools/cpu_expert_bench.cpp src/cpu_experts.cpp -lpthread -o bin/cpu_expert_bench" > $S/logs/build-cxb.log 2>&1
if [ ! -x $R/bin/cpu_expert_bench ]; then echo BENCH_BUILD_FAILED > $O; exit 1; fi
echo BENCH_BUILT > $O
for i in $(seq 1 720); do
  grep -qE "CHAIN_DONE|BUILD_FAILED" $S/logs/pin_chain.out 2>/dev/null && break
  sleep 10
done
grep -qE "CHAIN_DONE|BUILD_FAILED" $S/logs/pin_chain.out || { echo CHAIN_NOT_DONE >> $O; exit 1; }
sleep 5
docker run --rm -v $R:/grimoire -v $S:/bench -w /grimoire grimoire-dev:latest bash /bench/cxb_inner.sh >> $O 2>&1
echo CXB_DONE >> $O
