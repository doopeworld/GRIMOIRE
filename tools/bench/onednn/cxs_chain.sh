#!/bin/bash
# 10-27: after the fold tests: build with the decode-only CPU-expert statistics (GRIMOIRE_CX_STATS)
# and the prefetch option removed, run fn_cxs.sh (two cards), then the CPU pool alone in the bench
# with the caller's wait for the last worker (CPU only).
S=/mnt/user/isos/grimoire-runs/bench-1022
R=/mnt/storage/isos/grimoire-fuse
O=$S/logs/cxs_chain.out
cd $S || exit 2
for i in $(seq 1 240); do grep -q CHAIN_DONE logs/fold_chain.out 2>/dev/null && break; sleep 10; done
grep -q CHAIN_DONE logs/fold_chain.out || { echo "fold chain not done -- not started" > $O; exit 1; }
for f in src/grimoire.cpp src/cpu_experts.cpp include/b70/cpu_experts.hpp tools/cpu_expert_bench.cpp; do
  [ -e $R/$f.cxs ] && mv $R/$f.cxs $R/$f
done
$R/tools/dev_build.sh server > logs/build-cxs.log 2>&1
if ! grep -q "built: bin/grimoire-server" logs/build-cxs.log; then
  echo BUILD_FAILED > $O; tail -20 logs/build-cxs.log >> $O; exit 1
fi
CXB="source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; icpx -fsycl -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno -I include -I src tools/cpu_expert_bench.cpp src/cpu_experts.cpp -lpthread -o bin/cpu_expert_bench"
docker run --rm -v $R:/grimoire -w /grimoire grimoire-dev:latest bash -lc "$CXB" > logs/build-cxb3.log 2>&1
echo BUILD_OK > $O
./fn_cxs.sh >> $O 2>&1
echo "== the CPU pool alone (bench): one rank, then both ranks at once" >> $O
docker run --rm -v $R:/grimoire -w /grimoire grimoire-dev:latest bash -lc \
  "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; bin/cpu_expert_bench ffn 0 1 0 2; bin/cpu_expert_bench ffn 0 1 0 2 4000 & bin/cpu_expert_bench ffn 0 1 1 2 4000; wait" >> $O 2>&1
echo CHAIN_DONE >> $O
