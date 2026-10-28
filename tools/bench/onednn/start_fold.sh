#!/bin/bash
# 10-27: when the CPU bench (cxb.sh) is done, put the staged sources in place and run fold_chain.sh.
S=/mnt/user/isos/grimoire-runs/bench-1022
R=/mnt/storage/isos/grimoire-fuse
for i in $(seq 1 240); do
  grep -qE "CXB_DONE|BENCH_BUILD_FAILED|CHAIN_NOT_DONE" $S/logs/cxb.out 2>/dev/null && break
  sleep 10
done
grep -q CXB_DONE $S/logs/cxb.out || { echo "CPU bench not done -- fold chain not started" > $S/logs/fold_chain.out; exit 1; }
mv $R/src/grimoire.cpp.fold $R/src/grimoire.cpp
mv $R/tools/cpu_expert_bench.cpp.usm $R/tools/cpu_expert_bench.cpp
chmod +x $S/fold_chain.sh $S/fn_fold.sh
exec $S/fold_chain.sh
