#!/bin/bash
# 10-27 evening: build the server with the CPU-partial fold and the CPU bench with USM=1, then
# (1) the CPU bench on sycl::malloc_host memory vs plain memory (gpu0 opened for its allocator
# only, no kernel runs; gpu0.lock held), (2) fn_fold.sh (two cards: byte check, then speed).
S=/mnt/user/isos/grimoire-runs/bench-1022
R=/mnt/storage/isos/grimoire-fuse
O=$S/logs/fold_chain.out
cd $S || exit 2
$R/tools/dev_build.sh server > logs/build-fold.log 2>&1
if ! grep -q "built: bin/grimoire-server" logs/build-fold.log; then
  echo BUILD_FAILED > $O; tail -20 logs/build-fold.log >> $O; exit 1
fi
docker run --rm -v $R:/grimoire -w /grimoire grimoire-dev:latest bash -lc \
  "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; icpx -fsycl -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno -I include -I src tools/cpu_expert_bench.cpp src/cpu_experts.cpp -lpthread -o bin/cpu_expert_bench" > logs/build-cxb2.log 2>&1
echo BUILD_OK > $O
N0=$(basename $(readlink /dev/dri/by-path/pci-0000:03:00.0-render))
run_b() {
  docker run --rm --init --device /dev/dri/$N0 -v /dev/dri/by-path:/dev/dri/by-path:ro -v $R/bin:/grimoire/bin:ro \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu -e ZE_AFFINITY_MASK=0 "$@"
}
(
  flock -n 8 || { echo "gpu0.lock is held -- USM bench skipped"; exit 0; }
  echo "== CPU experts on plain memory vs sycl::malloc_host memory ($N0 opened, no kernel)"
  run_b --entrypoint /grimoire/bin/cpu_expert_bench grimoire-b70:latest ffn 0 1 0 2
  run_b -e USM=1 --entrypoint /grimoire/bin/cpu_expert_bench grimoire-b70:latest ffn 0 1 0 2
  run_b --entrypoint /grimoire/bin/cpu_expert_bench grimoire-b70:latest dot
  run_b -e USM=1 --entrypoint /grimoire/bin/cpu_expert_bench grimoire-b70:latest dot
) 8>/mnt/user/isos/grimoire-runs/gpu0.lock >> $O 2>&1
./fn_fold.sh >> $O 2>&1
echo CHAIN_DONE >> $O
