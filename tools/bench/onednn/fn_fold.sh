#!/bin/bash
# 10-27: (1) CPU experts with software prefetch on the GPU's pinned memory (USM=1) and on plain
# small-page memory (PAGES=4k): gpu0 opened for its allocator only, no kernel runs.  (2) two cards,
# decode: the CPU experts' partial added on the host inside the MoE all-reduce (GRIMOIRE_CX_FOLD,
# default on) -- byte check first (GRIMOIRE_CX_FOLD_CHECK=1: the old device add must give the same
# bits), CPU experts per call without / with prefetch, then llama-benchy fold / no fold / fold +
# prefetch / fold.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
R=/mnt/storage/isos/grimoire-fuse
N0=$(basename $(readlink /dev/dri/by-path/pci-0000:03:00.0-render))
run_b() {
  docker run --rm --init --device /dev/dri/$N0 -v /dev/dri/by-path:/dev/dri/by-path:ro -v $R/bin:/grimoire/bin:ro \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu -e ZE_AFFINITY_MASK=0 "$@" --entrypoint /grimoire/bin/cpu_expert_bench \
    grimoire-b70:latest ffn 0 1 0 2
}
(
  flock -n 8 || { echo "gpu0.lock is held -- prefetch bench skipped"; exit 0; }
  for pf in 0 2 4 8; do
    echo "== CPU bench, prefetch $pf rows"
    run_b -e USM=1 -e GRIMOIRE_CPU_PREFETCH=$pf 2>&1 | grep -E "median|pool|AnonHuge"
    run_b -e PAGES=4k -e GRIMOIRE_CPU_PREFETCH=$pf 2>&1 | grep -E "median|pool|AnonHuge"
  done
) 8>/mnt/user/isos/grimoire-runs/gpu0.lock
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
RP="RP_WIN=1000 RP_RUN=1000000"
for v in chk:0 pf4:4; do
  t=${v%%:*}; p=${v##*:}
  env $RP LOAD_MAX=1500 ./dual.sh fn-fold-$t TP $F bf16 65536 "$B $PLE GRIMOIRE_CX_FOLD_CHECK=1 GRIMOIRE_CPU_PREFETCH=$p GRIMOIRE_Q4_TIMING=1" gen,pleready,gen > logs/fn-fold-$t.run 2>&1
  echo "== fold byte check, CPU prefetch $p rows"
  grep -E "\[gen\]|STOP|LEFT|ready=0" logs/fn-fold-$t.run | cut -c1-160
  grep -E "fold check" logs/fn-fold-$t.server.log | tail -4
  grep -E "qwen4_exp decode per token" logs/fn-fold-$t.server.log | tail -2 | cut -c1-300
  sleep 3
done
for v in fold:1:0 nofold:0:0 foldpf:1:4 fold2:1:0; do
  IFS=: read -r t p f <<< "$v"
  env $RP LOAD_MAX=1500 ./dual.sh fn-$t-tp TP $F bf16 65536 "$B $PLE GRIMOIRE_CX_FOLD=$p GRIMOIRE_CPU_PREFETCH=$f" gen,pleready,benchy --pp 2048 4096 --tg 128 --runs 2 > logs/fn-$t-tp.run 2>&1
  echo "== $t (fold $p, prefetch $f): llama-benchy"
  grep -E "\[gen\]|STOP|LEFT|ready=0" logs/fn-$t-tp.run | cut -c1-160
  grep -E "^\| /" logs/fn-$t-tp.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-8s %-18s\n", $3, $4}'
  sleep 3
done
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_FOLD_DONE
