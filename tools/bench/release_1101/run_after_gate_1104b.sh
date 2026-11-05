#!/bin/bash
# After the combined v1.10.4 gate ends (done or stopped), on a quiet machine, one GPU job at a time: the B580
# comparison, GRIMOIRE's G21 build of the candidate (grimoire-release-1105) vs llama.cpp SYCL and Vulkan, Ornith-1.5-35B-A3B
# with experts partly in RAM: GRIMOIRE with a 6 GB expert budget and with v1.10.4's default (<= 60% of the card).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
until grep -qE "^V1104B_ALL_(DONE|STOP)" "$B/logs/v1104b-all.out" 2>/dev/null; do sleep 30; done
until [ -z "$(docker ps -q)" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
GTREE=/mnt/storage/isos/grimoire-release-1105 NCPU=26 bash "$B/run_b580_1104.sh" v1104 grimoire sycl vulkan grimoire-default
