#!/bin/bash
# After the v1.10.4 gate ends (done or stopped), on a quiet machine, one GPU job at a time:
# the B580 comparison: GRIMOIRE v1.10.4's G21 build (grimoire-release-1102b, 35201c2) vs llama.cpp SYCL and Vulkan,
# Ornith-1.5-35B-A3B, experts partly in RAM.
# (11-05: the two-B70 Qwen3.8-27B GPTQ-Int4 + MTP run that used to be step 1 runs earlier, run_gptq_tp_1105.sh.)
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
until grep -qE "^V1104_ALL_(DONE|STOP)" "$B/logs/v1104-all.out"; do sleep 30; done
until [ -z "$(docker ps -q)" ]; do sleep 10; done
GTREE=/mnt/storage/isos/grimoire-release-1102b NCPU=26 bash "$B/run_b580_1104.sh" v1104 grimoire grimoire-default sycl vulkan
