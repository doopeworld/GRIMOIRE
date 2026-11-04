#!/bin/bash
# After the v1.10.4 gate ends (done or stopped): the B580 comparison on a quiet machine -- GRIMOIRE v1.10.4's
# G21 build (grimoire-release-1102b), then llama.cpp SYCL and Vulkan, Ornith-1.5-35B-A3B, experts partly in RAM.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
until grep -qE "^V1104_ALL_(DONE|STOP)" "$B/logs/v1104-all.out"; do sleep 30; done
until [ -z "$(docker ps -q)" ]; do sleep 10; done
GTREE=/mnt/storage/isos/grimoire-release-1102b NCPU=26 bash "$B/run_b580_1104.sh" v1104 grimoire sycl vulkan
