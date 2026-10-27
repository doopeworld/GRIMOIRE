#!/bin/bash
# Runs inside grimoire-dev (CPU only): the CPU-expert pool scenarios.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
B=/grimoire/bin/cpu_expert_bench
echo "== dot speed, one thread"
$B dot
echo "== one rank alone: pinned (engine list), unpinned (8 threads)"
$B ffn 0 1 0 2
$B ffn 0 0 0 2
echo "== two ranks at once: pinned"
$B ffn 0 1 0 2 4000 & $B ffn 0 1 1 2 4000; wait
echo "== two ranks at once: unpinned"
$B ffn 0 0 0 2 4000 & $B ffn 0 0 1 2 4000; wait
echo "== two experts a call, one rank alone, pinned"
$B ffn 0 1 0 2 3000 2
echo "== thread counts on chosen CPUs (CPUS[0] = the caller, unpinned)"
CPUS=0,1,2 $B ffn 3 0 0 2
CPUS=0,1,2,6,7,8,9 $B ffn 7 0 0 2
CPUS=0,1,2,3,4,5 $B ffn 6 0 0 1
CPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13 $B ffn 14 0 0 1
echo "== no gap between calls, pinned"
$B ffn 0 1 0 2 3000 1 0
