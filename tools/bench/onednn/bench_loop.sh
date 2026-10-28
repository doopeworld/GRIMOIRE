#!/bin/bash
# Inside the runtime container: one benchdnn call per configuration line of a .cases file
# (KNOBS ;; PROBLEMS), so no knob carries over from one configuration to the next.
cd /onednn/build-3.13.4/tests/benchdnn || exit 2
while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    knobs="${line%%;;*}"; prbs="${line#*;;}"
    echo "### $knobs"
    ./benchdnn --matmul --engine=gpu --mode=P --cold-cache=wei --max-ms-per-prb=1500 \
        "--perf-template=perf,%impl%,%prb%,%Gops%,%-time%,%-Gflops%,%0time%,%0Gflops%,%+ctime%" \
        $knobs $prbs 2>&1
    echo "### rc=$?"
done < "$1"
