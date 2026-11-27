#!/bin/bash
# run_bench.sh ARGS -- bench_ink_sm on gpu0 (03:00.0) only, container removed after
X=/mnt/storage/isos/grimoire-runs/ink-1123
if docker ps --format "{{.Names}}" | grep -q .; then echo "OTHER CONTAINERS RUNNING"; docker ps; exit 1; fi
N=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render)
timeout -k 10 300 docker run --rm --init --name ink-1123-bench --network none --device "$N:/dev/dri/renderD128" -v "$X:/work" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 $(env | grep -oE "^INK_[A-Z0-9_]+=[^ ]*" | sed "s/^/-e /" | tr "\n" " ") \
  --entrypoint /bin/bash grimoire-b70:v1.10.10 -c "cd /work/src && ./bin/bench_ink_sm $*"
echo "exit $?"
