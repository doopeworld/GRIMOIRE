#!/bin/bash
# run_llamacpp.sh -- Phase 1 baseline: llama.cpp (SYCL or Vulkan) on gpu0, Ornith Q4_K_M GGUF.
#
#   run_llamacpp.sh sycl|vulkan [OUTDIR]
#
# COMPETITOR measurement only: llama.cpp's own published image, pinned by digest, in its own
# container; nothing shared with GRIMOIRE.  Same container-lifecycle rules as tools/b70run.sh
# (CLAUDE.md rule 6): --init, the time limit lives INSIDE the container, detached + docker wait,
# long stop timeout.  gpu0 is resolved by PCI device id (tools/gpunode.sh), never by a hard-coded
# renderD number -- on 2026-10-15 renderD128 was the iGPU.
#
# Produces in OUTDIR/<backend>/:
#   env.txt          image digest, llama.cpp build, device list, driver/runtime packages, PCIe link
#   coherence.txt    greedy generation of a fixed prompt -- READ IT before trusting any number (rule 8)
#   bench_*.json     llama-bench JSON (pp512/2048/4096/8192 and tg32/128, -r 5), default and ub=2048
set -u
BACKEND=${1:?sycl|vulkan}
OUT=${2:-/mnt/user/isos/grimoire-runs/gguf-baseline}/$BACKEND
REPO=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_DIR=/mnt/models/Ornith-1.5-35B-A3B-GGUF
MODEL=Ornith-1.5-35B-Q4_K_M.gguf
case $BACKEND in
  sycl)   IMAGE=ghcr.io/ggml-org/llama.cpp:full-intel@sha256:c68de96b6d45634398b648b1ad9efb2f5cb0d1d7e95fa45e9657cd78610d427f
          BENV=(-e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e ZES_ENABLE_SYSMAN=1) ;;
  vulkan) IMAGE=ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:267614252d76fa9e08599483f86f45d2472739ed557e90298ef1ac5ffdabd223
          BENV=() ;;
  *) echo "backend must be sycl or vulkan" >&2; exit 2 ;;
esac
mkdir -p "$OUT"

NODE=$(bash "$REPO/tools/gpunode.sh" gpu0) || exit 2
PCI=""
for l in /dev/dri/by-path/pci-*-render; do
  [ "$(readlink -f "$l")" = "/dev/dri/$NODE" ] && { PCI=${l##*/pci-}; PCI=${PCI%-render}; }
done
[ "$PCI" = 0000:03:00.0 ] || echo "WARNING: gpu0 resolved to $PCI, expected 0000:03:00.0 (Gen4 x8 slot)" >&2

# Refuse if ANY container or host process holds this render node (not only grim-*).
holders=$(for id in $(docker ps -q); do
            docker inspect --format '{{.Name}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$id"
          done | awk -v d="/dev/dri/$NODE" '{for(i=2;i<=NF;i++) if($i==d) print $1}')
procs=$(for f in /proc/[0-9]*/fd/*; do [ "$(readlink "$f" 2>/dev/null)" = "/dev/dri/$NODE" ] && echo "${f#/proc/}"; done | cut -d/ -f1 | sort -u | tr '\n' ' ')
if [ -n "$holders$procs" ]; then echo "REFUSING: /dev/dri/$NODE busy: containers=[$holders] pids=[$procs]" >&2; exit 3; fi

# One container run, the b70run.sh way.  $1 = name suffix, $2 = limit (s), rest = command.
crun() {
  local name="gguf-llama-$BACKEND-$1" limit=$2; shift 2
  docker rm -f "$name" >/dev/null 2>&1
  docker run -d --name "$name" --init --stop-timeout 300 \
      --device /dev/dri/$NODE --group-add "$(stat -c %g /dev/dri/$NODE)" \
      -v $MODEL_DIR:/models:ro "${BENV[@]}" \
      --entrypoint /usr/bin/timeout "$IMAGE" --signal=TERM --kill-after=60 "$limit" "$@" >/dev/null || return 4
  local rc; rc=$(docker wait "$name" 2>/dev/null || echo wait-failed)
  docker logs "$name" 2>&1
  docker rm "$name" >/dev/null 2>&1
  [ "$rc" = 0 ]
}

{
  echo "date: $(date -Is)"
  echo "backend: $BACKEND"
  echo "image: $IMAGE"
  echo "gpu0: $NODE pci=$PCI"
  for p in 0000:00:01.0 "$PCI"; do
    [ -n "$p" ] && echo "link $p: $(cat /sys/bus/pci/devices/$p/current_link_speed 2>/dev/null) x$(cat /sys/bus/pci/devices/$p/current_link_width 2>/dev/null)"
  done
  echo "kernel: $(uname -r)"
  echo "model: $MODEL_DIR/$MODEL sha256=42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f"
  echo "--- llama-cli --version"; crun ver 120 /app/llama-cli --version
  if [ "$BACKEND" = sycl ]; then echo "--- sycl-ls"; crun syclls 120 sycl-ls; fi
  echo "--- packages"; crun pkgs 120 sh -c 'dpkg -l 2>/dev/null | grep -iE "level-zero|intel-opencl|libigc|intel-ocloc|mesa-vulkan|libvulkan|intel-oneapi-runtime-dpcpp" | awk "{print \$2, \$3}"'
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

echo "--- coherence (greedy, 96 tokens)"
crun coh 900 /app/llama-completion -m /models/$MODEL -ngl 99 -fa on -c 4096 --temp 0 --seed 1 -n 96 -no-cnv \
    -p "<|im_start|>user
Explain in three sentences why the sky is blue.<|im_end|>
<|im_start|>assistant
<think>

</think>

" > "$OUT/coherence.txt" 2>&1 \
  || crun coh2 900 /app/llama-cli -m /models/$MODEL -ngl 99 -fa on -c 4096 --temp 0 --seed 1 -n 96 -no-cnv -st \
    -p "Explain in three sentences why the sky is blue." > "$OUT/coherence.txt" 2>&1
echo "coherence rc=$?  (read $OUT/coherence.txt)"; grep -v -E "^(llama|load|print_info|ggml|common|main|sampler|generate|system_info|build|.*: )" "$OUT/coherence.txt" | head -20

for UB in 512 2048; do
  echo "--- llama-bench ub=$UB"
  crun bench-ub$UB 5400 /app/llama-bench -m /models/$MODEL -ngl 99 -fa 1 -ub $UB -b 2048 \
      -p 512,2048,4096,8192 -n 32,128 -r 5 -o json > "$OUT/bench_ub$UB.raw" 2>&1
  echo "rc=$?"
  python3 - "$OUT/bench_ub$UB.raw" "$OUT/bench_ub$UB.json" <<'EOF'
import json, sys
lines = open(sys.argv[1]).read().splitlines()
# the JSON array starts at a line that is exactly "[" (log lines before it may contain '[')
starts = [i for i, l in enumerate(lines) if l.strip() == "["]
ends = [i for i, l in enumerate(lines) if l.strip() == "]"]
rows = json.loads("\n".join(lines[starts[0]:ends[-1] + 1])) if starts and ends else []
json.dump(rows, open(sys.argv[2], "w"), indent=1)
for r in rows:
    test = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
    print(f"  {test:8s} {r['avg_ts']:10.2f} ± {r['stddev_ts']:7.2f} tok/s  (n_ubatch={r.get('n_ubatch')}, build={r.get('build_commit')})")
EOF
done
echo "LLAMACPP $BACKEND DONE -> $OUT"
