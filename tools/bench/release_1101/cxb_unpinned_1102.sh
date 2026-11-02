#!/bin/bash
# CPU only. The non-hybrid desktop layout: no pinning, hardware threads - 1 workers. Busy cores = a compile or a browser.
set -u
W=/mnt/storage/isos/grimoire-cxpool-1102/tree
one() { # TAG "busy cores"
  local pids=() c
  for c in $2; do taskset -c "$c" sh -c "while :; do :; done" & pids+=($!); done
  echo "== $1 unpinned busy=[${2:-none}]"
  docker run --rm --name grim-cxb-up-$$ -v "$W":/grimoire -w /grimoire grimoire-dev:latest bash -lc \
    "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; bin/cpu_expert_bench.$1 ffn 0 0 0 1 1500 2 600" 2>&1 | grep -vE "^pool:|AnonHuge" | cut -c1-170
  for c in "${pids[@]:-}"; do [ -n "$c" ] && kill "$c" 2>/dev/null; done
}
for t in old new; do one $t ""; one $t "3 7 9 11"; done
