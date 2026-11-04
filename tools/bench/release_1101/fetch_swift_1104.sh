#!/bin/bash
# Download Wrapzii/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8 at a pinned revision (18.3 GB) to /mnt/models.
# Large files: pfetch.sh (parallel ranges, sha256 checked); small files: curl. Input: "path size sha256|-" lines.
set -u
REPO=Wrapzii/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8
REV=408925577f800b05e1b09dc47761d203ab4e030f
D=/mnt/models/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8
PF=/mnt/user/isos/grimoire-gguf/benchmarks/gguf/pfetch.sh
mkdir -p "$D"
bad=0
while read -r p s h; do
  [ -n "$p" ] || continue
  mkdir -p "$D/$(dirname "$p")"
  U=https://huggingface.co/$REPO/resolve/$REV/$p
  if [ "$h" = "-" ]; then
    curl -sfL -o "$D/$p" "$U" && echo "small ok $p" || { echo "small FAILED $p"; bad=1; }
  elif [ -f "$D/$p" ] && [ "$(sha256sum "$D/$p" | cut -d' ' -f1)" = "$h" ]; then
    echo "have $p"
  else
    bash "$PF" "$U" "$D/$p" "$s" "$h" 12 | tail -1 || bad=1
    [ "$(sha256sum "$D/$p" | cut -d' ' -f1)" = "$h" ] || { echo "SHA_BAD $p"; bad=1; }
  fi
done < "${1:?file list}"
[ "$bad" = 0 ] && echo SWIFT_FETCH_DONE || echo SWIFT_FETCH_FAIL
