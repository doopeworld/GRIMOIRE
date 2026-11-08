#!/bin/bash
set -euo pipefail
A=/mnt/storage/Models/grimoire-acc
B=/mnt/storage/isos/grimoire-runs/bench-1022
exec >"$B/logs/fetch-calibration-1108.out" 2>&1
fetch() {
  local name=$1 url=$2
  if [ -s "$A/$name" ]; then echo "SKIP $name"; return; fi
  curl -fL --retry 2 --max-time 1800 -o "$A/$name.part" "$url"
  python3 - "$A/$name.part" <<'CHECK'
import sys
with open(sys.argv[1], 'rb') as f:
    assert f.read(4) == b'GGUF', 'not a GGUF calibration file'
CHECK
  mv "$A/$name.part" "$A/$name"
  sha256sum "$A/$name"
}
fetch imatrix-qwen38-27b.gguf https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/imatrix_unsloth.gguf
fetch imatrix-flash-next.gguf https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/imatrix_unsloth.gguf_file
echo CALIBRATION_FETCH_PASS
