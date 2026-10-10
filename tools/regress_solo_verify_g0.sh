#!/bin/bash
# Check actual output IDs against ordinary greedy decoding. This fixture catches
# the old single-slot verify lengths that omitted each query's own key.
# Run foreground; g0run handles GPU0 mapping, completion and AER guards.
set -euo pipefail
cd /mnt/storage/isos/grimoire-fuse
out=${SOLO_TEST_OUT:-/mnt/storage/isos/grimoire-runs/solo-verify-$(date +%m%d-%H%M%S)}
mkdir -p "$out"
binary=${SOLO_TEST_BIN:-/grimoire/bin/grimoire}
model=${SOLO_TEST_MODEL:-Qwen3.8-27B-GPTQ-Int4-MTP-BF16}
tag="solo-$(date +%m%d-%H%M%S)"
for variant in plain exact legacy-exact; do
 envs="GRIMOIRE_SEQ_SLOTS=1
GRIMOIRE_PRINT_IDS=1"
 if [ "$variant" != plain ]; then
  envs+="
GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=4
GRIMOIRE_MTP_DRAFT_VOCAB=65536
GRIMOIRE_MTP_EXACT_VERIFY=1
GRIMOIRE_SPEC_STATS=1"
 fi
 if [ "$variant" = legacy-exact ]; then envs+="
GRIMOIRE_LEGACY_VERIFY_ATTN=1"; fi
 EXTRA_ENV="$envs" TAILN=7 LIM=900 bash tools/g0run.sh "$tag-$variant" "$binary" -m "/models/$model" --proj int4 --ctx 8192 -p "Write a detailed story about a lighthouse keeper." -n 96 | tee "$out/$variant.out"
 cp "/tmp/grim-$tag-$variant.log" "$out/$variant.log"
done
python3 - "$out" <<'CHECK'
import re
import sys
from pathlib import Path
out=Path(sys.argv[1])
ids={}
for name in ['plain','exact','legacy-exact']:
 s=(out/f'{name}.log').read_text()
 m=re.search(r'greedy ids \(\d+\):([^\n]+)',s)
 if not m: raise SystemExit(f'{name}: missing output token ids')
 ids[name]=[int(x) for x in m.group(1).split()]
for name in ['exact','legacy-exact']:
 same=ids[name]==ids['plain']
 print(f'{name}: {len(ids[name])} tokens; identical to plain={same}')
 if not same: raise SystemExit(1)
CHECK
