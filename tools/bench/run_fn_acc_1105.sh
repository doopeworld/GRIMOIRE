#!/bin/bash
# 11-05 19:15: what the dense part's precision costs Flash-Next in accuracy, one card (gpu0), candidate build.
# Reference = GRIMOIRE's own BF16 dense part (the original BF16 Flash-Next is not on this machine); the routed experts
# are the checkpoint's NVFP4 in every run, the same placement (90 a layer in VRAM, frozen hits), so the comparison
# isolates the dense part: --proj int4 (one card) and --proj int8 (the two-card candidate), each against --proj bf16.
# 512 token ids of the Sherlock text (Flash-Next's tokenizer, in the grimoire-ref:cpu container), fed one at a time
# through the decode path (grimoire --score-ids); cmp_logits.py: cosine, KL(bf16 || test), top-1. Holds HOLD-fnacc.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1105b
A=/mnt/storage/Models/grimoire-acc
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/fn-acc-1105.out" 2>&1
echo "FN_ACC_START $(date +%T)"
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-fnacc
trap 'rm -f "$B/logs/HOLD-fnacc"' EXIT
mkdir -p "$A"
cp /mnt/storage/isos/grimoire-fuse/real4k_ascii.txt "$A/text.txt"
cp /mnt/storage/isos/grimoire-fuse/tools/bench/accuracy/cmp_logits.py "$A/" 2>/dev/null || cp "$B/cmp_logits.py" "$A/"
docker run --rm --network none -v /mnt/models:/models2:ro -v "$A":/acc grimoire-ref:cpu python3 -c '
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("/models2/Qwen3.8-Flash-Next-NVFP4")
ids = t(open("/acc/text.txt", encoding="utf-8", errors="replace").read(), add_special_tokens=False)["input_ids"][:512]
open("/acc/fn512.ids", "w").write("\n".join(str(i) for i in ids) + "\n")
print("ids", len(ids), "vocab", len(t))'
[ -s "$A/fn512.ids" ] || { echo "FN_ACC_STOP no ids"; exit 9; }
cd /mnt/storage/isos/grimoire-fuse || exit 9
for pf in bf16 int4 int8; do
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) score $pf"
  EXTRA_ENV="GRIMOIRE_POWER_LIMIT_W=0
GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin
GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab
GRIMOIRE_EXPERT_VRAM_PER_LAYER=90" GRIM_TREE=$T LIM=1500 TAILN=4 \
    bash tools/g0run.sh "fnacc-$pf" /grimoire/bin/grimoire -m /models2/Qwen3.8-Flash-Next-NVFP4 --proj "$pf" --ctx 1024 \
      --score-ids /models/grimoire-acc/fn512.ids --score-out "/models/grimoire-acc/fn-$pf.bin"
  ls -la "$A/fn-$pf.bin" 2>&1 | cut -c1-120
done
cd "$B"
for pf in int4 int8; do
  docker run --rm --network none -v "$A":/acc grimoire-ref:cpu python3 /acc/cmp_logits.py /acc/fn-bf16.bin "/acc/fn-$pf.bin" 248320 "Flash-Next dense $pf vs bf16"
done
cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
echo "FN_ACC_END $(date +%T)"
