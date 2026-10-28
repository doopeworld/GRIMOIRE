#!/bin/bash
# 10-28: Flash-Next two cards: the router top-k on the register kernel for 512 experts and the
# head split over the vocabulary, against the old kernels in the SAME build
# (GRIMOIRE_TOPK_ESIMD=0 GRIMOIRE_TP_HEAD_SPLIT=0).
#   1) new + GRIMOIRE_TOPK_CHECK=1 (both top-k kernels on the same logits, bit for bit), answers
#   2) old, 3) new: answers + llama-benchy pp 2048 / 4096, tg 128; the client on CPUs 14-15 in both
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
RP="RP_WIN=1000 RP_RUN=1000000"
run() {  # run NAME "ENV" TESTS [benchy args]
  local n=$1 e=$2 t=$3; shift 3
  env $RP LOAD_MAX=1500 CLIENT_CPUS=14,15 ./dual.sh $n TP $F bf16 65536 "$B $PLE $e" $t "$@" > logs/$n.run 2>&1
  echo "== $n [$e]"
  grep -E "\[gen\]|STOP|LEFT|ready=0" logs/$n.run | cut -c1-160
  grep -E "lm_head|top-k check|split-head" logs/$n.server.log | head -6 | cut -c1-200
  grep -E "^\| /" logs/$n.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-8s %-18s\n", $3, $4}'
  sleep 3
}
run fn-chk "GRIMOIRE_TOPK_CHECK=1" gen
run fn-old "GRIMOIRE_TOPK_ESIMD=0 GRIMOIRE_TP_HEAD_SPLIT=0" gen,pleready,benchy --pp 2048 4096 --tg 128 --runs 2
run fn-new "" gen,pleready,benchy --pp 2048 4096 --tg 128 --runs 2
echo "== answers: the same text?"
python3 - <<'EOF'
import json
def load(n):
    try: return json.load(open("logs/%s.gen.json" % n))
    except Exception as e: return {}
old, new, chk, tl3 = load("fn-old"), load("fn-new"), load("fn-chk"), load("fn-tl3")
for k in old:
    t = old[k]["text"]
    print("  %-8s new %s, check run %s, 10-28 morning run %s" % (k,
          "SAME" if new.get(k, {}).get("text") == t else "DIFFERENT",
          "SAME" if chk.get(k, {}).get("text") == t else "DIFFERENT",
          "SAME" if tl3.get(k, {}).get("text") == t else "DIFFERENT"))
EOF
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_AB3_DONE
