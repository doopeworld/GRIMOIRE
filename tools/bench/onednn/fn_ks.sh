#!/bin/bash
# 10-27: decode copies out by kernel + pinned flag (ksend) instead of copy-engine memcpy + event wait.
# Same arithmetic, so the greedy answers must be IDENTICAL to before (logs/fn-tl-ref.gen.json two
# cards, logs/fn-final-g0.gen.json one card).  1) two cards, all-reduce timers + timeline, answers;
# 2) two cards llama-benchy; 3) one card answers + llama-benchy.
# Before: TP tg128 31.0 / 30.2, all-reduce copy out 38-56 us a call; G0 tg128 22.2 / 21.9.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
BEN="--pp 2048 4096 --tg 128 --runs 2"
RP="RP_WIN=1000 RP_RUN=1000000"
same() {  # same NAME REF.json: identical answers?
python3 - "$1" "$2" <<'EOF'
import json, sys
name, ref = sys.argv[1], sys.argv[2]
try:
    a = json.load(open(ref)); b = json.load(open("logs/%s.gen.json" % name))
except Exception as e:
    print("no answers to compare:", e); sys.exit(0)
for k in a:
    x, y = a[k]["text"], b[k]["text"]
    print("%-8s %s: %s (%d vs %d chars)" % (k, name, "IDENTICAL" if x == y else "DIFFERENT", len(x), len(y)))
EOF
}
show() {
  grep -E "\[gen\]|\[code\] \| ~|PLE table in memory|STOP|LEFT|ready=0" logs/$1.run | cut -c1-170
  grep -E "^\| /" logs/$1.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-8s %-18s\n", $3, $4}'
}
env $RP LOAD_MAX=1500 ./dual.sh fn-ks-tl TP $F bf16 65536 "$B GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 $PLE GRIMOIRE_TIMELINE=1 GRIMOIRE_TP_SMALL_AR_TIME=1" gen > logs/fn-ks-tl.run 2>&1
echo "== two cards, timers"; show fn-ks-tl; same fn-ks-tl logs/fn-tl-ref.gen.json
grep -E "small all-reduce .*calls" logs/fn-ks-tl.server.log | tail -2 | cut -c1-260
grep -E "rank0" logs/fn-ks-tl.server.log | grep -E "TOTAL|sum of|layer marks only" | cut -c1-120
sleep 3
env $RP LOAD_MAX=1500 ./dual.sh fn-ks-tp TP $F bf16 65536 "$B GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 $PLE" gen,code,pleready,benchy $BEN > logs/fn-ks-tp.run 2>&1
echo "== two cards"; show fn-ks-tp; same fn-ks-tp logs/fn-tl-ref.gen.json
sleep 3
LOAD_MAX=1500 ./dual.sh fn-ks-g0 G0 $F bf16 65536 "$B GRIMOIRE_EXPERT_VRAM_PER_LAYER=112 $PLE" gen,code,benchy $BEN > logs/fn-ks-g0.run 2>&1
echo "== one card"; show fn-ks-g0; same fn-ks-g0 logs/fn-final-g0.gen.json
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_KS_DONE
