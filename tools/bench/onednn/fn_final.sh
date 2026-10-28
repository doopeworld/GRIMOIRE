#!/bin/bash
# 10-27: Flash-Next final numbers with today's defaults (layer-major prefill, HC inject kernel,
# per-call sub-chunks, decode HC kernels B/C; ring prefetch and cold split off): two cards (TP,
# 170 experts/layer each) and one card (G0, 112).  gen, code, wait for the PLE copy, llama-benchy
# pp2048/pp4096 tg128 --runs 2.  Morning: TP 974 / 1,183, 31.2 / 30.1; G0 464 / 654, 21.95 / 21.75.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
BEN="--pp 2048 4096 --tg 128 --runs 2"
RP="RP_WIN=1000 RP_RUN=1000000"
cmp() {  # cmp NAME REF.json
python3 - "$1" "$2" <<'EOF'
import json, sys
name, ref = sys.argv[1], sys.argv[2]
try:
    a = json.load(open(ref)); b = json.load(open("logs/%s.gen.json" % name))
except Exception as e:
    print("no answers to compare:", e); sys.exit(0)
for k in a:
    x, y = a[k]["text"], b[k]["text"]
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]:
        n += 1
    print("%-8s %s: same first %d of %d chars vs %s; %d tokens in %.1f s" % (k, name, n, len(x), ref, b[k]["tokens"], b[k]["seconds"]))
EOF
}
show() {
  grep -E "\[gen\]|\[code\] \| ~|PLE table in memory|PLE copy|STOP|LEFT|ready=0" logs/$1.run | cut -c1-170
  grep -E "^\| /" logs/$1.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-8s %-18s\n", $3, $4}'
}
env $RP LOAD_MAX=1500 ./dual.sh fn-final-tp TP $F bf16 65536 "$B GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 $PLE" gen,code,pleready,benchy $BEN > logs/fn-final-tp.run 2>&1
echo "== two cards (TP)"; show fn-final-tp; cmp fn-final-tp logs/fn-tp-lm.gen.json
sleep 3
LOAD_MAX=1500 ./dual.sh fn-final-g0 G0 $F bf16 65536 "$B GRIMOIRE_EXPERT_VRAM_PER_LAYER=112 $PLE" gen,code,pleready,benchy $BEN > logs/fn-final-g0.run 2>&1
echo "== one card (G0)"; show fn-final-g0; cmp fn-final-g0 logs/fn-g0-lm.gen.json
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_FINAL_DONE
