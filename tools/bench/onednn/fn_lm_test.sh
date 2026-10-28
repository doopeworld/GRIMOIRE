#!/bin/bash
# 10-26 layer-major Flash-Next prefill: two cards (TP, 170 experts/layer each) then one card (G0, 112),
# each: the three fixed greedy answers, the code probe, llama-benchy pp2048/pp4096 tg128 --runs 2.
# Before (speed table 10-26 / first TP run): TP 713 / 664 pp, 31.7 / 30.6 tg; PP 905 / 1,150, 25.2 / 24.4;
# G0 478 / 752, 21.7 / 22.1.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
BEN="--pp 2048 4096 --tg 128 --runs 2"
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
LOAD_MAX=1500 ./dual.sh fn-tp-lm TP $F bf16 65536 "$B GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 $PLE GRIMOIRE_BUDGET_DEBUG=1" gen,code,benchy $BEN 2>&1
grep -E "layer-major|chunk cap|qwen4_exp prefill budget" logs/fn-tp-lm.server.log | head -6 | cut -c1-220
cmp fn-tp-lm logs/fn-tp.gen.json
sleep 3
LOAD_MAX=1500 ./dual.sh fn-g0-lm G0 $F bf16 65536 "$B GRIMOIRE_EXPERT_VRAM_PER_LAYER=112 $PLE GRIMOIRE_BUDGET_DEBUG=1" gen,code,benchy $BEN 2>&1
grep -E "layer-major|qwen4_exp prefill budget" logs/fn-g0-lm.server.log | head -4 | cut -c1-220
cmp fn-g0-lm logs/pp-flashnext.gen.json
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_LM_DONE
