#!/bin/bash
# Flash-Next on two cards, Megatron TP with expert-parallel MoE (10-26): the three fixed greedy answers
# against the PP reference (logs/pp-flashnext.gen.json, 10-22), then the code probe and llama-benchy
# pp2048/pp4096 tg128 --runs 2 exactly as the speed table (st-fn-pp: 905 / 1,150 pp, 25.2 / 24.4 tg).
# NV = experts per layer in VRAM on EACH card (of its 256); 180 = the same VRAM as PP's 360 x 24 layers.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
NV=${NV:-170}
NAME=${NAME:-fn-tp}
LOAD_MAX=1500 ./dual.sh $NAME TP $F bf16 65536 "$B GRIMOIRE_EXPERT_VRAM_PER_LAYER=$NV $PLE ${EXTRA:-}" \
  ${TESTS:-gen,code,benchy} --pp 2048 4096 --tg 128 --runs 2 2>&1
grep -E "tiered experts|VRAM|PLE|all-reduce|Megatron|error|fail" logs/$NAME.server.log 2>/dev/null | head -20 | cut -c1-200
python3 - "$NAME" <<'EOF'
import json, sys
name = sys.argv[1]
try:
    a = json.load(open("logs/pp-flashnext.gen.json")); b = json.load(open("logs/%s.gen.json" % name))
except Exception as e:
    print("no answers to compare:", e); sys.exit(0)
for k in a:
    x, y = a[k]["text"], b[k]["text"]
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]:
        n += 1
    print("%-8s same first %d of %d chars; TP %d tokens in %.1f s" % (k, n, len(x), b[k]["tokens"], b[k]["seconds"]))
    if n < len(x):
        print("   PP: %r" % x[max(0, n - 40):n + 60])
        print("   TP: %r" % y[max(0, n - 40):n + 60])
EOF
echo FN_TP_DONE
