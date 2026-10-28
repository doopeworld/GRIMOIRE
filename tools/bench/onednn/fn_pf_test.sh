#!/bin/bash
# 10-27: Flash-Next after the HC fixes (prompt inject kernel, per-call sub-chunk, decode kernels B/C)
# and the expert ring prefetch across layers.  Runs: two cards (TP, 170 experts/layer each) full test;
# the same with GRIMOIRE_Q4_TIMING + GRIMOIRE_HC_TIMING (pp4096 once: where the time goes now); one
# card (G0, 112) full test.  Full test = the three greedy answers, the code probe, llama-benchy
# pp2048/pp4096 tg128 --runs 2.  GRIMOIRE_HC_CHECK=1 prints new-vs-old decode HC kernel differences.
# Before (10-27 morning, layer-major): TP 974 / 1,183 pp, 31.2 / 30.1 tg; G0 464 / 654, 21.95 / 21.75.
# gpu1's root-port receive errors come in bursts while it idles (10-25..27, never a drop, 0
# uncorrectable; Ian 10-25: run two-card tests anyway): log them, the other stop rules stay.
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
env $RP LOAD_MAX=1500 ./dual.sh fn-tp-pf TP $F bf16 65536 "$B GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 $PLE GRIMOIRE_BUDGET_DEBUG=1 GRIMOIRE_HC_CHECK=1" gen,code,benchy $BEN 2>&1
grep -E "layer-major|HC check|error|fail" logs/fn-tp-pf.server.log | head -12 | cut -c1-220
cmp fn-tp-pf logs/fn-tp-lm.gen.json
sleep 3
env $RP LOAD_MAX=1500 ./dual.sh fn-tp-pfprof TP $F bf16 65536 "$B GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 $PLE GRIMOIRE_Q4_TIMING=1 GRIMOIRE_HC_TIMING=1" benchy --pp 4096 --tg 16 --runs 1 2>&1
grep -E "qwen4_exp prefill" logs/fn-tp-pfprof.server.log | tail -8 | cut -c1-300
sleep 3
LOAD_MAX=1500 ./dual.sh fn-g0-pf G0 $F bf16 65536 "$B GRIMOIRE_EXPERT_VRAM_PER_LAYER=112 $PLE GRIMOIRE_BUDGET_DEBUG=1 GRIMOIRE_HC_CHECK=1" gen,code,benchy $BEN 2>&1
grep -E "HC check|error|fail" logs/fn-g0-pf.server.log | head -8 | cut -c1-220
cmp fn-g0-pf logs/fn-g0-lm.gen.json
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_PF_DONE
