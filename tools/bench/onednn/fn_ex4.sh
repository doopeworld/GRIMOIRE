#!/bin/bash
# 10-28: Flash-Next two cards -- the four decode changes (router top-k register kernel for 512
# experts, split head, lean tiered-expert kernels, HC norm in registers) against all four
# switched off (= the code before), for the SAME answers and for speed.
# Every run starts from the same expert placement: a frozen copy of the hits file, put back
# before each run.  A served run adds its routes to that file, so the next run's VRAM / RAM
# split -- and with it which experts the CPU sums instead of the GPU, in another order -- and
# so the greedy text moved from run to run (10-28 morning: same build, different stories).
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
PD=/mnt/storage/Models/grimoire-ple
[ -e $PD/flash-next.hits.frozen-1028 ] || cp $PD/flash-next.hits $PD/flash-next.hits.frozen-1028
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab"
RP="RP_WIN=1000 RP_RUN=1000000"
OLD="GRIMOIRE_TOPK_ESIMD=0 GRIMOIRE_TP_HEAD_SPLIT=0 GRIMOIRE_TMOE_LEAN=0 GRIMOIRE_HC_NORM_REGS=0"
run() {  # run NAME "ENV" TESTS [benchy args]
  local n=$1 e=$2 t=$3; shift 3
  cp $PD/flash-next.hits.frozen-1028 $PD/flash-next.hits.ab
  env $RP LOAD_MAX=1500 CLIENT_CPUS=14,15 ./dual.sh $n TP $F bf16 65536 "$B $PLE $e" $t "$@" > logs/$n.run 2>&1
  echo "== $n [${e:-new defaults}]"
  grep -E "\[gen\]|STOP|LEFT|ready=0" logs/$n.run | cut -c1-160
  grep -E "lm_head|split-head|exchange" logs/$n.server.log | head -4 | cut -c1-200
  grep -E "^\| /" logs/$n.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-8s %-18s\n", $3, $4}'
  sleep 3
}
run fx-old1 "$OLD" gen
run fx-old2 "$OLD" gen
run fx-new1 "" gen,pleready,benchy --pp 2048 4096 --tg 128 --runs 2
run fx-new2 "" gen
echo "== answers: the same text?"
python3 - <<'EOF'
import json
def load(n):
    try: return json.load(open("logs/%s.gen.json" % n))
    except Exception: return {}
r = {n: load(n) for n in ("fx-old1", "fx-old2", "fx-new1", "fx-new2")}
for k in r["fx-old1"]:
    t = r["fx-old1"][k]["text"]
    same = lambda n: "SAME" if r[n].get(k, {}).get("text") == t else "DIFFERENT"
    print("  %-8s old run 2 %s, new run 1 %s, new run 2 %s" % (k, same("fx-old2"), same("fx-new1"), same("fx-new2")))
EOF
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_EX4_DONE
