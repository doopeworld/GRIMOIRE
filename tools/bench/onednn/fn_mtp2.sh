#!/bin/bash
# 10-28: Flash-Next MTP (mtp4), step 2 = real speculation on two cards (GRIMOIRE_MTP=1, no shadow):
# drafts K tokens with the head, verifies them in one batched pass (prefill_qwen4_exp verify mode,
# DeltaNet verify + replay, conv ring / PLE history rebuilt at the commit).  Same placement in
# every arm (frozen hits, 160 experts a layer in VRAM to make room for the head):
#   base  : no MTP -- the reference answers and speed
#   k3/k4 : GRIMOIRE_MTP_K=3 / 4 -- answers compared with base (a batched verify adds in another
#           order than decode, so a near-tie can flip: count, do not assume), llama-benchy tg128,
#           GRIMOIRE_SPEC_STATS=1 accepted tokens per step
# Run AFTER fn_mtp1.sh showed the head drafts sensibly (shadow acceptance well above chance).
S=/mnt/user/isos/grimoire-runs/bench-1022
cd $S || exit 2
O=$S/logs/fn-mtp2.outer
[ -z "$(docker ps -q --filter ancestor=grimoire-b70:latest)" ] || { echo "a GPU container is up" > $O; exit 1; }
PD=/mnt/storage/Models/grimoire-ple
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=160 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab"
F=/models2/Qwen3.8-Flash-Next-NVFP4
RP="RP_WIN=1000 RP_RUN=1000000"
{
for v in base: k3:"GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 GRIMOIRE_SPEC_STATS=1" k4:"GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=4 GRIMOIRE_SPEC_STATS=1"; do
  t=${v%%:*}; e=${v#*:}
  cp $PD/flash-next.hits.frozen-1028 $PD/flash-next.hits.ab
  env $RP LOAD_MAX=1500 CLIENT_CPUS=14,15 ./dual.sh fn-mtp2-$t TP $F bf16 65536 "$B $e" \
    gen,pleready,benchy --pp 2048 --tg 128 --runs 2 > logs/fn-mtp2-$t.run 2>&1
  echo "== fn-mtp2-$t [$e]"
  grep -E "\[gen\]|STOP|LEFT|ready=0" logs/fn-mtp2-$t.run | cut -c1-160
  grep -E "mtp4 head|speculation|accepted|error|failed" logs/fn-mtp2-$t.server.log | head -8 | cut -c1-200
  grep -E "^\| /" logs/fn-mtp2-$t.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-8s %-18s\n", $3, $4}'
  sleep 3
done
echo "== answers vs base"
python3 - <<'EOF'
import json
def load(n):
    try: return json.load(open("logs/%s.gen.json" % n))
    except Exception: return {}
b = load("fn-mtp2-base")
for arm in ("fn-mtp2-k3", "fn-mtp2-k4"):
    a = load(arm)
    for k in b:
        x, y = b[k]["text"], a.get(k, {}).get("text", "")
        n = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
        print("  %-12s %-8s %s" % (arm, k, "SAME" if x == y else "first difference at character %d of %d" % (n, len(x))))
EOF
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_MTP2_DONE
} > $O 2>&1
