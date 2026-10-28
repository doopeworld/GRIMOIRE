#!/bin/bash
# 10-28: build with the tiered-expert kernels reading activations from L1 (GX), then
#   fx-gx : frozen placement, answers (compared with fx-old1) + llama-benchy, client on 14-15
#   fn-tl5: per-kernel timeline of the same build
S=/mnt/user/isos/grimoire-runs/bench-1022; R=/mnt/storage/isos/grimoire-fuse
cd $S || exit 2
O=$S/logs/gx.outer
[ -z "$(docker ps -q --filter ancestor=grimoire-b70:latest)" ] || { echo "a GPU container is up" > $O; exit 1; }
t0=$(date +%s)
(cd $R && bash tools/dev_build.sh server) > logs/build-gx.log 2>&1
if ! grep -q "built: bin/grimoire-server" logs/build-gx.log || [ "$(stat -c %Y $R/bin/grimoire-server)" -le "$t0" ]; then
  { echo BUILD_FAILED; grep -E "error:" -A 3 logs/build-gx.log | head -30; } > $O; exit 1
fi
cp $R/bin/grimoire-server $S/grimoire-server.gx
PD=/mnt/storage/Models/grimoire-ple
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
F=/models2/Qwen3.8-Flash-Next-NVFP4
RP="RP_WIN=1000 RP_RUN=1000000"
{
cp $PD/flash-next.hits.frozen-1028 $PD/flash-next.hits.ab
env $RP LOAD_MAX=1500 CLIENT_CPUS=14,15 ./dual.sh fx-gx TP $F bf16 65536 \
  "$B GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab" \
  gen,pleready,benchy --pp 2048 4096 --tg 128 --runs 2 > logs/fx-gx.run 2>&1
echo "== fx-gx"
grep -E "\[gen\]|STOP|LEFT|ready=0" logs/fx-gx.run | cut -c1-160
grep -E "^\| /" logs/fx-gx.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-8s %-18s\n", $3, $4}'
python3 - <<'EOF'
import json
a = json.load(open("logs/fx-old1.gen.json")); b = json.load(open("logs/fx-gx.gen.json"))
for k in a: print("  %-8s %s" % (k, "SAME as the old code" if a[k]["text"] == b.get(k, {}).get("text") else "DIFFERENT"))
EOF
sleep 3
sed -e "s/fn-tl3/fn-tl5/g" -e "s/FN_TL3_DONE/FN_TL5_DONE/" fn_tl3.sh > fn_tl5.sh
bash fn_tl5.sh
grep -E "^\[rank0\]   (sum of .(moe|deltanet|attn|hc-attn|hc-mlp|ple). |head )" logs/fn-tl5.server.log | cut -c1-100
echo GX_DONE
} > $O 2>&1
