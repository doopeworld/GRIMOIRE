#!/bin/bash
# 11-06 evening: Flash-Next MTP on ONE card again, now that the heat engine and the expert fill leave far fewer experts in
# RAM -- the MTP verify's floor (11-06 16:00: MoE 58-93 ms of a 4-row verify = the rows' RAM experts on the CPU; K=3
# 28-31 tok/s vs plain 45 at 120 a layer, no heat). Release tree grimoire-release-1106 (HOSTBIN), gpu0, Flash-Next
# NVFP4, INT4 dense part, 120 a layer at load + fill + heat (the defaults), frozen hits before every run:
#   plain / K=3 / K=2 (both with the 64K draft vocabulary), gen + llama-benchy pp512 / tg128 x2 + spec stats.
# Waits for idle GPUs and no HOLD files. Holds logs/HOLD-fnmtpfill. Log: logs/fn-mtp-fill-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1106
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/fn-mtp-fill-1106.out" 2>&1
echo "FN_MTP_FILL_START $(date +%T) tree=$T ($(git -C "$T" rev-parse --short HEAD 2>/dev/null))"
grep -qx RELEASE_BUILD_PASS logs/release-build-1106.out || { echo "FN_MTP_FILL_STOP no passing build"; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-fnmtpfill
trap 'rm -f "$B/logs/HOLD-fnmtpfill"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1"
run() { # name env
  local name=$1 env=$2
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name [$env]"
  bash ./dual_engine_1102.sh "$name" G0 "$F" int4 8192 "$E $env" pleready,gen,benchy --pp 512 --tg 128 \
    --concurrency 1 --runs 2 > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg128|pp512" "logs/$name.run" | cut -c1-170
  grep -hE "expert fill|spec: MTP|speculation|CPU experts in decode \(1[0-9]{3}|error|failed" \
    "logs/$name.server.log" 2>/dev/null | tail -4 | cut -c1-200
}
run fnmf-plain ""
run fnmf-k3 "GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 GRIMOIRE_MTP_DRAFT_VOCAB=65536 GRIMOIRE_SPEC_STATS=1"
run fnmf-k2 "GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=2 GRIMOIRE_MTP_DRAFT_VOCAB=65536 GRIMOIRE_SPEC_STATS=1"
python3 - "$B/logs" <<'PY'
import json, sys
d = sys.argv[1]
try:
    a = json.load(open(f"{d}/fnmf-plain.gen.json"))
    for n in ("fnmf-k3", "fnmf-k2"):
        b = json.load(open(f"{d}/{n}.gen.json"))
        print("  %s vs plain: %s" % (n, ", ".join("%s %s" % (k, "SAME" if a[k]["text"] == b.get(k, {}).get("text") else "reworded") for k in a)))
except Exception as e: print("  no gen json", e)
PY
echo "FN_MTP_FILL_END $(date +%T)"
