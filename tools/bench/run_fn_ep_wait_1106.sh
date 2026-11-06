#!/bin/bash
# 11-06: does Radiance's width split (every expert halved across the two ranks, RESULTS-2026-11-05.md "Radiance port
# study") pay on Flash-Next?  Measure first: two cards (the DUAL-FLASH-NEXT profile's settings: Megatron TP + expert-
# parallel tiered experts, ctx 65536, 170 experts a layer of the rank's 256 in VRAM), the dense part INT8, with
# GRIMOIRE_TP_SMALL_AR_TIME=1 (diagnostic: two extra waits a call, so tokens/s here is NOT a speed number).  Every
# 2048 all-reduces each rank prints where a call's time went; "peer wait" on the rank that finished first is the
# imbalance between the ranks' expert work (GPU experts + the CPU experts the rank owns) -- what an even width split
# would remove.  Radiance (2x R9700, its experts streamed from a host pool too): ~15 ms a decode step with MTP 3.
# Frozen hits as in run_fn_dual_1105.sh.  Holds logs/HOLD-fnep.  Log: logs/fn-ep-wait-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=${EP_TREE:-/mnt/storage/isos/grimoire-release-1105b}
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/fn-ep-wait-1106.out" 2>&1
echo "FN_EP_START $(date +%T) tree=$T"
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-\|^grim-get-")" ] &&
      ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-fnep
trap 'rm -f "$B/logs/HOLD-fnep"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 GRIMOIRE_TP_SMALL_AR_TIME=1"
name=fnep1106-int8
cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
echo "== $(date +%T) $name"
bash ./dual_engine_1102.sh "$name" TP "$F" int8 65536 "$E" pleready,gen,benchy --pp 512 --tg 256 --concurrency 1 --runs 1 \
  > "logs/$name.run" 2>&1
echo "rc=$?"
grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg256|pp512" "logs/$name.run" | cut -c1-170
echo "-- the last timing line of each rank and row class (us per call):"
grep -h "TP small all-reduce" "logs/$name.server.log" 2>/dev/null |
  awk 'match($0, /rank [0-9]+, [^ ]+ rows/) {last[substr($0, RSTART, RLENGTH)]=$0} END{for(k in last) print last[k]}' |
  sort | cut -c1-260
grep -hiE "CPU experts in decode|tiered experts" "logs/$name.server.log" 2>/dev/null | head -4 | cut -c1-200
echo "FN_EP_END $(date +%T)"
