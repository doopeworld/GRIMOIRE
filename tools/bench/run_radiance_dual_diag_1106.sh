#!/bin/bash
# 11-06 evening: Radiance's Flash-Next shape on two B70s (TP2 + MTP 3; its verify of 4 rows costs about one plain step,
# ~15 ms; ours 43-52 ms against ~19 ms a plain token, fnmd1106-k3-time).  Where does the two-card verify go, and what
# does the heat engine do on two cards?  radiance-heat-1106b build (verify small-M path + heat engine v2), Megatron TP,
# INT8 dense part, 170 experts a layer of the rank's 256 in VRAM, frozen hits:
#   fnr-k3-q4t    MTP K=3 + GRIMOIRE_SPEC_TIME / GRIMOIRE_Q4_TIMING / GRIMOIRE_TP_SMALL_AR_TIME (fenced; gen only)
#   fnr-plain-h8  GRIMOIRE_EXPERT_HEAT=8, no MTP                          gen + llama-benchy pp2048 / tg128 x2
#   fnr-k3-h8     GRIMOIRE_EXPERT_HEAT=8 + MTP K=3                        gen + benchy
# (today, same link: plain 52.86, MTP K=3 41.60.)  gpu1 at Gen3 x4: dual_engine_1102.sh's link guard; the script stops
# at the first guard stop.  Holds logs/HOLD-fnrdiag. Log: logs/radiance-dual-diag-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/radiance-dual-diag-1106.out" 2>&1
echo "FNR_START $(date +%T) tree=$T ($(git -C "$T" rev-parse --short HEAD 2>/dev/null))"
[ -x "$T/bin/grimoire-server" ] || { echo "FNR_STOP no server in $T"; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-fnrdiag
trap 'rm -f "$B/logs/HOLD-fnrdiag"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
K3="GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 GRIMOIRE_SPEC_STATS=1"
run() { # name env tests...
  local name=$1 env=$2; shift 2
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name [$env]"
  bash ./dual_engine_1102.sh "$name" TP "$F" int8 65536 "$E $env" "$@" > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg128|pp2048|RxErr during" "logs/$name.run" | cut -c1-200
  [ -e "logs/$name.STOP" ] && { echo "FNR_STOP link guard: $(cat "logs/$name.STOP")"; exit 1; }
  return 0
}
run fnr-k3-q4t "$K3 GRIMOIRE_SPEC_TIME=1 GRIMOIRE_Q4_TIMING=1 GRIMOIRE_TP_SMALL_AR_TIME=1" pleready,gen
grep -hE "spec time per step|qwen4_exp prefill M=" logs/fnr-k3-q4t.server.log | tail -8 | cut -c1-230
grep -h "TP small all-reduce (BAR" logs/fnr-k3-q4t.server.log |
  awk 'match($0, /rank [0-9]+, [^ ]+ rows/) {last[substr($0, RSTART, RLENGTH)]=$0} END{for(k in last) print last[k]}' |
  sort | cut -c1-260
run fnr-plain-h8 "GRIMOIRE_EXPERT_HEAT=8" pleready,gen,benchy --pp 2048 --tg 128 --concurrency 1 --runs 2
grep -hE "expert heat engine|CPU experts in decode \(768" logs/fnr-plain-h8.server.log | tail -4 | cut -c1-200
run fnr-k3-h8 "$K3 GRIMOIRE_EXPERT_HEAT=8" pleready,gen,benchy --pp 2048 --tg 128 --concurrency 1 --runs 2
grep -hE "expert heat engine: [0-9]+ steps|spec: MTP" logs/fnr-k3-h8.server.log | tail -4 | cut -c1-200
echo "FNR_END $(date +%T)"
