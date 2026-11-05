#!/bin/bash
# 11-05 18:30: where a Flash-Next MTP round goes (fnmtp-1105, one card, gpu0): K=3 decoded 5.3 tok/s against 25.4
# plain (fnmtp-test-1105.out). GRIMOIRE_SPEC_TIME=1 (snapshot / draft / verify / commit per step, fenced) and
# GRIMOIRE_Q4_TIMING=1 (the Qwen4-Exp prefill -- the verify -- per stage kind, waits between stages), the three
# fixed answers only. Same placement as the test. Holds logs/HOLD-fnmtp2.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/fnmtp-1105
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/fnmtp-test-1105.out" 2>&1
echo "FNMTP_DIAG_START $(date +%T)"
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-fnmtp2
trap 'rm -f "$B/logs/HOLD-fnmtp2"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=90 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1"
cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
name=fnmtp1105-k3time
echo "== $(date +%T) $name"
bash ./dual_engine_1102.sh "$name" G0 "$F" bf16 8192 "$E GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 GRIMOIRE_SPEC_TIME=1 GRIMOIRE_Q4_TIMING=1" \
  pleready,gen > "logs/$name.run" 2>&1
echo "rc=$?"
grep -E "\[gen\]|STOP|LEFT RUNNING|server stopped" "logs/$name.run" | cut -c1-170
grep -iE "spec time|spec:|q4|qsa|hc timing|CPU experts|cx |prefill" "logs/$name.server.log" | tail -30 | cut -c1-230
cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
echo "FNMTP_DIAG_END $(date +%T)"
