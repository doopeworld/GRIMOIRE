#!/bin/bash
# 11-06: Radiance's Flash-Next serving shape on the B70s -- two cards (Megatron TP + expert-parallel tiered experts,
# the DUAL-FLASH-NEXT profile's settings, ctx 65536, 170 experts a layer of the rank's 256 in VRAM), the dense part
# INT8, and the model's own MTP head drafting 3 (Radiance: TP2 + MTP 3, c1 177 tok/s, ~15 ms a step on 2x R9700).
# fnmtp-verify-1106's build (the verify on decode's small-M kernels); the MTP head loads whole on every rank.
#   plain    no MTP (11-05 int8-170: tg128 53.94, pp2048 1,349)              gen + llama-benchy pp2048 / tg128 x2
#   k3       GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3                                  gen + benchy; answers vs plain
#   k3-time  + GRIMOIRE_SPEC_TIME=1 (fenced phases: draft / verify / commit)  gen
#   k2       GRIMOIRE_MTP_K=2                                                 benchy
# gpu1 runs at Gen3 x4 today (Ian 11-06: "be careful"): dual_engine_1102.sh's link guard stops a run on any
# uncorrectable error and logs gpu1's corrected ones every 10 s (logs/NAME.link); the link state before / after each
# run is printed. Frozen hits before every run. Holds logs/HOLD-fnmtpdual. Log: logs/fnmtp-dual-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/fnmtp-verify-1106
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/fnmtp-dual-1106.out" 2>&1
echo "FNMTP_DUAL_START $(date +%T) tree=$T ($(git -C "$T" rev-parse --short HEAD 2>/dev/null))"
[ -x "$T/bin/grimoire-server" ] || { echo "FNMTP_DUAL_STOP no server in $T"; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-fnmtpdual
trap 'rm -f "$B/logs/HOLD-fnmtpdual"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
K3="GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 GRIMOIRE_SPEC_STATS=1"
run() { # name env tests [benchy args]
  local name=$1 env=$2; shift 2
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name [$env]"
  bash ./dual_engine_1102.sh "$name" TP "$F" int8 65536 "$E $env" "$@" > "logs/$name.run" 2>&1
  local rc=$?
  echo "rc=$rc"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg128|pp2048|link (before|after)|RxErr during" \
    "logs/$name.run" | cut -c1-200
  grep -hE "spec:|accept|verify|draft|CPU experts in decode|error|failed|refus" "logs/$name.server.log" 2>/dev/null |
    grep -v "^ *$" | tail -8 | cut -c1-230
  [ -e "logs/$name.STOP" ] && { echo "FNMTP_DUAL_STOP link guard: $(cat "logs/$name.STOP")"; exit 1; }
  return 0
}
cmp_gen() { # name: its answers against plain
  python3 - "$B/logs" "$1" <<'PY'
import json, sys
d, n = sys.argv[1], sys.argv[2]
try:
    b, a = (json.load(open(f"{d}/{x}.gen.json")) for x in ("fnmd1106-plain", n))
except Exception as e:
    print("  no gen json", e); sys.exit()
for k in b:
    x, y = b[k]["text"], a.get(k, {}).get("text", "")
    i = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
    print("  %s vs plain %-8s %s" % (n, k, "SAME" if x == y else "first difference at character %d of %d" % (i, len(x))))
PY
}
BENCHY="benchy --pp 2048 --tg 128 --concurrency 1 --runs 2"
run fnmd1106-plain "" pleready,gen,$BENCHY
run fnmd1106-k3 "$K3" pleready,gen,$BENCHY; cmp_gen fnmd1106-k3
run fnmd1106-k3-time "$K3 GRIMOIRE_SPEC_TIME=1" pleready,gen; cmp_gen fnmd1106-k3-time
run fnmd1106-k2 "GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=2 GRIMOIRE_SPEC_STATS=1" pleready,$BENCHY
echo "FNMTP_DUAL_END $(date +%T)"
