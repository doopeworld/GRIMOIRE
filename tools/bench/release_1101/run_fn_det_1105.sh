#!/bin/bash
# 11-05: is two-card Flash-Next text deterministic run to run? The combined gate stopped because the candidate with
# fp32 partials differed from v1.10.3 -- and the candidate with bf16 partials differed from both (in the reasoning
# preamble). Same settings as run_dual_1104.sh fn: v1.10.3 again, then the candidate with fp32 partials again.
# Holds logs/HOLD-det (the B580 queue waits); waits for the CPU-only probe builds to finish first.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
exec >"$B/logs/fn-det-1105.out" 2>&1
touch "$B/logs/HOLD-det"
trap 'rm -f "$B/logs/HOLD-det"' EXIT
until grep -qE "^GGUFBENCH_BUILD_(PASS|FAIL)" logs/build-ggufbench-1105.out 2>/dev/null; do sleep 15; done
until [ -z "$(docker ps -q)" ]; do sleep 10; done
IMG=grimoire-b70:v1.10.3 HOSTBIN=0 bash "$B/run_dual_1104.sh" v1103b fn || echo "DET_STOP v1103b"
DUAL_EXTRA=GRIMOIRE_TP_SMALL_BF16=0 GRIM_TREE=/mnt/storage/isos/grimoire-release-1105 HOSTBIN=1 IMG=grimoire-b70:v1.10.3 \
  bash "$B/run_dual_1104.sh" v1104f32b fn || echo "DET_STOP v1104f32b"
python3 - "$B/logs" <<'PY'
import json, sys
d = sys.argv[1]
runs = ["v1103", "v1103b", "v1104", "v1104f32", "v1104f32b"]
g = {r: json.load(open(f"{d}/d-{r}-fn-tp.gen.json")) for r in runs}
for k in g["v1103"]:
    groups = {}
    for r in runs:
        groups.setdefault(g[r][k]["text"], []).append(r)
    print(f"{k}: {len(groups)} distinct texts: " + " | ".join(",".join(v) for v in groups.values()))
PY
echo "FN_DET_DONE $(date +%T)"
