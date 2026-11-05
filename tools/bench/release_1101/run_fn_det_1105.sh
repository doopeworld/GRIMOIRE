#!/bin/bash
# 11-05: is two-card Flash-Next text deterministic run to run? The combined gate stopped because the candidate with
# fp32 partials differed from v1.10.3 -- and the candidate with bf16 partials differed from both (reasoning preamble).
# Cause found in the source: a served process ADDS its routing counts to GRIMOIRE_EXPERT_HITS (tm_save_hits), and the
# next load places the hot experts in VRAM from that file -- so every run started from a different placement (GPU vs
# CPU arithmetic for the experts that moved). Here every run starts from the SAME hits file (a snapshot restored
# before each run): v1.10.3 twice, the candidate with fp32 partials, the candidate as shipped (bf16 + peer). At the end
# the snapshot is put back (the test runs' counts are dropped). Holds logs/HOLD-det; waits for the probe builds.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
H=/mnt/storage/Models/grimoire-ple/flash-next.hits
S=/mnt/storage/Models/grimoire-ple/flash-next.hits.det-1105
cd "$B" || exit 9
exec >"$B/logs/fn-det-1105.out" 2>&1
touch "$B/logs/HOLD-det"
trap 'rm -f "$B/logs/HOLD-det"' EXIT
until grep -qE "^GGUFBENCH_BUILD_(PASS|FAIL)" logs/build-ggufbench-1105.out 2>/dev/null; do sleep 15; done
until [ -z "$(docker ps -q)" ]; do sleep 10; done
[ -e "$S" ] || cp -p "$H" "$S"
echo "hits snapshot $(sha256sum "$S" | cut -c1-16) $(date +%T)"
run() { # TAG + env for run_dual_1104.sh
  local tag=$1; shift
  cp "$S" "$H"
  env "$@" bash "$B/run_dual_1104.sh" "$tag" fn || echo "DET_STOP $tag"
}
run v1103c  IMG=grimoire-b70:v1.10.3 HOSTBIN=0
run v1103d  IMG=grimoire-b70:v1.10.3 HOSTBIN=0
run v1104f32c DUAL_EXTRA=GRIMOIRE_TP_SMALL_BF16=0 GRIM_TREE=/mnt/storage/isos/grimoire-release-1105 HOSTBIN=1 IMG=grimoire-b70:v1.10.3
run v1104c  GRIM_TREE=/mnt/storage/isos/grimoire-release-1105 HOSTBIN=1 IMG=grimoire-b70:v1.10.3
cp "$S" "$H"
echo "hits file restored to the snapshot $(date +%T)"
python3 - "$B/logs" <<'PY'
import json, sys
d = sys.argv[1]
runs = ["v1103c", "v1103d", "v1104f32c", "v1104c"]
g = {r: json.load(open(f"{d}/d-{r}-fn-tp.gen.json")) for r in runs}
for k in g["v1103c"]:
    groups = {}
    for r in runs:
        groups.setdefault(g[r][k]["text"], []).append(r)
    print(f"{k}: {len(groups)} distinct texts: " + " | ".join(",".join(v) for v in groups.values()))
PY
echo "FN_DET_DONE $(date +%T)"
