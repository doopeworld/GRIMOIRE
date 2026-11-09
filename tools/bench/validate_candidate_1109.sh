#!/bin/bash
set -euo pipefail
B=/mnt/storage/isos/grimoire-runs/bench-1022
X=/mnt/storage/isos/grimoire-release-1109
PD=/mnt/storage/Models/grimoire-ple
A=/mnt/storage/Models/grimoire-acc
cd "$B"
exec >>logs/candidate-validation1109.out 2>&1
exec 6>logs/candidate-validation1109.lock
flock -n 6 || exit 9
grep -qx CANDIDATE_IMAGE_ASSEMBLY_PASS logs/candidate-assembly1109.out
export IMG=grimoire-b70:v1.10.5-rc3 GRIM_IMAGE=grimoire-b70:v1.10.5-rc3 HOSTBIN=0 GRIM_TREE=$X LOAD_MAX=1800 CLIENT_CPUS=14,15
EXPECTED_IMAGE=$(docker image inspect "$IMG" --format '{{.Id}}')
python3 ./verify_candidate_image1109.py
echo "CANDIDATE_VALIDATION_START $(date -Iseconds) image=$EXPECTED_IMAGE source=$(cat "$X/SOURCE_REVISION") own_binaries=1"
run() {
 local name=$1; shift
 if [ -f "logs/$name.VERIFIED" ];then echo "SKIP verified $name";return;fi
 test ! -e "logs/$name.run"
 test "$(docker image inspect "$IMG" --format '{{.Id}}')" = "$EXPECTED_IMAGE"
 cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.candidate1109"
 echo "CANDIDATE_CASE_START $name $(date -Iseconds)"
 bash ./dual_candidate_1109.sh "$name" "$@" > "logs/$name.run" 2>&1
 python3 ./verify_candidate_1109.py "$B/logs/$name"
 touch "logs/$name.VERIFIED"
 tail -10 "logs/$name.run"
}
# The current user profile: preserve checkpoint, MTP and 131K context.
export CODE_LENGTHS=4000 CODE_REPEATS=1
run v1109-orca-mtp TP /models2/Qwen3.8-27B-Uncensored-FP8 int8 131072 \
 'GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=1 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536 GRIMOIRE_P2P=0' \
 gen,api1109,code,benchy --pp 4096 --tg 128 --concurrency 1 --runs 2
# Prove host-budget clamping, actual TP fallback, and a real long prompt in the packaged image.
export CODE_LENGTHS=4000,128000 CODE_REPEATS=0
run v1109-q27-host-auto TP /models2/Qwen3.8-27B-FP8 int8 131072 \
 'GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2 GRIMOIRE_DFLASH_TAPS_HOST=1 GRIMOIRE_P2P=auto GRIMOIRE_SPEC_STATS=1' \
 gen,code,benchy --pp 4096 --tg 128 --concurrency 1 --runs 2
F=/models2/Qwen3.8-Flash-Next-NVFP4
E='GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=1 GRIMOIRE_MTP=0 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.candidate1109'
run v1109-fn-tp-auto TP "$F" int8 65536 "$E GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 GRIMOIRE_P2P=auto" \
 pleready,gen,benchy --pp 512 2048 --tg 128 --concurrency 1 --runs 2
run v1109-fn-single G0 "$F" int4 65536 "$E GRIMOIRE_EXPERT_VRAM_PER_LAYER=112" \
 pleready,gen,benchy --pp 512 --tg 128 --concurrency 1 --runs 2
run v1109-fn-imx G0 "$F" int4 65536 "$E GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 GRIMOIRE_IMATRIX=/models/grimoire-acc/imatrix-flash-next.gguf" \
 pleready,gen,benchy --pp 512 --tg 128 --concurrency 1 --runs 2
# Combined-loader/new-library integration check: do not repeat the 11-case accuracy sweep.
name=v1109-cli-fn-g64
if [ ! -f "logs/$name.VERIFIED" ];then
 test ! -e "$A/$name.bin"
 cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.candidate1109"
 EXTRA_ENV="GRIMOIRE_POWER_LIMIT_W=0
GRIMOIRE_MTP=0
GRIMOIRE_EXPERT_HEAT=0
GRIMOIRE_EXPERT_FILL=0
GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin
GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.candidate1109
GRIMOIRE_EXPERT_VRAM_PER_LAYER=90
GRIMOIRE_IMATRIX=/models/grimoire-acc/imatrix-flash-next.gguf
GRIMOIRE_INT4_RTN_GROUP=64" LIM=1500 TAILN=8 \
 flock -n /mnt/storage/isos/grimoire-runs/gpu0.lock bash /mnt/storage/isos/grimoire-fuse/tools/g0run.sh "$name" /grimoire/bin/grimoire \
 -m "$F" --proj int4 --ctx 1024 --score-ids /models/grimoire-acc/fn512.ids --score-out "/models/grimoire-acc/$name.bin"
 cp "/tmp/grim-$name.log" "logs/$name.log"
 uv run --offline --no-project --with numpy python "$B/compare_candidate_logits_1109.py" \
 "$A/fn-bf16.bin" "$A/$name.bin" "$B/logs/$name.metrics.json"
 touch "logs/$name.VERIFIED"
fi
# The native transport tests intentionally allow SAME-device IPC; not a bandwidth claim.
for tool in test_peer_setup test_peer_copy;do
 name=v1109-$tool
 if [ -f "logs/$name.VERIFIED" ];then continue;fi
 HOSTBIN=1 LIM=300 TAILN=20 flock -n /mnt/storage/isos/grimoire-runs/gpu0.lock \
 bash /mnt/storage/isos/grimoire-fuse/tools/g0run.sh "$name" "/grimoire/bin/$tool"
 cp "/tmp/grim-$name.log" "logs/$name.log"
 grep -Eq 'PEER_SETUP_PASS|PEER_COPY_SAME_GPU_PASS' "logs/$name.log"
 touch "logs/$name.VERIFIED"
done
# Strict mode must fail clearly before any peer payload on this unsupported topology.
name=v1109-peer-required
if [ ! -f "logs/$name.VERIFIED" ];then
 test ! -e "logs/$name.run"
 set +e
 bash ./dual_candidate_1109.sh "$name" TP /models/grimoire-acc/grimoire-peer-fixture1109 int8 1024 \
 'GRIMOIRE_SEQ_SLOTS=1 GRIMOIRE_MTP=0 GRIMOIRE_P2P=required GRIMOIRE_POWER_LIMIT_W=0' gen > "logs/$name.run" 2>&1
 rc=$?
 set -e
 test "$rc" = 8
 docker logs "$name" > "logs/$name.server.log" 2>&1
 python3 - "$name" <<'PY'
import json,subprocess,sys
from pathlib import Path
name=sys.argv[1];p=Path('logs')/name
state=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]['State']
s=p.with_suffix('.server.log').read_text();r=p.with_suffix('.run').read_text()
assert not state['Running'] and state['ExitCode']==1,state
assert s.count('TP P2P: peer IPC import unsupported by driver/topology')==2,s
assert not p.with_suffix('.STOP').exists() and 'uncorrectable(all ports)=0' in r
assert 'Segmentation fault' not in s
print('P2P_REQUIRED_TOPOLOGY_REFUSAL_PASS')
PY
 docker rm "$name"
 touch "logs/$name.VERIFIED"
fi
cmp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.frozen-1028.backup-1109"
echo CANDIDATE_SCOPE_PASS
