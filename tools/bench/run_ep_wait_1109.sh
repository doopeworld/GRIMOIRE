#!/bin/bash
# Finish the previously prepared expert-parallel wait diagnostic. Rates here
# include extra synchronization and must not be reported as performance.
set -euo pipefail
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B"
exec >>logs/ep-wait1109.out 2>&1
exec 6>logs/ep-wait1109.lock
flock -n 6 || exit 9
grep -qx CANDIDATE_SCOPE_PASS logs/candidate-validation1109.out
export IMG=grimoire-b70:v1.10.5-rc3 HOSTBIN=0 GRIM_TREE=/mnt/storage/isos/grimoire-release-1109 LOAD_MAX=1800 CLIENT_CPUS=14,15
name=v1109-fn-epwait
if [ -f "logs/$name.VERIFIED" ];then echo 'SKIP verified expert-wait diagnostic';exit;fi
test ! -e "logs/$name.run"
cp /mnt/storage/Models/grimoire-ple/flash-next.hits.frozen-1028 /mnt/storage/Models/grimoire-ple/flash-next.hits.epwait1109
bash ./dual_candidate_1109.sh "$name" TP /models2/Qwen3.8-Flash-Next-NVFP4 int8 65536 \
 'GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=1 GRIMOIRE_MTP=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 GRIMOIRE_EXPERT_HEAT=0 GRIMOIRE_P2P=0 GRIMOIRE_CX_STATS=1 GRIMOIRE_TP_SMALL_AR_TIME=1 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.epwait1109' \
 pleready,gen >"logs/$name.run" 2>&1
python3 - "$name" <<'PY'
import json,re,sys
from pathlib import Path
p=Path('logs')/sys.argv[1];s=p.with_suffix('.server.log').read_text();r=p.with_suffix('.run').read_text()
assert 'ready=1' in r and 'server stopped gracefully' in r and not p.with_suffix('.STOP').exists()
assert 'uncorrectable(all ports)=0' in r
assert not re.search(r'\b(?:FAIL|FAILED|STOP|Traceback)\b',r)
last={}
for line in s.splitlines():
 m=re.search(r'TP small all-reduce.*rank ([0-9]+), ([^ ]+) rows',line)
 if m:last[m.groups()]=line
assert {k[0] for k in last}=={'0','1'},last
for k in sorted(last):print(last[k])
a=json.loads(p.with_suffix('.gen.json').read_text());b=json.loads(Path('logs/v1109-fn-tp-auto.gen.json').read_text())
assert all(a[k]['text']==b[k]['text'] for k in a),'diagnostic changed fixed answers'
print('EXPERT_WAIT_DIAGNOSTIC_PASS (timings are instrumented)')
PY
touch "logs/$name.VERIFIED"
