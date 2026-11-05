#!/bin/bash
# v1.10.4-rc1, combined candidate (11-05) = the v1.10.3 image with the two executables AND libgrimoire_gemm.so rebuilt
# from mxfp4-smallm-1105 315167c (executables built at dd33cbe; the library after the 9-16-row plan rule, fold_lib_1105.sh). libgrimoire_attn.so, oneDNN and the runtime stay v1.10.3's.
# Run only after RELEASE_BUILD_PASS and the GPU checks.
set -euo pipefail
R=/mnt/storage/isos/grimoire-release-1105
B=/mnt/storage/isos/grimoire-runs/bench-1022
C=$B/release-1104b-context
BASE=sha256:3663f7a7c5cc745b7a16ee5de8b57ba7da37f8bace87cc4ad238af3f9b761e45
SRC=315167c9f8e8512ef2dbacc123ca04adfa9d7918
[ "$(docker image inspect grimoire-b70:v1.10.3 --format '{{.Id}}')" = "$BASE" ]
grep -qx RELEASE_BUILD_PASS "$B/logs/release-build-1105.out"
grep -qx FOLD_PASS "$B/logs/fold-lib-1105.out" 2>/dev/null || grep -q "^FOLD_PASS" "$B/logs/fold-lib-1105.out"
[ "$(git -C "$R" rev-parse HEAD)" = "$SRC" ]
[ -z "$(git -C "$R" status --porcelain)" ]
if docker image inspect grimoire-b70:v1.10.4-rc1 >/dev/null 2>&1; then echo "v1.10.4-rc1 already exists"; exit 1; fi
rm -rf "$C"
mkdir -p "$C/bin" "$C/base"
cp "$R/bin/grimoire" "$R/bin/grimoire-server" "$R/bin/libgrimoire_gemm.so" "$C/bin/"
# Inspect a stopped container; never start its server or expose a GPU.
docker create --name grimoire-runtime-inspect-1104b grimoire-b70:v1.10.3 >/dev/null
trap 'docker rm grimoire-runtime-inspect-1104b >/dev/null' EXIT
docker cp "grimoire-runtime-inspect-1104b:/grimoire/bin/libgrimoire_attn.so" "$C/base/libgrimoire_attn.so"
cmp "$C/base/libgrimoire_attn.so" "$R/bin/libgrimoire_attn.so"
# the rebuilt gemm library must hold both targets (G31 + G21): about twice the G31-only test build
docker cp "grimoire-runtime-inspect-1104b:/grimoire/bin/libgrimoire_gemm.so" "$C/base/libgrimoire_gemm.so.v1103"
python3 - "$C/base/libgrimoire_gemm.so.v1103" "$R/bin/libgrimoire_gemm.so" <<'PY_GEMM'
import os, sys
a, b = (os.path.getsize(p) for p in sys.argv[1:3])
print(f"libgrimoire_gemm.so v1.10.3 {a} bytes, rebuilt {b} bytes")
assert 0.8 * a < b < 1.25 * a, "rebuilt gemm library size far from v1.10.3's (targets?)"
PY_GEMM
rm "$C/base/libgrimoire_gemm.so.v1103"
# The image's oneDNN is the 10-30 compile, the tree's the 10-28 compile of the same version: they may differ only in
# the assembler temp-file name left in the symbol table (as checked for v1.10.3).
docker cp "grimoire-runtime-inspect-1104b:/grimoire/bin/libdnnl.so.3" "$C/base/libdnnl.so.3"
python3 - "$C/base/libdnnl.so.3" "$R/bin/libdnnl.so.3" <<'PY_DNNL'
import re,sys
a,b=(open(p,"rb").read() for p in sys.argv[1:3])
pat=re.compile(rb"ittptmark64-[0-9a-f]{6}\.o")
assert len(pat.findall(a))==1 and len(pat.findall(b))==1
assert pat.sub(b"X",a)==pat.sub(b"X",b),"oneDNN libraries differ beyond the temp-file name"
PY_DNNL
export GRIM_RELEASE_CONTEXT="$C" GRIM_RELEASE_TREE="$R"
python3 - <<'MANIFEST'
import hashlib,json,os,subprocess
from pathlib import Path
r=Path(os.environ['GRIM_RELEASE_TREE']); c=Path(os.environ['GRIM_RELEASE_CONTEXT'])
def output(*args):return subprocess.check_output(args,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m={'source_commit':output('git','-C',str(r),'rev-parse','HEAD'),
   'targets':['intel_gpu_bmg_g31','intel_gpu_bmg_g21'],
   'builder_image':output('docker','image','inspect','grimoire-dev:latest','--format','{{.Id}}'),
   'runtime_base':output('docker','image','inspect','grimoire-b70:v1.10.3','--format','{{.Id}}'),
   'rebuilt':{p.name:sha(p) for p in sorted((c/'bin').iterdir())},
   'unchanged_from_base':{p.name:sha(p) for p in sorted((c/'base').iterdir())},
   'cpu_build':{'cpuset':'6-13','gpu_devices':[],'privileged':False},
   'default_changes':['native GGUF experts grouped for 2-16 rows (GRIMOIRE_MOE_NATIVE_SMALLM=0 restores the old path)',
                      'CPU-expert pool no longer waits for workers that did not start a job',
                      'one card: the decode thread is not pinned (GRIMOIRE_CPU_PIN_CALLER=1 restores the pin)',
                      'two cards: bf16 partials in the decode / verify all-reduce (GRIMOIRE_TP_SMALL_BF16=0 restores fp32)',
                      'two cards: the peer partial added on the card (GRIMOIRE_TP_SMALL_PEER=0 restores the CPU sum)',
                      'two cards: 4-bit FFN weights on the prompt wavefront (GRIMOIRE_TP_WAVE_4BIT=0 restores the per-layer path)',
                      'MXFP4 small-M (MTP verify, batched decode): 128-register kernel for <= 8 rows, nibble-pair dequant '
                      '(GRIMOIRE_MX_SMALLM_256=1 restores the library kernel for <= 8 rows)',
                      'tiered experts: default VRAM budget at most 60% of the card (a B70 keeps 16 GB)'],
   'options_default_off':['GRIMOIRE_ONEDNN_W8A8','GRIMOIRE_MTP_GPU_DRAFT']}
(c/'build-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
MANIFEST
rm -rf "$C/base"
cp "$B/Dockerfile.runtime-v1104b" "$C/Dockerfile"
docker build --network none --pull=false -t grimoire-b70:v1.10.4-rc1 "$C"
docker image inspect grimoire-b70:v1.10.4-rc1 --format '{{.Id}}'
echo IMAGE_ASSEMBLY_PASS
