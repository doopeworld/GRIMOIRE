#!/bin/bash
# v1.10.5-rc1 = the v1.10.4 image with the two executables AND libgrimoire_gemm.so rebuilt from mxfp4-smallm-1105 a3f81a1
# (run_release_build_v1105b.sh, grimoire-release-1105b, G31+G21, CPU only). libgrimoire_attn.so, oneDNN and the runtime
# stay v1.10.4's (= v1.10.3's). Run only after RELEASE_BUILD_PASS and the GPU checks.
set -euo pipefail
R=/mnt/storage/isos/grimoire-release-1105b
B=/mnt/storage/isos/grimoire-runs/bench-1022
C=$B/release-1105-context
BASE=sha256:e0101d360a04f397cd5b1731321b6cd28e9e5c48cae6d3f3e95caf5cf9fbe3ba
SRC=a3f81a1d98365127081f22c0dd47f20c46e3d09c
[ "$(docker image inspect grimoire-b70:v1.10.4 --format '{{.Id}}')" = "$BASE" ]
grep -qx RELEASE_BUILD_PASS "$B/logs/release-build-1105b.out"
[ "$(git -C "$R" rev-parse HEAD)" = "$SRC" ]
[ -z "$(git -C "$R" status --porcelain)" ]
if docker image inspect grimoire-b70:v1.10.5-rc1 >/dev/null 2>&1; then echo "v1.10.5-rc1 already exists"; exit 1; fi
rm -rf "$C"
mkdir -p "$C/bin" "$C/base"
cp "$R/bin/grimoire" "$R/bin/grimoire-server" "$R/bin/libgrimoire_gemm.so" "$C/bin/"
# Inspect a stopped container; never start its server or expose a GPU.
docker create --name grimoire-runtime-inspect-1105 grimoire-b70:v1.10.4 >/dev/null
trap 'docker rm grimoire-runtime-inspect-1105 >/dev/null' EXIT
docker cp "grimoire-runtime-inspect-1105:/grimoire/bin/libgrimoire_attn.so" "$C/base/libgrimoire_attn.so"
cmp "$C/base/libgrimoire_attn.so" "$R/bin/libgrimoire_attn.so"
# the rebuilt gemm library must hold both targets (G31 + G21): about the size of v1.10.4's
docker cp "grimoire-runtime-inspect-1105:/grimoire/bin/libgrimoire_gemm.so" "$C/base/libgrimoire_gemm.so.v1104"
python3 - "$C/base/libgrimoire_gemm.so.v1104" "$R/bin/libgrimoire_gemm.so" <<'PY_GEMM'
import os, sys
a, b = (os.path.getsize(p) for p in sys.argv[1:3])
print(f"libgrimoire_gemm.so v1.10.4 {a} bytes, rebuilt {b} bytes")
assert 0.9 * a < b < 1.1 * a, "rebuilt gemm library size far from v1.10.4's (targets?)"
PY_GEMM
rm "$C/base/libgrimoire_gemm.so.v1104"
docker cp "grimoire-runtime-inspect-1105:/grimoire/bin/libdnnl.so.3" "$C/base/libdnnl.so.3"
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
   'runtime_base':output('docker','image','inspect','grimoire-b70:v1.10.4','--format','{{.Id}}'),
   'rebuilt':{p.name:sha(p) for p in sorted((c/'bin').iterdir())},
   'unchanged_from_base':{p.name:sha(p) for p in sorted((c/'base').iterdir())},
   'cpu_build':{'cpuset':'6-13','gpu_devices':[],'privileged':False},
   'default_changes':['tiered experts, one card: the experts kept in system RAM are computed by the CPU pool during '
                      'decode while the GPU runs the VRAM ones (GRIMOIRE_CPU_SPLIT_DECODE=0 restores reading them over PCIe)',
                      'small-M split-K plans from the 11-05 sweeps: INT4 (MTP verify / batched decode, <= 16 rows), '
                      'native GGUF (<= 8 rows), MXFP4 o_proj / z shapes'],
   'options_default_off':['GRIMOIRE_ONEDNN_W8A8','GRIMOIRE_MTP_GPU_DRAFT']}
(c/'build-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
MANIFEST
rm -rf "$C/base"
cp "$B/Dockerfile.runtime-v1105" "$C/Dockerfile"
docker build --network none --pull=false -t grimoire-b70:v1.10.5-rc1 "$C"
docker image inspect grimoire-b70:v1.10.5-rc1 --format '{{.Id}}'
echo IMAGE_ASSEMBLY_PASS
