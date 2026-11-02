#!/bin/bash
# Run only after the CPU-only G31/G21 build has completed successfully.
set -euo pipefail
R=/mnt/storage/isos/grimoire-release-1101
B=/mnt/storage/isos/grimoire-runs/bench-1022
C=$B/release-1101-context
BASE=sha256:b14de3b48ba009c1ad6eb2cdf9af6684438825f172a34e1fb76553e037769bbd
[ "$(docker image inspect grimoire-b70:v1.10.2 --format '{{.Id}}')" = "$BASE" ]
grep -qx RELEASE_BUILD_PASS "$B/logs/release-build-1101.out"
[ -z "$(git -C "$R" status --porcelain)" ]
mkdir -p "$C/bin"
# Copy immutable output into the Docker context before assembly.
for f in grimoire grimoire-server libgrimoire_gemm.so libgrimoire_attn.so; do
  cp "$R/bin/$f" "$C/bin/$f"
done
# Inspect a stopped container; never start its server or expose a GPU.
docker create --name grimoire-runtime-inspect-1101 grimoire-b70:v1.10.2 >/dev/null
trap 'docker rm grimoire-runtime-inspect-1101 >/dev/null' EXIT
docker cp grimoire-runtime-inspect-1101:/grimoire/bin/libdnnl.so.3 "$C/libdnnl-base.so.3"
docker cp grimoire-runtime-inspect-1101:/usr/local/bin/docker-entrypoint.sh "$C/entrypoint-base.sh"
# The image compiled oneDNN on 10-30 and this tree holds the 10-28 compile of the same version. The only allowed
# difference is the assembler temp-file name left in the symbol table (ittptmark64-XXXXXX.o).
python3 - "$C/libdnnl-base.so.3" "$R/bin/libdnnl.so.3" <<PY_DNNL
import re,sys
a,b=(open(p,"rb").read() for p in sys.argv[1:3])
pat=re.compile(rb"ittptmark64-[0-9a-f]{6}\\.o")
assert len(pat.findall(a))==1 and len(pat.findall(b))==1
assert pat.sub(b"X",a)==pat.sub(b"X",b),"oneDNN libraries differ beyond the temp-file name"
PY_DNNL
cmp "$C/entrypoint-base.sh" "$R/docker-entrypoint.sh"
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
   'runtime_base':output('docker','image','inspect','grimoire-b70:v1.10.2','--format','{{.Id}}'),
   'build_scripts':{p:sha(r/p) for p in ['tools/build_grimoire_only_b70.sh','tools/build_server_only_b70.sh']},
   'artifacts':{p.name:sha(p) for p in sorted((c/'bin').iterdir())},
   'onednn_sha256':sha(c/'libdnnl-base.so.3'),
   'entrypoint_sha256':sha(c/'entrypoint-base.sh'),
   'cpu_build':{'cpuset':'6-13','gpu_devices':[],'privileged':False},
   'new_options_default_off':['GRIMOIRE_ONEDNN_W8A8','GRIMOIRE_MTP_GPU_DRAFT','GRIMOIRE_MOE_NATIVE_SMALLM']}
(c/'build-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
MANIFEST
cp "$B/Dockerfile.runtime-1101" "$C/Dockerfile"
docker build --network none --pull=false -t grimoire-b70:v1.10.3-rc1 "$C"
docker image inspect grimoire-b70:v1.10.3-rc1 --format '{{.Id}}'
echo IMAGE_ASSEMBLY_PASS
