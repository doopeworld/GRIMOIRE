#!/bin/bash
# Overlay only changed executables onto the exact validated rc2 runtime.
set -euo pipefail
REV=${1:?source revision}
B=/mnt/storage/isos/grimoire-runs/bench-1022
X=/mnt/storage/isos/grimoire-release-1109
TAG=grimoire-b70:v1.10.5-rc3
BASE=grimoire-b70:v1.10.5-rc2
BASE_ID=sha256:717960860fad7cd194ef9a7bf6d35787059f7c7dc084e8b04348c9290b112210
exec >"$B/logs/candidate-assembly1109.out" 2>&1
grep -q '^CANDIDATE_BUILD_PASS ' "$B/logs/candidate-build1109.out"
test "$(cat "$X/SOURCE_REVISION")" = "$REV"
test "$(docker image inspect "$BASE" --format '{{.Id}}')" = "$BASE_ID"
if docker image inspect "$TAG" >/dev/null 2>&1;then echo "$TAG already exists; inspect before rebuilding";exit 1;fi
C="$B/release1109-context-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$C/bin" "$C/base"
cp "$X/bin/grimoire" "$X/bin/grimoire-server" "$C/bin/"
INSPECT=grimoire-runtime-inspect1109
docker create --name "$INSPECT" "$BASE" >/dev/null
trap 'docker rm "$INSPECT" >/dev/null' EXIT
for lib in libgrimoire_gemm.so libgrimoire_attn.so libdnnl.so.3;do
 docker cp "$INSPECT:/grimoire/bin/$lib" "$C/base/$lib"
done
cmp "$C/base/libgrimoire_gemm.so" "$X/bin/libgrimoire_gemm.so"
cmp "$C/base/libgrimoire_attn.so" "$X/bin/libgrimoire_attn.so"
python3 - "$C" "$X" "$REV" "$BASE_ID" <<'PY'
from pathlib import Path
import hashlib,json,re,subprocess,sys
c,x=map(Path,sys.argv[1:3]);rev,base=sys.argv[3:5]
a=(c/'base/libdnnl.so.3').read_bytes();b=(x/'bin/libdnnl.so.3').read_bytes()
p=re.compile(rb'ittptmark64-[0-9a-f]{6}\.o')
assert a==b or p.sub(b'X',a)==p.sub(b'X',b),'oneDNN differs beyond the known temporary object name'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
m={
 'source_commit':rev,'targets':['intel_gpu_bmg_g31','intel_gpu_bmg_g21'],
 'builder_image':subprocess.check_output(['docker','image','inspect','grimoire-dev:latest','--format','{{.Id}}'],text=True).strip(),
 'runtime_base':base,'rebuilt':{f.name:sha(f) for f in (c/'bin').iterdir()},
 'unchanged_runtime_libraries':{f.name:sha(f) for f in (c/'base').iterdir()},
 'source_files':{str(f.relative_to(x)):sha(f) for d in ['src','include'] for f in sorted((x/d).rglob('*')) if f.is_file()},
 'cpu_build':{'gpu_devices':[],'privileged':False,'network':'none'},
 'precision_flags':['-O3','-fno-fast-math','-ffp-contract=fast','-fno-math-errno'],
 'device_code_split':{'server':'off','cli':'per_kernel'},
 'options_default_off':['GRIMOIRE_P2P','GRIMOIRE_DFLASH_TAPS_HOST','GRIMOIRE_IMATRIX','GRIMOIRE_IMATRIX_FOLD','GRIMOIRE_INT4_RTN_SEARCH'],
 'int4_default_group':128,'validation_scope':['Qwen3.8-27B','Qwen3.8-Flash-Next'],
 'experimental_features':['Level Zero IPC peer copy transport','DFlash pinned-host FP32 history with bounded GPU staging']}
(c/'build-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
PY
cat > "$C/Dockerfile" <<EOF
FROM $BASE
LABEL org.opencontainers.image.source="https://github.com/doopeworld/GRIMOIRE" \\
      org.opencontainers.image.revision="$REV" \\
      org.opencontainers.image.version="v1.10.5-rc3" \\
      com.grimoire.validation-scope="Qwen3.8-27B,Qwen3.8-Flash-Next"
COPY --chmod=0755 bin/grimoire bin/grimoire-server /grimoire/bin/
COPY build-manifest.json /usr/share/doc/grimoire/build-manifest.json
EOF
docker build --network none --pull=false -t "$TAG" "$C"
docker image inspect "$TAG" --format '{{.Id}}'
printf '%s\n' "$C" > "$B/logs/candidate-context1109.path"
echo CANDIDATE_IMAGE_ASSEMBLY_PASS
