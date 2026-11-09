#!/bin/bash
# Inside a CPU-only builder. Keep validated release splitting/precision flags.
set -eo pipefail
export OCL_ICD_FILENAMES="${OCL_ICD_FILENAMES:-}"
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
cd /work
test ! -e /dev/dri
mkdir -p bin
cp -a /release/bin/libgrimoire_gemm.so /release/bin/libgrimoire_attn.so /release/bin/libdnnl.so* bin/
echo "CANDIDATE_BUILD_START $(date -Iseconds) source=$GRIM_SOURCE_COMMIT targets=G31,G21 no_gpu_devices"
icpx -O2 -std=c++17 -pthread -I include tools/test_imatrix.cpp src/quantize.cpp -o bin/test_imatrix
./bin/test_imatrix
icpx -O2 -std=c++17 -I include tools/test_int4_rtn.cpp src/quantize.cpp -o bin/test_int4_rtn
./bin/test_int4_rtn plain | tee /tmp/rtn-plain.txt
GRIMOIRE_INT4_RTN_GROUP=64 ./bin/test_int4_rtn g64 | tee /tmp/rtn-g64.txt
GRIMOIRE_INT4_RTN_GROUP=64 GRIMOIRE_INT4_RTN_SEARCH=1 ./bin/test_int4_rtn g64-search | tee /tmp/rtn-g64-search.txt
python3 - <<'PY'
import re
from pathlib import Path
v=[float(re.search(r'MSE ([0-9.]+)',Path('/tmp/rtn-'+x+'.txt').read_text())[1]) for x in ['plain','g64','g64-search']]
assert v[1]<.9*v[0] and v[2]<v[1],v
print('RTN_COMPARISON_PASS',v)
PY
icpx -O2 -std=c++17 -I include tools/test_ipc_socket.cpp -o bin/test_ipc_socket
./bin/test_ipc_socket
icpx -O2 -std=c++17 -I include tools/test_host_history_budget.cpp -o bin/test_host_history_budget
./bin/test_host_history_budget
icpx -fsycl -fsycl-targets=spir64 -O2 -std=c++20 -I include tools/test_peer_copy.cpp -ldl -o bin/test_peer_copy
icpx -fsycl -fsycl-targets=spir64 -O2 -std=c++20 -I include tools/test_peer_setup.cpp -ldl -o bin/test_peer_setup
SOURCES=(src/grimoire.cpp src/qwen35_loader.cpp src/native_model.cpp src/safetensors.cpp src/quantize.cpp src/gptq.cpp src/gemv_decode.cpp src/gguf.cpp src/gguf_decode.cpp src/gguf_qwen35.cpp src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp src/moe_kernels.cpp src/tiered_moe.cpp src/cpu_experts.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp src/tokenizer.cpp)
FLAGS=(-fsycl -fsycl-targets=intel_gpu_bmg_g31,intel_gpu_bmg_g21 -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno -Wmisleading-indentation -I include -I src)
icpx "${FLAGS[@]}" -fsycl-device-code-split=off tools/grimoire_server.cpp "${SOURCES[@]}" \
 -lpthread -ldl -Lbin -lgrimoire_gemm -lgrimoire_attn -ldnnl '-Wl,-rpath,$ORIGIN' -o bin/grimoire-server.new &
server_pid=$!
icpx "${FLAGS[@]}" -fsycl-device-code-split=per_kernel tools/grimoire_main.cpp "${SOURCES[@]}" \
 -lpthread -ldl -Lbin -lgrimoire_gemm -lgrimoire_attn -ldnnl '-Wl,-rpath,$ORIGIN' -o bin/grimoire.new &
cli_pid=$!
set +e
wait "$server_pid"; server_rc=$?
wait "$cli_pid"; cli_rc=$?
set -e
echo "CANDIDATE_LINK_RESULTS server=$server_rc cli=$cli_rc $(date -Iseconds)"
if [ "$server_rc" != 0 ] || [ "$cli_rc" != 0 ];then exit 1;fi
mv bin/grimoire-server.new bin/grimoire-server
mv bin/grimoire.new bin/grimoire
strip --strip-unneeded bin/grimoire bin/grimoire-server
sha256sum bin/grimoire bin/grimoire-server bin/libgrimoire_gemm.so bin/libgrimoire_attn.so bin/libdnnl.so.3
printf '%s\n' "$GRIM_SOURCE_COMMIT" > SOURCE_REVISION
echo "CANDIDATE_BUILD_PASS $(date -Iseconds)"
