#!/bin/bash
# =====================================================================
#  preflight_b70.sh -- one command, run on the Tower, that says whether
#  GRIMOIRE is actually working on this box.
#
#  Run this FIRST after a cold boot, a pull, or any kernel change, and
#  before trusting a single number.  It builds in the right order
#  (bridges first -- rule 3), runs every gate that does not need a
#  model, and only then touches a real checkpoint.
#
#  It never `docker run`s the GPU directly (rule 6): the model stages go
#  through tools/tune.sh, which wraps b70run.sh's safe --init + in-container
#  timeout + detached-wait pattern.
#
#  Usage:
#    tools/preflight_b70.sh                       # build + host/device gates
#    tools/preflight_b70.sh /models/<dir>         # ... and a real model
#    MODEL=/models/<dir> PROJ=int4 tools/preflight_b70.sh
#    SKIP_BUILD=1 tools/preflight_b70.sh          # gates only, no rebuild
#
#  Exit status is 0 only if every REQUIRED stage passed.
# =====================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
MODEL="${1:-${MODEL:-}}"
PROJ="${PROJ:-int4}"
GPU="${GPU:-gpu0}"

PASS=0; FAIL=0; SKIP=0
declare -a FAILED=()

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mPASS\033[0m  %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '   \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=$((FAIL+1)); FAILED+=("$*"); }
skip() { printf '   ....  skip  %s\n' "$*"; SKIP=$((SKIP+1)); }

# A stage is a name plus a command.  Output goes to a log; only the tail
# is shown on failure, so a passing run stays readable.
LOGDIR="${TMPDIR:-/tmp}/grimoire-preflight-$$"
mkdir -p "$LOGDIR"
stage() {
  local name="$1"; shift
  local log="$LOGDIR/${name//[^A-Za-z0-9]/_}.log"
  if "$@" >"$log" 2>&1; then ok "$name"; return 0; fi
  bad "$name  (log: $log)"
  tail -20 "$log" | sed 's/^/        /'
  return 1
}

printf '\033[1mGRIMOIRE preflight\033[0m   repo %s\n' "$REPO"
printf 'logs: %s\n' "$LOGDIR"

# ---------------------------------------------------------------------
say "1. build"
if [[ -n "${SKIP_BUILD:-}" ]]; then
  # Only for re-running the gates against a build you just made.  Never
  # for "the build is broken, let me see if the tests pass anyway": they
  # would be testing the previous binary.
  skip "SKIP_BUILD set -- bin/ and src/*.so are whatever was there"
else
# RULE 3: build_b70.sh does NOT rebuild the cutlass bridges.  A stale .so
# has already cost this project a DEVICE_LOST that looked like a kernel
# bug.  Always go through build_bridges_b70.sh, which rebuilds the
# bridges and then calls build_b70.sh.
#
# It runs INSIDE the container: its include paths are /src/... (the
# vllm-xpu-kernels checkout with cutlass-sycl under .deps) and it needs
# icpx and ocloc, none of which exist on the Unraid host.  No GPU is
# attached -- this is an ahead-of-time compile, so nothing here can wedge
# a card.
IMAGE="${GRIM_IMAGE:-my-vllm-xpu:latest}"
KERNELS="${KERNELS:-}"
if [[ -z "$KERNELS" ]]; then
  for cand in /mnt/user/appdata/vllm-xpu-kernels \
              /mnt/cache/appdata/vllm-xpu-kernels; do
    [[ -d "$cand" ]] && { KERNELS="$cand"; break; }
  done
fi
if [[ -z "$KERNELS" || ! -d "$KERNELS" ]]; then
  bad "vllm-xpu-kernels checkout not found (tried /mnt/user and /mnt/cache appdata)"
  echo "   Set KERNELS=/path/to/vllm-xpu-kernels and re-run."
  echo "   Without it the bridges cannot be built, and building the engine"
  echo "   alone would leave STALE .so files -- rule 3."
  exit 1
fi
echo "   image   : $IMAGE"
echo "   kernels : $KERNELS"

if ! stage "bridges + engine (in $IMAGE)" \
        docker run --rm --entrypoint bash \
          -v "$REPO":/grimoire -v "$KERNELS":/src -w /grimoire \
          "$IMAGE" -lc 'bash tools/build_bridges_b70.sh /grimoire'; then
  echo
  echo "   Build failed -- everything below would test a STALE bin/."
  echo "   Stopping here."
  exit 1
fi

fi

for b in bin/grimoire bin/grimoire-server; do
  [[ -x "$b" ]] && ok "$b present" || bad "$b missing (build it before trusting anything below)"
done

# RULE: GRIMOIRE is C++/SYCL/Level Zero.  A bridge that links Torch drags
# a Python runtime into the hot path and is not what this project is.
say "2. bridges link no Torch"
# Read the ELF's own NEEDED list rather than ldd: ldd RESOLVES, so on the
# host (where the container's libraries do not exist) it can fail outright
# and report nothing at all.  objdump -p reads what the file declares.
shopt -s nullglob
sofiles=(src/libgrimoire_*.so)
if (( ${#sofiles[@]} == 0 )); then
  bad "no src/libgrimoire_*.so -- the bridges are not built"
else
  reader=""
  command -v objdump >/dev/null && reader="objdump -p"
  [[ -z "$reader" ]] && command -v readelf >/dev/null && reader="readelf -d"
  for so in "${sofiles[@]}"; do
    if [[ -z "$reader" ]]; then skip "$(basename "$so") (no objdump/readelf here)"; continue; fi
    if $reader "$so" 2>/dev/null | grep -i 'NEEDED' | grep -qi 'libtorch\|libc10'; then
      bad "$(basename "$so") links Torch"
    else
      ok "$(basename "$so")"
    fi
  done
fi
shopt -u nullglob

# ---------------------------------------------------------------------
say "3. host tests (no GPU)"
# These build with plain g++ on the HOST.  Unraid does not necessarily
# have a host toolchain -- skip rather than fail if it does not, since
# the same suites also run wherever the code was developed.
if command -v make >/dev/null && command -v g++ >/dev/null; then
  stage "make test"             make test
  stage "make test-correctness" make test-correctness
else
  skip "make/g++ not on the host -- host suites not run here"
fi

# ---------------------------------------------------------------------
# These need a device but no model, and they are the ones that catch the
# expensive mistakes: a kernel that drifted from its reference, a path
# that was never executed, a parallel split that changes the answer.
say "4. device gates (no model)"
# tools/tune.sh forwards to b70run.sh, whose argument order is
#   <renderNode> <limit> <NAME> <command...>
# and tune.sh supplies the node and the limit -- so the first thing WE
# pass is the container NAME, not the binary.  Getting that wrong makes
# the binary the container name and runs nothing.
# A gate that was not BUILT is not a gate that passed.  build_b70.sh
# treats device-test compile failures as warnings, so a missing binary
# here means the build silently dropped a required check -- and skipping
# it let preflight finish clean and exit 0.  These are required: treat a
# missing one as a failure, not a skip.
run_gate() {
  local bin="$1"; shift
  local name="${bin##*/}"
  if [[ ! -x "$bin" ]]; then
    bad "$bin was not built -- the build dropped a required gate"
    return
  fi
  stage "$name" env GPU="$GPU" LIM=900 bash tools/tune.sh \
        "pf-${name//_/-}" "/grimoire/${bin}" "$@"
}
# MULTI-RANK GATES NEED MULTI-DEVICE VISIBILITY (external audit F9,
# 2026-09-21).  test_parallel_e2e, test_spec_e2e and test_pp_server each
# fork their OWN child processes and give each one a DIFFERENT
# GRIMOIRE_PP_RANK/GRIMOIRE_TP_RANK -- and Grimoire's own device
# selector explicitly THROWS when a requested rank has no visible GPU
# (src/grimoire.cpp: "rank_requested && !pick.empty()"), rather than
# silently reusing rank 0's card.  run_gate() above exposes exactly ONE
# render node (tune.sh -> b70run.sh --device /dev/dri/$NODE), which is
# correct for a single-process binary but means rank 1 of any of these
# three gates throws immediately once real GPUs are visible in the
# container -- a container with only ONE node exposed is not "no GPU at
# all", so GRIMOIRE_DEVICE_ANY's fallback does not rescue it either.
# Off the card this was never visible: with NO GPU present at all,
# every rank falls back the same way, uniformly, and the gate reports
# green -- which is a real, useful correctness check, but not a check
# that this gate's actual multi-device launch path works.
#
# This wrapper reuses pp2run.sh's OWN device-exposure flags -- full
# /dev/dri, not one node, the exact shape a real Tower run already uses
# successfully for bin/grimoire -- rather than inventing new container
# flags untested against real hardware. It runs the gate binary ONCE
# (not pp2worker.sh's double-launch): these binaries fork their own
# children internally, so doubling the outer launch would run each rank
# twice over.
run_multigpu_gate() {
  local bin="$1"; shift
  local name="${bin##*/}"
  if [[ ! -x "$bin" ]]; then
    bad "$bin was not built -- the build dropped a required gate"
    return
  fi
  if ! command -v docker >/dev/null; then
    skip "$name (no docker here -- this stage only runs from the Tower)"
    return
  fi
  local cname="pf-mgpu-${name//_/-}"
  docker rm -f "$cname" >/dev/null 2>&1 || true
  local log="$LOGDIR/${name//[^A-Za-z0-9]/_}.log"
  {
    echo "== devices visible to this container =="
    ls -la /dev/dri/ 2>&1
    echo "== running $name =="
  } >"$log" 2>&1
  local cid
  cid=$(docker run -d --name "$cname" -w /grimoire --init --stop-timeout 300 \
      --ipc=host --privileged --shm-size=10g \
      --device /dev/dri:/dev/dri \
      -v /dev/dri/by-path:/dev/dri/by-path \
      -v "$REPO:/grimoire" \
      -e ZE_AFFINITY_MASK=0,1 \
      --entrypoint /usr/bin/timeout "${GRIM_IMAGE:-my-vllm-xpu:latest}" \
      --signal=TERM --kill-after=60 900 \
      "/grimoire/${bin}" "$@" 2>>"$log")
  if [[ -z "$cid" ]]; then
    bad "$name (container launch failed, see $log)"
    return
  fi
  {
    echo "== devices as seen INSIDE the running container =="
    docker exec "$cname" ls -la /dev/dri/ 2>&1
  } >>"$log" 2>&1
  local rc
  rc=$(docker wait "$cname" 2>/dev/null || echo "wait-failed")
  docker logs "$cname" >>"$log" 2>&1
  docker rm -f "$cname" >/dev/null 2>&1 || true
  if [[ "$rc" == "0" ]]; then
    ok "$name"
  else
    bad "$name  (log: $log)"
    tail -20 "$log" | sed 's/^/        /'
  fi
}
run_gate bin/test_k2_kernels
run_gate bin/test_k2_e2e
run_gate bin/test_model_matrix
run_multigpu_gate bin/test_parallel_e2e
run_multigpu_gate bin/test_spec_e2e
# Both of these compare a BATCHED prefill against sequential decode.  They
# were built by build_b70.sh and not run here, which is the same shape of
# gap rule 14 is about: the code existed, the gate existed, and nothing
# ran it.
run_gate bin/test_gemma4_prefill
run_gate bin/test_qwen4_exp_e2e
run_gate bin/test_nvfp4_e2e
run_gate bin/test_prefix_reuse
run_gate bin/test_batch_prefix
run_gate bin/test_batch_decode
run_gate bin/test_scheduler
run_multigpu_gate bin/test_pp_server

# ---------------------------------------------------------------------
say "5. real model"
if [[ -z "$MODEL" ]]; then
  skip "no model given -- pass one as \$1 or MODEL=..."
  skip "rule 8: nothing is verified until text has been generated and READ"
else
  # RULE 8: a numeric self-check is not sufficient.  Generate text and
  # put it on screen; a human decides whether it is English.
  LOG="$LOGDIR/generate.log"
  if env GPU="$GPU" LIM=1200 bash tools/tune.sh \
        pf-generate /grimoire/bin/grimoire -m "$MODEL" --proj "$PROJ" \
        -p "Explain in two sentences why the sky is blue." -n 64 \
        >"$LOG" 2>&1; then
    ok "generation completed ($MODEL, --proj $PROJ)"
    echo
    echo "   ---- READ THIS.  If it is not coherent English, nothing else"
    echo "        in this run counts (rule 8). ----"
    sed 's/^/   /' "$LOG" | tail -30
    echo "   ----"
  else
    bad "generation failed ($MODEL, --proj $PROJ)  (log: $LOG)"
    tail -20 "$LOG" | sed 's/^/        /'
  fi
fi

# ---------------------------------------------------------------------
say "summary"
printf '   %d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
if (( FAIL )); then
  printf '\n   failed stages:\n'
  for f in "${FAILED[@]}"; do printf '     - %s\n' "$f"; done
  printf '\n   Do not benchmark anything until these are green.\n'
  exit 1
fi
printf '\n   Preflight clean.\n'
if [[ -z "$MODEL" ]]; then
  printf '   NOTE: no model was run.  Re-run with a model directory before\n'
  printf '         trusting this box -- rule 8.\n'
fi
printf '   Dual-GPU next: tools/pp2run.sh (pipeline) or tools/tp2run.sh (tensor).\n'
exit 0
