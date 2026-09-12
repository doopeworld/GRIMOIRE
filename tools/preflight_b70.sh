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
# RULE 3: build_b70.sh does NOT rebuild the cutlass bridges.  A stale .so
# has already cost this project a DEVICE_LOST that looked like a kernel
# bug.  Always go through build_bridges_b70.sh, which rebuilds the
# bridges and then calls build_b70.sh.
if ! stage "bridges + engine (tools/build_bridges_b70.sh)" \
        bash tools/build_bridges_b70.sh "$REPO"; then
  echo
  echo "   Build failed -- everything below would test a STALE bin/."
  echo "   Stopping here."
  exit 1
fi

for b in bin/grimoire bin/grimoire-server; do
  [[ -x "$b" ]] && ok "$b present" || bad "$b missing after a successful build"
done

# RULE: GRIMOIRE is C++/SYCL/Level Zero.  A bridge that links Torch drags
# a Python runtime into the hot path and is not what this project is.
say "2. bridges link no Torch"
shopt -s nullglob
for so in src/libgrimoire_*.so; do
  if ldd "$so" 2>/dev/null | grep -qi 'libtorch\|libc10'; then
    bad "$(basename "$so") links Torch"
  else
    ok "$(basename "$so")"
  fi
done
shopt -u nullglob

# ---------------------------------------------------------------------
say "3. host tests (no GPU)"
stage "make test"             make test
stage "make test-correctness" make test-correctness

# ---------------------------------------------------------------------
# These need a device but no model, and they are the ones that catch the
# expensive mistakes: a kernel that drifted from its reference, a path
# that was never executed, a parallel split that changes the answer.
say "4. device gates (no model)"
run_gate() {
  local bin="$1"; shift
  if [[ ! -x "$bin" ]]; then skip "$bin not built"; return; fi
  stage "$(basename "$bin")" env GPU="$GPU" LIM=900 bash tools/tune.sh \
        "/grimoire/${bin}" "$@"
}
run_gate bin/test_k2_kernels
run_gate bin/test_k2_e2e
run_gate bin/test_model_matrix
run_gate bin/test_parallel_e2e

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
        /grimoire/bin/grimoire -m "$MODEL" --proj "$PROJ" \
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
