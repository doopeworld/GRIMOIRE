#!/bin/bash
# v1.10.4 gate, one GPU job at a time, stops at the first failure. Only what the change touches:
# wait for the build -> one-card Flash-Next + Ornith checks -> two-card Flash-Next, v1.10.3 image vs candidate
# (the CPU-expert pool changed for two ranks too) -> the regression cases the change can reach (regress_fn_v1104.sh)
# -> assemble -> validate the image.
# 11-05: the candidate is 35201c2 (+ bf16 two-card small all-reduce); the CLI pin matrix is not rerun (passed 11-04,
# one-card code unchanged); if the bf16 two-card text differs, fp32 partials must still match v1.10.3.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
exec >>"$B/logs/v1104-all.out" 2>&1
stop() { echo "V1104_ALL_STOP $1 $(date +%T)"; exit 9; }
echo "V1104_ALL_START $(date +%T)"
until grep -qE "^RELEASE_BUILD_(PASS|FAIL)" "$B/logs/release-build-1102b.out"; do sleep 30; done
grep -qx RELEASE_BUILD_PASS "$B/logs/release-build-1102b.out" || stop "build failed"
until [ -z "$(docker ps -q --filter name=grim-release-build-1102b)" ]; do sleep 5; done
# a B580 run or an A/B may still be going: the checks need a quiet machine, so wait for every container to end
# and for logs/HOLD-v1104 (set while another planned GPU job runs) to be gone
until [ -z "$(docker ps -q)" ] && [ ! -e "$B/logs/HOLD-v1104" ]; do sleep 10; done

SKIP_MATRIX=1 bash "$B/run_v1104_checks.sh"; rc=$?
grep -E "^\||IDENTICAL|DIFFERS|identical answers|tg128|V1104_CHECKS_END" "$B/logs/v1104-checks.out" | cut -c1-160
[ "$rc" = 0 ] || stop "one-card checks rc=$rc"

IMG=grimoire-b70:v1.10.3 HOSTBIN=0 bash "$B/run_dual_1104.sh" v1103 fn > "$B/logs/dual-v1103-fn.out" 2>&1 || stop "two-card Flash-Next on v1.10.3"
cat "$B/logs/dual-v1103-fn.out"
GRIM_TREE=/mnt/storage/isos/grimoire-release-1102b HOSTBIN=1 IMG=grimoire-b70:v1.10.3 \
  bash "$B/run_dual_1104.sh" v1104 fn > "$B/logs/dual-v1104-fn.out" 2>&1 || stop "two-card Flash-Next on the candidate"
cat "$B/logs/dual-v1104-fn.out"
same() { # TAG: the candidate's two-card Flash-Next answers vs v1.10.3's
python3 - "$B/logs/d-v1103-fn-tp.gen.json" "$B/logs/d-$1-fn-tp.gen.json" "$1" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
same = sum(a[k]["text"] == b.get(k, {}).get("text") for k in a)
for k in a:
    print(k, "two cards, v1.10.3 vs", sys.argv[3] + ":", "IDENTICAL" if a[k]["text"] == b[k]["text"] else "DIFFERS",
          "seconds %.2f -> %.2f" % (a[k]["seconds"], b[k]["seconds"]))
sys.exit(0 if same == len(a) else 1)
PY
}
if ! same v1104; then
  # 35201c2's bf16 partials may move a near-tie token. The same binary with fp32 partials must equal v1.10.3 exactly
  # (that isolates the CPU-expert pool change); the bf16 answers are then left for review (marker below).
  DUAL_EXTRA=GRIMOIRE_TP_SMALL_BF16=0 GRIM_TREE=/mnt/storage/isos/grimoire-release-1102b HOSTBIN=1 IMG=grimoire-b70:v1.10.3 \
    bash "$B/run_dual_1104.sh" v1104f32 fn > "$B/logs/dual-v1104f32-fn.out" 2>&1 || stop "two-card Flash-Next, candidate with fp32 partials"
  cat "$B/logs/dual-v1104f32-fn.out"
  same v1104f32 || stop "two-card Flash-Next answers differ with fp32 partials too"
  echo "TWO_CARD_BF16_TEXT_DIFFERS: review logs/d-v1104-fn-tp.gen.json against logs/d-v1103-fn-tp.gen.json"
fi

bash "$B/regress_fn_v1104.sh" > "$B/logs/regress-fn-v1104.out" 2>&1; rc=$?
cat "$B/logs/regress-fn-v1104.out"
[ "$rc" = 0 ] || stop "regression (the cases the change reaches) rc=$rc"

bash "$B/assemble-image-v1104.sh" > "$B/logs/assemble-v1104.out" 2>&1 || stop "assembly"
tail -2 "$B/logs/assemble-v1104.out"
bash "$B/validate_v1104.sh" > "$B/logs/validate-v1104.out" 2>&1
cat "$B/logs/validate-v1104.out"
grep -qx IMAGE_LOG_CHECK_PASS "$B/logs/validate-v1104.out" || stop "image validation"
echo "V1104_ALL_DONE $(date +%T)"
