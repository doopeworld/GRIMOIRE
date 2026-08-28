#!/bin/bash
# GEMV tuning sweep. One build, many configs -- no rebuild between runs.
#
#   docker run --rm --device /dev/dri --entrypoint bash b70:local /usr/local/bin/sweep.sh
#
# Sources setvars itself so it works whether or not it goes through the
# image entrypoint. Without this the binary cannot find the SYCL UR
# adapters and every run silently produces nothing.
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true

BIN=/usr/local/bin/b70_native_inference
printf '%-8s %-8s' "EPL" "UNROLL"
printf ' %10s' bf16 int8 int4 mxfp4 mxfp8 fp8
echo
for epl in 16 32 64; do
  for un in 2 4 8; do
    printf '%-8s %-8s' "$epl" "$un"
    OUT=$(B70_EPL=$epl B70_UNROLL=$un $BIN 2>&1)
    for f in bf16 int8 int4 mxfp4 mxfp8 fp8_e4m3; do
      V=$(echo "$OUT" | awk -v f="$f" '$1==f {print $4}')
      printf ' %10s' "${V:-  -}"
    done
    echo
  done
done
echo
echo "Pick the best per column, then set the defaults in"
echo "src/kernels.hpp GemvGeom<>::EPL_DEFAULT."
