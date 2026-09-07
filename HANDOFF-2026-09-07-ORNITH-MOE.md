# Ornith MoE checkpoint — 2026-09-07

## Current state

The Tower branch is `qwen-tg-restored`. The latest real prompt is 5,993
tokens (`real4k.txt`). Router validation at layer 2 reports valid expert IDs,
no per-token duplicates, all 256 experts active, and a highly skewed expert
load. The remap count sum matches `M * top_k` in the diagnostic path.

The old custom persistent/static-grid experiment hung on a 4,096-token
synthetic prefill and was reverted. Selecting vLLM's Xe2 policies by average
routed rows (`m8`, `m16`, `m32`, and full) was implemented, but the real prompt
still reached `UR_RESULT_ERROR_DEVICE_LOST` in the grouped path before this
checkpoint. The diagnostic run with `GRIMOIRE_MOE_ROUTE_CHECK_STOP=1` used the
sequential fallback because the grouped bridge was not loaded; its remap data
was therefore not a grouped-kernel test.

## vLLM comparison

The image contains `vllm 0.28.1.dev8+ga4f0b85eb.d20260831.xpu` and
`vllm-xpu-kernels 0.1.14.dev0+g07d44bc.d20260813`, newer than the previously
inspected v0.1.13.1 checkout. The installed native schema is:

`cutlass_grouped_gemm_interface(A, A_scale, B, B_scale, bias, D, rows_per_expert, expert_ids, N, K, num_experts)`.

vLLM remaps to exactly `num_rows * topk`, keeps `rows_per_expert` on XPU,
passes optional compact expert IDs, and dispatches MXFP4 through the native
operator. Grimoire's direct bridge has a separate SYCL ABI and currently passes
`experts=nullptr`, so the call contract still needs a direct apples-to-apples
check. The image's normal serving flags are not the cause of a prefill device
loss: `--max-num-seqs 16 --kv-cache-dtype fp8 --max-num-batched-tokens 16384
--block-size 64 --language-model-only --trust-remote-code
--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser
qwen3`. Parser/reasoning/tool flags affect request formatting and output
post-processing, not the grouped GEMM pointer/shape contract. KV cache is used
by attention/decode and is not the failing MoE prefill operation.

## Next safe checks

1. Add temporary `dlerror()` logging to every grouped-bridge candidate and run
   the safe `tools/tune.sh` launcher with the exact `GRIMOIRE_XE2_GROUPED_BRIDGE`
   path; confirm the exact-policy symbols are selected before any GPU test.
2. Compare the installed 0.1.14.dev0 MXFP4 dispatch source/compiled operator
   with the bridge's `N/K/E`, rows, and expert-ID behavior. In particular test
   the compact expert-ID path and the two GEMM calls separately.
3. Only after bridge loading is confirmed, rerun the 5,993-token prompt with
   the proper Ornith attention bridge and the vLLM serving flags above.

No vLLM image rebuild was performed; the current image already contains the
newer installed kernel package and was sufficient for source/schema inspection.
