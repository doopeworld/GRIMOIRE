# K2-Horizon-MoVA-36B-A4B — handoff

Session of 2026-09-11/12. Branch `codex/b70-audit-fixes-20260911`.

Everything below was written WITHOUT a SYCL toolchain or a B70: this
container has no `/dev/dri`, no oneAPI, and zero SYCL headers, so
`src/grimoire.cpp` was never compiled here. Host-side work is tested and
green; the four new kernels and the whole engine path are unbuilt. Treat the split below as the
boundary between "verified" and "needs a compiler".

## What the model is

Qwen3-MoE with three operator substitutions plus MoVA. Extracted from
`modeling_k2_horizon.py`, not inferred.

- **MoVA** — on a SPARSE layer `v_proj` does not exist. The value is a
  routed mixture over 64 experts of `[kv_heads*head_dim, hidden]` =
  `[1024, 2560]`, top-4, each output through **SiLU** and then scaled by
  the router weight. Dense layers keep an ordinary `v_proj`.
- **Grouped RMSNorm** — `layernorm_num_groups: 2`. Variance per
  contiguous half, weight still applied across the full row.
- **Softplus attention gate** — `beta = log 2`, applied to the attention
  output BEFORE `o_proj`. Keeps torch's `threshold=20` linear fallback.
- **Sigmoid router** — the bias steers **selection only**; the weight
  that scales an expert is the UNBIASED sigmoid. Then sum-normalised and
  multiplied by `router_scaling_factor` 2.5. Used by both the 100-expert
  MoE and MoVA.

Layers 0-2 dense, 3-47 sparse (`mlp_only_layers`). Both FFN shapes
coexist in one model; nothing may assume one.

No MTP head, no DFlash, no draft model — `__all__` in the modeling file
confirms it. K2 gets plain decode with no speculation available.

## Norm weight convention — the one that would silently ruin output

`K2HorizonRMSNorm` initialises its weight to **ones** and multiplies
directly. Qwen3.5 stores the weight centred on zero and applies
`(1 + w)`, and every existing `launch_rmsnorm_*` in `ops.cpp` hardcodes
that. Using the Qwen form for K2 shrinks every norm output by roughly
10x, 48 layers over. This is now carried by `set_norm_convention`, called
once in `build()`: `weight_offset` is 0.0 for K2 and 1.0 for everything
else, and both norm entry points delegate on it.

## Done and host-tested

- `include/b70/k2_horizon.hpp` — host reference for all four operators
  plus the layer-classification rule.
- `tests/test_k2_horizon.cpp` — pins the properties, not golden numbers.
  Includes a lane-for-lane host emulation of `launch_router_topk_k2`
  diffed against the reference over MoVA (64/top-4), MoE (100/top-8) and
  a ragged expert count. 400 tokens each, zero mismatches.
- `tests/test_k2_config.cpp` — parses the REAL checkpoint `config.json`
  through the production loader, pins every field, the derived layer
  plan, the split-rope refusal, the BF16 policy, and resolves a complete
  miniature checkpoint (2 dense + 2 sparse layers).
- Config parsing and per-layer tensor mapping in `qwen35_loader.cpp`.
- `keep_qwen_bf16` now protects `self_attn.v_router.weight`.

`make test` runs 9 suites, all passing, zero warnings.

## Written but NEVER COMPILED

Nothing in `src/grimoire.cpp`, `src/ops.cpp` or `src/prefill.cpp` has
been through a compiler in this session. The new kernels are:

- `launch_rmsnorm_grouped` (grouped, batched, weight_offset)
- `launch_softplus_gate`
- `launch_router_topk_k2`
- `launch_silu_scale_accum`

**If the native build fails, look here first.** They are the last ~150
lines of `ops.cpp` and can be commented out to unblock anything else in
the same translation unit. The `group_barrier` placement in the grouped
norm is the part with no host analogue and is entirely unproven.

## Engine wiring — DONE, uncompiled

`src/grimoire.cpp` now has a K2 path. Everything is gated on `cfg.is_k2`
or `d.k2_sparse`, so Qwen and Ornith paths are unchanged.

- **Upload** — a sparse layer's `v_proj` is not requested (it does not
  exist); `v_router` uploads BF16 and unsharded (N=64 cannot reach a
  W4A8 tile, rule 4); 64 `v_experts` upload per sparse layer at the
  projection format; `mlp.gate.bias` and the softplus `gate_proj` upload
  alongside.
- **MoVA value** — `mova_value_m1()`: router logits, top-k on
  `sigmoid + bias`, weight from the UNBIASED sigmoid, then
  `sum_j w_j * silu(expert_j @ x)` through `launch_silu_scale_accum`.
  Wired at all five decode sites and both prefill sites. The fused bf16
  QKV fast path excludes MoVA layers explicitly.
- **Softplus gate** — applied where `cfg.attn_gate == 2`, at both gate
  sites, instead of `launch_gate_sigmoid_mul`.
- **K2 router** — `launch_router_topk_k2` in the batched MoE path with
  the bias, `norm_topk_prob` and `router_scaling_factor`. The bf16
  router fast path is disabled for K2: it softmaxes the top-k and has no
  bias input, so it cannot express this routing.
- **Grouped norm** — handled by ONE convention set in `build()`
  (`set_norm_convention`) that both `launch_rmsnorm_residual` and
  `launch_rmsnorm_residual_batched` delegate on. Doing it there rather
  than at ~60 call sites is what stops prefill and decode ending up on
  different conventions, which would be silent. For non-K2 models the
  predicate is false and behaviour is unchanged.

### The one deliberate compromise

`mova_value_m1` reads the routing table back to the host so the selected
experts can go through the proven `gemv_any`. One sync per layer at M=1;
**O(M) syncs per layer in prefill**, which is fine for a short prompt and
useless at 4096 tokens. It is correct and it reuses kernels that already
work, which is the right trade for a first bring-up — get text worth
reading, then optimise.

The fix is to pack the 64 experts expert-major into a single weight so a
grouped GEMV can index them with no readback, exactly the shape the
routed MoE already uses (`d.gu_pack` / `d.moe.gate_up`). Do that before
quoting any PP number.

## Quantization

Every matrix divides cleanly — all `K % 128 == 0`, all `N % 256 == 0` —
so INT4 g128 is legal everywhere and unlocks the W4A8 path, except the
two routers above.

| | model | per token |
| --- | --- | --- |
| INT4 | 22.0 GB | 3.6 GB |
| FP8 E4M3 | 38.0 GB | 5.0 GB |

FP8 has working decode GEMV and prefill GEMM paths; use `fp8-e4m3`, not
`mxfp8` (measured 320 GB/s vs 214 in `gemv_decode.cpp`). INT4 for a
single card, FP8 across two when the VRAM is there and precision matters.

`lm_head` stays BF16 under `keep_qwen_bf16` — 1.28 GB of the 3.6 GB
per-token read. Quantizing it to INT4 would take that to ~2.6 GB/token,
worth roughly 1.6 ms, but that is a quality call already settled the
other way for Qwen and Ornith. Left alone deliberately.

## Do not quote a speed from this file

No K2 number in this session came from a run. Earlier estimates in the
conversation were anchored on Qwen's PP, which runs at ~15 TFLOP/s
against Ornith's ~61 on the same card, and were wrong because of it. The
repo's own measurements are the honest bracket: bf16 XMX peak ~180
TFLOP/s, GEMMs measured 100-134, Ornith `2 x 3e9 x 4096 / 401.8 ms` =
10,194 tok/s PP. K2's prefill is ~44.8 TFLOP at 4096 tokens; divide by a
measured rate and you get a range, not a number. Per rule 8, none of it
counts until something generates text you can read.

## Tower checklist

```
tools/build_bridges_b70.sh                 # rule 3, NOT build_b70.sh
./bin/test_tokenizer <model-dir>           # covers 392b85a, skipped on hosts
tools/tune.sh ... -p "prompt" -n 64        # rule 8, read the output
GRIMOIRE_W4A8=1 GRIMOIRE_PARALLEL_SHARED=1 # the path the audit fix closed
llama-benchy --pp 4096 --tg 32
```

Verify the audit fixes first — they are complete and independent of all
K2 work above.
