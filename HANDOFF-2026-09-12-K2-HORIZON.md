# K2-Horizon-MoVA-36B-A4B — handoff

Session of 2026-09-11/12. Branch `codex/b70-audit-fixes-20260911`.

Everything below was written WITHOUT a SYCL toolchain or a B70: this
container has no `/dev/dri`, no oneAPI, and zero SYCL headers, so
`src/grimoire.cpp` was never compiled here. Host-side work is tested and
green; the three new kernels are unbuilt. Treat the split below as the
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
10x, 48 layers over. `launch_rmsnorm_grouped` takes `zero_centered` as a
parameter; K2 passes **false**.

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

Three kernels at the end of `src/ops.cpp`, declared in `kernels.hpp`:

- `launch_rmsnorm_grouped`
- `launch_softplus_gate`
- `launch_router_topk_k2`

**If the native build fails, look here first.** They are the last ~150
lines of `ops.cpp` and can be commented out to unblock anything else in
the same translation unit. The `group_barrier` placement in the grouped
norm is the part with no host analogue and is entirely unproven.

## NOT started — the engine wiring

`src/grimoire.cpp` has no K2 path. **The model will not load.** This is
the remaining work, and it needs a compile loop; it was left undone
rather than written blind into the file the audit fixes also live in.

Measured surface:

| touch point | sites |
| --- | --- |
| `v_proj` | 39 |
| `launch_rmsnorm*` | 59 |
| attention gate / swiglu | 28 |
| `launch_router_topk*` | 8 |

Order of work:

1. **Upload MoVA weights.** `Qwen35Layer::v_experts[]` and `v_router`
   already resolve. They need a `DevQuant` bank per sparse layer, same
   treatment as `d.moe.gate_up`. 64 experts x 45 layers.
   Rule 4: `v_router` is `[64,2560]` and `mlp.gate` is `[100,2560]` —
   neither N divides 256, both must stay BF16 and off any W4A8 tile.
   `keep_qwen_bf16` covers them offline; make sure no runtime conversion
   path picks them up.
2. **Attention value path.** When `lay.k2_sparse`, replace the `v_proj`
   GEMM with `launch_router_topk_k2` over `v_router` then a grouped GEMM
   over the selected experts, SiLU on the expert output, weighted
   accumulate. The existing grouped-MoE dispatch is the closest model.
   Shapes are identical to an MoE expert bank, so the 64-row tile and
   the `B70_MOE_SLOTS` machinery should apply.
3. **Norms.** Route every K2 norm through `launch_rmsnorm_grouped` with
   `n_groups = cfg.norm_groups` and `zero_centered = false`.
4. **Attention gate.** `cfg.attn_gate == 2` means softplus, not silu.
5. **MoE router.** `launch_router_topk_k2` with `cfg.router_sigmoid`,
   `lay.router_bias`, `cfg.norm_topk_prob`, `cfg.router_scale`.

Gate all of it on `cfg.is_k2` so Qwen and Ornith paths stay bit-identical.

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
