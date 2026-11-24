# Grimoire handoff — PP and TP complete

Date: 2026-08-27

Planned backup: `grimoire-FULL-BACKUP-20260827-1730.tar.gz`

## Current stopping point

Functional two-process pipeline parallelism (PP) and tensor parallelism (TP) are complete. All experiment and build containers were stopped before this backup. Performance over GPU 1's present USB4 connection is not a target or acceptance criterion; repeat performance measurements after the OCuLink PCIe Gen4 x4 connection is installed.

The prefix-cache implementation in `src/grimoire.cpp` and its CLI test in `tools/grimoire_main.cpp` are work in progress. They were copied to the server, but the interrupted build did not produce a validated binary. Treat these changes as uncompiled and untested until proven otherwise.

## Required runtime configuration

Docker device arguments:

```text
--ipc=host --privileged --shm-size=10g --device /dev/dri:/dev/dri -v /dev/dri/by-path:/dev/dri/by-path
```

Required environment:

```text
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_USE_V2_MODEL_RUNNER=1
CCL_ATL_TRANSPORT=ofi
CCL_ZE_IPC_EXCHANGE=sockets
CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
ZE_AFFINITY_MASK=0,1
VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8=1
TORCH_COMPILE_BACKEND=inductor
ZE_SHARED_FORCE_DEVICE_ALLOC=1
VLLM_TARGET_DEVICE=xpu
ONECCL_BINDINGS_FOR_PYTORCH_ENV_MODE=p2p
```

Never test `ZE_AFFINITY_MASK=1` alone. That previously caused a Level Zero device-loss/segmentation failure and wedged the external GPU until reboot. Preserve the full `0,1` mask and select the local device inside each process.

## PP result

- Two independent processes use `GRIMOIRE_PP_RANK=0|1`.
- Default layer partition is 24/16: 60% of layers on GPU 0 and 40% on GPU 1.
- Hidden-state transfer uses a Unix socket once per stage; generated tokens synchronize rank 1 to rank 0.
- Batched prefill transfers M x H hidden states and handles the deferred MoE boundary.
- Graph execution is disabled on the socket PP path.
- Launch with `tools/pp2run.sh`; workers are in `tools/pp2worker.sh`.
- Correctness passed for prompt `Hello`, four generated tokens: both ranks produced `The user just said`, matching single-GPU output.
- Safe `--prefill-only` tests passed at 128, 1024, and 4096 prompt tokens.
- The 4096 fallback run measured about 1577.5 ms / 2596.5 prompt tokens/s, but this USB4 result is informational only.

See `MULTIPROCESS-PP-2026-08-27.md` for detailed evidence and safety notes.

## TP result

- Two independent processes use `GRIMOIRE_TP_RANK=0|1` and a TP Unix socket.
- Decode projections with even output width are split by output rows: rank 0 computes the first half, rank 1 the second half, followed by an all-gather that reconstructs the full result.
- `ffn_gemv` uses the same TP projection path.
- Graph execution is disabled on the TP path.
- Launch with `tools/tp2run.sh`; workers are in `tools/tp2worker.sh`.
- Correctness passed for prompt `Hello`: two tokens produced `The user`, and four tokens produced `The user just said` on both ranks.
- Both processes exited successfully, about 61 TG over USB4, and both XPUs remained visible.

Important scope: this is functional projection-compute sharding. Each process still holds full weights; MoE weights and batched-prefill compute remain replicated. Do not describe this version as fully memory-sharded TP.

See `MULTIPROCESS-TP-2026-08-27.md` for detailed evidence.

## Next work, in order

1. Finish exact prompt-prefix caching for prefill.
   - First compile the current WIP without launching a model.
   - Fix any compile errors.
   - Run a safe single-GPU 128-token `--prefix-cache-test` with `GRIMOIRE_PREFIX_CACHE=1`.
   - Verify cold and cache-hit next-token IDs match and record cold/hit latency.
   - Only then test prefix caching through PP; keep the safe dual-device mask.
   - The intended cache includes exact token matching plus device-resident K/V, DeltaNet recurrent state, convolution rings, hidden/logits, position, and sequence length.
2. Add and baseline the Muse Glimmer model.
   - Confirm architecture, tokenizer, weight formats, and loader mapping from the actual model files/configuration before coding assumptions.
   - Establish single-GPU correctness first, then PP/TP compatibility.
3. Check Muse Glimmer speculative decoding support.
   - Inspect its native MTP metadata/weights first.
   - Then test MTP, DSpark, and DFlash2 one at a time with identical prompts and acceptance/correctness reporting.
   - Do not infer compatibility merely from model names; verify required heads, draft checkpoints, and runtime APIs.
4. After OCuLink is installed, rerun PP/TP performance measurements. Until then, optimize for correctness and stability rather than throughput.

## Recovery reference

Previous known backup: `grimoire-FULL-BACKUP-20260827-1510.tar.gz`

Previous SHA-256:

```text
7f6f23e12f2a352f3c01ddf32f3e5670bb2f6576b2ce54df19f33a99d113c32b
```


---
## ITEM 1 DONE (2026-08-27): exact prompt-prefix cache validated

Compiles clean (no WIP errors). GRIMOIRE_PREFIX_CACHE=1, --prefix-cache-test N.

Single-GPU (Ornith MXFP4, gpu0), cold vs cache-hit, next-token IDs MATCH:
  128:  cold 1008.9 ms -> hit 0.8 ms   token 104311==104311
  1024: cold 1126.3 ms -> hit 0.7 ms   token  20600==20600
  4096: cold 1330.5 ms -> hit 0.8 ms   token  20600==20600
Hit is ~0.8 ms regardless of size (device state copy); cold scales with tokens.

Through PP (pp2run.sh, 24/16 split, ZE_AFFINITY_MASK=0,1), both ranks PASS:
  rank0: cold 977.3 ms  -> hit 60.3 ms   token 228565==228565
  rank1: cold 1100.1 ms -> hit 96.4 ms   token 228565==228565
PP hit slower (socket sync still runs on hit) but correct, exit 0, both XPUs OK.
Cache stores per-layer dn_state/conv_ring/K/V + hidden/logits/pos/seq_len.

---
## ITEM 2 — Muse Glimmer architecture CONFIRMED (2026-08-27, from actual files)

Models on box: Muse-Glimmer-30B-MXFP4, -INT4-W4A16, -assistant (drafter).
Only HF safetensors (compressed-tensors), NO .b70 native artifact yet.

muse_glimmer_text: DENSE transformer, Gemma-style. hidden 6656, 52 layers,
all full-attention, 32 heads / 2 KV (GQA), head_dim 128, intermediate 19968,
vocab 202048, rope_theta 500000. MXFP4 mxfp4-pack-quantized, group_size 32,
symmetric (weight_packed + weight_scale per matrix).

Per-layer tensors (prefix model.language_model.layers.N.):
  input_layernorm, post_attention_layernorm,
  pre_feedforward_layernorm, post_feedforward_layernorm   <-- 4 norms (Gemma sandwich)
  self_attn.{q,k,v,o}_proj + self_attn.gate_proj          <-- gated attention
  mlp.{gate,up,down}_proj.{weight_packed,weight_scale}
Top: lm_head.weight, model.language_model.embed_tokens.weight, model.language_model.norm.weight
No q_norm/k_norm listed. Plus a vision tower (skip for text).

DIFFERENCES vs GRIMOIRE Qwen35/Ornith loader+forward:
  - 2 EXTRA norms/layer (pre/post feedforward) wrapping the MLP.
  - all full-attn (no GDN linear layers) -> simpler than Qwen/Ornith.
  - gated attention: reuse existing  path.
  - compressed-tensors MXFP4 packing: verify vs GRIMOIRE safetensors MXFP4.
Integration = new config branch + 2 norms in LayerDev + sandwich in forward()
+ skip vision. Establish single-GPU correctness first, then PP/TP.

### Muse Glimmer integration plan (verified against loader src)

GRIMOIRE loads HF safetensors directly (no .b70 needed). Concrete steps:
1. FIX skip_vision: Muse Glimmer vision tensors are model.vision_adapter.* /
   contain "vision" (809 of them), NOT ".visual." -- the current filter misses
   them. Add "vision" to the skip in qwen35_loader.cpp (both b70 and HF paths)
   and grimoire.cpp expert/tap scans. WITHOUT THIS it tries to load 809 vision
   tensors.
2. Config: n_experts=0 -> is_moe() false -> dense MLP branch already loads
   mlp.gate_proj/up_proj/down_proj = Muse Glimmer MLP. layer_types absent ->
   all FULL_ATTN (correct). GQA 32/2, head_dim 128, hidden 6656, 52 layers.
   May need a model_type=muse_glimmer_text recognition to set the sandwich flag.
3. LayerDev: add pre_ff_norm, post_ff_norm (bf16), and attn gate handling for
   full-attn layers (self_attn.gate_proj -> reuse the  path: q_proj
   emits [q|gate], or a separate gate_proj tensor -- CHECK which).
4. loader: load b+"pre_feedforward_layernorm.weight",
   b+"post_feedforward_layernorm.weight"; full-attn also self_attn.gate_proj.
   q_norm/k_norm absent (ok, get() returns not-ok).
5. forward(): Gemma sandwich order --
   h1 = input_norm(x); a = gated_attn(h1); x = x + post_attn_norm(a);
   h2 = pre_ff_norm(x); m = mlp(h2);   x = x + post_ff_norm(m).
   (Confirm exact residual/norm placement against the HF muse_glimmer modeling
   code before finalizing -- do not assume.)
6. Validate single-GPU coherence, then PP/TP.

STATUS: architecture + loader mapping CONFIRMED; port not yet coded.

---
## ITEM 2 DONE (2026-08-27): Muse Glimmer 30B text — WORKING single-GPU

Coherent generation, 31.6 TG single-GPU (Muse-Glimmer-30B-MXFP4, gpu0):
  prompt "Write a haiku about the ocean." ->
  "We need to write a haiku about the ocean. Probably 5-7-5 syllable..."
  (reasoning-model style, on-topic, stable over 48 tokens, no degeneration)
  pp 30.7 tok/s (sequential), tg 31.6 tok/s.

Muse Glimmer = dense Gemma-style transformer (52 layers, hidden 6656, GQA 32/2,
head_dim 128, vocab 202048). Integrated from scratch. Read the EXACT forward
from vLLM nightly ref/muse_glimmer.py (do not guess).

What was added (all gated on cfg.is_muse; Qwen/Ornith paths untouched):
1. Config: parse nested "text_config" (Muse nests text dims there); set
   is_muse, query_prescale (qk_scale_factor=3.87), attn_out_gate.
2. skip_vision: also skip "vision" (Muse names vision model.vision_adapter.*,
   NOT .visual.). 809 vision tensors skipped.
3. Loader packed(): resolve compressed-tensors weight_packed + weight_scale.
   For the MXFP4 model, weight_packed IS GRIMOIRE MXFP4 (E2M1 + E8M0 g32) ->
   direct upload, NO re-quant. Added direct-MXFP4 branches to quantize_upload_t
   and concat_upload_t keyed on "weight_packed". (MXFP4 model: only MLP is
   packed; attention stays BF16 .weight. INT4 variant packs everything.)
4. Loader: load pre_feedforward_layernorm, post_feedforward_layernorm,
   self_attn.gate_proj (Muse sandwich norms + attn output gate).
5. LayerDev: pre_ff_norm, post_ff_norm, o_gate. build() uploads them.
6. forward_muse(): Gemma sandwich, exact order --
     h = scaleless_rms(embed(t))                 # embed norm, no sqrt(H)
     per layer:
       n=rmsnorm(h,input_norm); attn(gated); a=rmsnorm(attn,post_attn_norm); h+=a
       n=rmsnorm(h,pre_ff_norm); mlp(swiglu); m=rmsnorm(mlp,post_ff_norm); h+=m
     logits = lm_head(rmsnorm(h,final_norm))
   Scaleless norms (embed, qk) = zero-weight -> (1+0). QK-norm scaleless over
   head_dim BEFORE RoPE; query prescale folded into softmax_scale =
   qk_scale_factor/sqrt(head_dim). Separate gate_proj -> sigmoid * attn.
7. prefill() returns false for muse -> sequential forward() (no DeltaNet bufs).

Models on box: Muse-Glimmer-30B-{MXFP4 (working), INT4-W4A16 (compressed-tensors
int4, needs weight_packed-int4 wiring), GPTQ-INT4 (GRIMOIRE linear() handles
GPTQ natively -> should load), -assistant (drafter for speculative)}.

REMAINING for Muse Glimmer: batched prefill (fly on long prompts, currently
sequential); PP/TP compat; item 3 speculative (the -assistant drafter is
MuseGlimmerAssistantModel -- in vLLM's supported list).

---
## NEXT SESSION: make Muse Glimmer prompt-prefill work properly and FAST

### Current state (2026-08-27 backup)
Muse Glimmer 30B text WORKS single-GPU, coherent, 31.6 TG decode.
But PROMPT PREFILL IS SEQUENTIAL AND SLOW: 96 tokens in 3026 ms = 31.7 tok/s
(same rate as decode, because each prompt token runs a full forward_muse()
pass one at a time). A 1000-token prompt would take ~31 s -- unusable.
Ornith batched PP is ~10,000; Qwen ~2,500; Muse ~32. ~80-300x too slow.

Cause: `prefill()` returns false for cfg.is_muse (early-return I added to skip
the DeltaNet/linear-attn buffers, which Muse does not have), so generate falls
back to sequential `for each token: forward_muse(token)`.

### The task: implement BATCHED prefill for Muse (process all M tokens in one pass)

Build a `prefill_muse(tokens, next_tokens)` that mirrors forward_muse but over M
tokens, exactly as the Qwen/Ornith batched prefill does. Muse is DENSE full-attn
(no GDN), so it is SIMPLER than the existing batched prefill -- no DeltaNet, no
conv rings, no MoE gather.

Per-layer batched (M tokens), Gemma sandwich (verified order, do NOT change):
  n = rmsnorm(h, input_norm)                    # (1+w), no residual fold
  q=Wq n; k=Wk n; v=Wv n                        # batched GEMM, M rows
  qk_norm scaleless over head_dim (zero weight, zero_centered=true) BEFORE rope
  rope(q,k) over M positions; kv_append M tokens to K/V cache
  attn = FULL causal attention over M queries (batched flash / prefill attn)
         softmax_scale = query_prescale / sqrt(head_dim)   [= 3.87/sqrt(128)]
  gate = Wgate n; attn = sigmoid(gate) * attn   # separate gate_proj
  a = Wo attn; a = rmsnorm(a, post_attn_norm); h = h + a
  n = rmsnorm(h, pre_ff_norm); m = down(swiglu(gate_up n)); 
  m = rmsnorm(m, post_ff_norm); h = h + m
final: logits[last] = lm_head(rmsnorm(h[last], final_norm))
Embedding: h = scaleless_rms(embed(tokens))     # per row, no sqrt(H)

### Concrete steps
1. Remove/guard the `if (cfg.is_muse) return false;` in prefill(); instead route
   cfg.is_muse to a new prefill_muse() that allocates ONLY what a dense model
   needs (bh/bn/r0 hidden buffers, qkv M*QW, k/v M*KVW, attn M*QW, gate M*QW,
   sh_g M*2I, moe_y M*H, logits). Do NOT allocate alpha/beta/dn_state/conv.
   (The current failure was `prefill allocation failed: beta` -- deltanet gate.)
2. Reuse the batched kernels the Qwen/Ornith prefill uses:
   - launch_rmsnorm_*_batched (or plain rmsnorm over M*H)
   - the W4A8 M16-tile GEMM for the MLP (sh_gu/sh_down are MXFP4 -> use
     xe2_dense_mxfp4 / the small-M W4A8 path already wired), attention proj are
     BF16 -> launch_gemm_xmx or xe2_dense.
   - batched attention: the full-attn prefill path (xe2_attention or
     launch_flash over M queries with causal mask). Muse is standard causal
     full attention, GQA 32/2.
   - launch_swiglu_batched, launch_gate_sigmoid_mul_batched (exists for Ornith
     gated attn -- reuse), launch_rmsnorm_heads over M*heads.
3. qk-norm scaleless: launch_rmsnorm_heads(zero_centered=true, muse_zero) over
   M*heads. Query prescale folded into softmax_scale (already done in decode).
4. GATE: verify correctness against sequential forward_muse output on the SAME
   prompt -- next-token id MUST match. Then measure PP; target >1500 tok/s.
5. Then the prefix cache works for Muse too (already generic).

### Gotchas
- Sandwich norms add residual AFTER the post-norms (h += norm(x)), NOT
  residual-folded like Qwen's rmsnorm_residual. Use plain rmsnorm + launch_add.
- Scaleless norms = zero weight (muse_zero), zero_centered=true -> (1+0)=1.
- Muse MLP is compressed-tensors MXFP4 weight_packed (loaded direct, working).
  Attention is BF16 .weight. So MLP GEMM = MXFP4 path, attn GEMM = bf16 path.
- Only the last token's logits are needed for generation (argmax at pos M-1).

### After fast prefill: speculation (item 3)
Muse -assistant / incoai/Muse-Glimmer-30B-DFlash2 = DFlash block drafter
(5 layers, block 16, taps [1,13,25,37,49]). vLLM maps MuseGlimmerAssistantModel
-> DFlashQwen3ForCausalLM. vLLM single-card reference is UNMEASURABLE (30B target
+ drafter + KV > 32 GB, OOM; needs TP=2/USB4). vLLM Muse baseline = 27.2 TG
(GPTQ-INT4); GRIMOIRE Muse MXFP4 already = 31.6, beating it. DFlash2 port =
drafter forward + block propose + verify (large, same as unfinished Ornith
DFlash2); the batched prefill above IS the verify-batch foundation, and the
verifier per-step overhead is the known blocker to solve first.
