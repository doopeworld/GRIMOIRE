"""Dump the first DFlash proposal cycle's tensors from Fusion, for parity
comparison against Grimoire's GRIMOIRE_DFLASH_DUMP output.

Written as raw little-endian float32, name-matched to Grimoire's `g_*.f32`
with an `f_` prefix so tools/compare_dumps.py can pair them.
"""
import os, sys, json

os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
# Ian never runs enforce_eager: it is slower and degrades output. Eager is only
# needed to capture in-model tensors, since XPU graph capture stops the forward
# hooks firing per step. Probes that read slot mappings or block sizes sit
# outside the graph and must run in the real (graph-enabled) configuration.
os.environ.setdefault("VLLM_XPU_ENABLE_XPU_GRAPH", "1")

DUMP = os.environ["FUSION_DUMP"]
os.makedirs(DUMP, exist_ok=True)
TARGET = os.environ.get("TARGET", "/models/Muse-Glimmer-30B-INT4-W4A16")
DRAFT = os.environ.get("DRAFT", "/models/Muse-Glimmer-assistant")
K = int(os.environ.get("K", "15"))
PROMPT_IDS = os.environ.get("PROMPT_IDS", "")

import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.model_executor.models import qwen3_dflash as QD
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as SPEC

# The draft runs once per decode step; only the first cycle is comparable to
# Grimoire's, since every later cycle inherits the previous cycle's acceptance.
_cycle = [0]
# vLLM runs several dummy/profiling forward passes with synthetic inputs before
# the first real request. Those hit every hook below and would otherwise be
# captured as "the first cycle" -- they show up as an all-identical draft.
# Nothing is written until the real generate() call arms this.
_armed = [False]


def save(name, t):
    if not _armed[0] or _cycle[0] != 0:
        return
    a = t.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
    a.tofile(os.path.join(DUMP, "f_%s.f32" % name))
    print("  fusion dump: %-22s %s" % (name, tuple(t.shape)), flush=True)


# ---- stage 1/2: aux concat + fc -------------------------------------------
_orig_combine = QD.DFlashQwen3ForCausalLM.combine_hidden_states


def combine_hidden_states(self, hidden_states):
    save("01_aux_0", hidden_states)
    out = _orig_combine(self, hidden_states)
    save("02_fc_0", out)
    return out


QD.DFlashQwen3ForCausalLM.combine_hidden_states = combine_hidden_states

# ---- stage 3/4: context norm + fused context K/V ---------------------------
import vllm._custom_ops as vops


def _project_context_kv(self, context_states, num_ctx, num_layers, nkv, hd):
    normed = torch.empty_like(context_states)
    vops.rms_norm(normed, context_states, self._hidden_norm_weight,
                  self._rms_norm_eps)
    save("03_ctxnorm_0", normed)
    all_kv_flat = F.linear(normed, self._fused_kv_weight, self._fused_kv_bias)
    save("04_ctxkv_0", all_kv_flat)
    all_kv = (all_kv_flat.view(num_ctx, num_layers, 2, nkv, hd)
              .permute(2, 1, 0, 3, 4).contiguous())
    return all_kv[0], all_kv[1]


QD.DFlashQwen3Model._project_context_kv = _project_context_kv

_orig_precompute = QD.DFlashQwen3Model.precompute_and_store_context_kv


def precompute_and_store_context_kv(self, context_states, context_positions,
                                    context_slot_mapping=None):
    if context_slot_mapping is not None:
        save("00b_ctxstates", context_states)
        save("00c_ctxpositions", context_positions.to(torch.float32))
    return _orig_precompute(self, context_states, context_positions,
                            context_slot_mapping)


QD.DFlashQwen3Model.precompute_and_store_context_kv = \
    precompute_and_store_context_kv

# ---- stage 5/6: draft block forward ---------------------------------------
def forward(self, input_ids, positions, input_embeds=None):
    if input_embeds is None:
        input_embeds = self.embed_input_ids(input_ids)
    save("07_blockembed", input_embeds)
    save("07b_positions", positions.to(torch.float32))
    save("07c_input_ids", input_ids.to(torch.float32))

    hidden_states = input_embeds
    residual = None
    for li, layer in enumerate(self.layers):
        attn = layer.self_attn
        # Recompute the input norm out-of-band purely to observe it; the real
        # value is produced again inside layer() from the same inputs.
        if residual is None:
            peek = layer.input_layernorm(hidden_states)
        else:
            peek = layer.input_layernorm(hidden_states.clone(),
                                         residual.clone())[0]
        save("08_L%d_innorm" % li, peek)
        qkv = attn.qkv_proj(peek)
        if isinstance(qkv, tuple):
            qkv = qkv[0]
        save("09_L%d_qkv" % li, qkv)

        _layer_ix[0] = li
        hidden_states, residual = layer(positions=positions,
                                        hidden_states=hidden_states,
                                        residual=residual)
        save("15_L%d_mlp" % li, hidden_states)
        save("15b_L%d_residual" % li, residual)
    hidden_states, _ = self.norm(hidden_states, residual)
    save("16_finalnorm", hidden_states)
    return hidden_states


# DFlashLagunaModel subclasses this and is decorated with @support_torch_compile,
# which inspects the inherited forward's annotations at import time. Without them
# the registry scan dies with "No dynamic dimensions found in the forward method".
_orig_model_forward = QD.DFlashQwen3Model.forward
forward.__annotations__ = dict(getattr(_orig_model_forward, "__annotations__", {}))
forward.__name__ = _orig_model_forward.__name__
forward.__qualname__ = _orig_model_forward.__qualname__
# ---- inside-layer taps: post-norm/RoPE Q/K/V, attention out, o_proj out ----
# L0 innorm and qkv already match Grimoire exactly, so the first divergence is
# somewhere between the QKV split and the MLP. These taps bisect that span.
_layer_ix = [0]
_orig_attn_forward = QD.DFlashQwen3Attention.forward


def attn_forward(self, positions, hidden_states):
    li = _layer_ix[0]
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    qs, ks = q.shape, k.shape
    q = self.q_norm(q.view(*qs[:-1], qs[-1] // self.head_dim,
                           self.head_dim)).view(qs)
    k = self.k_norm(k.view(*ks[:-1], ks[-1] // self.head_dim,
                           self.head_dim)).view(ks)
    q, k = self.rotary_emb(positions, q, k)
    save("10_L%d_q" % li, q)
    save("11_L%d_k" % li, k)
    save("12_L%d_v" % li, v)
    attn_output = self.attn(q, k, v)
    save("13_L%d_attn" % li, attn_output)
    output, _ = self.o_proj(attn_output)
    save("14_L%d_o" % li, output)
    return output


attn_forward.__annotations__ = dict(
    getattr(_orig_attn_forward, "__annotations__", {}))
QD.DFlashQwen3Attention.forward = attn_forward

QD.DFlashQwen3Model.forward = forward

# ---- draft logits ----------------------------------------------------------
# ---- how Fusion places the draft query K/V into the draft KV cache --------
# Grimoire writes the query block linearly at [position, position+16) and reads
# it back through an identity block table. Dump Fusion's real query/context
# slot mappings so the port copies the placement instead of assuming it.
_orig_prepare = SPEC.prepare_dflash_inputs


def prepare_dflash_inputs(input_buffers, query_slot_mapping,
                          context_positions, context_slot_mapping, *a, **kw):
    r = _orig_prepare(input_buffers, query_slot_mapping, context_positions,
                      context_slot_mapping, *a, **kw)
    if _armed[0] and _cycle[0] == 0:
        qs = query_slot_mapping[:16]
        save("21_query_slots", qs.to(torch.float32))
        save("22_ctx_positions", context_positions[:64].to(torch.float32))
        cs = context_slot_mapping
        if isinstance(cs, (list, tuple)):
            cs = cs[0]
        if cs is not None:
            save("23_ctx_slots", cs[:64].to(torch.float32))
            print("  fusion dump: ctx slots[:8]=%s ctx slots[56:64]=%s"
                  % (cs[:8].tolist(), cs[56:64].tolist()), flush=True)
        print("  fusion dump: query slots = %s" % qs.tolist(), flush=True)
        # Authoritative: the block size the DRAFT kv-cache group resolves to,
        # independent of the target's --block-size. a[-?] positions vary, so
        # read it off the speculator instead of the call args.
        try:
            sp = _spec_ref[0]
            gids = sp.draft_kv_cache_group_ids
            print("  fusion dump: draft group ids=%s kernel_block_sizes=%s"
                  % (list(gids),
                     [int(sp.block_tables.kernel_block_sizes[g]) for g in gids]),
                  flush=True)
            print("  fusion dump: target kernel_block_sizes=%s"
                  % [int(x) for x in sp.block_tables.kernel_block_sizes],
                  flush=True)
        except Exception as e:
            print("  fusion dump: block size read failed: %r" % (e,), flush=True)
        print("  fusion dump: query positions = %s"
              % input_buffers.positions[:16].tolist(), flush=True)
        print("  fusion dump: query input_ids = %s"
              % input_buffers.input_ids[:16].tolist(), flush=True)
    return r


SPEC.prepare_dflash_inputs = prepare_dflash_inputs

_spec_ref = [None]
_orig_spec_init = SPEC.DFlashSpeculator.__init__


def _spec_init(self, *a, **kw):
    _orig_spec_init(self, *a, **kw)
    _spec_ref[0] = self


SPEC.DFlashSpeculator.__init__ = _spec_init

_orig_generate = SPEC.DFlashSpeculator._generate_draft


def _generate_draft(self, num_reqs, num_tokens_padded, *a, **kw):
    r = _orig_generate(self, num_reqs, num_tokens_padded, *a, **kw)
    if _armed[0] and _cycle[0] == 0:
        n = num_reqs * self.num_speculative_steps
        idx = self.sample_indices[:n]
        save("16b_sampled_indices", idx.to(torch.float32))
        logits = self.model.compute_logits(self.last_draft_hidden[idx]) \
            if hasattr(self, "last_draft_hidden") else None
        if logits is not None:
            save("17_logits", logits)
        toks = self.draft_tokens[:num_reqs].tolist()
        with open(os.path.join(DUMP, "f_17_argmax.txt"), "w") as f:
            f.write(json.dumps(toks))
        print("  fusion dump: draft_tokens %s" % toks, flush=True)
        _cycle[0] = 1
    return r


SPEC.DFlashSpeculator._generate_draft = _generate_draft

# Capture the draft hidden states so the logits above can be recomputed on
# exactly the rows the sampler used.
_orig_run_model = SPEC.DFlashSpeculator._run_model


def _run_model(self, *a, **kw):
    out = _orig_run_model(self, *a, **kw)
    self.last_draft_hidden = out
    return out


SPEC.DFlashSpeculator._run_model = _run_model

# ---- run -------------------------------------------------------------------
llm = LLM(
    model=TARGET,
    dtype="float16",
    max_model_len=int(os.environ.get("MAX_LEN", "2048")),
    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.90")),
    max_num_seqs=16,
    max_num_batched_tokens=16384,
    block_size=64,
    trust_remote_code=True,
    enforce_eager=os.environ.get("ENFORCE_EAGER", "0") == "1",
    speculative_config={"method": "dflash", "model": DRAFT,
                        "num_speculative_tokens": K},
)

if PROMPT_IDS:
    ids = [int(x) for x in PROMPT_IDS.replace("\n", "").split(",") if x.strip()]
    with open(os.path.join(DUMP, "f_00_prompt_ids.txt"), "w") as f:
        f.write(",".join(str(i) for i in ids))
    print("prompt ids: %d tokens" % len(ids), flush=True)
    prompt = TokensPrompt(prompt_token_ids=ids)
else:
    prompt = "Write a haiku about the ocean."

_armed[0] = True
out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))
print("OUTPUT TOKENS:", list(out[0].outputs[0].token_ids))
print("OUTPUT TEXT:", repr(out[0].outputs[0].text))
