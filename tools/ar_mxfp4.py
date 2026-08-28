# AutoRound -> MXFP4 (weight-only) compressed-tensors export for the B70.
import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

SRC   = os.environ.get("AR_SRC", "/models/Qwen3.8-27B")
OUT   = os.environ.get("AR_OUT", "/models/Qwen3.8-27B-MXFP4-AutoRound")
DEV   = os.environ.get("AR_DEVICE", "xpu")
ITERS = int(os.environ.get("AR_ITERS", "0"))       # 0 = RTN export, ~10 min
NSAMP = int(os.environ.get("AR_NSAMPLES", "128"))
SEQLEN= int(os.environ.get("AR_SEQLEN", "2048"))
FMT   = os.environ.get("AR_FORMAT", "llm_compressor")

print(f"[ar] src={SRC} out={OUT} dev={DEV} iters={ITERS} fmt={FMT}", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    SRC, torch_dtype=torch.bfloat16, trust_remote_code=True,
    low_cpu_mem_usage=True, device_map="cpu")
print(f"[ar] loaded in {time.time()-t0:.0f}s", flush=True)

# AutoRound 0.14.2 bug: it reverses transformers' _checkpoint_conversion_mapping
# and the reversed target keeps a dangling \\1 group reference, so re.subn raises
# "invalid group reference 1" on the first packed tensor.  We rename nothing, so
# make the revert a no-op.  shard_writer imported the symbol directly, so the
# module-level binding must be patched there too.
import auto_round.utils.common as _arc
import auto_round.compressors.shard_writer as _sw
_noop = lambda name, mapping=None, *a, **k: name
_arc.revert_checkpoint_conversion_mapping = _noop
_sw.revert_checkpoint_conversion_mapping = _noop
for _o in (model, getattr(model, "config", None)):
    if _o is not None and hasattr(_o, "_checkpoint_conversion_mapping"):
        try: _o._checkpoint_conversion_mapping = {}
        except Exception: pass
print("[ar] neutralised checkpoint_conversion_mapping (module + shard_writer)", flush=True)

# Build layer_config from REAL module names -- no regex, AutoRound re.sub-es keys.
# These tensors must stay BF16: the delta-rule gates are exponentiated into alpha
# and feed the matrix inverted by chunk_inverse_opt_kernel; the MoE router and
# shared-expert gate pick experts.  Established 2026-08-25; Intel's own
# int4-AutoRound config pins in_proj_a to bits:16 for the same reason.
KEEP = ("linear_attn.in_proj_a", "linear_attn.in_proj_b",
        "mlp.gate", "mlp.shared_expert_gate")
BF16 = {"bits": 16, "act_bits": 16, "data_type": "float"}
layer_config = {}
for name, mod in model.named_modules():
    if not isinstance(mod, torch.nn.Linear):
        continue
    if name.endswith("lm_head") or name == "lm_head" or any(name.endswith(k) for k in KEEP):
        layer_config[name] = dict(BF16)
print(f"[ar] pinned {len(layer_config)} tensors to BF16", flush=True)
for n in list(layer_config)[:4]:
    print("   ", n, flush=True)

ar = AutoRound(model, tokenizer=tok, scheme="MXFP4",
               act_bits=16,                    # weight-only; activations stay BF16
               iters=ITERS, nsamples=NSAMP, seqlen=SEQLEN,
               low_gpu_mem_usage=True, device=DEV, layer_config=layer_config)
print("[ar] quantizing...", flush=True)
ar.quantize_and_save(output_dir=OUT, format=FMT)
print(f"[ar] DONE in {time.time()-t0:.0f}s -> {OUT}", flush=True)
