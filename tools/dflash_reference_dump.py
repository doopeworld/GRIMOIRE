#!/usr/bin/env python3
"""Dump the reference DFlash draft tensors for one step, to diff against
GRIMOIRE's.

Pairs with tools/dflash_compare.py.  GRIMOIRE writes g_<name>.f32 under
GRIMOIRE_DFLASH_DUMP; this writes r_<name>.f32 with the same names and
the same element order, so the comparator can find the first stage that
disagrees.

WHY: Ornith DFlash commits 2.21 tokens per step against a reference that
reports 6.1-7.7.  HANDOFF-2026-08-28-ORNITH-DFLASH.md says plainly that
kernel tuning cannot create the missing accepted tokens, and that the
next step is a one-step tensor comparison.  This is the reference half of
that comparison.

RUN IT IN THE vLLM CONTAINER on the Tower -- it imports vllm's own
qwen3_dflash module so the numbers come from the reference
implementation, not from a re-derivation of it.  Re-deriving the forward
pass by hand is exactly how earlier sessions lost days to a wrong norm
order.

    docker run --rm --entrypoint bash \
      -v /mnt/storage/isos/grimoire-fuse:/grimoire \
      -v /mnt/storage/Models:/models \
      --device /dev/dri/renderD129:/dev/dri/renderD129 \
      my-vllm-xpu:latest -lc \
      'python3 /grimoire/tools/dflash_reference_dump.py \
         --target /models/<ornith> --draft /models/<z-lab-dflash> \
         --out /tmp/dfl --prompt "Explain why the sky is blue."'

then:

    tools/dflash_compare.py /tmp/dfl

HONESTY NOTE: this script has NEVER BEEN RUN.  There is no GPU and no
vLLM in the container it was written in.  The tensor names, shapes and
element order below were taken from ref/qwen3_dflash.py and from the
dump sites in src/grimoire.cpp, but the vLLM entry points will very
likely need adjusting on first contact.  Treat it as a starting point
that encodes WHICH tensors to capture and in what layout -- that part is
researched -- not as working code.
"""
import argparse
import os
import sys

import numpy as np
import torch


def w(out_dir, name, t):
    """Write one tensor as raw little-endian float32, flattened.

    Flattening order must match GRIMOIRE's.  Where GRIMOIRE dumps a
    per-layer stack it is layer-major ([L][tokens][kv]); where it dumps a
    per-token buffer it is token-major.  The comments at each call site
    below say which.
    """
    a = t.detach().to(torch.float32).cpu().contiguous().numpy().ravel()
    a.astype("<f4").tofile(os.path.join(out_dir, f"r_{name}.f32"))
    print(f"  reference dump: {name} [{a.size}]", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="target model dir")
    ap.add_argument("--draft", required=True, help="DFlash drafter dir")
    ap.add_argument("--out", required=True, help="same dir GRIMOIRE dumped into")
    ap.add_argument("--prompt", required=True,
                    help="MUST be the identical prompt GRIMOIRE was given")
    ap.add_argument("--num-spec", type=int, default=15,
                    help="mask rows; GRIMOIRE's k (default 15)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- the reference, not a re-derivation of it --------------------
    from vllm import LLM
    from vllm.model_executor.models.qwen3_dflash import (  # noqa: F401
        dflash_has_any_non_causal,
    )

    llm = LLM(
        model=args.target,
        speculative_config={"model": args.draft, "num_speculative_tokens": args.num_spec},
        dtype="bfloat16",          # the drafter was TRAINED against a bf16 target
        enforce_eager=True,        # graphs would hide the intermediates
    )

    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    spec = runner.speculator                     # DFlashSpeculator
    draft = spec.draft_model                     # Qwen3DFlashForCausalLM
    model = draft.model

    taps = []
    ctx = {}

    # 01_aux: the eight target hidden taps, concatenated per token.
    # GRIMOIRE dumps token-major [tokens][n_taps * H].
    def tap_hook(_m, _i, o):
        taps.append((o[0] if isinstance(o, tuple) else o).detach())

    tap_ids = getattr(draft.config, "target_layer_ids", None) or \
              getattr(draft.config, "aux_hidden_state_layer_ids", None)
    if tap_ids is None:
        raise SystemExit("cannot find target_layer_ids on the draft config -- "
                         "GRIMOIRE reads the same key, so check both")
    target_layers = runner.model.model.layers
    handles = [target_layers[i].register_forward_hook(tap_hook) for i in tap_ids]

    # 02_fc / 03_ctxnorm: fc() over the taps, then the hidden RMSNorm that
    # produces the draft's context input.
    model.fc.register_forward_hook(
        lambda _m, _i, o: ctx.__setitem__("02_fc", o.detach()))
    if hasattr(model, "hidden_norm"):
        model.hidden_norm.register_forward_hook(
            lambda _m, _i, o: ctx.__setitem__("03_ctxnorm", o.detach()))

    # 04/05/06: the fused context K/V projection, then K after per-layer
    # K-norm and RoPE.  Capture by wrapping the reference's own method so
    # the layout is whatever IT produces -- layer-major [L][num_ctx][kv].
    orig_project = model._project_context_kv
    orig_normk = model._normalize_context_k

    def project(context_states, num_ctx, L, nkv, hd):
        k, v = orig_project(context_states, num_ctx, L, nkv, hd)
        # 04 is all_kv_flat as the fused GEMM produced it, BEFORE the
        # model's own permute to layer-major: [num_ctx, L, 2, nkv, hd].
        # That is the layout GRIMOIRE dumps on both the fused (Muse) and
        # the per-layer path, so undo the permute rather than invent a
        # third layout.
        kv = torch.stack([k, v], dim=0)          # [2, L, num_ctx, nkv, hd]
        ctx["04_ctxkv"] = kv.permute(2, 1, 0, 3, 4).reshape(num_ctx, -1).detach()
        ctx["06_ctxv"] = v.detach()              # [L, num_ctx, nkv, hd]
        return k, v

    def normk(all_k):
        out = orig_normk(all_k)
        # GRIMOIRE's 05_ctxk is AFTER k-norm AND RoPE.  The reference
        # rotates all_k_flat -- a VIEW of what this returns -- in place
        # right after this call, so holding the tensor WITHOUT cloning it
        # yields the post-RoPE values by the time it is written out.  Do
        # not .clone() here: a clone freezes the pre-RoPE values and there
        # is nothing on GRIMOIRE's side to compare those against -- its
        # K-norm and rotation are one fused kernel.
        ctx["05_ctxk"] = out
        return out

    model._project_context_kv = project
    model._normalize_context_k = normk

    out = llm.generate([args.prompt], sampling_params=None)
    for h in handles:
        h.remove()

    if taps:
        # token-major [tokens][n_taps * H]
        w(args.out, "01_aux_0", torch.cat([t for t in taps], dim=-1))
    for name in ("02_fc", "03_ctxnorm", "04_ctxkv", "05_ctxk", "06_ctxv"):
        if name in ctx:
            w(args.out, name + "_0", ctx[name])

    print(f"\nwrote reference tensors to {args.out}", file=sys.stderr)
    print(f"now: tools/dflash_compare.py {args.out}", file=sys.stderr)
    print(f"generated: {out[0].outputs[0].text!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
