#!/usr/bin/env python3
"""Find the FIRST tensor where GRIMOIRE's DFlash draft diverges from the
reference.

This is step 1 of the acceptance investigation in
HANDOFF-2026-08-28-ORNITH-DFLASH.md.  Ornith DFlash commits 2.21 tokens
per step where the reference reports 6.1-7.7, and the handoff is explicit
that kernel tuning cannot create the missing accepted tokens: something
in the draft forward differs, and the only way to find it is to compare
the same tensors from both implementations on ONE step of ONE prompt.

GRIMOIRE already writes its half:

    GRIMOIRE_DFLASH_DUMP=/tmp/dfl ./bin/grimoire -m <target> ... -n 1

writes <dir>/g_<name>.f32, raw little-endian float32, one file per
tensor, on the FIRST draft call only.  tools/dflash_reference_dump.py
writes the reference's half as r_<name>.f32 into the same directory.

This script pairs them up and reports the first stage that disagrees, in
pipeline order -- which is the only comparison that means anything.  A
mismatch at stage 3 makes every later stage meaningless, so reporting
"12 tensors differ" would be noise; what matters is which one went first.

    tools/dflash_compare.py /tmp/dfl

Exit status is 0 when every paired tensor agrees.
"""
import argparse
import os
import re
import sys

import numpy as np

# Pipeline order.  The numeric prefixes GRIMOIRE writes already encode it;
# this table is what each stage MEANS, so the first divergence names a
# suspect rather than a filename.
STAGES = [
    ("01_aux",        "target hidden taps, concatenated per token"),
    ("02_fc",         "draft fc() over the taps"),
    ("03_ctxnorm",    "hidden RMSNorm after fc -- the draft's context input"),
    ("04_ctxkv",      "fused context K/V projection"),
    ("05_ctxk",       "context K after per-layer K-norm and RoPE"),
    ("06_ctxv",       "context V"),
    ("07_blockembed", "embedding of the 1 anchor + N mask query rows"),
    ("16_finalnorm",  "draft final norm, the lm_head input"),
    ("17_logits",     "draft logits"),
    ("18_cache_ctx_k","draft KV cache, context region (FP8 vs BF16 lives here)"),
    ("19_cache_qry_k","draft KV cache, query region, K"),
    ("20_cache_qry_v","draft KV cache, query region, V"),
]


def load(path):
    return np.fromfile(path, dtype="<f4")


def collect(d, prefix):
    """Map bare stage name -> list of (file, chunk-suffix).

    Several stages are dumped once per context chunk and carry a start
    offset in the name (04_ctxkv_0, 04_ctxkv_16, ...).  Compare them
    chunk by chunk, in order, rather than concatenating: a chunk-boundary
    bug is exactly the kind of thing this is looking for.
    """
    out = {}
    for fn in sorted(os.listdir(d)):
        m = re.fullmatch(re.escape(prefix) + r"([0-9]{2}_[a-z_]+?)_?([0-9]*)\.f32", fn)
        if not m:
            continue
        out.setdefault(m.group(1).rstrip("_"), []).append((fn, m.group(2)))
    for k in out:
        out[k].sort(key=lambda t: int(t[1]) if t[1] else -1)
    return out


def compare(a, b, rtol, atol):
    """Return (ok, description). Shape first -- a length mismatch is a
    different bug from a value mismatch and deserves to be said so."""
    if a.size != b.size:
        return False, f"LENGTH {a.size} vs {b.size}"
    if a.size == 0:
        return True, "empty"
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.all():
        return False, (f"non-finite: grimoire {np.count_nonzero(~np.isfinite(a))}, "
                       f"reference {np.count_nonzero(~np.isfinite(b))}")
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = diff / np.maximum(scale, 1e-6)
    worst = int(np.argmax(rel))
    # Cosine too: a pure scale error (a wrong norm convention, a missing
    # 1+w) shows up as cos ~= 1 with a large relative error, which points
    # somewhere completely different from a reordering.
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")
    ok = bool(np.all(diff <= atol + rtol * np.abs(b)))
    desc = (f"max|d| {diff.max():.3e}  max rel {rel.max():.3e} at [{worst}] "
            f"(g {a[worst]:+.6g} vs r {b[worst]:+.6g})  cos {cos:.6f}  "
            f"rms g {np.sqrt((a*a).mean()):.4g} r {np.sqrt((b*b).mean()):.4g}")
    return ok, desc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="directory holding g_*.f32 and r_*.f32")
    ap.add_argument("--rtol", type=float, default=2e-2,
                    help="relative tolerance (default 2e-2: the target runs "
                         "quantized, so exact equality is not the bar)")
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    g = collect(args.dir, "g_")
    r = collect(args.dir, "r_")
    if not g:
        print(f"no g_*.f32 in {args.dir} -- run GRIMOIRE with "
              f"GRIMOIRE_DFLASH_DUMP={args.dir} first", file=sys.stderr)
        return 2
    if not r:
        print(f"no r_*.f32 in {args.dir} -- run tools/dflash_reference_dump.py "
              f"first", file=sys.stderr)
        return 2

    print(f"{'stage':<16} {'result':<8} detail")
    print("-" * 100)
    first_bad = None
    for stage, meaning in STAGES:
        gf, rf = g.get(stage), r.get(stage)
        if not gf and not rf:
            continue
        if not gf or not rf:
            side = "reference" if gf else "grimoire"
            print(f"{stage:<16} {'MISSING':<8} only {side} dumped it")
            continue
        if len(gf) != len(rf):
            print(f"{stage:<16} {'CHUNKS':<8} grimoire {len(gf)} chunks, "
                  f"reference {len(rf)}")
            first_bad = first_bad or (stage, meaning)
            continue
        for (gn, gc), (rn, rc) in zip(gf, rf):
            a = load(os.path.join(args.dir, gn))
            b = load(os.path.join(args.dir, rn))
            ok, desc = compare(a, b, args.rtol, args.atol)
            label = stage + (f"[{gc}]" if gc else "")
            print(f"{label:<16} {'ok' if ok else 'DIFFERS':<8} {desc}")
            if not ok and first_bad is None:
                first_bad = (stage, meaning)

    print("-" * 100)
    if first_bad is None:
        print("every paired tensor agrees.")
        print("If acceptance is still low, the divergence is NOT in the draft")
        print("forward -- look at the verifier, the sampled rows, or the")
        print("target's own precision (the drafter was trained against a BF16")
        print("target; GRIMOIRE runs it quantized).")
        return 0
    stage, meaning = first_bad
    print(f"FIRST DIVERGENCE: {stage}  --  {meaning}")
    print()
    print("Fix only this one, then re-run this comparison.  Every later stage")
    print("is downstream of it and its numbers mean nothing until this agrees.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
