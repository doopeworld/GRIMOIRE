"""Compare Grimoire g_*.f32 dumps against Fusion f_*.f32 dumps, in stage order.

Grimoire walks the 64-token context in 16-row chunks (g_01_aux_0/16/32/48),
while Fusion projects all 64 rows in one call, so the chunked stages are
reassembled here before comparison.
"""
import os, sys, glob, re
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/dump"
NL, KVW = 5, 1024          # draft layers, kv width per layer

# stage -> axis along which its 16-row chunks concatenate
CHUNKED = {"01_aux": 0, "02_fc": 0, "03_ctxnorm": 0, "04_ctxkv": 0,
           "05_ctxk": 1, "06_ctxv": 1}


def load(p):
    return np.fromfile(p, dtype=np.float32)


def gimoire_stage(stem):
    """Return Grimoire's tensor for a stage, reassembling chunks if needed."""
    base = re.sub(r"_\d+$", "", stem)
    if base not in CHUNKED:
        p = os.path.join(D, "g_%s.f32" % stem)
        return load(p) if os.path.exists(p) else None
    parts = sorted(glob.glob(os.path.join(D, "g_%s_*.f32" % base)),
                   key=lambda p: int(re.search(r"_(\d+)\.f32$", p).group(1)))
    if not parts:
        return None
    axis = CHUNKED[base]
    if axis == 0:
        return np.concatenate([load(p) for p in parts])
    arrs = [load(p).reshape(NL, -1, KVW) for p in parts]
    return np.concatenate(arrs, axis=1).ravel()


fusion = sorted(os.path.basename(p)[2:-4]
                for p in glob.glob(os.path.join(D, "f_*.f32")))

print("%-22s %10s %10s %12s %12s %10s" %
      ("stage", "g_len", "f_len", "max_abs", "rel_l2", "cos"))
print("-" * 82)

first_bad = None
for stem in fusion:
    f = load(os.path.join(D, "f_%s.f32" % stem))
    g = gimoire_stage(stem)
    if g is None:
        print("%-22s %10s %10d %12s" % (stem, "-", f.size, "NO GRIMOIRE"))
        continue
    if g.size != f.size:
        print("%-22s %10d %10d %12s" % (stem, g.size, f.size, "SHAPE"))
        if first_bad is None:
            first_bad = stem
        continue
    rel = np.linalg.norm(g - f) / (np.linalg.norm(f) or 1.0)
    gn, fn = np.linalg.norm(g), np.linalg.norm(f)
    cos = float(g @ f / (gn * fn)) if gn and fn else float("nan")
    print("%-22s %10d %10d %12.5g %12.5g %10.6f" %
          (stem, g.size, f.size, np.abs(g - f).max(), rel, cos))
    if rel > 5e-2 and first_bad is None:
        first_bad = stem

print("-" * 82)
if not first_bad:
    print("ALL COMPARED STAGES MATCH within tolerance")
    sys.exit(0)

print("FIRST DIVERGENT STAGE: %s" % first_bad)
g, f = gimoire_stage(first_bad), load(os.path.join(D, "f_%s.f32" % first_bad))
if g is not None and g.size == f.size:
    i = int(np.argmax(np.abs(g - f)))
    print("  worst element %d: grimoire=%.6g fusion=%.6g" % (i, g[i], f[i]))
    print("  grimoire[:8] %s" % np.array2string(g[:8], precision=5))
    print("  fusion  [:8] %s" % np.array2string(f[:8], precision=5))
    for rows in (64, 16, 5):
        if g.size % rows == 0:
            e = np.abs(g - f).reshape(rows, -1).max(axis=1)
            print("  max abs err per row (%d): %s" %
                  (rows, np.array2string(e, precision=4, threshold=100)))
            break
