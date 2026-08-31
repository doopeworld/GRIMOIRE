"""Compare Grimoire g_*.f32 dumps against Fusion f_*.f32 dumps, in stage order.

Prints one line per stage and stops describing detail after the first stage
that is materially unequal, since every later stage inherits that error.
"""
import os, sys, glob
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/dump"


def load(p):
    return np.fromfile(p, dtype=np.float32)


names = sorted({os.path.basename(p)[2:-4]
                for p in glob.glob(os.path.join(D, "g_*.f32"))} |
               {os.path.basename(p)[2:-4]
                for p in glob.glob(os.path.join(D, "f_*.f32"))})

# Draft-block stages are only meaningful once the context stages agree.
print("%-22s %10s %10s %12s %12s %10s" %
      ("stage", "g_len", "f_len", "max_abs", "rel_l2", "cos"))
print("-" * 80)

first_bad = None
for n in names:
    gp = os.path.join(D, "g_%s.f32" % n)
    fp = os.path.join(D, "f_%s.f32" % n)
    if not os.path.exists(gp) or not os.path.exists(fp):
        print("%-22s %10s %10s %12s" % (
            n,
            len(load(gp)) if os.path.exists(gp) else "-",
            len(load(fp)) if os.path.exists(fp) else "-",
            "MISSING"))
        continue
    g, f = load(gp), load(fp)
    if g.size != f.size:
        print("%-22s %10d %10d %12s" % (n, g.size, f.size, "SHAPE"))
        if first_bad is None:
            first_bad = n
        continue
    d = np.abs(g - f)
    denom = np.linalg.norm(f) or 1.0
    rel = np.linalg.norm(g - f) / denom
    gn, fn = np.linalg.norm(g), np.linalg.norm(f)
    cos = float(g @ f / (gn * fn)) if gn and fn else float("nan")
    print("%-22s %10d %10d %12.5g %12.5g %10.6f" %
          (n, g.size, f.size, d.max(), rel, cos))
    # fp16 round-trip noise floor is ~1e-3 relative; flag well above it
    if rel > 5e-2 and first_bad is None:
        first_bad = n

print("-" * 80)
if first_bad:
    print("FIRST DIVERGENT STAGE: %s" % first_bad)
    g = load(os.path.join(D, "g_%s.f32" % first_bad))
    f = load(os.path.join(D, "f_%s.f32" % first_bad))
    if g.size == f.size:
        i = int(np.argmax(np.abs(g - f)))
        print("  worst element %d: grimoire=%.6g fusion=%.6g" % (i, g[i], f[i]))
        print("  grimoire[:8] %s" % np.array2string(g[:8], precision=5))
        print("  fusion  [:8] %s" % np.array2string(f[:8], precision=5))
        # per-row error helps localise to a token or a head
        for rows in (16, 64):
            if g.size % rows == 0:
                m = np.abs(g - f).reshape(rows, -1).max(axis=1)
                print("  max abs err per row (%d rows): %s" %
                      (rows, np.array2string(m, precision=4, threshold=80)))
                break
else:
    print("ALL COMPARED STAGES MATCH within tolerance")
