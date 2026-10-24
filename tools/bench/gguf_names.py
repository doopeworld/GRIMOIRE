#!/usr/bin/env python3
# gguf_names.py FILE... -- read-only: GGUF header -> tensor names / MTP-related tensors and metadata
import struct, sys
def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
def rstr(f):
    n = rd(f, "Q"); return f.read(n).decode("utf-8", "replace")
SZ = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
def rval(f, t):
    if t == 8: return rstr(f)
    if t == 9:
        et = rd(f, "I"); n = rd(f, "Q")
        if et == 8: return [rstr(f) for _ in range(n)][:4]
        f.seek(struct.calcsize("<" + SZ[et]) * n, 1); return "[%d x type%d]" % (n, et)
    return rd(f, SZ[t])
for path in sys.argv[1:]:
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver = rd(f, "I"); nt = rd(f, "Q"); nkv = rd(f, "Q")
        meta = {}
        for _ in range(nkv):
            k = rstr(f); t = rd(f, "I"); meta[k] = rval(f, t)
        names = []
        for _ in range(nt):
            nm = rstr(f); nd = rd(f, "I"); f.seek(8 * nd, 1); rd(f, "I"); rd(f, "Q"); names.append(nm)
    arch = meta.get("general.architecture")
    mtp = [n for n in names if any(k in n.lower() for k in ("mtp", "nextn", "eh_proj", "enorm", "hnorm", "shared_head"))]
    blks = sorted({int(n.split(".")[1]) for n in names if n.startswith("blk.")})
    print("== %s\n   arch %s, %d tensors, blocks %s..%s, block_count=%s" % (path.split("/")[-1], arch, nt,
          blks[0] if blks else "-", blks[-1] if blks else "-", meta.get("%s.block_count" % arch)))
    for k, v in meta.items():
        if "nextn" in k or "mtp" in k: print("   meta %s = %s" % (k, v))
    print("   MTP-like tensors: %d %s" % (len(mtp), mtp[:6]))
