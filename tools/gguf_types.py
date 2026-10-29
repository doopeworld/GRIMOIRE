#!/usr/bin/env python3
# gguf_types.py FILE.gguf [blk.N ...] -- the ggml type of every tensor (header only, no deps).
import struct, sys
T = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
     11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
     19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
     27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16"}
f = open(sys.argv[1], "rb")
def u32(): return struct.unpack("<I", f.read(4))[0]
def u64(): return struct.unpack("<Q", f.read(8))[0]
def s(): n = u64(); return f.read(n).decode("utf-8", "replace")
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def val(t):
    if t == 8: return s()
    if t == 9:
        et = u32(); n = u64()
        for _ in range(n): val(et)
        return None
    f.read(SZ[t]); return None
assert f.read(4) == b"GGUF"; ver = u32(); nt = u64(); nkv = u64()
for _ in range(nkv): s(); val(u32())
want = sys.argv[2:]
for _ in range(nt):
    name = s(); nd = u32(); dims = [u64() for _ in range(nd)]; t = u32(); u64()
    if not want or any(name.startswith(w + ".") for w in want):
        print(f"{name:40s} {T.get(t, t):8s} {'x'.join(map(str, dims))}")
