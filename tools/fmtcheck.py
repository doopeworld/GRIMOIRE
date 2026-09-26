# fmtcheck.py -- is our reading of a quantized checkpoint right?  Dequantize
# tensors from the NVFP4/FP8 checkpoint in numpy and compare them with the
# same tensors in the original BF16 checkpoint (relative L2 error).
import json, struct, glob, sys
import numpy as np
Q, B = sys.argv[1], sys.argv[2]
def index(d):
    m = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        b = open(f, "rb"); n = struct.unpack("<Q", b.read(8))[0]; h = json.loads(b.read(n))
        for k, v in h.items():
            if k != "__metadata__": m[k] = (f, 8 + n, v)
    return m
qi, bi = index(Q), index(B)
def raw(m, k):
    f, base, v = m[k]; a, b = v["data_offsets"]
    with open(f, "rb") as fh:
        fh.seek(base + a); return v, np.frombuffer(fh.read(b - a), dtype=np.uint8)
def bf16(u8): return (u8.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
def e4m3(u8):
    u = u8.astype(np.int32); s = np.where(u & 0x80, -1.0, 1.0); e = (u >> 3) & 15; m = u & 7
    v = np.where(e == 0, m / 8.0 * 2.0**-6, (1 + m / 8.0) * 2.0**(e - 7.0))
    return (s * v).astype(np.float32)
def f32(u8): return u8.view(np.float32)
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
def get_bf16(name):
    if name in bi:
        v, u = raw(bi, name); return bf16(u).reshape(v["shape"])
    # fused BF16 experts: experts.gate_up_proj [E][2I][H], experts.down_proj [E][H][I]
    import re
    m = re.match(r"(.*\.experts)\.(\d+)\.(gate|up|down)_proj\.weight", name)
    pre, e, which = m.group(1), int(m.group(2)), m.group(3)
    if which == "down":
        v, u = raw(bi, pre + ".down_proj"); E_, H_, I_ = v["shape"]
        return bf16(u).reshape(E_, H_, I_)[e]
    v, u = raw(bi, pre + ".gate_up_proj"); E_, I2, H_ = v["shape"]
    t = bf16(u).reshape(E_, I2, H_)[e]
    return t[: I2 // 2] if which == "gate" else t[I2 // 2:]
def scal(m, k):
    v, u = raw(m, k)
    return float(bf16(u)[0]) if v["dtype"] == "BF16" else float(f32(u)[0])
def rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
for name in sys.argv[3:]:
    base = name[:-len(".weight")]
    v, u = raw(qi, name); ref = get_bf16(name)
    if v["dtype"] == "U8":                       # modelopt NVFP4
        N, Kh = v["shape"]; K = 2 * Kh
        lo = E2M1[u & 15]; hi = E2M1[u >> 4]
        w = np.empty((N, K), np.float32); w[:, 0::2] = lo.reshape(N, Kh); w[:, 1::2] = hi.reshape(N, Kh)
        sv, su = raw(qi, base + ".weight_scale"); sc = e4m3(su).reshape(sv["shape"])
        g2 = scal(qi, base + ".weight_scale_2")
        wb = w * np.repeat(sc, 16, axis=1)
        print(f"{name}: NVFP4 {N}x{K} g2={g2:.4g}  mul err {rel(wb * g2, ref):.4f}  div err {rel(wb / g2, ref):.4f}"
              f"  swapped-nibble mul err {rel((w[:, ::-1].reshape(N, K//2, 2)[:, :, ::-1].reshape(N, K)[:, ::-1] * np.repeat(sc,16,axis=1)) * g2, ref):.4f}")
    elif v["dtype"] == "F8_E4M3":
        w = e4m3(u).reshape(v["shape"])
        sk = base + ".weight_scale" if base + ".weight_scale" in qi else base + ".weight_scale_inv"
        sv, su = raw(qi, sk)
        s = bf16(su) if sv["dtype"] == "BF16" else f32(su)
        if s.size == 1: wd = w * s[0]; kind = "per-tensor"
        elif s.size == w.shape[0]: wd = w * s.reshape(-1, 1); kind = "per-channel"
        else:
            S = s.reshape(sv["shape"]); wd = w * np.repeat(np.repeat(S, 128, 0), 128, 1)[:w.shape[0], :w.shape[1]]; kind = "block128"
        print(f"{name}: FP8 {kind} {v['shape']}  err {rel(wd, ref):.4f}")
