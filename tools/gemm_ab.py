"""Run the EXACT five Muse projection shapes through Fusion's oneDNN W4A16 op
(torch.ops._xpu_C.int4_gemm_w4a16) at the prefill batch size, so the numbers
are directly comparable to Grimoire's per-stage prefill profile.

Shapes are derived from the live Muse config: hidden 6656, inter 19968,
32 q heads / 2 kv heads / head_dim 128, 52 layers.
"""
import time, torch, json
import vllm._custom_ops  # registers torch.ops._xpu_C

M = 3672
H, I, GS = 6656, 19968, 128
QW = 32 * 128
QKV = (32 + 2 * 2) * 128

SHAPES = [
    ("qkv proj",     QKV,     H),
    ("o_gate proj",  QW,      H),
    ("o_proj",       H,       QW),
    ("FFN gate_up",  2 * I,   H),
    ("FFN down",     H,       I),
]
LAYERS = 52
dev = "xpu"

print("Fusion oneDNN W4A16, M=%d, per-call and x%d layers" % (M, LAYERS))
print("%-14s %7s %7s %10s %12s %12s" % ("stage", "N", "K", "ms/call", "ms x52", "TFLOP/s"))
total52 = 0.0
rows = {}
for name, N, K in SHAPES:
    a = torch.randn((M, K), device=dev, dtype=torch.float16)
    b_storage = torch.randint(-(2**31), 2**31 - 1, (N, K // 8), device=dev, dtype=torch.int32)
    b = b_storage.t()
    s = torch.full((K // GS, N), 1.0 / 128.0, device=dev, dtype=torch.float16)
    zp = torch.tensor([8], device=dev, dtype=torch.int8)

    def run():
        return torch.ops._xpu_C.int4_gemm_w4a16(a, b, None, s, zp, GS, None)

    for _ in range(3):
        run()
    torch.xpu.synchronize()
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        run()
        torch.xpu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    ms = sum(ts[2:8]) / 6
    flops = 2.0 * M * N * K
    tfs = flops / (ms * 1e9)
    print("%-14s %7d %7d %10.3f %12.1f %12.1f" % (name, N, K, ms, ms * LAYERS, tfs))
    total52 += ms * LAYERS
    rows[name] = ms * LAYERS
    del a, b_storage, b, s, zp
    torch.xpu.empty_cache()

print("%-14s %7s %7s %10s %12.1f" % ("TOTAL", "", "", "", total52))
print("JSON=" + json.dumps(rows))
