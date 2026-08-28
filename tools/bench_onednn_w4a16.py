import time
import torch
import vllm._custom_ops  # registers torch.ops._xpu_C

M, N, K, GS = 4096, 8192, 2048, 128
dev = "xpu"
a = torch.randn((M, K), device=dev, dtype=torch.float16)
# GPTQ NT contract: eight K nibbles per int32, logical [K, N].
b_storage = torch.randint(
    -(2**31), 2**31 - 1, (N, K // 8), device=dev, dtype=torch.int32
)
b = b_storage.t()  # [K/8,N], stride[-2] == 1: oneDNN GPTQ NT contract
s = torch.full((K // GS, N), 1.0 / 128.0, device=dev, dtype=torch.float16)
zp = torch.tensor([8], device=dev, dtype=torch.int8)

def run():
    return torch.ops._xpu_C.int4_gemm_w4a16(a, b, None, s, zp, GS, None)

for _ in range(3):
    out = run()
torch.xpu.synchronize()
times = []
for _ in range(10):
    t0 = time.perf_counter()
    out = run()
    torch.xpu.synchronize()
    times.append((time.perf_counter() - t0) * 1000.0)
times.sort()
ms = sum(times[2:8]) / 6
tops = 2.0 * M * N * K / (ms * 1e9)
print(f"oneDNN W4A16 {M}x{N}x{K}: {ms:.3f} ms, {tops:.1f} TOPS, all={times}")
print(float(out[0, 0]))
