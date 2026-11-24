# B70-native image checkpoint — 2026-08-29

## Result

Built and validated `grimoire:b70-native`, a B70-only runtime image derived
from the working `my-vllm-xpu:latest` recipe.

- Image size: 2,312,298,326 bytes (2.31 GB), down from 24,067,460,705 bytes.
- No Python executable.
- No Torch/C10 libraries.
- No CUDA or HIP Unified Runtime adapters.
- Contains only the native Grimoire CLI and raw Xe2 grouped-GEMM, attention,
  and GDN bridges plus Intel SYCL/Unified Runtime/Level Zero dependencies.
- Compiles against the live customized vLLM XPU kernel tree supplied as a
  read-only Docker build context. Kernel math and launch policy are not
  rewritten by the image recipe.
- Uses `intel_gpu_bmg_g31` AOT compilation.

The Tower requires the exact known-working user-mode driver stack from
`my-vllm-xpu:latest` (compute-runtime 39122.11 and IGC 2.38.2). The newer PPA
pair installed successfully but enumerated zero B70 devices. The final image
copies only those verified driver shared libraries from the discarded builder
stage. Unified Runtime also requires the UMF directory in the dynamic-loader
configuration; without it the Level Zero adapter silently fails to load.

## Validation

Muse-Glimmer-30B-MXFP4 on renderD128:

```text
device: Intel(R) Arc(TM) Pro B70 Graphics | driver 1.15.39122+11 | 256 EUs | 31.9 GiB
checkpoint resolved: 52 layers, vocab 202048, hidden 6656
FULL E2E PP ONLY: PASS, 64 tokens in 186.2 ms -> 343.7 tok/s
FULL E2E PP ONLY: PASS, 4096 tokens in 8316.1 ms -> 492.5 tok/s
```

The packaging goal is complete; performance parity is not. The 492.5 PP
result proves that deleting Python/Torch alone does not reproduce vLLM's
~2100 PP. Grimoire still uses its own per-layer execution graph and command
schedule above the raw kernels. Continue by reproducing vLLM's batching,
fusion, and submission schedule in native C++, while keeping the customized
Xe2 kernels unchanged.
