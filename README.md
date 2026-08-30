# b70 — bare-metal LLM inference engine for Intel Arc Pro B70 (Battlemage G31)

An implementation of the blueprint, extended to the seven weight formats you asked for,
with the blueprint's numerical and API bugs fixed. No Python, no PyTorch, no OpenVINO —
SYCL, USM device pointers, and AOT-compiled kernels.

```
make test          # host validation, no GPU needed
./build_b70.sh     # AOT GPU build, needs oneAPI DPC++ + ocloc
```

---

## Format support

All seven share one decode path (`include/b70/formats.hpp`), compiled into both the
GPU kernels and the host tests, so what gets validated is what runs.

| Format | Element | Scale | bits/elem | XMX pipe | rel-RMS¹ | 7B weights |
|---|---|---|---|---|---|---|
| `BF16` | bfloat16 | — | 16.00 | bf16 | 0.0017 | 14.0 GB |
| `FP8_E4M3` | OCP E4M3 | fp32 / channel | 8.00 | bf16 | 0.0264 | 7.0 GB |
| `FP8_E5M2` | OCP E5M2 | fp32 / channel | 8.00 | bf16 | 0.0526 | 7.0 GB |
| `INT8` | int8 sym | fp32 / channel | 8.00 | **int8** | 0.0080 | 7.0 GB |
| `INT4` | uint4 asym | bf16+zero / 128 | 4.19 | **int8** | 0.1009 | 3.7 GB |
| `MXFP8` | OCP E4M3 | E8M0 / 32 | 8.25 | bf16 | 0.0292 | 7.2 GB |
| `MXFP4` | OCP E2M1 | E8M0 / 32 | 4.25 | bf16 | 0.1143 | 3.7 GB |

¹ measured, 256×1024 gaussian weights, `make test`.

**Two things worth knowing before you pick a format.**

*MXFP4 cannot use the int8 XMX path.* E2M1 is a floating-point grid
`{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`. The blueprint unpacks MXFP4 nibbles into
`int8_t` operands and feeds them to an INT8 DPAS — that collapses 0.5 and 1.5 onto
integers and destroys the format. MXFP4 here gets 4-bit *memory traffic* with bf16-rate
*math*. If you want 4-bit weights **and** the 367 TOPS int8 rate, use `INT4`.

*Block scaling is what you're actually buying.* On clean gaussian weights INT4 (0.101)
slightly beats MXFP4 (0.114). Put one 100× outlier in each row and the ranking inverts
hard — a per-channel scale is hostage to its worst element, a per-32 scale is not:

```
              clean     one outlier per row
  int8       0.0080  ->  0.0499     (6x worse)
  int4       0.1009  ->  0.1135
  mxfp8      0.0292  ->  0.0064     (better — outlier gets its own block)
  mxfp4      0.1143  ->  0.0595
```

Real transformer weights have outlier channels. That is the entire argument for MX.

---

## What was wrong in the blueprint

Working through it, in rough order of severity.

**1. The FlashDecoding online softmax is broken.** Two independent bugs. When the
running maximum moves, the code rescales the denominator but *not* the output
accumulator — `acc` keeps terms normalized against a stale maximum. And each lane keeps
a *private* maximum, so the final `reduce_over_group` sums partial results that were
normalized against different constants.

`tests/test_attention.cpp` runs both recurrences against a materialized softmax:

```
  seq_len  spread          corrected      blueprint
  512      1.0             5.397e-07      5.332e+00
  4096     8.0             1.613e-06      4.392e+00
```

Not marginally wrong — **150–700% wrong**. Fixed in `src/attention.cpp`: the max is
reduced across the sub-group first so `(m, l)` stay uniform, and the accumulator is
rescaled by `exp(m_old − m_new)` alongside the denominator.

**2. `local_b` is never initialized.** The B fragment is loaded from uninitialized
stack memory, so the GEMM multiplies weights by garbage. Relatedly, `x_fp16` — the
activations — is passed in and never read at all. The kernel as written cannot compute
a linear layer.

**3. `MXFP4Block` is the wrong size.** `{uint8_t elements; uint8_t scale;}` is one byte
of payload, but the code indexes blocks as `[...]/32`, implying 32 elements per block.
A real MX block is 16 bytes of payload + 1 byte E8M0. Every element in a block also
decodes from the same byte, so 32 weights collapse to 2 distinct values.

**4. MXFP4 is decoded as if it were an integer format.** `(raw_4bit & 0x0F)` reads a
nibble as an unsigned integer 0–15. E2M1 is sign-magnitude float: bit 3 is the sign, and
the low 3 bits index `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. The sign is silently dropped and
every magnitude is wrong.

**5. The AOT target is the wrong architecture.**

```bash
# blueprint
-fsycl-targets=intel_gpu_xe_hpg       # Alchemist / A-series
-Xsycl-target-backend "-device bmg"   # conflicts with the triple above

# correct for Arc Pro B70
-fsycl-targets=intel_gpu_bmg_g31      # Battlemage G31
```

`intel_gpu_bmg_g21` is the *smaller* Battlemage die (B580, Arc Pro B50/B60). The B70 and
B65 are G31. Getting this wrong doesn't fail loudly — you get a binary that JIT-compiles
at first launch, which is precisely what the "Zero JIT Latency" goal was about.

**6. `-ffast-math` breaks the attention kernel.** It implies `-ffinite-math-only`,
which permits the compiler to assume infinities don't exist. FlashDecoding initializes
its running max to `-inf` and relies on `exp(-inf) == 0` for masked lanes. The build
script uses `-ffp-contract=fast -fno-math-errno` instead — the parts you want, without
the part that quietly degrades output.

**7. Host math functions in device code.** `std::ldexp`, `std::exp`, `std::sqrt` should
be `sycl::` equivalents. `std::ldexp(1.0f, scale - 127)` also mis-decodes E8M0 at the
boundaries: code `0x00` is 2⁻¹²⁷ (an fp32 *subnormal*, not constructible by shifting the
exponent field) and `0xFF` is NaN. `e8m0_to_f32()` handles both; `test_formats.cpp`
checks every one of the 256 codes is an exact power of two.

**8. `M / XMX_M` is zero when `M = 1`.** `main()` calls the prefill GEMM with `M = 1`,
giving `nd_range` a zero-size dimension. Which points at the real issue —

**9. Decode should not use XMX at all.** At `M = 1` an 8×16×16 DPAS wastes 7 of 8 A-rows
on padding, and the layer is bound by weight streaming regardless. `src/gemv_decode.cpp`
is a separate bandwidth-optimal kernel; XMX is for prefill only. Roofline per decode
step at 608 GB/s, 7B params:

| | bytes | ms | tok/s ceiling |
|---|---|---|---|
| MXFP4 / INT4 | 3.7 GB | 6.1 | ~164 |
| INT8 / FP8 | 7.0 GB | 11.5 | ~87 |
| BF16 | 14.0 GB | 23.0 | ~43 |

**10. Every lane writes the same output addresses.** The GEMM epilogue computes
`out[out_offset + i]` with no `lane_id` term, so all 16 lanes of the sub-group race on
one 8×16 tile. `joint_matrix_store` with a proper stride handles this.

**11. B operands need VNNI layout.** Intel's DPAS wants the B fragment packed —
`bf16: [k/2][n][2]`, `int8: [k/4][n][4]` — not row-major. `src/gemm_xmx.cpp` writes
directly in VNNI order while dequantizing into SLM, so there's no separate transpose
pass. Also: `joint_matrix_load` reads from local or global memory with a stride; loading
from a per-work-item private array, as the blueprint does, isn't valid.

**12. Tensor-parallel section conflates IPC with P2P.** `zeMemGetIpcHandle` /
`zeMemOpenIpcHandle` share an allocation **between processes**, not between devices —
it doesn't by itself let card A address card B's VRAM. Cross-device access needs
`zeDeviceCanAccessPeer` to return true, and P2P on Arc has historically returned
`-995 (unsupported)` through Level Zero. **Query it at runtime and keep a
device→host→device staging path**, because if P2P isn't there, an all-reduce that
assumes it will either fail or silently fall back to something much slower than the
PCIe Gen5 figure suggests. This is the one item I'd verify on your actual cards before
building anything on it.

---

## Layout

```
include/b70/formats.hpp    format codecs — E8M0, E2M1, E4M3, E5M2, bf16. No SYCL dep,
                           so the GPU and the tests share one decode path.
include/b70/weights.hpp    QuantWeight descriptor (raw device pointers, no ownership)
src/quantize.cpp           offline fp32 -> any format
src/gemv_decode.cpp        decode GEMV, bandwidth bound, in-register dequant
src/gemm_xmx.cpp           prefill GEMM, joint_matrix, bf16 + int8 pipelines
src/attention.cpp          FlashDecoding, corrected online softmax
src/main.cpp               device probe, per-format benchmark + correctness check
tests/                     host validation, no GPU required
```

**KV cache layout.** `k_cache` is D-major `[head][head_dim][seq_cap]` so lane L's read
of `K[s0+L][d]` is contiguous across the sub-group — one 64-byte transaction per score
step instead of 16 scattered ones. `v_cache` stays D-minor because the output
accumulator is partitioned over `d`, which already gives lane-contiguous reads.
Appending a token writes K strided; a few hundred bytes, irrelevant against streaming
the whole cache.

**Why bf16 for MX dequant is free.** bf16 carries the full 8-bit fp32 exponent, so
multiplying by an E8M0 scale — a pure power of two — is *exact*. Folding the block scale
into the dequantized tile loses nothing versus accumulating per block in fp32, and keeps
the scale multiply out of the MAC loop entirely. `test_formats.cpp` asserts this over
2⁻⁶⁰..2⁶⁰.

---

## Status, honestly

**Validated by running it:** the whole format layer (all 256 codes of each FP8 format
round-trip exactly, E8M0 exhaustive, E2M1 round-to-nearest-even ties), quantization error
against theory for all seven formats, the outlier behaviour, end-to-end GEMV per format,
and the online-softmax recurrence against a materialized softmax at six sequence lengths.
`make test` — zero warnings under `-Wall -Wextra`.

**Not compiled here:** the SYCL kernels. This container has no `icpx` and no network
route to Intel's package repos, so `src/gemv_decode.cpp`, `src/gemm_xmx.cpp`,
`src/attention.cpp` and `src/main.cpp` have never been through a compiler. Expect to
fix things on first build — `joint_matrix` is still in
`sycl::ext::oneapi::experimental` and its signatures have moved between oneAPI releases,
so `matrix::ext_intel_packed`, `get_wi_data`, and the `get_multi_ptr` calls are the most
likely to need adjusting to your toolchain version. The algorithms are validated; the
API surface is the part to check.

Start with `./build_b70.sh && ./b70_native_inference`. `main.cpp` checks every GEMV
against a host reference computed through the same decode functions, so a format that
comes out wrong on device will say so rather than quietly producing bad tokens.
