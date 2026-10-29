# Native GGUF block formats (current)

The original Q4_K/Q6_K implementation is documented below as Phase 3 step 1.
The current native set is Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS and IQ4_NL.
`--proj native` retains raw GGML blocks for eligible projections. Q2_K,
Q3_K, IQ1_*, IQ2_* and IQ3_* still decode at load and use the MXFP4 base.
The banner prints native and fallback tensor counts and GiB per GGUF type.
Critical tensors continue to use the BF16 policy. A gate/up pair with
mismatched GGUF types uses the existing decode and repack fallback.

`NativeGGUFBlock<F>` in `include/b70/formats.hpp` provides each format's
block width, byte stride, type ID and decode lane count. K must be divisible
by the block width: 256 for K quants and IQ4_XS, 32 for Q8_0 and IQ4_NL.
The raw readers and upload paths refuse an unaligned named tensor. Q5_K is
176 bytes per 256 values, Q8_0 34 bytes per 32, IQ4_XS 136 bytes per 256,
and IQ4_NL 18 bytes per 32. Both IQ4 formats use the ggml IQ4_NL value table.

The shared element decoder, 16-value VNNI tile decoder, grouped MoE tile
writer and 16-lane GEMV arithmetic dispatch through the trait. The GEMV
applies the block or sub-block scale after accumulating the lane's dot; Q5_K
also applies its minimum term after the input sum, as Q4_K does. Q8_0 and
IQ4_NL use two 16-value lanes per block. The MXFP4/INT4 ESIMD and bridge
paths still require their exact formats and side-scale pointers.

### Q5_K packed decode lane: B70 root cause and fix (task `q5k`)

Symptom: with Q5_K in the packed fast GEMV, Qwen3.8-27B UD-Q4_K_XL
`--proj native` kept the prefill token but produced garbage on later decode
tokens, while every host test passed (commit 1f6a438 routed Q5_K through the
element GEMV as a workaround, ~4.5x slower decode).

Root cause: a **compiler (icpx/IGC) miscompile on the B70**, not a layout or
alignment bug. The old 8-lane form took the high-bit plane straight from the
block with unrolled `uchar` loads shifted by a per-lane amount,
`((p[32 + i] >> lane) & 1) << 4`. For `lane != 0` the device returns wrong bits
(lane 0, shift 0, is right, which is why the first token looked fine).
`tools/test_q5k_bits.cpp` reproduces it in isolation on gpu0: summing
`(p[16+i] >> lane) & 1` over 32 bytes is wrong in 7 of 8 lanes for the plain
form, with or without the `lane < 8` guard, with a shift of `lane >> 1` or
`lane & 7`, and even when the values go through a local array. It is right
when the loop is `#pragma unroll 1` or when the bytes are first extracted
from a 64-bit pair (the `native_qk_byte` form). Host emulation cannot see it.
The unaligned 176-byte block base, padded row strides and lane mapping were
all ruled out by `tools/test_q5k_device.cpp` (below).

Fix (`include/b70/native_qk_gemv.hpp`): Q5_K now uses all 16 sub-group lanes
(`NativeGGUFBlock<Q5_K>::lanes = 16`): lane `2*g + h` owns the 16 values
`[g*32 + h*16, +16)` of sub-block `g`. The header (d, dmin, scales/mins), the
16 qs bytes and the 16 qh bytes are each taken with one `native_qk_load16`
and every byte comes out through `native_qk_byte`; no raw `p[i] >> variable`
remains in the Q5_K lane. The workaround `native_qk_fast_format()` is gone:
Q5_K uses the packed GEMV like every other native type, and
`GRIMOIRE_GGUF_FAST_GEMV=0` still selects the checked element GEMV as the
reference. The same A/B on Q4_K/Q6_K/Q8_0/IQ4 lanes never needed this (they
already read bytes through `native_qk_byte`).

B70 verification (gpu0; logs under `/mnt/user/isos/grimoire-runs/gguf-q5k/`):
`bin/test_q5k_device` runs the real `native_qk_block_lane<Q5_K>` in a 16-lane
sub-group against `decode_q5_k` in double: 94 cases = real-model N/K shapes
(3x256 ... 17408x5120), 64-byte-aligned and +1/+4-byte offset bases, row
padding 0/1/16 bytes, MB=1 and MB=4 (mc=2,4); 0 errors, worst error 2e-8 of
the term magnitude.
End to end on Qwen3.8-27B UD-Q4_K_XL `--proj native`: identical 48-token
greedy ids and coherent text, fast versus reference, on two prompts, and
decode 9.0 tok/s versus 2.0 tok/s (graph replay) with the reference; WikiText
ppl (512 ctx, 1 chunk, decode-path scoring) 4.28985 fast vs 4.28959 reference.

Ornith-1.5-35B Q4_K_M `--proj native` (Q4_K and Q6_K only, no Q5_K tensors):
48-token greedy ids with this branch are byte-identical to the 10-18 build of
the base commit (main-clone `bin/grimoire`) on prompt 1, text coherent
("The sky is blue because of a phenomenon called Rayleigh scattering...");
prompt 2 fast = reference ids. Prompt 1 fast vs `GRIMOIRE_GGUF_FAST_GEMV=0`
differs from token 16 on (float summation order, both coherent), as it did
before this change. Graph-replay decode 43.7 tok/s fast vs 11.8 reference.

**Separate open bug found on the way (not Q5_K):** with the *IQ4_NL* tensors
(3 native tensors in this file) on the packed fast GEMV, decode vs batched
prefill logits drift by 3.5-4.0 (max abs) and ppl rises; the reference path
and every other type combination are at 0.2-0.8. Allowlisting
`GRIMOIRE_GGUF_NATIVE_TYPES=q4_k,q6_k,q5_k,q8_0,iq4_xs` avoids it.

`make test_gguf_native` checks each new format against the checked-in upstream
golden block and independent `gguf_dequantize`, tests the VNNI pair layout,
fast GEMV at K=512/2048/17408 including M=4, grouped top-k MoE math, raw
permutations, and dense gate/up/down row handling. The full `make test` host
gate must pass. No SYCL compiler is available on the Mac; on the B70 build
with icpx, then compare a Qwen3.8-27B UD-Q4_K_XL native run against BF16 for
KLD and measure prefill and decode throughput. Check the banner's per-type
native/fallback counts before interpreting the result.

## Phase 3 step 1: original Q4_K/Q6_K implementation

`--proj native` (CLI or server) or `GRIMOIRE_GGUF_NATIVE=1` enables raw GGUF
Q4_K/Q6_K weights. Raw retention applies to GGUF inputs only. Without it, the
Strategy A decode and requantize path is unchanged. The existing
`keep_qwen_bf16` policy still applies to embeddings, DeltaNet a/b, routers,
and other critical tensors; `output.weight` is explicitly allowed to remain
native Q6_K for logits. The GGUF MTP layer remains excluded by the adapter.
Native mode uses **MXFP4 as the base projection** for any other projection
that the BF16 policy does not preserve. The CLI and server reject an explicit
`--proj` other than `native` or `mxfp4` when the native environment flag is
set; the engine also applies MXFP4 to API callers whose upload options still
default to INT4. The load banner reports this base and the actual native
Q4_K/Q6_K tensor counts and bytes.

## Storage and transforms

Each runtime row is one GGML row with K/256 consecutive super-blocks.
Q4_K uses 144 bytes per block (f16 `d`, f16 `dmin`, packed scale/min bytes,
then 128 nibble bytes); Q6_K uses 210 (128 low-nibble bytes, 64 high-bit
bytes, 16 signed scales, f16 `d`). Scales and minima are inside `payload`.
`QuantWeight.scales` and `.zeros` are null, and `row_scales` is zero. Native
upload refuses a named tensor if K is not divisible by 256.

QKV V rows, z rows, and other T4 row permutations move complete compressed
rows. Expert gate and up rows are concatenated as bytes per expert; down has
its own runtime format so a Q4_K gate/up and a Q6_K down work together.
An expert tensor in another GGUF format uses the MXFP4 base. If gate and up
would require different runtime formats, loading refuses them by tensor name
because their single packed matrix has one `Fmt`.
The stored `ssm_out` column permutation cannot be reversed on compressed
blocks. Immediately before that projection, a gather puts the activation
from grouped V-head order into GGUF tiled V-head order, only when the actual
weight is native. The float and BF16 prefill inputs and float decode input
use the same host-tested index map. Decode writes the gather to `s.bbuf`:
its allocation starts at `Hv*Dv` floats and grows for every layer's `la_z`,
`k_proj`, and `v_proj` output. In the DAG path, the projection waits on the
gather, and the next layer's input waits on that projection through `e_h`
and `e_moe`; a later full-attention `v_proj` cannot overwrite it early.

## Dispatch and fast-path guard inventory

The inventory below covers the projection-bearing hits from
`grep -n "Fmt::MXFP4\|Fmt::INT4\|gemm_fast_supported\|smallm\|esimd\|ESIMD\|W4A8\|has_i4\|bridge\|\.scales\\b" src/*.cpp`.
Attention/router ESIMD and benchmarks in that grep do not read a `QuantWeight`.

| Site | Disposition |
|---|---|
| `src/gemv_decode.cpp` dense and batched dispatch | FP8/INT4/MXFP4 ESIMD attempts require exact formats, intact payload and side scales. Q4_K/Q6_K first try the ESIMD GEMV (`src/native_qk_esimd.hpp`; needs K % 512 == 0, packed rows, 16 B / 4 B aligned payload; `GRIMOIRE_GGUF_ESGEMV=0` skips it), then the native super-block GEMV when its format, payload, K and row stride pass the guard; `GRIMOIRE_GGUF_FAST_GEMV=0` selects `GemvStepGGUF`. |
| `src/gemv_decode.cpp:1391-1408` | FP8 small-M requires its exact format; the INT4 small-M caller uses `int4_smallm_ok`. |
| `src/gemm_fast.cpp:127-228,399-440` | VNNI simple and tiled dequant stage Q4_K/Q6_K with `scale=1` and `decode_elem`; no native side-scale access. MXFP4 and INT4 streaming branches require exact formats. |
| `src/gemm_fast.cpp:348-353,1883-1904` | INT4 stream, fused MXFP4, and grouped MXFP4/ESIMD predicates require their own formats, payload and side scales. |
| `src/gemm_fast.cpp:2960-3034` | `gemm_fast_supported` intentionally accepts native weights for generic VNNI staging. `EPI=0/2` uses `dequant_any`; `EPI=1` is entered only after `gemm_fast_swiglu_supported` accepts MXFP4 or INT4. |
| `src/gemm_fast.cpp:3609-3674` | MXFP4, BF16, INT4, and grouped MXFP4 small-M predicates check exact formats and pointers. |
| `src/gemm_xmx.cpp:145-165,473-487` | Float XMX staging starts at `scale=1`, branches on scale-bearing formats, and dispatches Q4_K/Q6_K. The integer pipeline at `:525` dispatches INT8/INT4 only. |
| `src/prefill.cpp:70-140` | Plain SYCL batched GEMM calls `GemvStepGGUF` for Q4_K/Q6_K and uses no side scale. |
| `src/moe_kernels.cpp` routed, shared, batched and MoVA decode | Single-token decode with Ornith geometry (H 2048, I 512, top_k <= 15) and Q4_K/Q6_K gate_up/down of matching shared-expert type uses the fused ESIMD pair `moe_gu_native_esimd` / `moe_dn_native_esimd` (same layout guard as the dense ESIMD GEMV; `GRIMOIRE_MOE_NATIVE_ESIMD=0` skips). Otherwise each Q4_K/Q6_K matrix uses the native super-block dot when its own descriptor passes the guard. The original `GemvStepGGUF` path remains the reference under `GRIMOIRE_GGUF_FAST_GEMV=0`. |
| `src/moe_kernels.cpp:630-954` | ESIMD kernels read side scales, but `mx4_rows_ok` at `:941` requires MXFP4, payload and scales. |
| `src/ops.cpp` MoVA packed fallback | Q4_K/Q6_K use the host-tested super-block dot under the same guard; the element decoder remains the reference. Router ESIMD reads logits, not weights. |
| `src/grimoire.cpp:1388-1423,1747-1763` | Raw upload and gate/up concatenation copy complete rows; mismatched gate/up types or K not divisible by 256 are refused by tensor name. |
| `src/grimoire.cpp:4038-4072` | TP row sharding slices by `row_bytes`; null native side scales and zeros remain null. |
| `src/grimoire.cpp:4861,5369-5419,5457` | Fused DeltaNet upload excludes GGUF. W4A8 conversion requires MXFP4, payload and N multiple of 256; native weights are skipped. |
| `src/grimoire.cpp:11220-11264,12263-12293,13093-13113,14241-14293` | Draft, sandwich, Qwen4-Exp and main prefill small-M/bridge selections either check exact formats or call format-aware generic GEMV/XMX. |
| `src/grimoire.cpp:13870-13904,14490-14586,14657-14670` | Native mode disables MXFP4-only parallel modes and norm shortcut. Dense/aux/BF16 bridges check format and payload; grouped FFN bridge checks both gate/up and down. |
| `src/grimoire.cpp:14980-15000,15339,15471-15479` | Native `ssm_out` gathers before float or BF16 GEMM. Attention/router bridge uses MXFP4 predicate; native input takes generic GEMM. |
| `src/grimoire.cpp:15628-15958` | Small-M grouped predicates require MXFP4. At M >= 32 native Q4_K/Q6_K experts use the grouped QK path when both matrices and dimensions pass its guards; `GRIMOIRE_MOE_NATIVE_GROUPED=0` restores the per-expert reference. Shared expert uses generic GEMM. |
| `src/grimoire.cpp:16033-16055` | Native lm_head misses the MXFP4/INT4/FP8-only batch shortcuts and reaches native GEMV for each row. |
| `src/native_model.cpp:124-132` | `slice_quant_rows` advances payload by native `row_bytes` and touches side scales only when non-null. |
| `src/quantize.cpp:83-110` | Generic quantizer explicitly refuses Q4_K/Q6_K; raw GGUF blocks enter through the native upload path. |

The oneDNN W4A8 bridge in `src/grimoire.cpp:3190-3265` requires the
converted INT4 companion (`has_i4`) or exact MXFP4 plus payload and scales.
The remaining oneDNN FP16 paths are Muse-only. `GRIMOIRE_FUSE_DN_PROJECTIONS`
is excluded for GGUF at `src/grimoire.cpp:4861`. The `ssm_out` gather sites
are `src/grimoire.cpp:9167,9891,10341,13260,14981-14998`.

## Grouped native expert prefill

At M >= 32, `launch_moe_route_grouped` sorts the top-8 routes on the device
and builds the same padded 256-row tile table used by the MXFP4 grouped path.
`launch_moe_native_qk_grouped` runs once for gate/up and once for down. It
reuses the grouped joint-matrix DPAS tile and its fused SwiGLU epilogue; each
matrix dispatches its own Q4_K or Q6_K format. The down result is fp32 and
uses the existing weighted unpermute. Empty table entries have expert -1 and
return before reading the weight or entering a work-group barrier.

Each B-tile writer covers 16 K values of one output row. It resolves all
Q4_K scales/mins or Q6_K signed scales once when it enters a 256-value
super-block, then uses the cached values for every 16-value slice of that
block. It rounds decoded values to bf16 RNE in the same VNNI pair layout as
the MXFP4 tile. Two B-tile writers cover the two halves of each 32-K step.
The grouped path requires exact native formats, payloads without side scales,
K divisible by 256, output width divisible by 256, the expected expert matrix
shapes, top_k 8, and at most 1024 experts. A failed guard takes the existing
per-expert path. Q4_K matrices with a 16-byte-aligned row stride run in `moe_mxfp4_esimd<EPI, Q4_K>` (ESIMD, SLM-staged DPAS); `GRIMOIRE_MOE_NATIVE_ESIMD_GROUPED=0` selects the joint_matrix kernel; Q6_K stays on joint_matrix. `GRIMOIRE_MOE_NATIVE_GROUPED=0` selects that reference for
B70 A/B; `GRIMOIRE_MOE_GROUPED_OFF=1` also disables grouped dispatch.

The host test checks the VNNI pair layout against `gguf_dequantize` rounded
to bf16, checks cached decoding across two super-blocks, and compares a small
grouped top-k/SwiGLU/down reference with independent per-expert dequant and
matmul for Q4_K gate/up with both Q4_K and Q6_K down. No SYCL compiler is
available on the Mac; compile and measure this kernel on the B70.

## Validation

`make test` includes `test_gguf_native`: the exact device decoder functions
run on the host and match both the independent ggml-style decoder and the
checked-in GGUF goldens bit for bit, including a two-super-block row;
native-format host GEMV/GEMM match dequantized f32 products; raw QKV V-row
permutation retains whole blocks; row slicing retains null side scales; and
the ssm_out gather equals inverse column permutation.

Only the B70 can verify SYCL compilation and execution here. On the B70:

For the decode update, compare Ornith Q4_K_M `--proj native` decode tokens/s
with `GRIMOIRE_GGUF_FAST_GEMV=1` and `=0` on the same 80-token prompt. Then
run the WikiText-2 `--ppl` KLD check on two chunks against BF16: that command
uses decode, so it checks the changed summation order in the live path.

1. Compile `build_b70.sh` with `icpx`, recording any compiler errors and the
   exact command. Use `tools/build_bridges_b70.sh` if a bridge changes.
2. Load a real GGUF with `--proj native` and with the environment flag. Check
   the banner for MXFP4 base, raw Q4_K/Q6_K counts and named BF16-policy
   tensors. Check named refusals with an unsupported or malformed tensor.
3. Generate from two different prompts and inspect both continuations; also
   compare decode and prefill logits/token sequences against Strategy A and
   llama.cpp. Exercise short and long prefill, full attention, DeltaNet,
   routed and shared MoE, and a Q6_K lm_head.
4. Exercise `GRIMOIRE_W4A8=1`, the small-M verify path, and native mode with
   bridge flags set; check no freed MXFP4 payload is read. Exercise TP/PP if
   the production configuration uses them.
5. Measure KLD and throughput only after generation is coherent. The GGUF
   Q4_K_M KLD target is near llama.cpp's 0.061 against BF16, compared with
   Strategy A's measured 0.216 for Q4_K_M and 0.196 for BF16 GGUF
   re-quantized to MXFP4.

## Phase 3 step 3: decode and prefill speed (2026-10-19)

See `PERFORMANCE_REPORT.md` section 6 for numbers. Host coverage: `test_gguf_native` mirrors the byte/bit arithmetic of the
ESIMD kernels (vector Q4_K scale decode from the header bytes, the Q6_K u8 bit tricks, the 4-byte-aligned 212-byte window
for even/odd blocks incl. alignment and end-of-row bounds) against `gguf_dequantize`. Device coverage (B70 only):
`tools/bench_native_esgemv.cpp`, `tools/bench_native_moe.cpp`, `tools/bench_native_moe_grouped.cpp` (each checks against a
double-precision host reference built from `decode_elem` and prints GB/s or TFLOP/s). The fp32 summation order of the
ESIMD kernels differs from the SIMT reference (the reference stays selectable with the env switches above).
