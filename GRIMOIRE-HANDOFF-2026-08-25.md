# GRIMOIRE — STATE, 2026-08-25

Everything here is measured on the Tower today. Nothing inherited, nothing assumed.
For the blow-by-blow log (including three theories of mine that turned out wrong),
see `GRIMOIRE-HANDOFF-2026-08-25-FULL.md`. You do not need it to work.

---

## 1. Where things stand

| model | PP @4096 | TG | gate | status |
|---|---|---|---|---|
| **Ornith** | **9,822** (GPU1) / 9,792 (GPU0) | 111 | 10,000 | 1.8% short |
| **Qwen** | **1,694** | 23.0 | 2,000 / 28 | 15% / 18% short |

Reference, measured on the same card: **vLLM = 2,015.6 PP / 27.95 TG** on
`Qwen3.8-27B-int4-AutoRound`, single GPU, llama-benchy, pp4096/tg32.

Both models produce coherent output on long real prompts. Both GPUs healthy.

---

## 2. How to run — copy these exactly

Ornith and Qwen need **different flags**. Mixing them silently costs 2-20% and looks
like a code regression. This cost a day; do not improvise.

```bash
cd /mnt/storage/isos/grimoire-fuse

# ORNITH  -> 9,822 tok/s
GPU=gpu1 EXTRA_ENV="GRIMOIRE_BENCH_PREFILL_TOKENS=4096
GRIMOIRE_DEFER_MOE_GATHER=1
GRIMOIRE_BF16_QKV=1
GRIMOIRE_BF16_DN_QKV=1
GRIMOIRE_XE2_ATTN_BRIDGE=/grimoire/src/libgrimoire_xe2_attention_raw.so" \
  bash tools/tune.sh orn /grimoire/bin/grimoire \
  -m /models/Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE --proj mxfp4 --ctx 8192

# QWEN  -> 1,694 tok/s
GPU=gpu0 EXTRA_ENV="GRIMOIRE_XE2_ATTN_BRIDGE=/grimoire/src/libgrimoire_xe2_attention_raw.so" \
  bash tools/tune.sh qwen /grimoire/bin/grimoire \
  -m /models/Qwen3.8-27B-MXFP4-GRIMOIRE --proj mxfp4 --ctx 8192
```

- `--proj mxfp4`, NOT `int4`. `int4` costs Ornith ~7%, Qwen ~2%.
- `attention_raw.so`, NOT `attention_bridge.so`. `tune.sh` defaults to the wrong one.
- **Never** give Qwen `DEFER_MOE_GATHER` / `BF16_QKV` / `BF16_DN_QKV` — Qwen is dense,
  and those flags produce degenerate repeating output.
- **Never** hand-roll `docker run`. `timeout N docker run` orphans the container and
  wedges the GPU. Always `tools/tune.sh` (wraps `b70run.sh`).
- **Never** hardcode a `renderD` number — they shuffle on reboot. `GPU=gpu0|gpu1`
  resolves by PCI address via `tools/gpunode.sh`.

Build after editing a bridge (~4m25s); `bash build_b70.sh` alone (~1m) if only the
binary changed:
```bash
docker run --rm -v /mnt/storage/isos/grimoire-fuse:/grimoire \
  -v /mnt/cache/appdata/vllm-xpu-kernels:/src -v /mnt/storage/isos/sycl-cache:/cache \
  grimoire:queuefix /usr/local/bin/entry.sh bash -c \
  "ln -sf /usr/lib/x86_64-linux-gnu/libze_loader.so.1 /usr/lib/x86_64-linux-gnu/libze_loader.so && \
   cd /grimoire && bash tools/build_bridges_b70.sh /grimoire"
```
The `ln -sf` is required — the image ships only `libze_loader.so.1`.

---

## 3. The open problem: Qwen needs 420 ms

Qwen prefill is 2,418 ms. The gate needs 2,048 ms.

| where | ms | notes |
|---|---|---|
| FFN gate_up GEMM | 935 | at parity with vLLM |
| FFN down GEMM | 447 | vLLM is 107 ms better here |
| full attention | 229 | **not investigated** |
| 7 DN regions | 715 | **not investigated** |
| SwiGLU + norms | 120 | fixed today, near roofline |

**Located: ~190 ms.** int4 g128 on both FFN weights = 81 ms (needs converter work);
vLLM's unexplained ffn-down edge = 107 ms.

**Not located: ~230 ms.** It is **not** in the dense GEMM — that is measured, not
assumed. It is in `full attention` (229 ms) and the DN regions (715 ms), which
together are 39% of prefill and which nobody has opened yet. vLLM runs FlashAttention
and a fused Triton/FLA GDN prefill kernel for exactly these. **Start there.**

TG is a separate, easier problem: `k+v` (N=2048) runs at 108 GB/s and `ab` (N=96) at
**9 GB/s** against a 602 GB/s card. Split-K on those low-N GEMV shapes is worth
~3 ms of the 43 ms/token. Untouched, no dependency on the PP work.

---

## 4. Ruled out — do not spend time re-testing

| claim | verdict |
|---|---|
| A better dense tile exists | **No.** 13 policies measured in production; `p128x256` wins, and it is byte-identical to vLLM's own `w4a16_policy`. |
| Our GEMM is worse than vLLM's | **No.** Ours 14.65 ms vs theirs 14.67 ms under identical streaming conditions. |
| int4 is much faster than MXFP4 | **No, ~3%.** The real variable is *group size*: MXFP4 is locked to g32, so ffn-down (K=17408) pays 544 dequant-rescales per tile vs 136 at g128. |
| The FFN has large non-GEMM overhead | **No.** gate_up 935 + swiglu 75 + down 447 accounts for the whole region. 84% of prefill is GEMM. |
| oneDNN W4A16 is faster for prefill | **No.** Slightly slower (1650 vs 1658 tok/s E2E). |
| Longer/shorter prompts help | **No.** Flat: 1576/1643/1663/1635 at M=1024/2048/4096/8192. |
| One GPU is faster | **No.** GPU0 and GPU1 are equal within noise. |
| Tensor parallelism is required for 2000 | **No.** vLLM does 2,015 on ONE card. |
| The BF16-critical-tensor fix cost speed | **No.** ~0.7% on Ornith. |

**Two measurement traps that invalidated earlier conclusions in this project:**
1. **Constant-filled buffers lie.** The B70 does lossless memory compression, so
   `q.fill(x, 0.125f)` reads unrealistically fast. Use random data.
2. **Cache-hot microbenches lie.** Reusing one weight matrix keeps it in L2; production
   streams 64 distinct ones. Measure with `tools/bench_stream.cpp`, which reproduces
   production timing to 0.3% **without a 20 s model load**. Every tuning number in this
   project predating today was taken cache-hot and is suspect.

---

## 5. Fixed today

| change | effect |
|---|---|
| FFN **down** projection tile — a second entry point (`dense_mxfp4_f32`) that no previous tuning pass had ever touched | largest single win |
| dense bf16 tile -> `p128x256` | +27 tok/s |
| SwiGLU did a 64-bit divide **and** modulo per element, 71M times | ~30 ms |
| L2 swizzle `GRIMOIRE_GEMM_SWIZZLE_GM` (default 4) | +31 tok/s |
| 11 elementwise kernels off 64-bit div/mod | +3 tok/s |
| `--proj mxfp4` + `attention_raw` for Qwen | +32 tok/s |

**Qwen 1,556 -> 1,694 tok/s (+8.9%).** All numerically exact. Ornith unaffected
(verified by reverting each change and re-measuring).

---

## 6. Tools added today

- `tools/bench_stream.cpp` — production-condition GEMM timing, no model load. **Use
  this to iterate.**
- `tools/bench_sweep.cpp` — policy + int4 group-size sweep under streaming.
- `tools/gpunode.sh` — `gpu0|gpu1` -> current render node by PCI address.
- `tools/tune.sh` — canonical runner. `GPU=`, `LIM=`, `EXTRA_ENV=`.
- `tools/ar_mxfp4.py` — AutoRound MXFP4 export (works, 232 s; output needs the visual
  tower pruned before it is usable — currently 66 GB).
- `launch_dense_int4` in `xe2_grouped_raw_launcher.hpp` — works, unused, ready if the
  int4 converter gets written.

Backups: `.backups-20260825/` in the repo; full tarball on the Tower and Ian's Desktop.
