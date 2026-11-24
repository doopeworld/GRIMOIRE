# vLLM XPU reference build — DFlash on Muse-Glimmer, for GRIMOIRE

Reference doc for porting the working DFlash speculative-decode setup into
GRIMOIRE. Everything here was built and measured today on the Tower
(`root@192.168.8.225`), `my-vllm-xpu:latest`, single Arc Pro B70 (GPU0,
PCI `03:00.0`). Source: `/mnt/user/appdata/vllm-src`, branch `pr-51655`,
build: `docker/Dockerfile.xpu`.

## Findings

- **vLLM core is now 0.28.0**, merged from the upstream tag into `pr-51655`
  (commit `a4f0b85eb`, resolving 2 conflicts: upstream's new xpu-smi install
  step kept additively, and `qwen3_5_dflash.py`'s `_dflash_layer_causal` now
  checks upstream's new explicit `config.is_causal` override *before*
  falling back to our own `dflash_config.causal` override, then the legacy
  `layer_types` inference — both mechanisms preserved, not one dropped for
  the other. Installed version string:
  `vllm==0.28.1.dev8+ga4f0b85eb.d20260829.xpu`.
- **PyTorch is now nightly 2.15.0** (`torch==2.15.0.dev20260825+xpu`), paired
  with `torchvision==0.30.0.dev20260826+xpu` and
  `torchaudio==2.11.0.dev20260826+xpu`. These three don't share a build date
  with each other despite the naming — had to walk each wheel's own
  `Requires-Dist: torch==...` metadata to find a set that's actually
  mutually consistent; matching filename dates is not sufficient.
  `triton-xpu==3.8.0+git1e2d42a0` is the version torch 2.15 actually
  requires (was `3.7.2` before); the `TRITON_XPU_VERSION` Dockerfile ARG and
  every `requirements/*xpu.txt` pin had to move together or `uv`'s resolver
  reports the mismatch as unsatisfiable.
- **`vllm-xpu-kernels` needs to stay the custom-patched build, not the
  official release.** Both are built from the same commit (`g07d44bc`, the
  exact commit `v0.1.13.1` tags) so it's not a version gap — it's that the
  official release wheel ships a reduced kernel-tuple table ("use default
  config to control wheel size" per its own release notes). Confirmed
  concretely: Muse-Glimmer's DFlash config (`--block-size 64`) fatally
  crashes EngineCore on the official wheel with
  `RuntimeError: Paged decode kernel tuple not compiled for this
  configuration`, naming the exact missing tuple (`16,128,64,false,true,false`
  — the `64` is our block-size). No such crash on the custom wheel, which
  was compiled with broader tuple coverage. **The custom kernel is not a
  performance regression vs. stock** — `patch_caller.py` adds sparse MoE
  expert dispatch (measured 232µs→94µs per grouped-GEMM call at batch 1 on
  this B70, same work), `patch_workspace2.py` shares one MoE scratch
  allocation across all 40 layers instead of malloc/free per layer per
  step. Relevant for GRIMOIRE's own MoE dispatch if it ever wraps this
  kernel package rather than its own.
- **`torchcodec` from plain PyPI links CUDA's `libnvrtc.so.13`** and crashes
  every `vllm serve` at import time (`vllm.multimodal.video` imports it
  unconditionally, even for text-only serving) with
  `OSError: Could not load this library: .../torchcodec/libtorchcodec_image.so`.
  Fixed by pinning to the CPU-only nightly build instead:
  `torchcodec==0.17.0.dev20260828+cpu` from
  `download.pytorch.org/whl/nightly/xpu` (that index carries `+cpu` wheels
  too, not just `+xpu`). No functional loss — we don't use torchcodec's
  video/image decode path.
- **oneCCL requires `--privileged --ipc=host --security-opt label=disable
  --shm-size=10g`** on top of `--device /dev/dri`. Without `--privileged`
  specifically, engine init fails immediately with
  `oneCCL: ze_fd_manager.cpp:144 init_device_fds: EXCEPTION: opendir failed:
  could not open device directory` — this reproduced identically whether
  `/dev/dri` was exposed as the full directory or as a single
  `--device /dev/dri/renderDxxx` node, so it isn't a device-node-count
  issue, it's specifically the missing capabilities/IPC namespace.
- **Ornith's checkpoint format matters, not just the quant method.** The
  `INT4-W4A16-AutoRound` variant declares `architectures:
  ["Qwen3_5MoeForCausalLM"]` (text-only) in `config.json`, but its
  safetensors shards still carry 549 leftover `model.visual.*` weight keys
  from the source multimodal checkpoint — `--language-model-only` does
  *not* fix this, vLLM's strict loader still errors
  (`ValueError: There is no module or parameter named 'visual' in
  Qwen3_5Model`) because the instantiated class has nowhere to put them.
  The `GPTQ-Int4` variant instead declares `architectures:
  ["Qwen3_5MoeForConditionalGeneration"]` (the multimodal class, which has
  a `visual` submodule to absorb those same keys) — load that one with
  `--language-model-only` to skip building/running the vision tower. Not
  yet benchmarked (ran out of time this session before hitting this).
- **B70 device mapping is not stable across reboots.** `renderD128`/`129`/`130`
  swapped which physical card they pointed to after a power-outage reboot
  today. The reliable identifier is the level-zero device UUID's embedded
  PCI bus id (`868023e2-0000-0000-{bus}-000000000000`) — confirmed via
  `torch.xpu.get_device_properties(i).uuid` with only one render node
  exposed at a time. GPU0 = PCI `03:00.0`; GPU1 = PCI `35:00.0`.
- **GPU1 (`35:00.0`) is currently hung** (`xe` driver forcewake failures,
  `MMIO unreliable (forcewake register returns 0xFFFFFFFF)`, continuously
  spamming dmesg) — happened after a container got SIGKILL'd while still
  initializing the device. Needs a Tower reboot to clear; not a software
  problem, don't retry container restarts against it.

## Results (llama-benchy, `--pp 4096`, single B70/GPU0)

| model | pp4096 t/s | tg32 t/s | notes |
|---|---|---|---|
| Qwen3.8-27B-int4-AutoRound | 1965.00 ± 11.92 | 28.23 ± 0.01 (peak 29.00) | no spec-decode |
| Muse-Glimmer-30B-INT4-W4A16 | 2121.32 ± 16.12 | 46.85 ± 3.62 (peak 51.88) | **DFlash active**: `--speculative-config '{"method":"dflash","model":"/models/Muse-Glimmer-assistant","num_speculative_tokens":15}'`, `--block-size 64`, custom kernel |
| Ornith-1.5-35B-A3B | — | — | not run yet; use `GPTQ-Int4` variant + `--language-model-only` |

Both coherence tests passed. Full Muse launch command that produced the
above (custom-kernel image, GPU0 only):

```bash
docker run -d --name VLLM-XPU-FUSION \
  --privileged --security-opt label=disable --ipc=host --shm-size=10g \
  --device /dev/dri --network host \
  -v /mnt/storage/Models:/models \
  my-vllm-xpu:latest \
  /models/Muse-Glimmer-30B-INT4-W4A16 \
  --dtype float16 --port 3552 --host 0.0.0.0 \
  --max-model-len 12288 --gpu-memory-utilization 0.95 \
  --max-num-seqs 16 --kv-cache-dtype auto \
  --max-num-batched-tokens 16384 --block-size 64 \
  --language-model-only --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --speculative-config '{"method": "dflash", "model": "/models/Muse-Glimmer-assistant", "num_speculative_tokens": 15}'
```

## Next steps

1. Benchmark Ornith with the `GPTQ-Int4` variant + `--language-model-only`
   (`--pp 4096`, same as above) to complete the three-model comparison.
2. For GRIMOIRE's own DFlash path: vLLM's reference `_dflash_layer_causal`
   (now dual-mechanism after the 0.28.0 merge — `config.is_causal` checked
   before `dflash_config.causal`) is the authoritative causal-mask source
   GRIMOIRE's C++ engine should keep matching; re-diff it after any future
   vLLM merge since this is exactly the function that drove the 2.67→3.17
   committed-tokens/step fix.
3. Decide whether GRIMOIRE wants to depend on `vllm_xpu_kernels`'
   sparse-dispatch/shared-scratch pattern directly (same technique, same
   measured win) rather than reinventing it — the patch scripts
   (`patch_caller.py`, `patch_workspace2.py`) in this repo are a clean
   reference implementation against `fused_moe_interface.py`.
4. If vLLM ever needs to move off the custom kernel wheel (maintenance
   burden), the concrete blocker to clear first is filing/tracking the
   missing paged-decode tuple (`16,128,64,false,true,false`) against
   vllm-project/vllm-xpu-kernels issue #364, so the official release covers
   Muse's block-size=64 case.
5. Reboot the Tower before any further GPU1 (`35:00.0`) use.
