# GGUF Phase 1 — benchmark matrix (plan)

Status: **plan; no measurements yet.** `baseline.csv` is filled only from real runs, each row
traceable to a log under `/mnt/storage/isos/grimoire-runs/gguf-baseline/` (CLAUDE.md rule 8:
no number from a run whose generated text was not read).

## Hardware and placement

- **gpu0 only**: Arc Pro B70, PCI `0000:03:00.0`, Gen4 x8 (root port 00:01.0). Render node is
  resolved **by PCI path at run time** (`/dev/dri/by-path/pci-0000:03:00.0-render`); on
  2026-10-15 it was `renderD129` — `renderD128` is now the Arrow Lake iGPU, so any script that
  hard-codes `renderD128` for gpu0 is wrong on this boot (e.g. `tools/bench/vllm_mtp_up.sh`,
  which correctly refuses).
- gpu1 (`0b:00.0`, Gen4 x4 OCuLink) and the B580 (`07:00.0`, correctable AER errors on its
  upstream port) are not used.
- One engine at a time; a GPU-busy check (`busy()` from `tools/bench/ab_ornith_pp.sh`) runs
  before each start; every container is stopped as soon as its run ends.
- Containers come from images with recorded digests; GRIMOIRE runs go through the existing
  launchers (`srv2.sh`), never a bare `timeout docker run` (rule 6).

## Model

Ornith-1.5-35B-A3B, `ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612`, file
`Ornith-1.5-35B-Q4_K_M.gguf` (sha256 `42739874…d41f`; Q4_K 379 / Q6_K 64 / F32 310 tensors —
see `docs/gguf/FORMAT_SUPPORT.md`). Text only (no mmproj), MTP not used in any engine.

## Engines

| ID | Engine | Weights | Comparison label |
|---|---|---|---|
| L-SYCL | llama.cpp SYCL (`ghcr.io/ggml-org/llama.cpp` intel image, pinned tag + digest) | Q4_K_M GGUF | same file |
| L-VK | llama.cpp Vulkan (same release, vulkan image) | Q4_K_M GGUF | same file |
| V | vLLM XPU (`my-vllm-xpu:latest`, settings of `tools/bench/vllm_conc4k.sh`) | Ornith GPTQ-Int4 safetensors (4-bit weights, fp16 act.) | **cross-format** (vLLM cannot run this GGUF arch); nearest 4-bit precision |
| G-ST | GRIMOIRE (current main image) | Ornith safetensors → `--proj mxfp4` | **cross-format**; the existing ~9.7k pp4096 reference |
| G-GGUF | GRIMOIRE GGUF candidate (Phase 2/3) | Q4_K_M GGUF | same file |

## Workloads

Two harnesses, because they measure different things:

1. **Engine-internal (llama.cpp only):** `llama-bench -m <gguf> -ngl 99 -fa 1 -p 512,2048,4096,8192
   -n 32,128 -r 5 -o json` — prefill-only and decode-only throughput, batch 1, median of 5 with
   stddev. Gives the "llama.cpp SYCL is ~5× slower than vLLM" hypothesis its most favorable
   llama.cpp number.
2. **Served, same harness for every engine (the comparison of record):**
   `uvx llama-benchy@0.4.0 --pp 512 2048 4096 8192 --tg 32 128 --concurrency 1 --runs 5
   --tokenizer <HF Ornith dir>` against each engine's OpenAI endpoint (llama.cpp via
   `llama-server`). Reports prompt-processing tok/s, TTFT and decode tok/s. Concurrency 8 is a
   second pass once batch 1 is complete.

Pinned for every run: context 16384, KV precision (llama.cpp f16 KV; vLLM `--kv-cache-dtype fp8`
as in its reference script — recorded as a difference), flash attention on where offered,
1 GPU, warm-up run discarded, identical prompt source (llama-benchy book text, local server).

## Recorded per run

Engine commit/tag + image digest, driver/compute-runtime/oneAPI/Level Zero versions (from the
container), kernel `6.18.54-Unraid`, GPU PCI link at run time (`current_link_speed/width` on
00:01.0), GPU power/frequency (if exposed by `xe` sysfs/hwmon), peak VRAM (from engine logs or
`/sys/class/drm/card1/device/tile0/...` if available), load time, prefill tok/s, TTFT, decode
tok/s, and a **coherence check**: llama-benchy's coherence test plus one fixed prompt whose
output text is saved beside the numbers.

## Output

`benchmarks/gguf/baseline.csv` columns:
`date,engine,engine_version,image_digest,model_file,model_sha256,weights_format,cross_format,gpu,pcie,harness,workload,concurrency,runs,metric,median,stddev,unit,coherent,log_path`
