# Installing GRIMOIRE and loading a model

This guide covers the two ways to get the engine running on an Intel Arc Pro B70
(Battlemage G31): the prebuilt Docker image, and building from source. It ends with
loading a checkpoint and (optionally) a PEFT LoRA adapter. Adapter details and
restrictions are in [lora.md](lora.md).

## Requirements

- An Intel Arc Pro B70 (or any Battlemage card: B70/B65 = die `g31`, B580/B60/B50 = `g21`)
  with the `xe` kernel driver. The host needs nothing else -- the user-mode GPU driver,
  Level Zero, oneAPI and the engine all live in the container.
- Docker with the Compose plugin. Verify the card is visible on the host:

```
lspci | grep -i vga          # your card's PCI address, e.g. 08:00.0
ls -l /dev/dri/by-path/      # pci-0000:08:00.0-render -> ../renderDNNN (the render node)
```

The render node number can change after a reboot -- check it again every time.

- A checkpoint GRIMOIRE can read (see "Models" below): a Hugging Face safetensors
  folder (including GPTQ/AutoRound INT4) or a GGUF file.

## Option A -- Docker image (recommended)

### Prebuilt

```
docker pull ghcr.io/munikor/grimoire:latest     # if published; otherwise build (below)
```

### Build it yourself

The image installs the kobuk-PPA GPU user-mode driver, oneAPI DPC++ 2026.2 and `ocloc`,
runs the host test gates, then AOT-compiles the engine for the card die(s). Note: on some
boxes the stock `intel/oneapi:*-devel` image segfaults in `sycl::device::get_devices()`;
GRIMOIRE's own Dockerfile avoids that by pinning the PPA driver stack, so always build
from this repo's Dockerfile rather than running the engine in a stock oneAPI container.

```
git clone https://github.com/munikor/GRIMOIRE.git && cd GRIMOIRE
docker build -f docker/Dockerfile -t grimoire-b70 .
```

Both Battlemage dies are built by default. To build for one die only (faster):

```
docker build -f docker/Dockerfile --build-arg B70_TARGET=intel_gpu_bmg_g31 -t grimoire-b70 .
```

The image build runs `make test` (host-only gates, including the LoRA loader tests)
before the SYCL compile, so a broken tree fails early.

### Run

```
docker run -d --name grimoire --init --stop-timeout 300 \
    --device /dev/dri/renderD130 \
    -v /path/to/models:/models:ro \
    -p 8000:8000 \
    -e GRIMOIRE_SEQ_SLOTS=8 \
    grimoire-b70 \
    server --model /models/Qwen3.8-27B-GPTQ-Int4 --proj int4 --ctx 16384 --port 8000
```

- `--device` takes the render node you found above (`/dev/dri/by-path/pci-0000:08:00.0-render`
  is the stable spelling).
- `--init --stop-timeout 300` are required: killing a container with GPU work in flight can
  drop the card off the PCI bus until a power cycle.
- `GRIMOIRE_SEQ_SLOTS` = how many conversations decode together.
- The server is OpenAI-compatible at `http://<host>:8000/v1`.

### One-shot CLI (no server)

Same image, different command -- print a load report and benchmark:

```
docker run --rm --device /dev/dri/renderD130 -v /path/to/models:/models:ro \
    grimoire-b70 --model /models/Qwen3.8-27B-GPTQ-Int4 --proj int4
```

Generate from a prompt:

```
docker run --rm --device /dev/dri/renderD130 -v /path/to/models:/models:ro \
    grimoire-b70 --model /models/Qwen3.8-27B-GPTQ-Int4 --proj int4 \
    -p "The capital of France is" -n 128
```

## Option B -- build from source

You need oneAPI DPC++ 2026.x and an `ocloc` that knows Battlemage (Intel's
compute-runtime 24.35+; Ubuntu's stock `intel-ocloc` 23.43 is too old and answers
"Cannot get HW Info" for bmg_g31). Check with `ocloc ids bmg_g31` -- it must print
`Matched ids`.

```
./build_b70.sh                    # AOT build -> b70_native_inference, bin/grimoire, bin/grimoire-server
```

`build_b70.sh` probes ocloc per die and refuses to build if it cannot target what you
asked for. `B70_TARGET=intel_gpu_bmg_g31` (or `_g21`) narrows the dies when iterating.

For host-only iteration (no GPU, plain C++17):

```
make test                         # the full host gate suite, including bin/test_lora
```

For a quick SYCL build without ocloc (SPIR-V JIT at first launch instead of AOT):

```
make native SYCL_TARGET=spir64    # -> bin/native-spir64/grimoire, grimoire-server
```

Run the binary directly (the oneAPI env must be sourced so libsycl and the Level Zero
adapters are on the library path):

```
source /opt/intel/oneapi/setvars.sh
./bin/grimoire --model /models/Qwen3.8-27B-GPTQ-Int4 --proj int4 -p "hello" -n 32
```

## Models

Download a Hugging Face checkpoint into a folder:

```
hf download Qwen/Qwen3.8-27B-FP8 --local-dir /path/to/models/Qwen3.8-27B-FP8
```

`--proj` picks the format the projections are converted to at load (int4, int8, mxfp4,
mxfp8, fp8_e4m3, fp8_e5m2, bf16). Checkpoints already quantized (GPTQ INT4, MXFP4
compressed-tensors, GGUF Q4_K/Q6_K) keep their format when it matches. The table of
measured model/flag combinations is in the [README](README.md#models-downloads-launch-flags-and-measured-speed).

## Loading a LoRA adapter

Point `--lora` at a PEFT adapter directory (`adapter_config.json` +
`adapter_model.safetensors`); `GRIMOIRE_LORA=<dir>` is the environment equivalent. The
adapter is applied at every decode step and every prefill chunk by SYCL delta kernels --
it is not fused into the weights.

```
# CLI
docker run --rm --device /dev/dri/renderD130 -v /path/to/models:/models:ro \
    grimoire-b70 --model /models/Qwen3.8-27B --proj int4 \
    --lora /models/hermes-adapter -p "The capital of France is" -n 128

# server
docker run -d --name grimoire --init --stop-timeout 300 \
    --device /dev/dri/renderD130 -v /path/to/models:/models:ro -p 8000:8000 \
    grimoire-b70 server --model /models/Qwen3.8-27B --proj int4 \
        --lora /models/hermes-adapter --port 8000
```

On load the engine prints one line naming the adapter, its rank and scale:

```
LoRA adapter    /models/hermes-adapter (r=32, r16=32, scale=2.000) [Qwen/Qwen3.8-27B]
```

If the adapter is incompatible with the checkpoint (MoE trunk, TP/PP, a v-delta on a
router-routed layer, wrong shapes, non-F32 weights, ...) the load fails with a message
naming the offending tensor, layer or module -- see [lora.md](lora.md) for the full list.

## Two cards / picking a card

`ZE_AFFINITY_MASK` limits device visibility and `GRIMOIRE_DEVICES` remaps rank ->
device index (for a box mixing B70s and a smaller card). To keep a live service on one
card and run a test on the other, pass only the idle card's render node to the test
container:

```
docker run --rm --device /dev/dri/by-path/pci-0000:08:00.0-render ...
```

## Troubleshooting

- **No device / `sycl::device::get_devices()` returns 0** -- the container is missing the
  render node, or the render group. Add `--group-add $(stat -c %g /dev/dri/renderD130)`
  and check `ls -l /dev/dri/`.
- **Segfault at startup in a stock oneAPI image** -- use this repo's `docker/Dockerfile`;
  it pins a user-mode driver stack known to enumerate Battlemage (see the note above).
- **`ocloc ids bmg_g31` fails when building from source** -- Ubuntu's intel-ocloc is too
  old; install Intel's compute-runtime 24.35+ (the PPA the Dockerfile uses).
- **AOT binary prints a JIT compile on first launch** -- the binary was built with
  `SYCL_TARGET=spir64` or the card's die was not in `B70_TARGET`; rebuild with the right
  die. `SYCL_UR_TRACE=1` shows device compiles.
