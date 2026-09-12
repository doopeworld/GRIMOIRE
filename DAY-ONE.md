# Day one on the Tower

Read this when you sit down at the box. It is the short version; the
reasoning behind every step is in `CLAUDE.md` and the dated handoffs.

## 0. Before anything

```bash
ssh root@192.168.8.225
ls /mnt/storage/isos/grimoire-fuse    # if this is missing, the Unraid
                                      # ARRAY did not auto-start.  Start it.
                                      # It is NOT data loss.
cd /mnt/storage/isos/grimoire-fuse
git pull
```

## 1. One command

```bash
tools/preflight_b70.sh /models/<your-model-dir>
```

That builds in the right order (bridges first — rule 3), then runs, in
order, and stops at the first required failure:

| stage | what it proves |
| --- | --- |
| build | bridges + engine actually compile on this box |
| no-Torch check | the bridges are still pure SYCL/Level Zero |
| `make test`, `make test-correctness` | 12 host suites |
| `bin/test_k2_kernels` | every new kernel matches its host reference |
| `bin/test_k2_e2e` | the K2 engine path loads and generates |
| `bin/test_model_matrix` | 3 architectures × 7 projection formats |
| `bin/test_parallel_e2e` | PP and TP give the SAME tokens as one process |
| generate | real model, real prompt — **you read the output** |

Green preflight means the box is sane. It does not mean anything is
fast: no stage above produces a number, deliberately.

## 2. Dual GPU

GPU1 is now on OCuLink Gen4 x4, not USB4. The USB4 link tunnelled PCIe
and added large latency to every host/device transfer, and the pipeline
boundary is two of those per layer group. **Re-measure before redesigning
anything** — the thing that was slow may simply not be slow any more.

Pipeline parallel (the proven path, and what to use for a model that
does not fit on one card):

```bash
GRIMOIRE_PP_SPLIT=24 tools/pp2run.sh renderD129 renderD130 1800 pp \
  /grimoire/bin/grimoire -m /models/<dir> --proj fp8 -p "..." -n 64
```

Tensor parallel:

```bash
tools/tp2run.sh renderD129 renderD130 1800 tp \
  /grimoire/bin/grimoire -m /models/<dir> --proj fp8 -p "..." -n 64
```

Verified in a container, two processes, before you ever powered the box
on: PP and TP both reproduce the single-process tokens exactly, for a
dense model, a routed-MoE model and K2, in BF16 and FP8. What was NOT
verified there is the link itself — that is what these two commands
measure and nothing else can.

**Speculation is off whenever TP or PP is on.** `mtp_enabled() &&
!tp_enabled() && !pp_enabled()`. Dual-GPU today means no MTP and no
DFlash, so dual-GPU TG is slower per token than a single card that fits
the model. That is a real trade, not a bug to patch in a hurry. The load
banner says which mode you are in — read it.

## 3. Which format

One card, model fits: `--proj int4` (the W4A8 path, int8 XMX, ~367 TOPS).
Two cards, or precision matters: `--proj fp8` (fp8_e4m3; `mxfp8` measured
slower — 214 GB/s vs 320 in `gemv_decode.cpp`).

All seven formats load and generate on all three architectures
(`bin/test_model_matrix`), so a format that fails on a real checkpoint is
about that checkpoint's shapes, not about the format.

## 4. Before you believe any number

Rule 8, and it has caught every false lead in this project: **a config
that never generated readable text has not been verified.** A PASS on a
numeric self-check is not enough. Run `-p "<prompt>" -n 64` and read it.

`tools/benchy_qwen.sh` now stamps the sha256, mtime and git commit of the
binary it is about to measure, and takes `BENCHY_VERSION` to pin the
harness. Use it, so a number can be traced to a build.

## 5. If a card falls off the bus

`forcewake register returns 0xFFFFFFFF` needs a POWER CYCLE. There is no
software recovery. The known causes, all avoidable:

- a stale bridge `.so` (rule 3 — always `tools/build_bridges_b70.sh`)
- `timeout N docker run` (rule 6 — always go through `tools/tune.sh`)
- a W4A8 tile on a weight with `N % 256 != 0` (rule 4)
- reading `w.payload` after W4A8 freed it (rule 1)

## 6. What is still open

- **F07**: single-process `GRIMOIRE_PIPELINE` puts every kernel on
  device 0's queue. It is numerically correct but GPU1's weights cross
  the link every token. Use the multiprocess launchers above instead.
  Making it real means threading a per-layer queue through every
  launcher — architectural, not a patch.
- Speculation under TP/PP (section 2).
- MoVA's value projection reads its routing table back to the host once
  per layer. Fine at M=1, useless at 4096 tokens. Pack the experts
  expert-major before quoting a K2 prefill number.
