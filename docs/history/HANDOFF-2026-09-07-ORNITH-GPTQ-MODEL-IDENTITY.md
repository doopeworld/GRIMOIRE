# Ornith 1.5 GPTQ model identity backup — 2026-09-07

## Verified model in use

The model intended for the next session is:

`SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2`

Hugging Face: <https://huggingface.co/SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2>

Tower local path:

`/mnt/storage/Models/Ornith-1.5-35B-A3B-GPTQ-Int4`

This local directory is not an empty rename. Read-only verification on Tower
found the MixedCal-v2 README, six safetensors shards, `quant_log.csv`,
`quantize_config.json`, and `b70-artifact-contract.json`. The contract reports:

```text
tensor_keys:             124611
shards:                  6
qweight_tensors:         30720
expert_qweight_tensors:  30720
mtp_tensors:             785
mtp_quantized_tensors:   0
forbidden_qweight_tensors: 0
artifact_bytes:          24454916052
```

The installed `config.json` identifies `Qwen3_5MoeForConditionalGeneration`,
`model_type: qwen3_5_moe`, `dtype: float16`, 256 experts, top-8 routing, 40
layers, and one MTP layer. Its GPTQ configuration is symmetric INT4, group
size 128, `desc_act: false`, packed as int32. The dynamic exclusions leave
attention/GDN, the router, shared experts, embeddings/lm-head, vision, and the
entire `mtp.*` tree unquantized. Therefore this is an experts-only GPTQ INT4
artifact with BF16/FP16 non-expert and MTP weights, not an MXFP4 checkpoint.

The local README says this is a local Intel Arc Pro B70 conversion of
`ornith-ai/Ornith-1.5-35B-A3B`, not an official Ornith GPTQ conversion. It also
states that the published research default is MTP1; this is not a DFlash2
checkpoint.

## Important correction to previous experiments

The earlier Grimoire runs used:

`/models/Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE` with `--proj mxfp4`.

That is a separate 36G artifact on Tower. It is not the verified
`SergiioB/...MixedCal-v2` GPTQ model and must not be used as evidence about the
GPTQ loader, GPTQ MoE dispatch, MTP, or DFlash behavior. The earlier MXFP4
MoE/device-loss conclusions remain useful only for the MXFP4 path.

The repository’s latest pushed Tower commit at this backup is:

`973aef9 wip: align Ornith MoE dispatch with vLLM`

Branch: `qwen-tg-restored` on `root@192.168.8.225:/mnt/storage/isos/grimoire-fuse`.
Tower also has pre-existing untracked test/build artifacts; they are preserved
and intentionally excluded from this documentation backup.

## Correct next baseline

Before investigating MTP or any speculative method, load the verified GPTQ
artifact through the GPTQ path. The model card’s conservative baseline is:

```bash
vllm serve /model \
  --quantization gptq --dtype float16 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype auto --block-size 64 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching --language-model-only --trust-remote-code
```

Only after no-spec baseline generation works should MTP1 be enabled:

```bash
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Do not begin with `--proj mxfp4`, DFlash2, MTP2/MTP4, or a forced FP8 KV cache.
Those are different variables from the verified model identity. The user’s
usual parser flags (`--enable-auto-tool-choice`, `--tool-call-parser
qwen3_coder`, and `--reasoning-parser qwen3`) can be added after baseline
serving; they affect request parsing/output handling, not model weight loading.

## Image/version note

The custom image is rebuilt from:

```bash
cd /mnt/user/appdata/vllm-src
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.xpu \
  -t my-vllm-xpu:latest --shm-size=4g .
```

The previously inspected image contains the custom vLLM XPU stack. No rebuild,
GPU run, or source-code change was made during this identity backup.

## Sources

- Exact checkpoint: <https://huggingface.co/SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2>
- Checkpoint configuration: <https://huggingface.co/SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2/blob/main/config.json>
- Artifact contract: <https://huggingface.co/SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2/blob/main/b70-artifact-contract.json>

