# Swift comparison preparation — 2026-10-10

## Current constraint

Ian: **do not use GPUs for now**. Work is CPU-only until he resumes GPU use.
Preserve ComfyUI. No GPU benchmark, model upload or queued GPU job is authorized.

## Findings

- The preceding comparison used Qwen3.8-27B-GPTQ-Int4-MTP-BF16 because it was
  the checkpoint named in the original handoff. Wrapzii used Swift. The different
  weights limit that comparison; the next comparison target is his exact Swift bake.
- Already downloaded completely on Tower:
  `/mnt/storage/Models/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8`.
  Download log says DL_DONE, all sizes match. This inspection read headers and code
  only; no GPU program was started.
- Config: hidden5120, intermediate17408, 64layers, H24/KVH4/D256, GPTQ group128,
  symmetric, no activation ordering. Bake has GPTQ INT4 output head and MTP linears.
- Original index maps embedding to BF16[248320,5120] in model-00002-of-00005.
  Side file model-embed-int8.safetensors contains the SAME weight name as
  I8[248320,5120], with F16[248320,1] weight_scale.
- Critical static finding: HFModel::discover actually enumerates all sorted
  .safetensors files despite its index-preference comment. Qwen35Model::load
  overwrites duplicate names in that order. The INT8 side file thus supersedes
  the dense embedding. Embedding resolution is raw get(), with no scales attached.
- Grimoire::build uploads embed through dev_copy_t<bf16_t>. For I8, that calls
  SafeTensors::read_f32, whose conversion returns float(integer_code). It does
  not apply the separate embedding scales. Using this directory unchanged would
  load incorrect embedding values. This is a code-inspection conclusion, not
  a GPU reproduction. Do not benchmark it as a working matched checkpoint yet.
- GRIMOIRE supports ordinary Swift GPTQ fine-tunes (README), but that does not
  establish compatibility with Wrapzii's INT8 embedding bake.
- Upstream source: https://github.com/Wrapzii/k8v4-xpu/blob/main/docs/swift-1.5-bake.md.
  His runtime uses the side-file INT8 codes/scales and BF16 output; preserving its
  exact rounding needs inspection of his embedding implementation before coding.

## Next steps

1. Fix embedding resolution and scaled INT8 loading using pure C++/SYCL, with
   clear selection and duplicate handling. Match his actual dequantization arithmetic.
   CPU checks can be prepared now. Resident INT8 is needed to match its memory
   footprint; expanding scaled values to BF16 would match values but use more VRAM.
2. Check all baked GPTQ head/MTP projections retain the saved quantized values.
   Match thinking-off template and prompts. Do not infer a speed gain from fine-tuning.
3. After Ian explicitly resumes GPU use, validate the actual Swift model on GPU0
   only and rerun 4K/8K/16K/64K/128K. Keep only behavior-PASS decode rates.
4. Hardware remains one B70 versus his published two B60s. No GPU job queued.
