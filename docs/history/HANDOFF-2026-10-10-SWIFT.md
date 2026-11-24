# Swift comparison preparation — 2026-10-10

## Current status: PAUSED by Ian after validated Swift work

Ian asked to finish this work, make a backup and pause everything for tomorrow.
The fix and matched Swift table are complete. All agent GPU test containers have
exited normally; no queued job or automatic restart exists. Preserve ComfyUI.
Resume only when Ian asks. GPU1 remains off limits.

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

## Resumed implementation

Ian resumed on10-10, then asked to finish this work, back it up and pause for
tomorrow. GPU0 is still03:00.0[8086:e223]/renderD128 after the same09:38 reboot;
no GPU containers at resume. Preserve ComfyUI, GPU1 untouched, all jobs foreground.

- Upstream code rounds gathered scales to BF16 BEFORE multiplication and rounds
  its output to BF16. Implemented that exact arithmetic; store signed INT8 codes
  and pre-rounded BF16 scales (same16-bit footprint) on GPU, no dense expansion.
- Manifest selects side file and key names explicitly, independent of sort order.
  Require I8[vocab,H], F16[vocab,1], valid recipe, finite nonnegative scales.
  Zero scales support padding. Unscaled rawI8 embeddings now reject.
- Decode, batch prefill, graph replay, TP shard lookup, MTP and shared DFlash
  embedding dispatch use the resident format. Tied/Muse FP16 paths are refused
  for this new recipe, rather than silently changing their arithmetic.
- HF packed GPTQ/INT4 MTP keeps its saved encoding by default; explicit head-FMT
  overrides remain available. Old BF16 MTP defaults remain unchanged.
- CPU independent rational BF16 oracle51200 values PASS. Old raw-code negative
  control mismatches50490, late-scale-rounding control mismatches11058. Loader
  cases cover side-name order, dense legacy, malformed shape/dtype/recipe,
  missing side, negative/NaN scales, and no-recipeI8 rejection.
- Actual Swift metadata resolves correct INT8 source, attachedFP16 scales,
  GPTQ output head and all8 GPTQ MTP projections. No framework dependency added.
- Existing safetensors ingest suite ALL PASS. GPU production lookup atH256/H5120:
  single/batch/shard/dynamic-graph matches oracle BIT FOR BIT. Guard clean.
- Swift135168ctx/one-slot/K6/vocab65536/nativeINT4MTP:4K/8K four coded answers PASS;
  repeatdecode114.0/111.1. BF16 drafting gives SAME answers/acceptance but slows
  to95.2/93.4. Native retained. Full-vocab pilot running, then five-context table.
- Build failures corrected before measurement: detached worktree lacked bin/;
  CPU image has no python3 (generate fixtures on host); graph include is graph.hpp.
  No failed/missing test was promoted as a result.

Candidates in detached grimoire-swift-20261010, production mainbin still earlier
validated checkpoint. Logs/scripts bench-swift-1010 outside git. Next: complete
configuration/table checks, preserve original-model smoke, install when idle,
save source+raw data+handoff via git and Mac copies, then pause all agent work.

### Configuration check

Native INT4 MTP / draftvocab65536: repeat114.0/111.1 at4K/8K.
BF16 MTP / samevocab:95.2/93.4, identical answers and acceptance, allPASS.
Native INT4 MTP / fullvocab248320:110.7/102.3, acceptance5.32/5.06
versus4.89/4.91. AllPASS; added head traffic outweighs acceptance gain.
Selected native INT4/65536 for five-context curve. These are single observations
with one repeat, not randomized statistical estimates. Fullcurve running, eight
answers through64K PASS;128K fresh158.77s/61.4tok/s/5.07tokens per streamed update
PASS. Original-Qwen prior128K65.8/5.42; derived stream-update interval is similar
(~82ms) so lower observed tokens/s here tracks smaller output batches. This
is an SSE estimate, not a measured GPU-kernel or internal-step duration.

## Final checkpoint

- Actual Swift K6/65536/native INT4 drafting: 10/10 coded answers PASS, natural EOS.
  Fresh prompt tok/s at 4K/8K/16K/64K/128K: 1967/2000/1853/1265/827.
  Repeat decode: 113.9/111.0/107.2/82.1/61.3. Report RESULTS-2026-10-10-SWIFT.md.
  The model now matches Wrapzii's exact bake. Hardware/config/sample methods differ;
  his latest 128K is still 1255/78.0, so no universal performance win is claimed.
- Targeted original-model checks on candidate CLI: ordinary/exact/legacy-exact
  96-token GPTQ outputs IDENTICAL, speculation exercised. MXFP4 Sherlock n24
  IDENTICAL to saved reference. Logs legacy-solo/ and legacy-sherlock*.
- CPU oracle and loader cases PASS; existing safetensors suite ALL PASS; GPU lookup
  single/batch/shard/dynamic graph bit-exact at H256/H5120. Full 44-model sweep was
  done BEFORE this Swift patch; it has not been rerun after this patch. Report
  targeted evidence honestly. Multi-GPU behavior and new release image unvalidated.
- Installed validated CLI/server/libs in main bin with GPU0 idle. Old files kept in
  bench-swift-1010/main-before-swift. Installed SHA256 in installed-sha256.txt.
  Published v1.9.0 image/asset remains unchanged and its warning still applies.
- Source checkpoint, raw receipts, scripts, numerical fixtures and runtime files
  backed up via GitHub push and Mac copies. No tar archive, model rewrite or release
  publication. No GPU containers or queued GPU jobs; GPU0 guard clean, GPU1 untouched.
- Intel cloud still in review after 28h per Ian; no SSH details, no cloud job.

## Resume commands and priorities

1. SSH root@192.168.8.225. If LAN unreachable, check mullvad lan get is allow.
   cd /mnt/storage/isos/grimoire-fuse; branch longctx-attn-wip. Read CLAUDE.md
   and this handoff. Check PCI/by-path after any reboot; GPU0=03:00.0/renderD128.
2. CPU oracle: python3 tests/make_int8_embedding_fixtures.py <fixture-dir>.
   clang++ -O2 -std=c++20 -I include tests/test_int8_embedding.cpp
   src/qwen35_loader.cpp src/safetensors.cpp src/native_model.cpp -o <host-check>.
   <host-check> <fixture-dir> [<actual-Swift-directory>]. CPU image lacks python3:
   generate fixtures on host and run the compiled checker in CPU-only container.
3. GPU probe is tools/test_int8_embedding_gpu.cpp, linked with src/ops.cpp using
   -ffunction-sections/-fdata-sections/-Wl,--gc-sections. Build script in
   bench-swift-1010/build-swift-gpu-probe.sh; g0run wrapper run-swift-embedding-probe.sh.
4. Matched table: bench-swift-1010/run-swift-table.sh runs FOREGROUND, ctx135168,
   one slot, K6/vocab65536, native saved INT4 MTP, same Swift bake, no benchy.
   Never queue it behind another GPU workload; never kill in-flight work; stop server
   immediately after tests. Only behavior-PASS decode rates count.
5. Next performance work: profile retained Swift at128K with TIME_LAYER=all before
   picking attention/FFN target. PROFILE_PREFILL large-M event retention previously
   stalled. Direct FP8 prefill cache loading is only an idea, no measured gain.
   Do not redo rejected variants without new evidence. Consider full regression
   after this new format support before any release image work.
6. Commit bodies must list Findings and Next steps. Source stays pure C++/SYCL;
   do not import competitor vLLM/torch/oneDNN plug-ins into GRIMOIRE.
