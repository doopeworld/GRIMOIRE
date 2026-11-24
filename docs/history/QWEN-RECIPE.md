# Qwen3.8-27B canonical recipe (verified 2026-09-01)

    GPU=gpu0 LIM=900 EXTRA_ENV='GRIMOIRE_W4A8=1
    GRIMOIRE_MTP=1
    GRIMOIRE_MTP_K=3
    GRIMOIRE_MTP_PROFILE=1
    GRIMOIRE_MTP_DRAFT_VOCAB=131072' \
      bash tools/tune.sh qwen /grimoire/bin/grimoire \
      -m /models/Qwen3.8-27B-MXFP4-GRIMOIRE \
      --proj mxfp4 --ctx 8192 -p 'Explain why the sky is blue.' -n 256

Measured: PP 4096 = 2539.8 tok/s, MTP(k=3) TG = 49.8 tok/s, 15.43 GiB,
text coherent. vLLM on the same box: 2015 PP / 58.1 TG (MTP k=3).

## Use tools/tune.sh for Qwen, NEVER tools/qrun.sh

qrun.sh injects GRIMOIRE_DEFER_MOE_GATHER / BF16_QKV / BF16_DN_QKV. Those are
ORNITH flags. On Qwen they cut MTP decode from 49.8 to 18.3 tok/s (verify
4.6 s -> 13.2 s) while leaving PP alone. This is the "mixing flags fakes a
regression" trap; it cost a full debugging session.

## GRIMOIRE_MTP_DRAFT_VOCAB=131072 is required

Omitting it lowers acceptance. 32K and 64K lower it further.

## Do not alter the verified fast-MTP recipe

Do not set `GRIMOIRE_MTP_EXACT_VERIFY`, `GRIMOIRE_MTP_ZERO_CACHE`, or
`GRIMOIRE_DECODE_GRAPH`. They are not part of the verified 49.8 TG path in
`bd9b4bf`. Do not add the Ornith-only flags listed above.

## Bridges must actually load

libgrimoire_xe2_attention_bridge.so provides BOTH chunk GDN and chunk prefill.
If it fails to dlopen, PP silently halves: 1032 vs 2540 tok/s. Check for
'Xe2 chunk prefill unavailable' in the log -- it must NOT appear.
'Xe2 grouped GEMM unavailable' IS expected and benign (stale Aug-23
libgrimoire_xe2_bridge.so, superseded by libgrimoire_xe2_grouped.so).

The bridge needs libtorch + vllm_xpu_kernels linked in (see
tools/build_bridges_b70.sh) and the runtime LD_LIBRARY_PATH that tune.sh sets.

The server must use the same MTP implementation and variables as the proven
CLI result before any new MTP experiment is considered.
