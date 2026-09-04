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

## Do NOT set GRIMOIRE_MTP_EXACT_VERIFY on Qwen

It forces the accurate-but-slow lm_head path at M<=4. It was only ever needed
to mask the stale-a8-cache bug fixed in a1a92eb.

## Bridges must actually load

libgrimoire_xe2_attention_bridge.so provides BOTH chunk GDN and chunk prefill.
If it fails to dlopen, PP silently halves: 1032 vs 2540 tok/s. Check for
'Xe2 chunk prefill unavailable' in the log -- it must NOT appear.
'Xe2 grouped GEMM unavailable' IS expected and benign (stale Aug-23
libgrimoire_xe2_bridge.so, superseded by libgrimoire_xe2_grouped.so).

The bridge needs libtorch + vllm_xpu_kernels linked in (see
tools/build_bridges_b70.sh) and the runtime LD_LIBRARY_PATH that tune.sh sets.

## TG numbers are meaningless without the prompt length

Every high-TG record in this repo was measured on a SHORT prompt. The 49.8
above came from `-p 'Explain why the sky is blue.'` (~7 tokens).
MTP-FINAL-51.7TG-20260826.log records 51.7 TG on an 18-token prompt
(94 MTP steps, 2.76 committed/step, 165/222 = 74% acceptance).

llama-benchy measures tg32 after a ~4500-token prompt. That is a different
measurement and the two are NOT comparable. Measured 2026-09-04 on one
grimoire-server process, same build, same flags, in one run:

| prompt   | acceptance | tok/round | TG        |
|---------:|-----------:|----------:|----------:|
|   23 tok |      53.3% |      2.60 | 46.8 t/s  |
| 4500 tok |    11-26%  | 1.00-1.68 | 13-22 t/s |

Verify cost is FLAT across that range (48 -> 59 ms/round). The entire
difference is MTP draft acceptance.

The only long-context record that exists is a1a92eb: 4070-token prompt,
GRIMOIRE_MTP_EXACT_VERIFY=1, k=3, 92% acceptance, **TG 37.6**. That is the
number to beat at 4k context. Quoting 49.8 against a llama-benchy run is
comparing a 7-token prompt to a 4500-token one, and it cost a full day.

Note also that the "Do NOT set GRIMOIRE_MTP_EXACT_VERIFY" guidance above
contradicts a1a92eb, which documents TWO effects and fixed only one:
stale a8 cache 0%->69% (fixed), int8 precision 92%->69% (still open).
EXACT_VERIFY covers the second. It is the right flag at 4k context and the
wrong one at 7 tokens, which is why both claims look true in isolation.
