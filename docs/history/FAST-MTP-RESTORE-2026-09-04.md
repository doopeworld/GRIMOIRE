# Fast Qwen MTP restore handoff — 2026-09-04

## Proven source

Commit `bd9b4bf` is the known-good Qwen MTP result:

- PP 4096: 2539.8 tok/s
- MTP k=3: 49.8 tok/s sustained
- coherent output
- 15.43 GiB

The exact verified variables are:

```text
GRIMOIRE_W4A8=1
GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_MTP_DRAFT_VOCAB=131072
GRIMOIRE_MTP_PROFILE=1
```

These must be absent:

```text
GRIMOIRE_MTP_EXACT_VERIFY
GRIMOIRE_MTP_ZERO_CACHE
GRIMOIRE_DECODE_GRAPH
GRIMOIRE_DEFER_MOE_GATHER
GRIMOIRE_BF16_QKV
GRIMOIRE_BF16_DN_QKV
```

The last three are Ornith flags and were already measured to reduce Qwen MTP
from 49.8 to 18.3 tok/s. The first three are later experiments and were not
part of the successful run.

## Restored in this commit

- `Grimoire::mtp_draft()` is byte-identical to `bd9b4bf` (verified by SHA-1 of
  the extracted function: `fbc7568c447874c7307fffc26bed7b060b03b96f`).
- The later compact/zero MTP KV-cache experiment was removed completely.
- The Qwen image recipe now contains only the proven variables above.
- Server checkpoint synchronization matches the proven profiled CLI path.
- The command graph remains available as an opt-in experiment but is disabled
  for Qwen; it caused `UR_RESULT_ERROR_DEVICE_LOST` on the B70.
- The full attention bridge and runtime dependency closure remain in the image
  for the proven ~2540 PP path.

## Not run before handoff

No rebuild or GPU test was started after this restoration. The user requested
that GPU work be minimal and that the branch be pushed if the usage limit was
near.

## Exact next actions

1. Build this branch once as the unified image `grimoire:b70-native`.
2. In the Qwen Unraid container, delete these variables rather than setting
   them to zero: `GRIMOIRE_MTP_EXACT_VERIFY`, `GRIMOIRE_MTP_ZERO_CACHE`, and
   `GRIMOIRE_DECODE_GRAPH`. Delete the three Ornith-only variables if present.
3. Set `GRIMOIRE_MTP=1`, or delete the container override so the embedded Qwen
   recipe supplies it.
4. Keep post arguments:
   `--model /models/Qwen3.8-27B-MXFP4-GRIMOIRE --proj mxfp4 --ctx 12288 --host 0.0.0.0 --port 6887`
5. Run one Qwen benchmark only. Confirm the startup log says the Qwen recipe
   applied, MTP loaded, and no listed variable is shown as `OVERRIDDEN`.
6. Stop the container immediately after the benchmark if more source work is
   required.

Do not work on DSpark until this known-good MTP path is rebuilt and checked.
