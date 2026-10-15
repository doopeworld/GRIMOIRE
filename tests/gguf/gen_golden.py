#!/usr/bin/env python3
"""Generate independent GGUF decoder goldens from pinned upstream gguf-py.

Source: llama.cpp commit 711780ffc75b, gguf-py/gguf/quants.py and constants.py.
Run: /tmp/gguf-golden-venv/bin/python tests/gguf/gen_golden.py
"""
from pathlib import Path
import struct
import sys
import numpy as np

UPSTREAM_COMMIT = "711780ffc75b"
sys.path.insert(0, "/tmp/ggufpy")
from gguf import quants  # noqa: E402
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(711780)


def save(name, type_id, raw, expected):
    assert expected.dtype == np.float32
    assert np.isfinite(expected).all()
    blob = b"GGLD" + struct.pack("<III", type_id, expected.size, raw.size)
    blob += raw.tobytes() + expected.astype("<f4", copy=False).tobytes()
    (OUT / f"{name}.bin").write_bytes(blob)


for name in ("Q8_0", "Q4_K", "Q5_K", "Q6_K"):
    qtype = getattr(GGMLQuantizationType, name)
    block_elems, block_bytes = GGML_QUANT_SIZES[qtype]
    blocks = rng.integers(0, 256, size=(5, block_bytes), dtype=np.uint8)
    # Confirmed from upstream quants.py: Q8_0 d=0; K4/K5 d=0,dmin=2;
    # Q6_K has ql[0:128], qh[128:192], scales[192:208], d[208:210].
    offsets = (0, 2) if name in ("Q4_K", "Q5_K") else ((208,) if name == "Q6_K" else (0,))
    for off in offsets:
        finite = rng.uniform(-0.05, 0.05, size=5).astype("<f2")
        blocks[:, off:off + 2] = finite.view(np.uint8).reshape(5, 2)
    raw = blocks.reshape(-1)
    expected = quants.dequantize(raw, qtype).reshape(-1)
    assert expected.size == 5 * block_elems
    save(name, int(qtype), raw, expected)

for name in ("F16", "BF16"):
    values = rng.uniform(-200, 200, size=257).astype(np.float32)
    if name == "F16":
        raw = values.astype("<f2").view(np.uint8)
        expected = quants.dequantize(raw, GGMLQuantizationType.F16).reshape(-1)
        qtype = GGMLQuantizationType.F16
    else:
        # BF16 is defined as its 16 bits in the high half of an f32 word.
        bits = values.view(np.uint32)
        b16 = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype("<u2")
        raw = b16.view(np.uint8)
        expected = (b16.astype(np.uint32) << 16).view(np.float32)
        qtype = GGMLQuantizationType.BF16
    save(name, int(qtype), raw, expected)

(OUT / "README.md").write_text(
    "# GGUF decoder goldens\n\n"
    f"Source: llama.cpp gguf-py at commit `{UPSTREAM_COMMIT}`. "
    "Q8_0, Q4_K, Q5_K, Q6_K, and F16 expected floats come from "
    "`gguf.quants.dequantize`; BF16 uses the independent NumPy bit-layout conversion.\n\n"
    "Command: `/tmp/gguf-golden-venv/bin/python tests/gguf/gen_golden.py` "
    "from the repository root. The generator imports `/tmp/ggufpy`, a copy "
    "of upstream gguf-py at the recorded commit.\n\n"
    "Each `.bin` contains little-endian `GGLD`, u32 type, u32 element count, "
    "u32 raw byte count, raw blocks, and f32 expected elements.\n"
)
print(f"wrote {len(list(OUT.glob('*.bin')))} goldens from gguf-py {UPSTREAM_COMMIT}")
