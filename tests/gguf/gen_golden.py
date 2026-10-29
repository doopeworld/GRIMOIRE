#!/usr/bin/env python3
"""Independent decoder goldens from llama.cpp 711780ffc75b gguf-py/C layouts.

Run: /tmp/gguf-golden-venv/bin/python tests/gguf/gen_golden.py
"""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import numpy as np

UPSTREAM_COMMIT = "711780ffc75b"
sys.path.insert(0, "/tmp/ggufpy")
from gguf import quants  # noqa: E402
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(711780)
NAMES = ("F32 F16 BF16 F64 I8 I16 I32 I64 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1 "
         "Q2_K Q3_K Q4_K Q5_K Q6_K Q8_K IQ1_S IQ1_M IQ2_XXS IQ2_XS "
         "IQ2_S IQ3_XXS IQ3_S IQ4_NL IQ4_XS TQ1_0 TQ2_0 MXFP4 NVFP4 Q1_0 Q2_0").split()
C_REFERENCE = {"F64", "I8", "I16", "I32", "I64", "Q8_1", "Q8_K", "Q1_0", "Q2_0"}

# Keep the parser's layout table checked against the independent gguf-py constants.
parser = (Path(__file__).resolve().parents[2] / "src" / "gguf.cpp").read_text()
table = parser.split("static const GGUFTypeInfo types[]", 1)[1].split("};", 1)[0]
entries = re.findall(r'\{(\d+),(\d+),(\d+),"([^"]+)"\}', table)
assert len(entries) == len(NAMES), "GGUF parser type table is incomplete"
for type_id, elems, size, name in entries:
    qtype = GGMLQuantizationType(int(type_id))
    assert name == qtype.name and (int(elems), int(size)) == GGML_QUANT_SIZES[qtype]


def set_fp16(blocks, offset):
    vals = rng.uniform(-0.05, 0.05, size=blocks.shape[0]).astype("<f2")
    blocks[:, offset:offset+2] = vals.view(np.uint8).reshape(-1, 2)


def save(name, qtype, raw, expected):
    expected = np.asarray(expected, dtype="<f4").reshape(-1)
    assert np.isfinite(expected).all(), name
    blob = b"GGLD" + struct.pack("<III", int(qtype), expected.size, raw.size)
    blob += raw.tobytes() + expected.tobytes()
    (OUT / f"{name}.bin").write_bytes(blob)


with tempfile.TemporaryDirectory() as tmp:
    ref = Path(tmp) / "scalar_ref"
    subprocess.run(["cc", "-std=c11", "-O2", "-ffp-contract=off",
                    str(Path(__file__).with_name("reference_scalar.c")), "-o", str(ref)], check=True)
    for name in NAMES:
        qtype = getattr(GGMLQuantizationType, name)
        elems, size = GGML_QUANT_SIZES[qtype]
        count = 8 if elems > 1 else 257
        blocks = rng.integers(0, 256, size=(count, size), dtype=np.uint8)
        if name in ("F32", "F16", "BF16", "F64"):
            values = rng.uniform(-200, 200, size=count).astype(np.float32)
            if name == "F32": blocks = values.astype("<f4").view(np.uint8).reshape(-1, 4)
            if name == "F16": blocks = values.astype("<f2").view(np.uint8).reshape(-1, 2)
            if name == "BF16":
                bits = values.view(np.uint32)
                b16 = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype("<u2")
                blocks = b16.view(np.uint8).reshape(-1, 2)
            if name == "F64": blocks = values.astype("<f8").view(np.uint8).reshape(-1, 8)
        if name in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
                    "Q4_K", "Q5_K", "IQ2_XXS", "IQ2_XS", "IQ2_S",
                    "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ4_NL", "IQ4_XS", "Q1_0", "Q2_0"):
            set_fp16(blocks, 0)
        if name in ("Q4_1", "Q5_1", "Q4_K", "Q5_K"):
            set_fp16(blocks, 2)
        if name == "Q2_K":
            set_fp16(blocks, 80);set_fp16(blocks, 82)
        if name == "Q3_K": set_fp16(blocks, 108)
        if name == "Q6_K": set_fp16(blocks, 208)
        if name == "Q8_K":
            vals = rng.uniform(-0.05, 0.05, size=count).astype("<f4")
            blocks[:, :4] = vals.view(np.uint8).reshape(-1, 4)
        if name == "IQ1_M":
            # d is composed from four high nibbles of the last four u16s.
            d = rng.uniform(-0.05, 0.05, size=count).astype("<f2").view("<u2")
            for i in range(4):
                word = blocks[:, 48+2*i:50+2*i].copy().view("<u2").reshape(-1)
                word = (word & 0x0FFF) | (((d >> (4*i)) & 15) << 12)
                blocks[:, 48+2*i:50+2*i] = word.astype("<u2").view(np.uint8).reshape(-1, 2)
        if name in ("TQ1_0", "TQ2_0"): set_fp16(blocks, size-2)
        if name == "MXFP4": blocks[:, 0] = rng.integers(100, 150, size=count, dtype=np.uint8)
        if name == "NVFP4":
            blocks[:, :4] = rng.integers(0, 256, size=(count, 4), dtype=np.uint8)
            blocks[0, :4] = [0, 0x7F, 0x80, 0xFF]
        raw = blocks.reshape(-1)
        if name in C_REFERENCE:
            result = subprocess.run([str(ref), str(int(qtype)), str(elems*count)],
                                    input=raw.tobytes(), stdout=subprocess.PIPE, check=True)
            expected = np.frombuffer(result.stdout, dtype="<f4")
        else:
            expected = quants.dequantize(raw, qtype).reshape(-1)
        assert expected.size == elems*count, name
        save(name, qtype, raw, expected)

(OUT / "README.md").write_text(
    "# GGUF decoder goldens\n\n"
    f"Pinned source: llama.cpp gguf-py commit `{UPSTREAM_COMMIT}`. All expected floats "
    "except F64, I8/I16/I32/I64, Q8_1, Q8_K, Q1_0, and Q2_0 come from `gguf.quants.dequantize`. "
    "Those nine types lack gguf-py dequantizers; their expected floats come from "
    "the standalone C reference in `tests/gguf/reference_scalar.c`, compiled by "
    "the generator. Q8_K, Q1_0 and Q2_0 are copied from the pinned ggml "
    "dequantize_row_* arithmetic. Upstream has no dequantize_row_* for "
    "Q8_1 or scalar integers/F64, so their C conversions follow the pinned "
    "ggml-common.h layouts. The C reference shares no C++ decoder code.\n\n"
    "The IQ tables in `src/gguf_decode_grids.hpp` contain the exact literals "
    "from the pinned ggml-common.h.\n\n"
    "Command: `/tmp/gguf-golden-venv/bin/python tests/gguf/gen_golden.py` "
    "from the repository root. The generator imports `/tmp/ggufpy`.\n\n"
    "Each `.bin` has little-endian `GGLD`, u32 type, u32 element count, "
    "u32 raw byte count, raw blocks, then f32 expected elements.\n"
)
print(f"wrote {len(NAMES)} goldens ({sum(p.stat().st_size for p in OUT.glob('*.bin'))} bytes)")
