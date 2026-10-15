# GGUF decoder goldens

Source: llama.cpp gguf-py at commit `711780ffc75b`. Q8_0, Q4_K, Q5_K, Q6_K, and F16 expected floats come from `gguf.quants.dequantize`; BF16 uses the independent NumPy bit-layout conversion.

Command: `/tmp/gguf-golden-venv/bin/python tests/gguf/gen_golden.py` from the repository root. The generator imports `/tmp/ggufpy`, a copy of upstream gguf-py at the recorded commit.

Each `.bin` contains little-endian `GGLD`, u32 type, u32 element count, u32 raw byte count, raw blocks, and f32 expected elements.
