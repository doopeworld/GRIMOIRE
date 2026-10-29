# GGUF decoder goldens

Pinned source: llama.cpp gguf-py commit `711780ffc75b`. All expected floats except F64, I8/I16/I32/I64, Q8_1, Q8_K, Q1_0, and Q2_0 come from `gguf.quants.dequantize`. Those nine types lack gguf-py dequantizers; their expected floats come from the standalone C reference in `tests/gguf/reference_scalar.c`, compiled by the generator. Q8_K, Q1_0 and Q2_0 are copied from the pinned ggml dequantize_row_* arithmetic. Upstream has no dequantize_row_* for Q8_1 or scalar integers/F64, so their C conversions follow the pinned ggml-common.h layouts. The C reference shares no C++ decoder code.

The IQ tables in `src/gguf_decode_grids.hpp` contain the exact literals from the pinned ggml-common.h.

Command: `/tmp/gguf-golden-venv/bin/python tests/gguf/gen_golden.py` from the repository root. The generator imports `/tmp/ggufpy`.

Each `.bin` has little-endian `GGLD`, u32 type, u32 element count, u32 raw byte count, raw blocks, then f32 expected elements.
