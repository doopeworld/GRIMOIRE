// GRIMOIRE -- host GGUF reference dequantization. SPDX-License-Identifier: GPL-3.0-or-later
#ifndef B70_GGUF_DECODE_HPP
#define B70_GGUF_DECODE_HPP
#include <cstdint>
#include <string>

namespace b70 {
// Decode n_elements in a whole number of GGML blocks. dst has n_elements floats.
// F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K are supported; all others return false.
bool gguf_dequantize(uint32_t type, const void* src, uint64_t n_elements,
                     float* dst, std::string& err);
float gguf_fp16_to_f32(uint16_t bits);
}
#endif
