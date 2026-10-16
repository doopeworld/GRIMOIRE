// GRIMOIRE -- host GGUF reference dequantization. SPDX-License-Identifier: GPL-3.0-or-later
#ifndef B70_GGUF_DECODE_HPP
#define B70_GGUF_DECODE_HPP
#include <cstdint>
#include <string>

namespace b70 {
// Decode n_elements in a whole number of GGML blocks. dst has n_elements floats.
// Integer tensors require a weight/bias tensor name; metadata-like integer tensors are refused.
bool gguf_dequantize(uint32_t type, const void* src, uint64_t n_elements,
                     float* dst, std::string& err, const std::string& tensor_name = {});
float gguf_fp16_to_f32(uint16_t bits);
}
#endif
