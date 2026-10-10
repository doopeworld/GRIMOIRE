// GRIMOIRE — Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef B70_INT8_EMBEDDING_HPP
#define B70_INT8_EMBEDDING_HPP
#include "b70/formats.hpp"

namespace b70 {
// Swift's reference casts BOTH gathered operands to BF16 before multiplying.
// INT8 codes are exact in BF16. Scales are rounded at upload; round the product
// too, then expose that BF16 value to GRIMOIRE's FP32 activation buffers.
inline float int8_embedding_value(int8_t code, bf16_t scale) {
    return bf16_to_f32(f32_to_bf16(float(code) * bf16_to_f32(scale)));
}
} // namespace b70
#endif
