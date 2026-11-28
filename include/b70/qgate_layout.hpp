#pragma once

#include <cstdint>

namespace b70 {

struct QProjRows { int begin, count; };

// Select the checkpoint rows for the rank's contiguous range of Q heads.
constexpr QProjRows q_proj_head_rows(int rows, int all_heads, int dim,
                                     int head_begin, int head_count) {
    const int per_head = rows == 2 * all_heads * dim ? 2 * dim : dim;
    return {head_begin * per_head, head_count * per_head};
}

struct QGateOffsets { int64_t q, gate; };

// Input [token][head][q | gate][dim], output [token][head][dim].
constexpr QGateOffsets qgate_offsets(int64_t linear, int dim) {
    const int64_t head = linear / dim;
    const int64_t q = 2 * head * dim + linear % dim;
    return {q, q + dim};
}

} // namespace b70
