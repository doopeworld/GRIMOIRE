// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "b70/qfn_moe_routes.hpp"

namespace b70 {
// A worker visits strictly increasing (or, with multiple N tiles, repeated)
// global M tile indices. Each expert contributes ceil(live_rows / M32)
// tiles, not capacity rows. Carry the expert cursor between assignments.
struct QfnMoeOffsetCursor {
    int expert = 0;
    int first_tile = 0;

    QfnMoeTile find(int m_tile, const int32_t* offsets, int experts) {
        for (; expert < experts; ++expert) {
            const int begin = offsets[expert], end = offsets[expert + 1];
            const int rows = end - begin;
            const int tiles = (rows + qfn_moe_tile_rows - 1) / qfn_moe_tile_rows;
            if (m_tile < first_tile + tiles) {
                const int row = begin + (m_tile - first_tile) * qfn_moe_tile_rows;
                return {expert, row, std::min(qfn_moe_tile_rows, end - row)};
            }
            first_tile += tiles;
        }
        return {-1, 0, 0};
    }
};
} // namespace b70
