// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "b70/qfn_moe_routes.hpp"
#include <array>
#include <span>
#include <stdexcept>

namespace qfn_xe2 {
inline constexpr int K=2560, N=1280, experts=128, group=128, slots=2048*10;
struct Nibble { size_t byte; int shift; };
constexpr Nibble lane_code(int e, int n, int k) {
    // MoEGEMM B_offset and make_moe_tensor<int4_t,'R'>.
    const size_t subbyte = size_t(n) * K + k;
    return {size_t(e) * N * K / 2 + subbyte / 2, int(subbyte % 2) * 4};
}
constexpr size_t lane_scale(int e, int n, int k) {
    return (size_t(e) * N * K / 2) * 2 / group + size_t(n) * (K / group) + k / group;
}

struct Allocations {
    size_t a, b, scales, zeros, offsets, counts, tiles, live_tiles, atomic, bf16_d, fp32_d, folded;
};
inline constexpr std::array allocation_fields{&Allocations::a, &Allocations::b, &Allocations::scales,
    &Allocations::zeros, &Allocations::offsets, &Allocations::counts, &Allocations::tiles,
    &Allocations::live_tiles, &Allocations::atomic, &Allocations::bf16_d, &Allocations::fp32_d,
    &Allocations::folded};
constexpr Allocations layer0_allocations(size_t tiles, bool needs_fold = true) {
    return {size_t(slots)*K*2, size_t(experts)*N*K/2, size_t(experts)*N*(K/group)*2,
            size_t(experts)*N*(K/group), (experts+1)*4, experts*4, tiles*sizeof(b70::QfnMoeTile),
            4, 4, size_t(slots)*N*2, size_t(slots)*N*4, needs_fold ? size_t(experts)*N*K/2 : 0};
}
constexpr bool legal_block2d(size_t base, size_t width, size_t pitch) {
    return !(base % 64) && width && !(width % 4) && pitch >= width && !(pitch % 4);
}

inline void check_layout(std::span<const int32_t> counts, std::span<const int32_t> offsets,
                         std::span<const b70::QfnMoeTile> tiles, const Allocations& alloc,
                         bool needs_fold = true) {
    const auto need = layer0_allocations(tiles.size(), needs_fold);
    for (auto field : allocation_fields)
        if (alloc.*field < need.*field) throw std::runtime_error("Xe2 short allocation");
    if (counts.size()!=experts || offsets.size()!=experts+1 || offsets[0]!=0)
        throw std::runtime_error("Xe2 bad route shape");
    size_t row = 0, t = 0;
    for (int e = 0; e < experts; ++e) {
        if (counts[e] < 0 || int64_t(row)+counts[e] > slots || offsets[e] != int(row))
            throw std::runtime_error("Xe2 bad expert prefix");
        const size_t rows = counts[e], end = row + rows;
        if (need.b <= lane_code(e,N-1,K-1).byte || alloc.scales < 2*(lane_scale(e,N-1,K-1)+1) ||
            alloc.zeros <= lane_scale(e,N-1,K-1) || end*K*2 > alloc.a ||
            end*N*2 > alloc.bf16_d || end*N*4 > alloc.fp32_d)
            throw std::runtime_error("Xe2 expert surface exceeds allocation");
        if (!legal_block2d(row*K*2,K*2,K*2) || !legal_block2d(row*N*2,N*2,N*2) ||
            !legal_block2d(lane_code(e,0,0).byte,K/2,K/2))
            throw std::runtime_error("Xe2 block A/B/D base or pitch");
        // The reference's last M128 tile is clipped by its expert tensor
        // height; compare that row range and every N64 expert descriptor.
        for (size_t start = row; start < end; start += 128) {
            const size_t block_end = std::min(start + 128, end);
            for (size_t r = start; r < block_end; ++r) {
                const size_t lane_a = row*K + (r-row)*K;
                const size_t lane_d = row*N + (r-row)*N;
                if (lane_a != r*K || lane_d != r*N || (lane_d+N)*2 > alloc.bf16_d ||
                    (r*N+N)*4 > alloc.fp32_d)
                    throw std::runtime_error("Xe2/N64 A/D strides differ");
            }
        }
        for (size_t start = row; start < end; start += b70::qfn_moe_tile_rows) {
            if (t == tiles.size() || tiles[t].expert != e || tiles[t].first_row != int(start) ||
                tiles[t].rows != int(std::min(end-start,size_t(b70::qfn_moe_tile_rows))))
                throw std::runtime_error("Xe2/N64 expert span differs");
            ++t;
        }
        row = end;
    }
    if (row != size_t(offsets.back()) || t != tiles.size()) throw std::runtime_error("Xe2 bad live extent");
}
} // namespace qfn_xe2
