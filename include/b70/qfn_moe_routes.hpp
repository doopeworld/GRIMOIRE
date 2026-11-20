#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace b70 {

struct QfnMoeRoutePlan {
    std::vector<std::int32_t> counts;     // One row count per expert, including zeros.
    std::vector<std::int32_t> offsets;   // Expert e owns [offsets[e], offsets[e+1]).
    std::vector<std::int32_t> perm_token; // Compact expert-major row -> input token.
    std::vector<std::int32_t> inverse;   // Original token-major slot -> compact row, or -1.
};

// ids is a row-major [rows, top_k] table of global expert IDs. Exactly -1 means
// absent; every other ID must be in [0, num_experts). Active slots retain their
// input order within each expert. Gather with the original top_k stride and
// weights, skipping inverse == -1. offsets.back() is the active row count.
// Only experts in [expert_begin, expert_begin + expert_count) have local rows;
// tile expert indexes are relative to expert_begin, and nonlocal slots keep -1.
// Zero rows and all-absent tables are valid.
inline QfnMoeRoutePlan make_qfn_moe_route_plan(const std::int32_t* ids, std::size_t id_count,
                                              int rows, int top_k, int num_experts,
                                              int expert_begin, int expert_count) {
    if (rows < 0 || top_k <= 0 || num_experts <= 0 || expert_count <= 0 ||
        expert_begin < 0 || expert_begin > num_experts || expert_count > num_experts - expert_begin) {
        throw std::invalid_argument("invalid QFN MoE route shape");
    }
    if (rows > std::numeric_limits<std::int32_t>::max() / top_k) {
        throw std::length_error("QFN MoE route slots exceed int32 range");
    }
    const auto slots = static_cast<std::size_t>(rows) * top_k;
    if (id_count != slots || (slots != 0 && !ids)) {
        throw std::invalid_argument("QFN MoE route table does not match its shape");
    }

    QfnMoeRoutePlan plan;
    plan.counts.resize(static_cast<std::size_t>(expert_count));
    const int expert_end = expert_begin + expert_count;
    for (std::size_t r = 0; r < slots; ++r) {
        const auto e = ids[r];
        if (e == -1) continue;
        if (e < 0 || e >= num_experts) {
            throw std::out_of_range("invalid QFN MoE expert ID");
        }
        if (e >= expert_begin && e < expert_end) ++plan.counts[e - expert_begin];
    }

    plan.offsets.resize(plan.counts.size() + 1);
    for (std::size_t e = 0; e < plan.counts.size(); ++e) {
        plan.offsets[e + 1] = plan.offsets[e] + plan.counts[e];
    }
    plan.perm_token.resize(plan.offsets.back());
    plan.inverse.assign(slots, -1);
    auto next = plan.offsets;
    for (std::size_t r = 0; r < slots; ++r) {
        const auto e = ids[r];
        if (e < expert_begin || e >= expert_end) continue;
        const auto p = next[e - expert_begin]++;
        plan.perm_token[p] = static_cast<std::int32_t>(r / top_k);
        plan.inverse[r] = p;
    }
    return plan;
}

inline QfnMoeRoutePlan make_qfn_moe_route_plan(const std::int32_t* ids, std::size_t id_count,
                                             int rows, int top_k, int num_experts) {
    return make_qfn_moe_route_plan(ids, id_count, rows, top_k, num_experts, 0, num_experts);
}

// A work-group takes one contiguous expert-major tile. No per-expert
// launch or device search for a row's owner is needed.
inline constexpr int qfn_moe_tile_rows = 32;
struct QfnMoeTile {
    int32_t expert;
    int32_t first_row;
    int32_t rows;
};

inline std::vector<QfnMoeTile> make_qfn_moe_tiles(const QfnMoeRoutePlan& plan) {
    std::vector<QfnMoeTile> tiles;
    tiles.reserve(plan.counts.size() + plan.perm_token.size() / qfn_moe_tile_rows);
    for (size_t e = 0; e < plan.counts.size(); ++e) {
        const int32_t end = plan.offsets[e + 1];
        for (int32_t off = plan.offsets[e]; off < end;) {
            const int32_t rows = std::min<int32_t>(qfn_moe_tile_rows, end - off);
            tiles.push_back({static_cast<int32_t>(e), off, rows});
            off += rows;
        }
    }
    return tiles;
}

} // namespace b70
