#pragma once

#include <sycl/sycl.hpp>
#include "b70/qfn_moe_routes.hpp"

namespace b70 {

// The router guarantees IDs in [0,E) or -1. The resident expert window is
// contiguous. Slots and tile buffers have worst-case capacity; only the
// device-produced offsets[experts] and tile_count describe live entries.
inline sycl::event launch_qfn_device_routes(sycl::queue& q, const int32_t* ids,
    int slots, int top_k, int expert_begin, int experts, int32_t* counts,
    int32_t* offsets, int32_t* inverse, int32_t* perm_token,
    int32_t* tile_count, QfnMoeTile* tiles, const std::vector<sycl::event>& deps = {}) {
    constexpr int WG = 128;
    const auto counted = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(experts) * WG, WG), [=](sycl::nd_item<1> it) {
            const int e = int(it.get_group_linear_id());
            const int lane = int(it.get_local_linear_id());
            const auto g = it.get_group();
            int prior = 0;
            // Each slot belongs to at most one expert, so local ordinals can
            // be stored in inverse without atomics or cross-group ordering.
            for (int b = 0; b < slots; b += WG) {
                const int s = b + lane;
                const bool match = s < slots && ids[s] == expert_begin + e;
                const int prefix = sycl::inclusive_scan_over_group(g, int(match), sycl::plus<int>{});
                if (match) inverse[s] = prior + prefix - 1;
                prior += sycl::group_broadcast(g, prefix, WG - 1);
            }
            if (!lane) counts[e] = prior;
        });
    });
    const auto laid_out = q.submit([&](sycl::handler& h) {
        h.depends_on(counted);
        h.single_task([=] {
            int off = 0, nt = 0;
            for (int e = 0; e < experts; ++e) {
                offsets[e] = off;
                const int end = off + counts[e];
                for (; off < end; off += qfn_moe_tile_rows)
                    tiles[nt++] = {e, off, sycl::min(qfn_moe_tile_rows, end - off)};
                off = end;
            }
            offsets[experts] = off;
            *tile_count = nt;
        });
    });
    return q.submit([&](sycl::handler& h) {
        h.depends_on(laid_out);
        h.parallel_for(sycl::range<1>(size_t(slots)), [=](sycl::id<1> s) {
            const int e = ids[s] - expert_begin;
            if (e < 0 || e >= experts) {
                inverse[s] = -1;
                return;
            }
            const int row = offsets[e] + inverse[s];
            inverse[s] = row;
            perm_token[row] = int(s[0]) / top_k;
        });
    });
}

} // namespace b70
