// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>

namespace b70 {

// Router IDs need translation whenever the weight table is a slice. This is
// independent of the transport; tp_enabled() describes legacy TP only.
constexpr bool moe_routes_need_remap(int resident_count, int global_count) {
    return resident_count < global_count;
}

// Translate global expert IDs in place to the resident table on this rank.
// A rank with no matching routes must contribute zero; -1 means absent.
// Submit on the same in-order queue as the router and the resident GEMVs.
inline sycl::event launch_remap_moe_routes(sycl::queue& q, int32_t* ids,
                                          size_t slots, int begin, int end) {
    return q.parallel_for(sycl::range<1>(slots), [=](sycl::id<1> i) {
        const int32_t global = ids[i[0]];
        ids[i[0]] = (global >= begin && global < end) ? global - begin : -1;
    });
}

} // namespace b70
