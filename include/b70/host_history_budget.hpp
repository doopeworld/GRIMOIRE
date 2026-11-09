// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace b70 {
// Include container limits: /proc/meminfo alone describes the entire host.
inline uint64_t host_history_available_ram() {
    uint64_t available=0;
    std::ifstream mem("/proc/meminfo"); std::string line;
    while(std::getline(mem,line)) if(line.rfind("MemAvailable:",0)==0) {
        std::istringstream in(line.substr(13)); in>>available;
        available*=1024; break;
    }
    for(const auto& paths : {
        std::pair{"/sys/fs/cgroup/memory.max","/sys/fs/cgroup/memory.current"},
        std::pair{"/sys/fs/cgroup/memory/memory.limit_in_bytes","/sys/fs/cgroup/memory/memory.usage_in_bytes"}}) {
        std::ifstream lim(paths.first),cur(paths.second);
        uint64_t cap=0,used=0;
        if(lim>>cap && cur>>used)
            available=std::min(available,cap>used?cap-used:uint64_t(0));
    }
    return available;
}
// The first history on every local rank has already been allocated and
// touched. Additional slots must fit ALL ranks, sharing the same RAM pool.
inline int host_history_slot_limit(uint64_t available, uint64_t reserve,
                                   uint64_t history_bytes, unsigned ranks,
                                   int requested) {
    if(requested<=1 || !history_bytes)return std::max(0,requested);
    ranks=std::max(1u,ranks);
    if(history_bytes>std::numeric_limits<uint64_t>::max()/ranks)return 1;
    const uint64_t extra=available>reserve?(available-reserve)/(history_bytes*ranks):0;
    return 1+int(std::min(uint64_t(requested-1),extra));
}
} // namespace b70
