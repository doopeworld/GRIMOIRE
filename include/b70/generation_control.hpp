#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace b70 {
// One control word at each emitted token, after all token-selection messages.
// Only rank zero owns cancellation. Workers must receive its answer before
// starting another decode step; a private callback decision would desync them.
// PP forwards toward the last stage; TP broadcasts from rank zero.
template<class Read, class Write>
bool generation_continue(bool local_keep, int upstream,
                         const std::vector<int>& downstream,
                         Read read, Write write) {
    constexpr uint32_t tag = 0x47430000u;
    uint32_t word = tag | uint32_t(local_keep);
    if (upstream >= 0 && !read(upstream, &word, sizeof word))
        throw std::runtime_error("generation cancellation control receive failed");
    if (word != tag && word != (tag | 1u))
        throw std::runtime_error("generation cancellation control phase mismatch");
    for (int fd : downstream)
        if (!write(fd, &word, sizeof word))
            throw std::runtime_error("generation cancellation control send failed");
    return (word & 1u) != 0;
}
} // namespace b70
