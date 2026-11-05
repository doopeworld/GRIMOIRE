// The CPU-expert pool gives the same output whatever its thread count, across start/stop cycles.
// CPU only.  icpx -fsycl -O2 -std=c++20 -I include -I src tests/test_cpu_expert_pool.cpp src/cpu_experts.cpp -lpthread
// Add -fsanitize=thread to check the pool with the sanitizer (no reports expected; run it with busy cores around).
// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/cpu_experts.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace b70;
static void fill(uint8_t* b, const NvExpertLayout& L, std::mt19937& rng) {
    for (size_t i = 0; i + 4 <= L.gsc; i += 4) { const uint32_t r = rng(); std::memcpy(b + i, &r, 4); }
    auto sc = [&](size_t off, size_t n) { for (size_t i = 0; i < n; ++i) b[off + i] = uint8_t(0x30 + (rng() & 15)); };
    sc(L.gu_s, size_t(2) * L.I * (L.H / 16)); sc(L.dn_s, size_t(L.H) * (L.I / 16));
    const float g[4] = {1, 1, 1, 0}; std::memcpy(b + L.gsc, g, sizeof g);
}
int main() {
    const int H = 256, I = 64, POOL = 16;
    const NvExpertLayout L = NvExpertLayout::make(H, I);
    std::mt19937 rng(5);
    std::vector<uint8_t> pool(L.bytes * POOL);
    for (int e = 0; e < POOL; ++e) fill(pool.data() + L.bytes * e, L, rng);
    std::vector<float> x(H), y(H);
    for (auto& v : x) v = (float(rng() % 2001) - 1000.0f) * 5e-5f;
    uint64_t want = 0;
    for (int threads : {1, 2, 6, 14, 14}) {
        CpuExperts cx; cx.start(threads);
        std::mt19937 pick(9);
        uint64_t h = 14695981039346656037ull;
        for (int c = 0; c < 4000; ++c) {
            const uint8_t* blk[3]; const float w[3] = {0.1f, 0.2f, 0.3f};
            for (auto& b : blk) b = pool.data() + L.bytes * (pick() % POOL);
            cx.ffn(L, blk, w, 3, x.data(), y.data());
            for (float v : y) { uint32_t u; std::memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ull; }
        }
        if (!want) want = h;
        std::printf("threads %2d: output hash %016llx %s\n", threads, (unsigned long long)h, h == want ? "same" : "DIFFERS");
        if (h != want) return 1;
    }
    std::puts("POOL_STRESS_PASS");
    return 0;
}
