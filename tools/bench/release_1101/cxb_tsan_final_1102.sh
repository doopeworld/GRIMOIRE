#!/bin/bash
# CPU only: the pool under the thread sanitizer. Stress: many calls, no gap, four busy cores, then a restart cycle.
set -u
W=/mnt/storage/isos/grimoire-cxpool-1102/tree
pids=()
for c in 7 9; do taskset -c "$c" sh -c "while :; do :; done" & pids+=($!); done
trap "for p in \${pids[@]}; do kill \$p 2>/dev/null; done" EXIT
cat > "$W/bin/cxpool_tsan.cpp" <<SRC
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
SRC
docker run --rm --name grim-cxb-tsan --cpuset-cpus 6-13 --security-opt seccomp=unconfined -v "$W":/grimoire -w /grimoire grimoire-dev:latest bash -lc "
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  icpx -fsycl -g -O2 -std=c++20 -I include -I src bin/cxpool_tsan.cpp src/cpu_experts.cpp -lpthread -o bin/cxpool_plain 2>&1 | tail -3
  echo == plain build, start/stop cycles, thread counts 1 2 6 14 14
  timeout 120 bin/cxpool_plain 2>&1 | tail -8; echo plain rc=\$?
  icpx -fsycl -fsanitize=thread -g -O1 -std=c++20 -I include -I src bin/cxpool_tsan.cpp src/cpu_experts.cpp -lpthread -o bin/cxpool_tsan 2>&1 | tail -5
  echo == thread sanitizer
  TSAN_OPTIONS=halt_on_error=0 timeout 200 setarch -R bin/cxpool_tsan 2>&1 | grep -E \"threads|POOL_STRESS|WARNING: ThreadSanitizer|SUMMARY|FATAL\" | sort | uniq -c | sort -rn | head -24
  echo sanitizer done
" | cut -c1-170
