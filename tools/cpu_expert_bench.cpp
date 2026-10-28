// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

// =====================================================================
//  cpu_expert_bench.cpp -- the CPU-expert pool (b70/cpu_experts.hpp) on
//  its own, no GPU (10-27).  Flash-Next-sized NVFP4 experts (H 2560,
//  I 640, 2.7 MB) in a ~1 GB pool in RAM (far past the 24 MB L3), n
//  random experts per call at the decode cadence (the caller spins
//  `gap` us between calls, as it waits for the GPU there), per-call
//  time distribution.
//
//    cpu_expert_bench dot
//        one thread's cpu_nvfp4_dot speed on a P-, an E- and a
//        low-power E-core, row in L1 and rows streamed from RAM
//    cpu_expert_bench ffn <threads|0> <pin 0|1> <rank> <ranks> [calls] [n] [gap us]
//        threads 0 = the engine's choice (the pinned list, or cores/ranks)
//        CPUS=a,b,c overrides the pinned list (CPUS[0] = the caller's)
//    USM=1 (either mode): the pool is sycl::malloc_host memory of the first
//        GPU, as the engine's RAM experts are (needs the GPU device; no
//        kernel runs), instead of plain aligned_alloc memory
// =====================================================================
#include "b70/cpu_experts.hpp"

#include <sycl/sycl.hpp>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;
double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}
void pin_to(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}
void unpin() {
    cpu_set_t s; CPU_ZERO(&s);
    for (int c = 0; c < int(std::thread::hardware_concurrency()); ++c) CPU_SET(c, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

uint8_t* alloc_pool(size_t bytes) {
    if (std::getenv("USM")) {
        static sycl::queue q{sycl::gpu_selector_v};
        uint8_t* p = sycl::malloc_host<uint8_t>(bytes, q);
        std::printf("pool: %.2f GB of sycl::malloc_host memory (%s)\n", double(bytes) / 1e9,
                    q.get_device().get_info<sycl::info::device::name>().c_str());
        return p;
    }
    // 2 MB aligned so huge pages can back it; PAGES=4k|2m asks the kernel for small / huge pages
    const size_t al = size_t(2) << 20, rb = (bytes + al - 1) / al * al;
    uint8_t* p = static_cast<uint8_t*>(std::aligned_alloc(al, rb));
    const char* pg = std::getenv("PAGES");
    if (p && pg) ::madvise(p, rb, std::strcmp(pg, "4k") == 0 ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
    std::printf("pool: %.2f GB of aligned_alloc memory%s%s\n", double(bytes) / 1e9, pg ? ", pages " : "", pg ? pg : "");
    return p;
}

// how much of this process's memory huge pages back (after the pool is filled)
void print_huge() {
    std::FILE* f = std::fopen("/proc/self/smaps_rollup", "r");
    if (!f) return;
    char line[256];
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, "AnonHugePages:", 14) == 0) std::printf("  %s", line);
    std::fclose(f);
}

// Random payload; E4M3 scales 0.5..0.94 (no NaN, no denormals downstream);
// projection scales 1.
void fill_expert(uint8_t* b, const b70::NvExpertLayout& L, std::mt19937& rng) {
    for (size_t i = 0; i + 4 <= L.gsc; i += 4) { const uint32_t r = rng(); std::memcpy(b + i, &r, 4); }
    auto scales = [&](size_t off, size_t n) {
        for (size_t i = 0; i < n; ++i) b[off + i] = uint8_t(0x30 + (rng() & 0x0F));
    };
    scales(L.gu_s, size_t(2) * L.I * (L.H / 16));
    scales(L.dn_s, size_t(L.H) * (L.I / 16));
    const float g[4] = {1.0f, 1.0f, 1.0f, 0.0f};
    std::memcpy(b + L.gsc, g, sizeof g);
}

void report(const char* tag, std::vector<double>& t, double bytes) {
    std::sort(t.begin(), t.end());
    double sum = 0; for (double v : t) sum += v;
    const size_t n = t.size();
    const double med = t[n / 2];
    std::printf("%-28s calls %zu  median %7.1f  p90 %7.1f  p99 %7.1f  max %8.1f  mean %7.1f us"
                "  (%.1f GB/s at the median)\n", tag, n, med, t[n * 9 / 10], t[n * 99 / 100],
                t[n - 1], sum / double(n), bytes / med * 1e-3);
}

int dot_mode() {
    const int H = 2560;
    const b70::NvExpertLayout L = b70::NvExpertLayout::make(H, 640);
    std::mt19937 rng(7);
    std::vector<float> x(H);
    for (auto& v : x) v = (float(rng() % 2001) - 1000.0f) * 5e-5f;
    const int POOL = 384;
    uint8_t* pool = alloc_pool(L.bytes * POOL);
    if (!pool) return 1;
    for (int e = 0; e < POOL; ++e) fill_expert(pool + L.bytes * e, L, rng);
    print_huge();
    const int rows_per_expert = 2 * L.I;                 // gate + up rows, K = H
    for (int cpu : {0, 6, 14}) {
        pin_to(cpu);
        volatile float sink = 0;
        // cached: one gate row over and over
        const uint8_t* b = pool;
        float s = 0;
        for (int r = 0; r < 2000; ++r) s += b70::cpu_nvfp4_dot(b + L.gu_p, b + L.gu_s, x.data(), H);
        auto t0 = clk::now();
        const int REPS = 200000;
        for (int r = 0; r < REPS; ++r) s += b70::cpu_nvfp4_dot(b + L.gu_p, b + L.gu_s, x.data(), H);
        const double tc = us_since(t0);
        // streamed: every gate/up row of every expert in the pool once
        t0 = clk::now();
        for (int e = 0; e < POOL; ++e) {
            const uint8_t* be = pool + L.bytes * e;
            for (int i = 0; i < rows_per_expert; ++i)
                s += b70::cpu_nvfp4_dot(be + L.gu_p + size_t(i) * (H / 2), be + L.gu_s + size_t(i) * (H / 16),
                                        x.data(), H);
        }
        const double ts = us_since(t0);
        sink = s;
        (void)sink;
        const double macs_c = double(REPS) * H, macs_s = double(POOL) * rows_per_expert * H;
        const double bytes_s = double(POOL) * rows_per_expert * (H / 2 + H / 16);
        std::printf("cpu %2d: cached %.2f GMAC/s (%.1f ns a 2560 row) | streamed %.2f GMAC/s, %.1f GB/s\n",
                    cpu, macs_c / tc * 1e-3, tc * 1e3 / REPS, macs_s / ts * 1e-3, bytes_s / ts * 1e-3);
    }
    unpin();
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "dot") == 0) return dot_mode();
    if (argc < 6 || std::strcmp(argv[1], "ffn") != 0) {
        std::fprintf(stderr, "usage: cpu_expert_bench dot | ffn <threads|0> <pin 0|1> <rank> <ranks> "
                             "[calls] [n] [gap us]\n");
        return 2;
    }
    const int threads = std::atoi(argv[2]), pin = std::atoi(argv[3]);
    const int rank = std::atoi(argv[4]), ranks = std::atoi(argv[5]);
    const int calls = argc > 6 ? std::atoi(argv[6]) : 3000;
    const int nper  = argc > 7 ? std::atoi(argv[7]) : 1;
    const int gap   = argc > 8 ? std::atoi(argv[8]) : 600;
    const int H = 2560, I = 640, POOL = 384;
    const b70::NvExpertLayout L = b70::NvExpertLayout::make(H, I);
    std::mt19937 rng(1234 + rank);
    uint8_t* pool = alloc_pool(L.bytes * POOL);
    if (!pool) return 1;
    for (int e = 0; e < POOL; ++e) fill_expert(pool + L.bytes * e, L, rng);
    print_huge();
    std::vector<float> x(H), y(H);
    for (auto& v : x) v = (float(rng() % 2001) - 1000.0f) * 5e-5f;

    // the engine's choice (grimoire.cpp, CPU experts start)
    std::vector<int> cpus;
    if (const char* cl = std::getenv("CPUS")) {
        std::stringstream ss(cl);
        for (std::string p; std::getline(ss, p, ',');) cpus.push_back(std::atoi(p.c_str()));
    } else if (pin) {
        cpus = b70::hybrid_pool_cpus(ranks, rank);
    }
    const int share = std::max(1, int(std::thread::hardware_concurrency()) / ranks);
    const int nt = threads > 0 ? threads : !cpus.empty() ? int(cpus.size()) : ranks > 1 ? share : 0;
    b70::CpuExperts cx;
    cx.start(nt, cpus);
    std::string cl;
    for (size_t c = 1; c < cpus.size() && int(c) < nt; ++c) cl += (cl.empty() ? "" : ",") + std::to_string(cpus[c]);

    std::uniform_int_distribution<int> pick(0, POOL - 1);
    std::vector<double> t;
    t.reserve(size_t(calls));
    double w0 = 0.0;
    for (int c = 0; c < calls + 50; ++c) {
        const uint8_t* blk[16]; float w[16];
        for (int j = 0; j < nper && j < 16; ++j) { blk[j] = pool + L.bytes * size_t(pick(rng)); w[j] = 0.1f; }
        const auto g0 = clk::now();
        while (us_since(g0) < double(gap)) _mm_pause();
        if (c == 50) w0 = cx.wait_us();
        const auto t0 = clk::now();
        cx.ffn(L, blk, w, nper, x.data(), y.data());
        if (c >= 50) t.push_back(us_since(t0));       // the first 50 warm the pool up
    }
    char tag[96];
    std::snprintf(tag, sizeof tag, "rank %d/%d %d thr %s", rank, ranks, cx.threads(),
                  cl.empty() ? "unpinned" : ("pin " + cl).c_str());
    report(tag, t, double(nper) * double(L.bytes));
    std::printf("  the caller waited %.1f us a call for the last worker\n", (cx.wait_us() - w0) / double(calls));
    cx.stop();
    return 0;
}
